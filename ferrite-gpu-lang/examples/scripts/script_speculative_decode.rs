#![cfg(feature = "torch")]

//! Speculative Parallel Token Generation
//!
//! At each generation step, fork N candidate token computations across N parallel
//! streams simultaneously. Score all candidates, keep the best, discard the rest.
//! The O(1) TLSF allocation makes the fork-and-discard pattern essentially free.
//!
//! What this demonstrates:
//! - N parallel speculative branches running on N separate streams
//! - Each branch computes a full forward pass independently
//! - All branches sync, scores are compared on CPU, best is selected
//! - Discarded branches' memory freed in O(1) via TLSF
//! - Constant VRAM despite massive fork-discard churn

use anyhow::Result;
use aten_ptx::{
    get_fragmentation, init_pytorch_tlsf_ex, num_streams, print_stats, reset_torch_stream,
    set_torch_stream, sync_all_streams,
};
use std::time::Instant;
use tch::{Device, Kind, Tensor};

struct Config {
    hidden: i64,
    vocab_size: i64,
    num_candidates: usize,
    sequence_length: usize,
    streams: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            hidden: 1024,
            vocab_size: 32000,
            num_candidates: 8,
            sequence_length: 128,
            streams: 32,
        }
    }
}

fn parse_args() -> Config {
    let mut cfg = Config::default();
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hidden" => {
                cfg.hidden = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--vocab-size" => {
                cfg.vocab_size = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--num-candidates" => {
                cfg.num_candidates = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--sequence-length" => {
                cfg.sequence_length = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--streams" => {
                cfg.streams = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => {
                i += 1;
            }
        }
    }
    cfg
}

fn main() -> Result<()> {
    let cfg = parse_args();
    let device_id = 0i32;
    let device = Device::Cuda(device_id as usize);

    println!("=== Speculative Parallel Token Generation ===");
    println!(
        "hidden={} vocab={} candidates={} seq_len={} streams={}",
        cfg.hidden, cfg.vocab_size, cfg.num_candidates, cfg.sequence_length, cfg.streams
    );

    // --- Init runtime + TLSF pool ---
    init_pytorch_tlsf_ex(device_id, 0.70, cfg.streams).map_err(|e| anyhow::anyhow!("{}", e))?;

    let active_streams = num_streams();
    println!("PTX-OS streams: {}", active_streams);
    println!("script=speculative_decode");

    let _guard = tch::no_grad_guard();

    // --- Load "model" weights (PERSISTENT) ---
    // Embedding matrix: vocab_size x hidden
    let embedding = Tensor::randn([cfg.vocab_size, cfg.hidden], (Kind::Float, device))
        * (1.0 / (cfg.hidden as f64).sqrt());

    // Two transformer-like projection layers
    let proj1 = Tensor::randn([cfg.hidden, cfg.hidden], (Kind::Float, device))
        * (1.0 / (cfg.hidden as f64).sqrt());
    let proj2 = Tensor::randn([cfg.hidden, cfg.hidden], (Kind::Float, device))
        * (1.0 / (cfg.hidden as f64).sqrt());

    // Output head: project back to vocab for scoring
    let output_head = Tensor::randn([cfg.hidden, cfg.vocab_size], (Kind::Float, device))
        * (1.0 / (cfg.vocab_size as f64).sqrt());

    sync_all_streams();
    println!("model_weights_loaded");

    let frag_after_load = get_fragmentation();

    // --- Initialize sequence with random "prompt" tokens ---
    let mut sequence: Vec<i64> = (0..8).map(|i| (i * 1000) % cfg.vocab_size).collect();
    let prompt_len = sequence.len();

    // --- Speculative decoding loop ---
    let mut total_candidates_evaluated = 0usize;
    let mut top1_accepted = 0usize;
    let mut total_vram_churn_bytes = 0u64;

    let decode_start = Instant::now();

    for step in 0..cfg.sequence_length {
        // Get the current token's embedding as context
        let current_token = *sequence.last().unwrap();
        let context = embedding.get(current_token).unsqueeze(0); // [1, hidden]

        // --- Fork: launch num_candidates parallel branches ---
        let mut candidate_scores: Vec<(usize, f64)> = Vec::with_capacity(cfg.num_candidates);

        for c in 0..cfg.num_candidates {
            let stream_id = c % active_streams;
            set_torch_stream(stream_id);

            // Each candidate proposes a different token
            let candidate_token = ((current_token + c as i64 + 1) * 7919) % cfg.vocab_size;
            let candidate_embed = embedding.get(candidate_token).unsqueeze(0); // [1, hidden]

            // Combined input: context + candidate
            let combined = (&context + &candidate_embed) * 0.5;

            // Forward pass through layers
            let h1 = combined.matmul(&proj1).relu();
            let h2 = h1.matmul(&proj2).relu();

            // Score: project to vocab, get log-prob of candidate token
            let logits = h2.matmul(&output_head); // [1, vocab_size]
            let log_probs = logits.log_softmax(-1, Kind::Float);
            let score = log_probs
                .get(0)
                .get(candidate_token)
                .double_value(&[]);

            candidate_scores.push((c, score));

            // Estimate VRAM churn: each candidate allocates ~3 tensors of [1, hidden]
            total_vram_churn_bytes += (3 * cfg.hidden * 4) as u64;
            // Intermediates (combined, h1, h2, logits, log_probs) drop here — O(1) free
        }

        sync_all_streams();
        reset_torch_stream();

        // --- Select best candidate ---
        candidate_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let (best_idx, _best_score) = candidate_scores[0];

        // The "winner" token
        let winner_token =
            ((current_token + best_idx as i64 + 1) * 7919) % cfg.vocab_size;
        sequence.push(winner_token);

        total_candidates_evaluated += cfg.num_candidates;
        if best_idx == 0 {
            top1_accepted += 1;
        }

        // Progress every 32 steps
        if step % 32 == 0 && step > 0 {
            println!(
                "  step {}/{}: seq_len={} frag={:.6}",
                step,
                cfg.sequence_length,
                sequence.len(),
                get_fragmentation()
            );
        }
    }

    let decode_time = decode_start.elapsed();

    // --- Report ---
    let tokens_per_sec = cfg.sequence_length as f64 / decode_time.as_secs_f64();
    let hit_rate = top1_accepted as f64 / cfg.sequence_length as f64;
    let avg_candidates = total_candidates_evaluated as f64 / cfg.sequence_length as f64;
    let frag_after_decode = get_fragmentation();

    println!("\n--- Results ---");
    println!("RESULT script=speculative_decode");
    println!("RESULT hidden={}", cfg.hidden);
    println!("RESULT vocab_size={}", cfg.vocab_size);
    println!("RESULT num_candidates={}", cfg.num_candidates);
    println!("RESULT sequence_length={}", cfg.sequence_length);
    println!("RESULT streams={}", active_streams);
    println!("RESULT prompt_len={}", prompt_len);
    println!("RESULT generated_tokens={}", sequence.len() - prompt_len);
    println!(
        "RESULT total_time_ms={:.1}",
        decode_time.as_secs_f64() * 1000.0
    );
    println!("RESULT tokens_per_sec={:.1}", tokens_per_sec);
    println!(
        "RESULT avg_candidates_per_step={:.1}",
        avg_candidates
    );
    println!(
        "RESULT total_candidates_evaluated={}",
        total_candidates_evaluated
    );
    println!("RESULT top1_hit_rate={:.4}", hit_rate);
    println!(
        "RESULT vram_churn_mb={:.1}",
        total_vram_churn_bytes as f64 / 1e6
    );
    println!("RESULT fragmentation_after_load={:.6}", frag_after_load);
    println!("RESULT fragmentation_after_decode={:.6}", frag_after_decode);

    println!("\n--- TLSF Pool State ---");
    print_stats();

    println!("\nKey insight: {} candidates forked and discarded per step.", cfg.num_candidates);
    println!(
        "  Total VRAM churn: {:.1}MB across {} steps — yet fragmentation is {:.6}.",
        total_vram_churn_bytes as f64 / 1e6,
        cfg.sequence_length,
        frag_after_decode
    );
    println!("  On vanilla CUDA, this fork-discard pattern would fragment the heap.");
    println!("  With TLSF, each discard is O(1) and the pool stays clean.");

    Ok(())
}
