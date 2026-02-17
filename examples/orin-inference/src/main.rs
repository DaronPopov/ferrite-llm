//! Ferrite Direct Inference on Jetson Orin
//!
//! Bypasses the WASM layer for minimal overhead on embedded.
//! Uses ferriterc's TLSF O(1) memory pool with Orin unified memory.
//! Auto-detects GGUF architecture and uses the correct candle model loader.
//!
//! Usage:
//!   orin-infer [MODEL_NAME]
//!   orin-infer [MODEL_NAME] --bench        # run 15-prompt stability benchmark
//!
//! Examples:
//!   orin-infer                             # defaults to stablelm-zephyr-3b-q4
//!   orin-infer tinyllama-1.1b-q4           # small & fast (llama arch)
//!   orin-infer qwen2-1.5b-q4              # qwen2 arch
//!   orin-infer --list                      # show all available models
//!   orin-infer mistral-7b-q4 --bench       # benchmark with Mistral 7B

use anyhow::{Context, Result};
use candle_core::{quantized::gguf_file, Device, Tensor};
use ferrite_core::{
    ChatSession, ChatSessionConfig, GenerationConfig, InferenceModel,
    Tokenizer,
    registry::{ModelLoader, ChatTemplate},
};
use std::io::{self, Write};
use std::sync::Arc;

// ferriterc TLSF pool
use ptx_runtime::PtxRuntime;
use ptx_sys::GPUHotConfig;

// ═══════════════════════════════════════════════════════════════
// MULTI-ARCHITECTURE GGUF MODEL LOADER
// ═══════════════════════════════════════════════════════════════

enum ModelInner {
    Llama(candle_transformers::models::quantized_llama::ModelWeights),
    Phi3(candle_transformers::models::quantized_phi3::ModelWeights),
    Qwen2(candle_transformers::models::quantized_qwen2::ModelWeights),
    StableLM(candle_transformers::models::quantized_stable_lm::Model),
}

struct DynamicModel {
    inner: ModelInner,
    device: Device,
}

/// Remap GGUF tensor names (llama.cpp format) to HuggingFace format.
/// Used for architectures loaded via VarBuilder (e.g. StableLM) where
/// the model code expects HF-style names but GGUF uses llama.cpp names.
fn gguf_to_hf_tensor_name(name: &str) -> String {
    if name == "token_embd.weight" { return "model.embed_tokens.weight".into(); }
    if name == "output_norm.weight" { return "model.norm.weight".into(); }
    if name == "output_norm.bias" { return "model.norm.bias".into(); }
    if name == "output.weight" { return "lm_head.weight".into(); }

    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot_pos) = rest.find('.') {
            let layer_idx = &rest[..dot_pos];
            let suffix = &rest[dot_pos + 1..];
            let hf_suffix = match suffix {
                "attn_q.weight" => "self_attn.q_proj.weight",
                "attn_k.weight" => "self_attn.k_proj.weight",
                "attn_v.weight" => "self_attn.v_proj.weight",
                "attn_output.weight" => "self_attn.o_proj.weight",
                "attn_q.bias" => "self_attn.q_proj.bias",
                "attn_k.bias" => "self_attn.k_proj.bias",
                "attn_v.bias" => "self_attn.v_proj.bias",
                "attn_output.bias" => "self_attn.o_proj.bias",
                "attn_norm.weight" => "input_layernorm.weight",
                "attn_norm.bias" => "input_layernorm.bias",
                "ffn_gate.weight" => "mlp.gate_proj.weight",
                "ffn_up.weight" => "mlp.up_proj.weight",
                "ffn_down.weight" => "mlp.down_proj.weight",
                "ffn_norm.weight" => "post_attention_layernorm.weight",
                "ffn_norm.bias" => "post_attention_layernorm.bias",
                other => other,
            };
            return format!("model.layers.{}.{}", layer_idx, hf_suffix);
        }
    }
    name.to_string()
}

/// Load a model that uses VarBuilder (e.g. StableLM) from a GGUF file by
/// remapping tensor names from llama.cpp format to HuggingFace format.
/// Writes a remapped GGUF to a temp file on disk (not in memory) to avoid
/// OOM on memory-constrained devices like Jetson Orin with unified memory.
fn load_gguf_via_varbuilder(
    _gguf_path: &std::path::Path,
    content: &gguf_file::Content,
    file: &mut std::fs::File,
    device: &Device,
) -> Result<candle_transformers::quantized_var_builder::VarBuilder> {
    use candle_core::quantized::QTensor;

    println!("[Model] Remapping GGUF tensor names → HuggingFace format...");

    // Load all tensors from GGUF on CPU
    let mut named_tensors: Vec<(String, QTensor)> = Vec::new();
    for gguf_name in content.tensor_infos.keys() {
        let tensor = content.tensor(file, gguf_name, &Device::Cpu)
            .with_context(|| format!("Loading tensor '{}'", gguf_name))?;
        let hf_name = gguf_to_hf_tensor_name(gguf_name);
        named_tensors.push((hf_name, tensor));
    }

    let metadata: Vec<(&str, &gguf_file::Value)> = content.metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    let tensor_refs: Vec<(&str, &QTensor)> = named_tensors
        .iter()
        .map(|(name, t)| (name.as_str(), t))
        .collect();

    // Write remapped GGUF to a temp FILE on disk (saves ~1.6GB unified memory)
    let tmp_path = std::env::temp_dir().join("ferrite_remapped.gguf");
    {
        let mut tmp_file = std::fs::File::create(&tmp_path)
            .context("Creating temp GGUF file")?;
        gguf_file::write(&mut tmp_file, &metadata, &tensor_refs)
            .context("Writing remapped GGUF to disk")?;
    }

    let tmp_size = std::fs::metadata(&tmp_path).map(|m| m.len()).unwrap_or(0);
    println!("[Model] Remapped GGUF: {:.1} MB on disk", tmp_size as f64 / (1024.0 * 1024.0));

    // Drop CPU tensors to free memory BEFORE loading to device
    drop(named_tensors);

    // Load VarBuilder from the temp file (tensors go directly to target device)
    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(&tmp_path, device)
        .context("Loading VarBuilder from remapped GGUF")?;

    // Clean up temp file
    let _ = std::fs::remove_file(&tmp_path);

    Ok(vb)
}

/// Remap metadata keys to llama.* prefix for the llama-compatible fallback loader.
fn remap_metadata_to_llama(content: &mut gguf_file::Content, arch: &str) {
    use candle_core::quantized::gguf_file::Value;

    let prefix = format!("{}.", arch);
    let mut remapped: Vec<(String, Value)> = Vec::new();

    for (key, val) in content.metadata.iter() {
        if let Some(suffix) = key.strip_prefix(&prefix) {
            let llama_key = format!("llama.{}", suffix);
            if !content.metadata.contains_key(&llama_key) {
                let llama_key = if suffix == "attention.layer_norm_epsilon" {
                    "llama.attention.layer_norm_rms_epsilon".to_string()
                } else {
                    llama_key
                };
                remapped.push((llama_key, val.clone()));
            }
        }
    }

    let count = remapped.len();
    for (k, v) in remapped {
        content.metadata.insert(k, v);
    }

    if !content.metadata.contains_key("llama.attention.head_count_kv") {
        if let Some(hc) = content.metadata.get("llama.attention.head_count").cloned() {
            content.metadata.insert("llama.attention.head_count_kv".to_string(), hc);
        }
    }

    // Fix rope.dimension_count for partial-RoPE architectures
    if let (Some(embed_val), Some(heads_val)) = (
        content.metadata.get("llama.embedding_length"),
        content.metadata.get("llama.attention.head_count"),
    ) {
        if let (Ok(embed), Ok(heads)) = (embed_val.to_u32(), heads_val.to_u32()) {
            if heads > 0 {
                let head_dim = embed / heads;
                if let Some(rope_val) = content.metadata.get("llama.rope.dimension_count") {
                    if let Ok(rope_dim) = rope_val.to_u32() {
                        if rope_dim < head_dim {
                            println!("[Model] Fixing rope.dimension_count: {} → {}", rope_dim, head_dim);
                            content.metadata.insert(
                                "llama.rope.dimension_count".to_string(),
                                Value::U32(head_dim),
                            );
                        }
                    }
                }
            }
        }
    }

    if count > 0 {
        println!("[Model] Remapped {} metadata keys from {}* → llama.*", count, prefix);
    }
}

impl DynamicModel {
    fn load(gguf_path: &std::path::Path, device: &Device) -> Result<Self> {
        let mut f = std::fs::File::open(gguf_path)?;
        let mut content = gguf_file::Content::read(&mut f)
            .context("Reading GGUF file")?;

        let arch = content.metadata.get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .map(|s| s.to_string())
            .unwrap_or_else(|| "llama".to_string());

        println!("[Model] Detected GGUF architecture: {}", arch);

        let inner = match arch.as_str() {
            "llama" => {
                let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    content, &mut f, device,
                ).context("Loading llama weights")?;
                ModelInner::Llama(model)
            }
            "phi3" => {
                let model = candle_transformers::models::quantized_phi3::ModelWeights::from_gguf(
                    false, content, &mut f, device,
                ).context("Loading phi3 weights")?;
                ModelInner::Phi3(model)
            }
            "qwen2" => {
                let model = candle_transformers::models::quantized_qwen2::ModelWeights::from_gguf(
                    content, &mut f, device,
                ).context("Loading qwen2 weights")?;
                ModelInner::Qwen2(model)
            }
            "stablelm" => {
                // StableLM uses VarBuilder (LayerNorm + partial RoPE).
                // Remap GGUF tensor names to HF format via GGUF rewrite.
                println!("[Model] Loading StableLM via tensor name remapping...");
                let vb = load_gguf_via_varbuilder(gguf_path, &content, &mut f, device)?;
                let config = candle_transformers::models::stable_lm::Config::stablelm_3b_4e1t(false);
                let model = candle_transformers::models::quantized_stable_lm::Model::new(&config, vb)
                    .context("Creating StableLM model")?;
                ModelInner::StableLM(model)
            }
            other => {
                // Fallback: remap metadata keys and use llama loader.
                println!("[Model] Using llama-compatible loader for '{}' architecture", other);
                remap_metadata_to_llama(&mut content, other);
                let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
                    content, &mut f, device,
                ).context("Loading with llama-compatible fallback")?;
                ModelInner::Llama(model)
            }
        };

        Ok(DynamicModel { inner, device: device.clone() })
    }
}

impl InferenceModel for DynamicModel {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Ok(Tensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?);
        }

        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = match &mut self.inner {
            ModelInner::Llama(m) => m.forward(&input, pos)?,
            ModelInner::Phi3(m) => m.forward(&input, pos)?,
            ModelInner::Qwen2(m) => m.forward(&input, pos)?,
            ModelInner::StableLM(m) => m.forward(&input, pos)?,
        };
        Ok(logits.squeeze(0)?)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = match &mut self.inner {
                ModelInner::Llama(m) => m.forward(&input, pos)?,
                ModelInner::Phi3(m) => m.forward(&input, pos)?,
                ModelInner::Qwen2(m) => m.forward(&input, pos)?,
                ModelInner::StableLM(m) => m.forward(&input, pos)?,
            };
            pos += 1;
        }
        if let Some(&last_token) = tokens.last() {
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = match &mut self.inner {
                ModelInner::Llama(m) => m.forward(&input, pos)?,
                ModelInner::Phi3(m) => m.forward(&input, pos)?,
                ModelInner::Qwen2(m) => m.forward(&input, pos)?,
                ModelInner::StableLM(m) => m.forward(&input, pos)?,
            };
            Ok(logits.squeeze(0)?)
        } else {
            Ok(Tensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?)
        }
    }

    fn clear_cache(&mut self) {}
}

// ═══════════════════════════════════════════════════════════════
// TLSF POOL INIT
// ═══════════════════════════════════════════════════════════════

fn init_tlsf_pool() -> Result<PtxRuntime> {
    println!("[TLSF] Initializing ferriterc memory pool...");

    let mut config = GPUHotConfig::default();
    // Claim the ENTIRE unified memory pool - no fixed size cap.
    // pool_fraction=1.0 dynamically takes all available VRAM.
    config.pool_fraction = 1.0;
    config.fixed_pool_size = 0;
    config.reserve_vram = 0;
    config.prefer_orin_unified_memory = true;
    config.use_managed_pool = true;
    config.enable_pool_health = true;
    config.enable_leak_detection = false;
    config.warning_threshold = 0.95;
    config.max_streams = 16;
    config.quiet_init = false;

    let runtime = PtxRuntime::with_config(0, Some(config))
        .context("Failed to initialize TLSF pool")?;

    runtime.export_for_hook();
    runtime.enable_hooks(false);

    let stats = runtime.tlsf_stats();
    println!("[TLSF] Pool size:  {:.2} GB", stats.total_pool_size as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("[TLSF] Orin unified memory: enabled\n");

    Ok(runtime)
}

// ═══════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .with_target(false)
        .init();

    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--list" || a == "-l") {
        let loader = ModelLoader::new("./models");
        println!("Available models (Orin-friendly marked with *):\n");
        for spec in loader.catalog().list() {
            let fits = match spec.size.as_str() {
                s if s.contains("0.5B") || s.contains("1.1B") || s.contains("1.5B")
                    || s.contains("2B") || s.contains("3B") || s.contains("3.8B") => "*",
                s if s.contains("7B") || s.contains("6.7B") => " ",
                _ => " ",
            };
            println!("  {} {:30} {:6}  {}", fits, spec.name, spec.size, spec.description);
        }
        println!("\n  * = fits on 8GB Orin with TLSF pool");
        return Ok(());
    }

    let model_name = args.iter().skip(1)
        .find(|s| !s.starts_with('-'))
        .map(|s| s.as_str())
        .unwrap_or("stablelm-zephyr-3b-q4");

    println!("============================================");
    println!("  Ferrite Inference - Jetson Orin");
    println!("  with ferriterc TLSF memory pool");
    println!("============================================\n");

    // Init TLSF pool BEFORE any CUDA operations
    let _runtime = init_tlsf_pool()?;

    let device = Device::cuda_if_available(0).context("No compute device")?;
    match &device {
        Device::Cuda(_) => println!("[Device] CUDA (Jetson Orin sm_87) + TLSF pool"),
        Device::Cpu => println!("[Device] CPU fallback"),
        _ => println!("[Device] Unknown"),
    }

    let mem_info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
    if let Some(line) = mem_info.lines().find(|l| l.starts_with("MemAvailable")) {
        println!("[Memory] {}", line);
    }
    println!();

    // Download model
    println!("[Model] Downloading {} from HuggingFace...", model_name);
    let hf_token = std::env::var("HF_TOKEN").ok();
    let loader = ModelLoader::new("./models").with_auth(hf_token.clone());

    let spec = loader.get_spec(model_name)
        .with_context(|| format!("Unknown model: '{}'. Run with --list to see options.", model_name))?
        .clone();

    println!("[Model] {} - {}", spec.description, spec.size);

    let downloaded = loader.download_spec(&spec)
        .context("Failed to download model")?;

    println!("[Model] Weights: {}", downloaded.weights_path.display());
    println!("[Model] Loading into TLSF pool...");

    let load_start = std::time::Instant::now();
    let model = DynamicModel::load(&downloaded.weights_path, &device)
        .context("Failed to load model weights")?;
    println!("[Model] Loaded in {:.1}s", load_start.elapsed().as_secs_f32());

    // Pool stats after load
    let stats = _runtime.tlsf_stats();
    println!("[TLSF] After load: {:.2} GB used / {:.2} GB pool ({:.0}%)\n",
        stats.allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        stats.total_pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        stats.utilization_percent);

    // Load tokenizer
    println!("[Tokenizer] Downloading...");
    let tokenizer = load_tokenizer(&downloaded, hf_token)?;
    println!("[Tokenizer] Ready\n");

    // Chat format from model template
    let chat_format = match spec.chat_template {
        ChatTemplate::Mistral => ferrite_core::ChatFormat::Mistral,
        ChatTemplate::Llama2 | ChatTemplate::Llama3 => ferrite_core::ChatFormat::Llama,
        ChatTemplate::ChatML => ferrite_core::ChatFormat::ChatML,
        ChatTemplate::Phi3 => ferrite_core::ChatFormat::ChatML,
        ChatTemplate::Gemma => ferrite_core::ChatFormat::Gemma,
        ChatTemplate::Zephyr => ferrite_core::ChatFormat::Llama,
        _ => ferrite_core::ChatFormat::Mistral,
    };

    let ctx_len = spec.context_length.min(2048);

    let config = ChatSessionConfig::default()
        .with_chat_format(chat_format)
        .with_context_length(ctx_len)
        .with_generation(
            GenerationConfig::default()
                .with_max_tokens(512)
                .with_temperature(0.0)  // deterministic: always pick top token
        );

    let mut session = ChatSession::new(
        model,
        Arc::new(tokenizer),
        None,  // no system prompt — pure input/output
        config,
    ).context("Failed to create chat session")?;

    let bench_mode = args.iter().any(|a| a == "--bench");

    if bench_mode {
        run_bench(&mut session, &_runtime, model_name)?;
    } else {
        println!("============================================");
        println!("  Chat ready! Model: {}", model_name);
        println!("  Mode: stateless, deterministic (temp=0)");
        println!("  Type /quit to exit, /stats for pool info");
        println!("============================================\n");

        loop {
            print!("You> ");
            io::stdout().flush()?;

            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();

            if input.is_empty() { continue; }
            if input == "/quit" || input == "/exit" {
                println!("Goodbye!");
                break;
            }
            if input == "/stats" {
                let stats = _runtime.tlsf_stats();
                println!("[TLSF] Allocated: {:.2} GB / {:.2} GB ({:.0}%)",
                    stats.allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                    stats.total_pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
                    stats.utilization_percent);
                println!("[TLSF] Fragmentation: {:.4}", stats.fragmentation_ratio);
                continue;
            }

            // Stateless: clear history + KV cache before every turn
            session.clear();

            print!("\nAssistant> ");
            io::stdout().flush()?;

            match session.user_turn(input) {
                Ok(_response) => { println!(); }
                Err(e) => { eprintln!("\n[Error] {}", e); }
            }
        }
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// BENCHMARK MODE (--bench)
// ═══════════════════════════════════════════════════════════════

const BENCH_PROMPTS: &[&str] = &[
    "What is the capital of France?",
    "Explain how a transistor works in two sentences.",
    "Write a Python function that checks if a number is prime.",
    "What are the three laws of thermodynamics?",
    "Describe the difference between TCP and UDP.",
    "What causes the seasons on Earth?",
    "Write a haiku about the ocean.",
    "Explain what a hash table is and why it's useful.",
    "What is the significance of the Turing test?",
    "How does photosynthesis work?",
    "Write a SQL query to find duplicate emails in a users table.",
    "What is the difference between a stack and a queue?",
    "Explain the concept of recursion with a simple example.",
    "What are the main differences between HTTP/1.1 and HTTP/2?",
    "Describe how garbage collection works in modern languages.",
];

struct BenchResult {
    prompt_idx: usize,
    tokens: usize,
    elapsed_secs: f64,
    tps: f64,
    ok: bool,
    pool_util: f32,
    fragmentation: f32,
}

fn run_bench<M: InferenceModel>(
    session: &mut ChatSession<M>,
    runtime: &PtxRuntime,
    model_name: &str,
) -> Result<()> {
    let total_prompts = BENCH_PROMPTS.len();

    println!("============================================");
    println!("  BENCHMARK MODE - {} prompts", total_prompts);
    println!("  Model: {}", model_name);
    println!("  Deterministic (temp=0), stateless");
    println!("============================================\n");

    let mut results: Vec<BenchResult> = Vec::new();
    let bench_start = std::time::Instant::now();

    for (i, prompt) in BENCH_PROMPTS.iter().enumerate() {
        session.clear();

        println!("─── Prompt {}/{} ───", i + 1, total_prompts);
        println!("Q: {}", prompt);
        print!("A: ");
        io::stdout().flush()?;

        let t0 = std::time::Instant::now();
        let result = session.user_turn(prompt);
        let elapsed = t0.elapsed();

        let stats = runtime.tlsf_stats();

        match result {
            Ok(_response) => {
                // Parse token count from the [Stats] line the model printed.
                // We can estimate from the elapsed time. The model prints its own
                // stats, but we also measure externally for consistency.
                // Use a rough estimate: the model's own stats line is authoritative,
                // but we track wall-clock externally.
                let elapsed_secs = elapsed.as_secs_f64();
                // Count tokens from cached_tokens growth (clear -> generate)
                let tokens = session.token_count();
                let tps = if elapsed_secs > 0.0 { tokens as f64 / elapsed_secs } else { 0.0 };

                results.push(BenchResult {
                    prompt_idx: i + 1,
                    tokens,
                    elapsed_secs,
                    tps,
                    ok: true,
                    pool_util: stats.utilization_percent,
                    fragmentation: stats.fragmentation_ratio,
                });
            }
            Err(e) => {
                eprintln!("\n[Error] {}", e);
                results.push(BenchResult {
                    prompt_idx: i + 1,
                    tokens: 0,
                    elapsed_secs: elapsed.as_secs_f64(),
                    tps: 0.0,
                    ok: false,
                    pool_util: stats.utilization_percent,
                    fragmentation: stats.fragmentation_ratio,
                });
            }
        }
        println!();
    }

    let bench_elapsed = bench_start.elapsed();

    // ─── Summary Table ───
    println!("\n╔══════════════════════════════════════════════════════════════════╗");
    println!("║               BENCHMARK RESULTS - {}              ║", model_name);
    println!("╠═══════╦════════╦══════════╦═════════╦══════════╦═══════════════╣");
    println!("║ Run # ║ Tokens ║ Time (s) ║ Tok/s   ║ Pool %   ║ Frag          ║");
    println!("╠═══════╬════════╬══════════╬═════════╬══════════╬═══════════════╣");

    let mut total_tokens = 0usize;
    let mut total_decode_time = 0.0f64;
    let mut tps_values: Vec<f64> = Vec::new();
    let mut failures = 0usize;

    for r in &results {
        let status = if r.ok { " " } else { "!" };
        println!(
            "║ {:>3}{} ║ {:>6} ║ {:>8.2} ║ {:>7.1} ║ {:>7.1}% ║ {:<13.6} ║",
            r.prompt_idx, status, r.tokens, r.elapsed_secs, r.tps,
            r.pool_util, r.fragmentation,
        );
        if r.ok {
            total_tokens += r.tokens;
            total_decode_time += r.elapsed_secs;
            tps_values.push(r.tps);
        } else {
            failures += 1;
        }
    }

    println!("╠═══════╩════════╩══════════╩═════════╩══════════╩═══════════════╣");

    // Compute summary stats
    let avg_tps = if !tps_values.is_empty() {
        tps_values.iter().sum::<f64>() / tps_values.len() as f64
    } else { 0.0 };

    let min_tps = tps_values.iter().copied().fold(f64::INFINITY, f64::min);
    let max_tps = tps_values.iter().copied().fold(0.0f64, f64::max);

    let stddev = if tps_values.len() > 1 {
        let mean = avg_tps;
        let variance = tps_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (tps_values.len() - 1) as f64;
        variance.sqrt()
    } else { 0.0 };

    let final_stats = runtime.tlsf_stats();

    println!("║                                                                ║");
    println!("║  Total tokens:   {:>6}   in {:.1}s wall-clock                  ║", total_tokens, bench_elapsed.as_secs_f64());
    println!("║  Avg tok/s:      {:>7.1}   (decode throughput)                 ║", avg_tps);
    println!("║  Min tok/s:      {:>7.1}                                       ║", if min_tps.is_finite() { min_tps } else { 0.0 });
    println!("║  Max tok/s:      {:>7.1}                                       ║", max_tps);
    println!("║  Stddev:         {:>7.2}   ({:.1}% variation)                  ║", stddev, if avg_tps > 0.0 { stddev / avg_tps * 100.0 } else { 0.0 });
    println!("║  Failures:       {:>3}/{}                                        ║", failures, total_prompts);
    println!("║                                                                ║");
    println!("║  TLSF pool:      {:.2} GB / {:.2} GB ({:.0}%)                   ║",
        final_stats.allocated_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        final_stats.total_pool_size as f64 / (1024.0 * 1024.0 * 1024.0),
        final_stats.utilization_percent);
    println!("║  Fragmentation:  {:.6}                                       ║", final_stats.fragmentation_ratio);
    println!("║  Peak memory:    {:.2} GB                                     ║",
        final_stats.peak_allocated as f64 / (1024.0 * 1024.0 * 1024.0));
    println!("╚════════════════════════════════════════════════════════════════╝");

    // Stability verdict
    let stable = failures == 0 && stddev / avg_tps.max(0.001) < 0.15;
    if stable {
        println!("\n  VERDICT: STABLE  ({} prompts, <15% tok/s variation)", total_prompts);
    } else if failures > 0 {
        println!("\n  VERDICT: UNSTABLE  ({} failures in {} prompts)", failures, total_prompts);
    } else {
        println!("\n  VERDICT: VARIABLE  (>{:.0}% tok/s variation across {} prompts)",
            stddev / avg_tps * 100.0, total_prompts);
    }

    Ok(())
}

fn load_tokenizer(
    downloaded: &ferrite_core::registry::DownloadedModel,
    auth_token: Option<String>,
) -> Result<Tokenizer> {
    if downloaded.tokenizer_path.join("tokenizer.json").exists() {
        return Tokenizer::from_dir(&downloaded.tokenizer_path)
            .context("Loading tokenizer from local dir");
    }

    let tokenizer_repo = match &downloaded.spec.tokenizer {
        ferrite_core::registry::TokenizerSource::HuggingFace { repo } => repo.clone(),
        ferrite_core::registry::TokenizerSource::SameAsModel => {
            match &downloaded.spec.source {
                ferrite_core::registry::ModelSource::HuggingFace { repo, .. } => repo.clone(),
                _ => anyhow::bail!("Cannot determine tokenizer repo"),
            }
        }
        ferrite_core::registry::TokenizerSource::Local { path } => {
            return Tokenizer::from_dir(path).context("Loading local tokenizer");
        }
    };

    println!("[Tokenizer] Downloading from {}", tokenizer_repo);

    let mut api_builder = hf_hub::api::sync::ApiBuilder::new();
    if let Some(t) = auth_token {
        api_builder = api_builder.with_token(Some(t));
    }
    let api = api_builder.build()?;
    let repo = api.model(tokenizer_repo);

    let tokenizer_file = repo.get("tokenizer.json")?;
    let _ = repo.get("tokenizer_config.json").ok();

    let model_dir = tokenizer_file.parent()
        .context("No parent directory for tokenizer")?
        .to_path_buf();

    Tokenizer::from_dir(&model_dir).context("Loading downloaded tokenizer")
}
