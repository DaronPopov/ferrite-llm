use std::collections::VecDeque;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use anyhow::Result;
use tch::Tensor;

const DEFAULT_SEQ_LEN: usize = 2048;
const DEFAULT_PACK_PAD: i64 = 0;
const DEFAULT_TEXT_KEY: &str = "text";

#[derive(Clone)]
struct Config {
    dataset: PathBuf,
    format: DatasetFormat,
    text_key: String,
    seq_len: usize,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
    max_batches: usize,
    report_interval: usize,
}

#[derive(Clone, Copy)]
enum DatasetFormat {
    Auto,
    Txt,
    Jsonl,
}

impl DatasetFormat {
    fn parse(s: &str) -> Result<Self> {
        match s {
            "auto" => Ok(Self::Auto),
            "txt" => Ok(Self::Txt),
            "jsonl" => Ok(Self::Jsonl),
            _ => Err(anyhow::anyhow!("invalid format: {s}")),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Txt => "txt",
            Self::Jsonl => "jsonl",
        }
    }
}

fn detect_format(path: &PathBuf) -> DatasetFormat {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
    if ext == "jsonl" { DatasetFormat::Jsonl } else { DatasetFormat::Txt }
}

fn extract_json_value(line: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let idx = line.find(&needle)?;
    let mut i = idx + needle.len();
    let bytes = line.as_bytes();
    while i < bytes.len() && bytes[i] != b':' { i += 1; }
    i += 1;
    while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() { i += 1; }
    if i >= bytes.len() || bytes[i] != b'"' { return None; }
    i += 1;
    let mut out = String::new();
    let mut esc = false;
    while i < bytes.len() {
        let c = bytes[i] as char;
        i += 1;
        if esc { out.push(c); esc = false; continue; }
        if c == '\\' { esc = true; continue; }
        if c == '"' { return Some(out); }
        out.push(c);
    }
    None
}

/// Simple byte-level tokenizer for architecture demonstration.
/// In production, this would be replaced with a real tokenizer (sentencepiece, tiktoken, etc.).
fn tokenize(text: &str) -> Vec<i64> {
    text.bytes().map(|b| b as i64).collect()
}

/// Pack tokenized sequences into fixed-length blocks.
/// Concatenates tokens from multiple samples, splitting at seq_len boundaries.
/// Each block is exactly seq_len tokens, padded if necessary.
pub struct PackedBatcher {
    token_buffer: VecDeque<i64>,
    seq_len: usize,
    batch_size: usize,
    pad_token: i64,
    samples_consumed: usize,
    batches_produced: usize,
    tokens_produced: u64,
    padding_tokens: u64,
}

impl PackedBatcher {
    pub fn new(seq_len: usize, batch_size: usize, pad_token: i64) -> Self {
        Self {
            token_buffer: VecDeque::with_capacity(seq_len * batch_size * 2),
            seq_len,
            batch_size,
            pad_token,
            samples_consumed: 0,
            batches_produced: 0,
            tokens_produced: 0,
            padding_tokens: 0,
        }
    }

    pub fn feed(&mut self, text: &str) {
        let tokens = tokenize(text);
        self.token_buffer.extend(tokens.iter());
        self.samples_consumed += 1;
    }

    pub fn has_batch(&self) -> bool {
        self.token_buffer.len() >= self.seq_len * self.batch_size
    }

    pub fn next_batch(&mut self) -> Option<Tensor> {
        let total_needed = self.seq_len * self.batch_size;
        if self.token_buffer.len() < total_needed {
            return None;
        }

        let mut data = Vec::with_capacity(total_needed);
        for _ in 0..total_needed {
            data.push(self.token_buffer.pop_front().unwrap());
        }

        self.batches_produced += 1;
        self.tokens_produced += total_needed as u64;

        let tensor = Tensor::from_slice(&data).view([self.batch_size as i64, self.seq_len as i64]);
        Some(tensor)
    }

    /// Flush remaining tokens as a padded batch
    pub fn flush(&mut self) -> Option<Tensor> {
        if self.token_buffer.is_empty() {
            return None;
        }

        let total_needed = self.seq_len * self.batch_size;
        let mut data: Vec<i64> = self.token_buffer.drain(..).collect();
        let pad_count = total_needed.saturating_sub(data.len());
        data.resize(total_needed, self.pad_token);

        self.batches_produced += 1;
        self.tokens_produced += (total_needed - pad_count) as u64;
        self.padding_tokens += pad_count as u64;

        let tensor = Tensor::from_slice(&data).view([self.batch_size as i64, self.seq_len as i64]);
        Some(tensor)
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            samples_consumed: self.samples_consumed,
            batches_produced: self.batches_produced,
            tokens_produced: self.tokens_produced,
            padding_tokens: self.padding_tokens,
            packing_efficiency: if self.tokens_produced + self.padding_tokens > 0 {
                self.tokens_produced as f64 / (self.tokens_produced + self.padding_tokens) as f64
            } else {
                1.0
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct PipelineStats {
    pub samples_consumed: usize,
    pub batches_produced: usize,
    pub tokens_produced: u64,
    pub padding_tokens: u64,
    pub packing_efficiency: f64,
}

/// Shuffled sample iterator using Fisher-Yates with a seed
struct ShuffledSamples {
    samples: Vec<String>,
    indices: Vec<usize>,
    cursor: usize,
    seed: u64,
}

impl ShuffledSamples {
    fn new(samples: Vec<String>, seed: u64) -> Self {
        let n = samples.len();
        let mut indices: Vec<usize> = (0..n).collect();
        // Deterministic Fisher-Yates shuffle
        let mut h = seed;
        for i in (1..n).rev() {
            h ^= i as u64;
            h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (h as usize) % (i + 1);
            indices.swap(i, j);
        }
        Self { samples, indices, cursor: 0, seed }
    }

    fn next(&mut self) -> &str {
        if self.cursor >= self.indices.len() {
            // Re-shuffle for next epoch
            self.cursor = 0;
            let n = self.indices.len();
            self.seed = self.seed.wrapping_add(1);
            let mut h = self.seed;
            for i in (1..n).rev() {
                h ^= i as u64;
                h = h.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let j = (h as usize) % (i + 1);
                self.indices.swap(i, j);
            }
        }
        let idx = self.indices[self.cursor];
        self.cursor += 1;
        &self.samples[idx]
    }
}

fn load_samples(path: &PathBuf, fmt: DatasetFormat, key: &str) -> Result<Vec<String>> {
    let fmt = match fmt {
        DatasetFormat::Auto => detect_format(path),
        x => x,
    };
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut samples = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() { continue; }
        match fmt {
            DatasetFormat::Txt => samples.push(trimmed.to_string()),
            DatasetFormat::Jsonl => {
                if let Some(text) = extract_json_value(trimmed, key) {
                    let t = text.trim();
                    if !t.is_empty() { samples.push(t.to_string()); }
                }
            }
            DatasetFormat::Auto => {}
        }
    }
    if samples.is_empty() {
        return Err(anyhow::anyhow!("no usable samples in {}", path.display()));
    }
    Ok(samples)
}

fn parse_args() -> Result<Config> {
    let mut cfg = Config {
        dataset: PathBuf::new(),
        format: DatasetFormat::Auto,
        text_key: DEFAULT_TEXT_KEY.to_string(),
        seq_len: DEFAULT_SEQ_LEN,
        batch_size: 8,
        shuffle: true,
        seed: 42,
        max_batches: 0,
        report_interval: 10,
    };

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--dataset" => cfg.dataset = PathBuf::from(args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?),
            "--format" => cfg.format = DatasetFormat::parse(&args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?)?,
            "--text-key" => cfg.text_key = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?,
            "--seq-len" => cfg.seq_len = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--batch-size" => cfg.batch_size = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--no-shuffle" => cfg.shuffle = false,
            "--seed" => cfg.seed = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--max-batches" => cfg.max_batches = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "--report-interval" => cfg.report_interval = args.next().ok_or_else(|| anyhow::anyhow!("missing val"))?.parse()?,
            "-h" | "--help" => {
                println!("Usage: packed_pipeline.rs --dataset PATH [options]");
                println!("  --seq-len N       Packed sequence length (default: {})", DEFAULT_SEQ_LEN);
                println!("  --batch-size N    Batch size (default: 8)");
                println!("  --no-shuffle      Disable shuffling");
                println!("  --seed N          RNG seed for shuffling");
                println!("  --max-batches N   Stop after N batches (0 = exhaust dataset)");
                std::process::exit(0);
            }
            _ => return Err(anyhow::anyhow!("unknown arg: {arg}")),
        }
    }

    if cfg.dataset.as_os_str().is_empty() {
        return Err(anyhow::anyhow!("--dataset is required"));
    }
    Ok(cfg)
}

fn main() -> Result<()> {
    let cfg = parse_args()?;

    println!("=== Ferrite Packed Dataset Pipeline ===\n");
    println!("  dataset:     {}", cfg.dataset.display());
    println!("  seq_len:     {}", cfg.seq_len);
    println!("  batch_size:  {}", cfg.batch_size);
    println!("  shuffle:     {}", cfg.shuffle);
    println!("  seed:        {}", cfg.seed);
    println!();

    let samples = load_samples(&cfg.dataset, cfg.format, &cfg.text_key)?;
    println!("  raw samples: {}", samples.len());

    let total_chars: usize = samples.iter().map(|s| s.len()).sum();
    let avg_len = total_chars as f64 / samples.len() as f64;
    println!("  total chars: {}", total_chars);
    println!("  avg length:  {:.1}", avg_len);

    let mut batcher = PackedBatcher::new(cfg.seq_len, cfg.batch_size, DEFAULT_PACK_PAD);
    let mut source = ShuffledSamples::new(samples, cfg.seed);

    let t0 = std::time::Instant::now();
    let mut batch_count = 0usize;

    // Feed samples until we have enough batches
    let target_batches = if cfg.max_batches > 0 { cfg.max_batches } else { usize::MAX };

    loop {
        // Feed samples until a batch is ready
        while !batcher.has_batch() {
            let text = source.next();
            batcher.feed(text);

            // Safety: don't loop forever on tiny datasets
            if batcher.stats().samples_consumed > source.samples.len() * 3
                && !batcher.has_batch()
            {
                break;
            }
        }

        match batcher.next_batch() {
            Some(batch) => {
                batch_count += 1;
                if batch_count % cfg.report_interval == 0 {
                    let stats = batcher.stats();
                    println!(
                        "  batch {:>6} | shape={:?} | consumed={} | efficiency={:.4}",
                        batch_count,
                        batch.size(),
                        stats.samples_consumed,
                        stats.packing_efficiency,
                    );
                }
                if batch_count >= target_batches {
                    break;
                }
            }
            None => {
                // Try flushing
                if let Some(batch) = batcher.flush() {
                    batch_count += 1;
                    let stats = batcher.stats();
                    println!(
                        "  batch {:>6} (flush) | shape={:?} | consumed={} | efficiency={:.4}",
                        batch_count,
                        batch.size(),
                        stats.samples_consumed,
                        stats.packing_efficiency,
                    );
                }
                break;
            }
        }
    }

    let wall = t0.elapsed().as_secs_f64();
    let stats = batcher.stats();

    println!("\n  Pipeline Results:");
    println!("    Batches produced:    {}", stats.batches_produced);
    println!("    Samples consumed:    {}", stats.samples_consumed);
    println!("    Tokens produced:     {}", stats.tokens_produced);
    println!("    Padding tokens:      {}", stats.padding_tokens);
    println!("    Packing efficiency:  {:.4}", stats.packing_efficiency);
    println!("    Wall time:           {:.3}s", wall);
    println!("    Throughput:          {:.0} tokens/s", stats.tokens_produced as f64 / wall);

    println!("\nRESULT mode=packed_pipeline");
    println!("RESULT seq_len={}", cfg.seq_len);
    println!("RESULT batch_size={}", cfg.batch_size);
    println!("RESULT batches_produced={}", stats.batches_produced);
    println!("RESULT samples_consumed={}", stats.samples_consumed);
    println!("RESULT tokens_produced={}", stats.tokens_produced);
    println!("RESULT padding_tokens={}", stats.padding_tokens);
    println!("RESULT packing_efficiency={:.9}", stats.packing_efficiency);
    println!("RESULT wall_s={:.6}", wall);

    Ok(())
}
