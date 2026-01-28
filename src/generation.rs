//! Generation configuration and statistics
//!
//! Provides configuration for text generation and TPS tracking.

use std::time::{Duration, Instant};

/// Generation configuration
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum tokens to generate
    pub max_tokens: usize,
    /// Temperature for sampling (0.0 = greedy)
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    pub repetition_penalty: f32,
    /// Stop sequences (in addition to EOS)
    pub stop_sequences: Vec<String>,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            stop_sequences: vec![],
            seed: 42,
        }
    }
}

impl GenerationConfig {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            ..Default::default()
        }
    }

    pub fn creative() -> Self {
        Self {
            temperature: 1.0,
            top_p: 0.95,
            ..Default::default()
        }
    }

    pub fn with_max_tokens(mut self, n: usize) -> Self {
        self.max_tokens = n;
        self
    }

    pub fn with_temperature(mut self, t: f64) -> Self {
        self.temperature = t;
        self
    }

    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

/// Stop condition for generation
#[derive(Debug, Clone)]
pub enum StopCondition {
    /// Stop on EOS token
    Eos(u32),
    /// Stop on any of these token IDs
    TokenIds(Vec<u32>),
    /// Stop on text sequence
    Text(String),
    /// Stop after N tokens
    MaxTokens(usize),
}

impl StopCondition {
    pub fn should_stop(&self, token_id: u32, generated_text: &str, token_count: usize) -> bool {
        match self {
            StopCondition::Eos(eos) => token_id == *eos,
            StopCondition::TokenIds(ids) => ids.contains(&token_id),
            StopCondition::Text(seq) => generated_text.ends_with(seq),
            StopCondition::MaxTokens(max) => token_count >= *max,
        }
    }
}

/// Generation statistics and TPS tracking
#[derive(Debug, Clone)]
pub struct GenerationStats {
    /// Number of prompt tokens
    pub prompt_tokens: usize,
    /// Number of generated tokens
    pub generated_tokens: usize,
    /// Time spent on prefill (prompt processing)
    pub prefill_time: Duration,
    /// Time spent on decode (generation)
    pub decode_time: Duration,
    /// When generation started
    start_time: Option<Instant>,
    /// When decode phase started
    decode_start: Option<Instant>,
}

impl GenerationStats {
    pub fn new(prompt_tokens: usize) -> Self {
        Self {
            prompt_tokens,
            generated_tokens: 0,
            prefill_time: Duration::ZERO,
            decode_time: Duration::ZERO,
            start_time: None,
            decode_start: None,
        }
    }

    /// Start timing
    pub fn start(&mut self) {
        self.start_time = Some(Instant::now());
    }

    /// Mark end of prefill phase
    pub fn end_prefill(&mut self) {
        if let Some(start) = self.start_time {
            self.prefill_time = start.elapsed();
            self.decode_start = Some(Instant::now());
        }
    }

    /// Record a generated token
    pub fn record_token(&mut self) {
        self.generated_tokens += 1;
        if let Some(decode_start) = self.decode_start {
            self.decode_time = decode_start.elapsed();
        }
    }

    /// Tokens per second (decode phase only)
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_time.as_secs_f64() > 0.0 {
            self.generated_tokens as f64 / self.decode_time.as_secs_f64()
        } else {
            0.0
        }
    }

    /// Time to first token (prefill latency)
    pub fn time_to_first_token(&self) -> Duration {
        self.prefill_time
    }

    /// Total generation time
    pub fn total_time(&self) -> Duration {
        self.prefill_time + self.decode_time
    }

    /// Print stats summary
    pub fn print_summary(&self) {
        println!(
            "[Stats] {} tokens in {:.2}s ({:.1} tok/s) | TTFT: {:.0}ms",
            self.generated_tokens,
            self.decode_time.as_secs_f64(),
            self.tokens_per_second(),
            self.prefill_time.as_millis()
        );
    }
}
