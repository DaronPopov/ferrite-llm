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
    /// Min-p sampling threshold (0.0 = disabled, typical 0.05-0.1)
    pub min_p: f32,
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
            min_p: 0.05,
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

/// Trait for models that can perform streaming inference
pub trait InferenceModel {
    /// Perform forward pass for one or more tokens at a given position.
    /// Should return logits for all input tokens: [batch, seq_len, vocab_size] or [seq_len, vocab_size].
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<candle_core::Tensor, Box<dyn std::error::Error>>;
    /// Prefill the model with multiple tokens (prompt processing)
    fn prefill(&mut self, tokens: &[u32]) -> Result<candle_core::Tensor, Box<dyn std::error::Error>>;
    /// Clear the KV cache (for sliding window). Default is no-op for models without explicit cache.
    fn clear_cache(&mut self) {}
}

/// A high-level streaming inference engine
pub struct StreamingInference<'a, M: InferenceModel> {
    model: &'a mut M,
    sampler: crate::Sampler,
    config: GenerationConfig,
    stats: GenerationStats,
    decoder: crate::tokenizer::StreamDecoder,
    pos: usize,
    next_token: Option<u32>,
    eos_token: u32,
    all_tokens: Vec<u32>,
    finished: bool,
    /// Maximum context length (enables sliding window when set)
    context_length: Option<usize>,
}

impl<'a, M: InferenceModel> StreamingInference<'a, M> {
    /// Create a new streaming inference session
    pub fn new(
        model: &'a mut M,
        tokenizer: &'a crate::Tokenizer,
        prompt: &str,
        config: GenerationConfig,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encoding = tokenizer.encode(prompt)?;
        let tokens = encoding.ids;
        let eos_token = tokenizer.eos_token_id().unwrap_or(2);
        
        let mut stats = GenerationStats::new(tokens.len());
        stats.start();

        // Prefill phase
        let logits = model.prefill(&tokens)?;
        stats.end_prefill();

        let mut sampler = crate::Sampler::new(crate::SamplerConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: config.min_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        });

        // Sample first token
        let next_token = sampler.sample(&logits, &tokens)?;
        
        let decoder = tokenizer.decode_stream(&tokens, true);
        let pos = tokens.len();

        Ok(Self {
            model,
            sampler,
            config,
            stats,
            decoder,
            pos,
            next_token: Some(next_token),
            eos_token,
            all_tokens: tokens,
            finished: false,
            context_length: None,
        })
    }

    /// Set context length to enable sliding window KV-cache
    /// When tokens exceed this limit, oldest 20% are dropped and cache is rebuilt
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = Some(context_length);
        self
    }

    /// Perform sliding window: drop oldest 20% of tokens and rebuild cache
    fn slide_window(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let keep_ratio = 0.8;
        let keep_count = (self.all_tokens.len() as f32 * keep_ratio) as usize;
        let drop_count = self.all_tokens.len() - keep_count;

        // Keep only the most recent tokens
        self.all_tokens = self.all_tokens.split_off(drop_count);

        // Clear and rebuild cache
        self.model.clear_cache();
        let _ = self.model.prefill(&self.all_tokens)?;

        // Reset position to end of kept tokens
        self.pos = self.all_tokens.len();

        Ok(())
    }

    /// Generate the next token and return its text
    pub fn next(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if self.finished {
            return Ok(None);
        }

        let current_token = match self.next_token {
            Some(t) => t,
            None => return Ok(None),
        };

        if current_token == self.eos_token || self.stats.generated_tokens >= self.config.max_tokens {
            self.finished = true;
            return Ok(self.decoder.flush()?);
        }

        // Add token to history
        self.all_tokens.push(current_token);
        self.stats.record_token();

        // Sliding window: if approaching context limit, drop oldest tokens
        if let Some(max_ctx) = self.context_length {
            if self.pos >= max_ctx.saturating_sub(1) {
                self.slide_window()?;
            }
        }

        // Get text for current token
        let text = self.decoder.step(current_token)?;

        // Prepare for next token
        let logits = self.model.forward(&[current_token], self.pos)?;
        self.pos += 1;

        // Sample next token with repetition penalty
        self.next_token = Some(self.sampler.sample(&logits, &self.all_tokens)?);

        Ok(text.or(Some(String::new()))) // Return empty string if no full text yet
    }

    /// Get current generation statistics
    pub fn stats(&self) -> &GenerationStats {
        &self.stats
    }

    /// Check if complete
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// A high-level speculative decoding engine
pub struct SpeculativeInference<'a, T: InferenceModel, D: InferenceModel> {
    target: &'a mut T,
    draft: &'a mut D,
    sampler: crate::Sampler,
    config: GenerationConfig,
    stats: GenerationStats,
    decoder: crate::tokenizer::StreamDecoder,
    pos: usize,
    next_token: Option<u32>,
    eos_token: u32,
    all_tokens: Vec<u32>,
    finished: bool,
    draft_k: usize,
    /// Maximum context length (enables sliding window when set)
    context_length: Option<usize>,
}

impl<'a, T: InferenceModel, D: InferenceModel> SpeculativeInference<'a, T, D> {
    /// Create a new speculative inference session
    pub fn new(
        target: &'a mut T,
        draft: &'a mut D,
        tokenizer: &'a crate::Tokenizer,
        prompt: &str,
        config: GenerationConfig,
        draft_k: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let encoding = tokenizer.encode(prompt)?;
        let tokens = encoding.ids;
        let eos_token = tokenizer.eos_token_id().unwrap_or(2);
        
        let mut stats = GenerationStats::new(tokens.len());
        stats.start();

        // Prefill both models
        let target_logits = target.prefill(&tokens)?;
        let _ = draft.prefill(&tokens)?; // We don't need draft's initial logits, target's are authoritative
        
        stats.end_prefill();

        let mut sampler = crate::Sampler::new(crate::SamplerConfig {
            temperature: config.temperature,
            top_p: config.top_p,
            top_k: config.top_k,
            min_p: config.min_p,
            repetition_penalty: config.repetition_penalty,
            seed: config.seed,
        });

        // Sample first token from target logits
        let next_token = sampler.sample(&target_logits, &tokens)?;
        
        let decoder = tokenizer.decode_stream(&tokens, true);
        let pos = tokens.len();

        Ok(Self {
            target,
            draft,
            sampler,
            config,
            stats,
            decoder,
            pos,
            next_token: Some(next_token),
            eos_token,
            all_tokens: tokens,
            finished: false,
            draft_k,
            context_length: None,
        })
    }

    /// Set context length to enable sliding window KV-cache
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = Some(context_length);
        self
    }

    /// Perform sliding window: drop oldest 20% of tokens and rebuild cache
    fn slide_window(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let keep_ratio = 0.8;
        let keep_count = (self.all_tokens.len() as f32 * keep_ratio) as usize;
        let drop_count = self.all_tokens.len() - keep_count;

        self.all_tokens = self.all_tokens.split_off(drop_count);

        // Clear and rebuild both caches
        self.target.clear_cache();
        self.draft.clear_cache();
        let _ = self.target.prefill(&self.all_tokens)?;
        let _ = self.draft.prefill(&self.all_tokens)?;

        self.pos = self.all_tokens.len();
        Ok(())
    }

    /// Generate the next batch of tokens and return their text
    /// This might return multiple tokens in a single call.
    pub fn next(&mut self) -> Result<Option<String>, Box<dyn std::error::Error>> {
        if self.finished {
            return Ok(None);
        }

        let first_token = match self.next_token {
            Some(t) => t,
            None => return Ok(None),
        };

        if first_token == self.eos_token || self.stats.generated_tokens >= self.config.max_tokens {
            self.finished = true;
            return Ok(self.decoder.flush()?);
        }

        // Sliding window check before drafting
        if let Some(max_ctx) = self.context_length {
            if self.pos + self.draft_k >= max_ctx.saturating_sub(1) {
                self.slide_window()?;
            }
        }

        // 1. Draft K new tokens
        let mut draft_tokens = vec![first_token];
        let mut current_pos = self.pos;
        
        for _ in 0..self.draft_k {
            let last = *draft_tokens.last().unwrap();
            if last == self.eos_token { break; }
            
            let logits = self.draft.forward(&[last], current_pos)?;
            let sampled = self.sampler.sample(&logits, &[])?; // No penalty for draft Scout
            draft_tokens.push(sampled);
            current_pos += 1;
        }

        // 2. Verify all draft tokens in a single target forward pass
        // draft_tokens: [T_start, D1, D2, ..., Dk]
        let target_logits = self.target.forward(&draft_tokens, self.pos)?;
        
        // target_logits is [seq_len, vocab_size]
        // L0 predicts D1, L1 predicts D2, ..., Lk predicts bonus token
        let target_logits_vec = target_logits.chunk(draft_tokens.len(), 0)?;
        
        let mut accepted_text = String::new();
        let mut accepted_count = 0;
        let mut last_verified_token = first_token;

        for (i, draft_token) in draft_tokens.iter().enumerate().skip(1) {
            // Sample from target to see what it wanted at this position
            let target_token = self.sampler.sample(&target_logits_vec[i-1], &self.all_tokens)?;
            
            // Record and decode the verified token (the one from previous step)
            self.all_tokens.push(last_verified_token);
            self.stats.record_token();
            if let Some(t) = self.decoder.step(last_verified_token)? {
                accepted_text.push_str(&t);
            }
            accepted_count += 1;
            
            if target_token == *draft_token {
                // Correct guess! Keep going
                last_verified_token = target_token;
            } else {
                // Wrong guess! This target_token is the correction
                last_verified_token = target_token;
                break;
            }
        }

        // If we reached the end of draft successfully, the last target logit is our next "first_token"
        // and we might have one more verified token to sample.
        if accepted_count == draft_tokens.len() - 1 {
            let bonus_token = self.sampler.sample(target_logits_vec.last().unwrap(), &self.all_tokens)?;
            // The last_verified_token (Dk) is already set. 
            // We still need to record Dk and set bonus_token as next_token.
            self.all_tokens.push(last_verified_token);
            self.stats.record_token();
            if let Some(t) = self.decoder.step(last_verified_token)? {
                accepted_text.push_str(&t);
            }
            self.next_token = Some(bonus_token);
            self.pos += accepted_count + 1;
        } else {
            // We broke early. last_verified_token is the correction.
            self.next_token = Some(last_verified_token);
            self.pos += accepted_count;
        }

        // Update draft model position to match target
        let _ = self.draft.forward(&[], self.pos); // Sync draft's KV cache if needed (some models do)

        Ok(Some(accepted_text))
    }

    pub fn stats(&self) -> &GenerationStats { &self.stats }
    pub fn is_finished(&self) -> bool { self.finished }
}
