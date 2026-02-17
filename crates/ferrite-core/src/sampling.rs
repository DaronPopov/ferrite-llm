//! Token sampling utilities
//!
//! Provides sampling strategies for generation:
//! - Greedy (argmax)
//! - Temperature scaling
//! - Top-p (nucleus) sampling
//! - Top-k sampling

use candle_core::{Tensor, Result as CandleResult, D};

/// Sampler configuration
#[derive(Debug, Clone)]
pub struct SamplerConfig {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.05,
            repetition_penalty: 1.1,
            seed: 42,
        }
    }
}

/// Token sampler
pub struct Sampler {
    config: SamplerConfig,
    rng_state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
        }
    }

    pub fn greedy() -> Self {
        Self::new(SamplerConfig {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        })
    }

    /// Sample a token from logits with repetition penalty
    pub fn sample(&mut self, logits: &Tensor, previous_tokens: &[u32]) -> CandleResult<u32> {
        let logits = logits.squeeze(0)?; // Remove batch dim if present

        // Apply repetition penalty
        let logits = if self.config.repetition_penalty != 1.0 && !previous_tokens.is_empty() {
            self.apply_repetition_penalty(&logits, previous_tokens)?
        } else {
            logits
        };

        // Greedy decoding
        if self.config.temperature == 0.0 {
            return self.argmax(&logits);
        }

        // Apply temperature
        let scaled = (&logits / self.config.temperature)?;

        // Convert to probabilities
        let probs = candle_nn::ops::softmax(&scaled, D::Minus1)?;
        let probs_vec: Vec<f32> = probs.to_vec1()?;

        // Apply min-p filtering (quality gate based on top token)
        let filtered = if self.config.min_p > 0.0 {
            self.min_p_filter(&probs_vec)
        } else {
            probs_vec
        };

        // Apply top-p filtering
        let filtered = if self.config.top_p < 1.0 {
            self.top_p_filter(&filtered)
        } else {
            filtered
        };

        // Apply top-k filtering
        let filtered = if self.config.top_k > 0 {
            self.top_k_filter(&filtered)
        } else {
            filtered
        };

        // Sample from filtered distribution
        self.sample_from_probs(&filtered)
    }

    fn argmax(&self, logits: &Tensor) -> CandleResult<u32> {
        // GPU-native argmax — single scalar download instead of full vocab
        let idx = logits.argmax(D::Minus1)?;
        Ok(idx.to_scalar::<u32>()?)
    }

    fn apply_repetition_penalty(&self, logits: &Tensor, previous_tokens: &[u32]) -> CandleResult<Tensor> {
        let penalty = self.config.repetition_penalty as f64;
        let device = logits.device();
        let vocab_size = logits.dims()[0];

        // Deduplicate token indices and clamp to vocab range
        let mut token_set: Vec<u32> = previous_tokens.to_vec();
        token_set.sort_unstable();
        token_set.dedup();
        token_set.retain(|&t| (t as usize) < vocab_size);

        if token_set.is_empty() {
            return Ok(logits.clone());
        }

        // Build penalty mask on GPU: 1.0 everywhere, penalty at seen positions
        // For positive logits we divide (use 1/penalty), for negative we multiply (use penalty)
        // Strategy: extract values at indices, compute per-token penalty, scatter back
        let indices = Tensor::new(token_set.as_slice(), device)?;
        let selected = logits.index_select(&indices, 0)?;

        // positive → divide by penalty, negative → multiply by penalty
        // equiv: where(selected > 0, selected / penalty, selected * penalty)
        let zeros = Tensor::zeros_like(&selected)?;
        let pos_mask = selected.gt(&zeros)?;  // bool: true where > 0

        let divided = (&selected / penalty)?;
        let multiplied = (&selected * penalty)?;

        let penalized = pos_mask.where_cond(&divided, &multiplied)?;

        // Scatter penalized values back into logits clone
        // index_add with (penalized - selected) is the cleanest GPU approach
        let diff = (&penalized - &selected)?;
        logits.index_add(&indices, &diff, 0)
    }

    fn top_p_filter(&self, probs: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumsum = 0.0;
        let mut filtered = vec![0.0; probs.len()];

        for (idx, prob) in indexed {
            if cumsum < self.config.top_p as f32 {
                filtered[idx] = prob;
                cumsum += prob;
            }
        }

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter_mut().for_each(|p| *p /= sum);
        }

        filtered
    }

    fn top_k_filter(&self, probs: &[f32]) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut filtered = vec![0.0; probs.len()];
        for (idx, prob) in indexed.into_iter().take(self.config.top_k) {
            filtered[idx] = prob;
        }

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter_mut().for_each(|p| *p /= sum);
        }

        filtered
    }

    /// Min-P filtering: keep tokens with prob >= min_p * max_prob
    /// This is more surgical than top-p, scaling threshold based on confidence
    fn min_p_filter(&self, probs: &[f32]) -> Vec<f32> {
        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = self.config.min_p * max_prob;

        let mut filtered: Vec<f32> = probs
            .iter()
            .map(|&p| if p >= threshold { p } else { 0.0 })
            .collect();

        // Renormalize
        let sum: f32 = filtered.iter().sum();
        if sum > 0.0 {
            filtered.iter_mut().for_each(|p| *p /= sum);
        }

        filtered
    }

    fn sample_from_probs(&mut self, probs: &[f32]) -> CandleResult<u32> {
        let r = self.random_f32();

        let mut cumsum = 0.0;
        for (idx, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if r < cumsum {
                return Ok(idx as u32);
            }
        }

        // Fallback to last non-zero probability
        for (idx, &prob) in probs.iter().enumerate().rev() {
            if prob > 0.0 {
                return Ok(idx as u32);
            }
        }

        Ok(0)
    }

    // Simple xorshift64 PRNG
    fn random_f32(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state as f32) / (u64::MAX as f32)
    }
}
