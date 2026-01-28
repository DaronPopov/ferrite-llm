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
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 0,
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

        // Apply top-p filtering
        let filtered = if self.config.top_p < 1.0 {
            self.top_p_filter(&probs_vec)
        } else {
            probs_vec
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
        let logits_vec: Vec<f32> = logits.to_vec1()?;
        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        Ok(idx as u32)
    }

    fn apply_repetition_penalty(&self, logits: &Tensor, previous_tokens: &[u32]) -> CandleResult<Tensor> {
        let mut logits_vec: Vec<f32> = logits.to_vec1()?;
        let penalty = self.config.repetition_penalty;

        for &token_id in previous_tokens {
            if let Some(logit) = logits_vec.get_mut(token_id as usize) {
                // If logit > 0, divide by penalty; if < 0, multiply by penalty
                *logit = if *logit > 0.0 {
                    *logit / penalty
                } else {
                    *logit * penalty
                };
            }
        }

        Tensor::from_vec(logits_vec, logits.shape(), logits.device())
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
