//! ChatSession - Incremental KV-Cached Multi-Turn Chat
//!
//! A harness that properly maintains the KV cache across conversation turns.
//! Unlike naive approaches that re-encode the full conversation each turn,
//! this stores generated tokens directly and encodes only new user messages
//! incrementally.
//!
//! # Why This Matters
//!
//! BPE tokenization is non-deterministic at boundaries - re-encoding a conversation
//! produces different tokens than the original, causing KV cache misses.
//! This module solves that by:
//! 1. Encoding each user turn incrementally (just new message + formatting)
//! 2. Storing generated response tokens directly (never re-encode them)
//! 3. KV cache position always equals `cached_tokens.len()`
//!
//! # Example
//!
//! ```ignore
//! use ferrite::{ChatSession, ChatSessionConfig, InferenceModel};
//!
//! // Create session with system prompt
//! let mut session = ChatSession::new(
//!     model,
//!     tokenizer,
//!     Some("You are a helpful assistant."),
//!     config,
//! )?;
//!
//! // Multi-turn conversation with proper caching
//! let response = session.user_turn("Hello, my name is Daron")?;
//! let response = session.user_turn("What's my name?")?;  // Uses cached context!
//!
//! // Streaming API
//! for chunk in session.user_turn_streaming("Tell me a story")? {
//!     print!("{}", chunk?);
//! }
//! ```

use crate::generation::{GenerationConfig, InferenceModel};
use crate::models::ChatFormat;
use crate::sampling::{Sampler, SamplerConfig};
use crate::tokenizer::{StreamDecoder, Tokenizer};
use std::sync::Arc;
use std::time::Instant;

/// Configuration for a chat session
#[derive(Debug, Clone)]
pub struct ChatSessionConfig {
    /// Maximum tokens before sliding window kicks in
    pub context_length: usize,
    /// Ratio of tokens to keep when sliding (0.8 = keep 80%)
    pub keep_ratio: f32,
    /// Generation parameters
    pub generation: GenerationConfig,
    /// Chat format for message formatting
    pub chat_format: ChatFormat,
}

impl Default for ChatSessionConfig {
    fn default() -> Self {
        Self {
            context_length: 32768,
            keep_ratio: 0.8,
            generation: GenerationConfig::default(),
            chat_format: ChatFormat::Mistral,
        }
    }
}

impl ChatSessionConfig {
    /// Create config for Mistral models
    pub fn mistral() -> Self {
        Self {
            context_length: 32768,
            keep_ratio: 0.8,
            generation: GenerationConfig::default(),
            chat_format: ChatFormat::Mistral,
        }
    }

    /// Create config for Llama models
    pub fn llama(context_length: usize) -> Self {
        Self {
            context_length,
            keep_ratio: 0.8,
            generation: GenerationConfig::default(),
            chat_format: ChatFormat::Llama,
        }
    }

    /// Create config for ChatML models (Qwen, etc.)
    pub fn chatml(context_length: usize) -> Self {
        Self {
            context_length,
            keep_ratio: 0.8,
            generation: GenerationConfig::default(),
            chat_format: ChatFormat::ChatML,
        }
    }

    /// Builder: set context length
    pub fn with_context_length(mut self, context_length: usize) -> Self {
        self.context_length = context_length;
        self
    }

    /// Builder: set keep ratio for sliding window
    pub fn with_keep_ratio(mut self, keep_ratio: f32) -> Self {
        self.keep_ratio = keep_ratio;
        self
    }

    /// Builder: set generation config
    pub fn with_generation(mut self, generation: GenerationConfig) -> Self {
        self.generation = generation;
        self
    }

    /// Builder: set chat format
    pub fn with_chat_format(mut self, chat_format: ChatFormat) -> Self {
        self.chat_format = chat_format;
        self
    }
}

/// Tracks a message and its token range in cached_tokens
#[derive(Debug, Clone)]
pub struct CachedMessage {
    /// Role of the message
    pub role: MessageRole,
    /// Start index in cached_tokens (inclusive)
    pub start_idx: usize,
    /// End index in cached_tokens (exclusive)
    pub end_idx: usize,
}

/// Message role
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    System,
    User,
    Assistant,
}

/// Incremental KV-cached multi-turn chat session
///
/// The key insight: **never re-encode previously generated tokens**.
/// - Encode each user turn incrementally (just new message + formatting)
/// - Store generated response tokens directly (never re-encode them)
/// - KV cache position always equals `cached_tokens.len()`
pub struct ChatSession<M: InferenceModel> {
    /// The underlying model
    model: M,
    /// Tokenizer for encoding/decoding
    tokenizer: Arc<Tokenizer>,
    /// Session configuration
    config: ChatSessionConfig,
    /// THE SOURCE OF TRUTH - actual tokens in KV cache
    cached_tokens: Vec<u32>,
    /// High-level tracking with token ranges
    messages: Vec<CachedMessage>,
    /// Token sampler
    sampler: Sampler,
    /// Stop tokens (EOS + format-specific like <|im_end|>)
    stop_tokens: Vec<u32>,
    /// Stop strings checked against decoded text (fallback when tokenizer lacks special tokens)
    stop_strings: Vec<String>,
    /// System prompt (if any)
    system_prompt: Option<String>,
    /// Whether system prompt has been encoded
    system_encoded: bool,
}

impl<M: InferenceModel> ChatSession<M> {
    /// Create a new chat session
    ///
    /// # Arguments
    /// * `model` - The inference model (must implement InferenceModel)
    /// * `tokenizer` - The tokenizer
    /// * `system_prompt` - Optional system prompt
    /// * `config` - Session configuration
    pub fn new(
        model: M,
        tokenizer: Arc<Tokenizer>,
        system_prompt: Option<&str>,
        config: ChatSessionConfig,
    ) -> Result<Self, ChatSessionError> {
        let mut stop_tokens = Vec::new();
        if let Some(eos) = tokenizer.eos_token_id() {
            stop_tokens.push(eos);
        } else {
            stop_tokens.push(2); // fallback
        }
        // ChatML models must also stop on <|im_end|>
        let mut stop_strings = Vec::new();
        if matches!(config.chat_format, ChatFormat::ChatML) {
            if let Some(id) = tokenizer.token_to_id("<|im_end|>") {
                if !stop_tokens.contains(&id) {
                    stop_tokens.push(id);
                }
            } else {
                // Tokenizer doesn't have <|im_end|> — fall back to string matching
                stop_strings.push("<|im_end|>".to_string());
            }
        }

        let sampler = Sampler::new(SamplerConfig {
            temperature: config.generation.temperature,
            top_p: config.generation.top_p,
            top_k: config.generation.top_k,
            min_p: config.generation.min_p,
            repetition_penalty: config.generation.repetition_penalty,
            seed: config.generation.seed,
        });

        Ok(Self {
            model,
            tokenizer,
            config,
            cached_tokens: Vec::new(),
            messages: Vec::new(),
            sampler,
            stop_tokens,
            stop_strings,
            system_prompt: system_prompt.map(|s| s.to_string()),
            system_encoded: false,
        })
    }

    /// Process a user message and generate a response (blocking)
    ///
    /// This is the main API for multi-turn chat. It:
    /// 1. Encodes the system prompt (first time only)
    /// 2. Encodes the user message incrementally
    /// 3. Generates the response, storing tokens directly
    /// 4. Returns the response text
    pub fn user_turn(&mut self, content: &str) -> Result<String, ChatSessionError> {
        // Encode system prompt if needed (first turn only)
        if !self.system_encoded {
            self.encode_system_prompt()?;
        }

        // Check if we need to slide the window before encoding
        let estimated_tokens = self.estimate_new_tokens(content);
        if self.cached_tokens.len() + estimated_tokens + self.config.generation.max_tokens
            > self.config.context_length
        {
            self.slide_window()?;
        }

        // Encode the new user message
        let user_start = self.cached_tokens.len();
        self.encode_user_turn(content)?;
        let user_end = self.cached_tokens.len();

        // Track the user message
        self.messages.push(CachedMessage {
            role: MessageRole::User,
            start_idx: user_start,
            end_idx: user_end,
        });

        // Generate response
        let assistant_start = self.cached_tokens.len();
        let response = self.generate_response()?;
        let assistant_end = self.cached_tokens.len();

        // Track the assistant message
        self.messages.push(CachedMessage {
            role: MessageRole::Assistant,
            start_idx: assistant_start,
            end_idx: assistant_end,
        });

        Ok(response)
    }

    /// Process a user message and return a streaming iterator
    ///
    /// Use this for real-time output:
    /// ```ignore
    /// for chunk in session.user_turn_streaming("Hello")? {
    ///     print!("{}", chunk?);
    ///     io::stdout().flush()?;
    /// }
    /// ```
    pub fn user_turn_streaming(
        &mut self,
        content: &str,
    ) -> Result<StreamingResponse<'_, M>, ChatSessionError> {
        // Encode system prompt if needed
        if !self.system_encoded {
            self.encode_system_prompt()?;
        }

        // Check if we need to slide
        let estimated_tokens = self.estimate_new_tokens(content);
        if self.cached_tokens.len() + estimated_tokens + self.config.generation.max_tokens
            > self.config.context_length
        {
            self.slide_window()?;
        }

        // Encode user message
        let user_start = self.cached_tokens.len();
        self.encode_user_turn(content)?;
        let user_end = self.cached_tokens.len();

        self.messages.push(CachedMessage {
            role: MessageRole::User,
            start_idx: user_start,
            end_idx: user_end,
        });

        // Track assistant message start
        let assistant_start = self.cached_tokens.len();

        // Get initial logits for first token sampling
        let last_token = *self.cached_tokens.last().unwrap();
        let pos = self.cached_tokens.len() - 1;
        let logits = self
            .model
            .forward(&[last_token], pos)
            .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;

        let first_token = self
            .sampler
            .sample(&logits, &self.cached_tokens)
            .map_err(|e| ChatSessionError::SamplingError(e.to_string()))?;

        // Create streaming decoder
        let decoder = self.tokenizer.decode_stream(&self.cached_tokens, true);

        Ok(StreamingResponse {
            session: self,
            decoder,
            next_token: Some(first_token),
            assistant_start,
            generated_count: 0,
            finished: false,
            decoded_buf: String::new(),
        })
    }

    /// Get the current token count in the KV cache
    pub fn token_count(&self) -> usize {
        self.cached_tokens.len()
    }

    /// Get remaining capacity before context limit
    pub fn remaining_capacity(&self) -> usize {
        self.config.context_length.saturating_sub(self.cached_tokens.len())
    }

    /// Clear the conversation and reset KV cache
    pub fn clear(&mut self) {
        self.cached_tokens.clear();
        self.messages.clear();
        self.system_encoded = false;
        self.model.clear_cache();
    }

    /// Get the configuration
    pub fn config(&self) -> &ChatSessionConfig {
        &self.config
    }

    /// Update generation parameters for subsequent turns.
    pub fn set_generation_config(&mut self, generation: GenerationConfig) {
        self.sampler = Sampler::new(SamplerConfig {
            temperature: generation.temperature,
            top_p: generation.top_p,
            top_k: generation.top_k,
            min_p: generation.min_p,
            repetition_penalty: generation.repetition_penalty,
            seed: generation.seed,
        });
        self.config.generation = generation;
    }

    /// Get the cached tokens (for debugging/inspection)
    pub fn cached_tokens(&self) -> &[u32] {
        &self.cached_tokens
    }

    /// Get the message history
    pub fn messages(&self) -> &[CachedMessage] {
        &self.messages
    }

    // === Private Implementation ===

    /// Encode the system prompt (called once on first turn)
    fn encode_system_prompt(&mut self) -> Result<(), ChatSessionError> {
        if self.system_encoded {
            return Ok(());
        }

        if let Some(ref system) = self.system_prompt {
            let formatted = self.format_system_prompt(system);
            let encoding = self
                .tokenizer
                .encode(&formatted)
                .map_err(|e| ChatSessionError::TokenizerError(e.to_string()))?;

            let start_idx = self.cached_tokens.len();

            // Prefill the system tokens
            for &token in &encoding.ids {
                let pos = self.cached_tokens.len();
                let _ = self
                    .model
                    .forward(&[token], pos)
                    .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;
                self.cached_tokens.push(token);
            }

            let end_idx = self.cached_tokens.len();

            self.messages.push(CachedMessage {
                role: MessageRole::System,
                start_idx,
                end_idx,
            });
        }

        self.system_encoded = true;
        Ok(())
    }

    /// Encode a user turn incrementally
    fn encode_user_turn(&mut self, content: &str) -> Result<(), ChatSessionError> {
        let formatted = self.format_user_turn(content);
        let encoding = self
            .tokenizer
            .encode(&formatted)
            .map_err(|e| ChatSessionError::TokenizerError(e.to_string()))?;

        let new_tokens = &encoding.ids;
        let new_count = new_tokens.len();
        let cached_count = self.cached_tokens.len();

        println!(
            "[Generate] Prompt: {} tokens ({} cached, {} new)",
            cached_count + new_count,
            cached_count,
            new_count
        );

        // Process each new token
        for &token in new_tokens {
            let pos = self.cached_tokens.len();
            let _ = self
                .model
                .forward(&[token], pos)
                .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;
            self.cached_tokens.push(token);
        }

        Ok(())
    }

    /// Generate response tokens (blocking)
    fn generate_response(&mut self) -> Result<String, ChatSessionError> {
        let decode_start = Instant::now();
        let mut generated_tokens = 0usize;

        // Get initial logits
        let last_token = *self.cached_tokens.last().unwrap();
        let pos = self.cached_tokens.len() - 1;
        let logits = self
            .model
            .forward(&[last_token], pos)
            .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;

        let mut next_token = self
            .sampler
            .sample(&logits, &self.cached_tokens)
            .map_err(|e| ChatSessionError::SamplingError(e.to_string()))?;

        // Setup decoder for response
        let mut response_tokens = vec![next_token];
        let mut decoder = self.tokenizer.decode_stream(&self.cached_tokens, true);
        let mut decoded_text = String::new();
        let has_stop_strings = !self.stop_strings.is_empty();

        // Output first token
        if let Ok(Some(text)) = decoder.step(next_token) {
            if has_stop_strings {
                decoded_text.push_str(&text);
            }
            print!("{}", text);
            std::io::Write::flush(&mut std::io::stdout()).ok();
        }

        self.cached_tokens.push(next_token);
        generated_tokens += 1;

        // Autoregressive generation loop
        let mut hit_stop_string = false;
        for _ in 0..self.config.generation.max_tokens - 1 {
            if self.stop_tokens.contains(&next_token) {
                break;
            }

            let pos = self.cached_tokens.len() - 1;
            let logits = self
                .model
                .forward(&[next_token], pos)
                .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;

            next_token = self
                .sampler
                .sample(&logits, &self.cached_tokens)
                .map_err(|e| ChatSessionError::SamplingError(e.to_string()))?;

            if self.stop_tokens.contains(&next_token) {
                break;
            }

            self.cached_tokens.push(next_token);
            response_tokens.push(next_token);
            generated_tokens += 1;

            if let Ok(Some(text)) = decoder.step(next_token) {
                if has_stop_strings {
                    decoded_text.push_str(&text);
                    if self.stop_strings.iter().any(|s| decoded_text.contains(s.as_str())) {
                        hit_stop_string = true;
                        break;
                    }
                }
                print!("{}", text);
                std::io::Write::flush(&mut std::io::stdout()).ok();
            }
        }

        // Flush remaining text
        if !hit_stop_string {
            if let Ok(Some(text)) = decoder.flush() {
                print!("{}", text);
            }
        }
        println!();

        // Print stats
        let elapsed = decode_start.elapsed();
        let tps = generated_tokens as f64 / elapsed.as_secs_f64();
        println!(
            "[Stats] {} tokens in {:.2}s ({:.1} tok/s) | Cache: {}/{}",
            generated_tokens,
            elapsed.as_secs_f64(),
            tps,
            self.cached_tokens.len(),
            self.config.context_length
        );

        // Decode response, stripping any stop strings from the end
        let mut response = self.tokenizer
            .decode(&response_tokens, true)
            .map_err(|e| ChatSessionError::TokenizerError(e.to_string()))?;
        for stop in &self.stop_strings {
            if let Some(pos) = response.find(stop.as_str()) {
                response.truncate(pos);
            }
        }
        Ok(response)
    }

    /// Slide the context window when approaching limit
    fn slide_window(&mut self) -> Result<(), ChatSessionError> {
        let keep_count = (self.cached_tokens.len() as f32 * self.config.keep_ratio) as usize;
        let drop_count = self.cached_tokens.len() - keep_count;

        // Find nearest message boundary (don't split messages)
        let mut actual_drop = drop_count;
        for msg in &self.messages {
            if msg.end_idx <= drop_count {
                // This whole message can be dropped
                actual_drop = msg.end_idx;
            } else if msg.start_idx < drop_count {
                // This message would be split - keep it whole
                actual_drop = msg.start_idx;
                break;
            }
        }

        // Ensure we drop at least something
        if actual_drop == 0 && !self.messages.is_empty() {
            actual_drop = self.messages[0].end_idx;
        }

        println!(
            "\n[Context] Sliding window: dropping {} oldest tokens, keeping {}",
            actual_drop,
            self.cached_tokens.len() - actual_drop
        );

        // Keep only recent tokens
        self.cached_tokens = self.cached_tokens.split_off(actual_drop);

        // Update message indices
        self.messages.retain_mut(|msg| {
            if msg.end_idx <= actual_drop {
                false // Drop this message
            } else {
                msg.start_idx = msg.start_idx.saturating_sub(actual_drop);
                msg.end_idx -= actual_drop;
                true
            }
        });

        // Clear and rebuild KV cache
        self.model.clear_cache();
        for (pos, &token) in self.cached_tokens.iter().enumerate() {
            let _ = self
                .model
                .forward(&[token], pos)
                .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;
        }

        Ok(())
    }

    /// Format a system prompt according to chat format
    fn format_system_prompt(&self, content: &str) -> String {
        match self.config.chat_format {
            ChatFormat::Llama | ChatFormat::Mistral => {
                // Mistral/Llama wrap system in first [INST]
                format!("<s>[INST] {}\n\n", content)
            }
            ChatFormat::ChatML => {
                format!("<|im_start|>system\n{}<|im_end|>\n", content)
            }
            ChatFormat::Gemma => {
                format!("<start_of_turn>user\n{}<end_of_turn>\n", content)
            }
            ChatFormat::Phi => {
                format!("System: {}\n", content)
            }
        }
    }

    /// Format a user turn according to chat format
    fn format_user_turn(&self, content: &str) -> String {
        match self.config.chat_format {
            ChatFormat::Llama | ChatFormat::Mistral => {
                if self.messages.is_empty() && self.system_prompt.is_none() {
                    // First turn without system prompt
                    format!("<s>[INST] {} [/INST]", content)
                } else if self.messages.iter().any(|m| m.role == MessageRole::Assistant) {
                    // Subsequent turn (after at least one assistant response)
                    format!("[INST] {} [/INST]", content)
                } else {
                    // First user turn after system prompt
                    format!("{} [/INST]", content)
                }
            }
            ChatFormat::ChatML => {
                format!("<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n", content)
            }
            ChatFormat::Gemma => {
                format!(
                    "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                    content
                )
            }
            ChatFormat::Phi => {
                format!("Instruct: {}\nOutput:", content)
            }
        }
    }

    /// Estimate token count for a new message (rough heuristic)
    fn estimate_new_tokens(&self, content: &str) -> usize {
        // Rough estimate: ~4 chars per token + overhead for formatting
        (content.len() / 4) + 20
    }
}

/// Streaming response iterator
pub struct StreamingResponse<'a, M: InferenceModel> {
    session: &'a mut ChatSession<M>,
    decoder: StreamDecoder,
    next_token: Option<u32>,
    assistant_start: usize,
    generated_count: usize,
    finished: bool,
    decoded_buf: String,
}

impl<'a, M: InferenceModel> StreamingResponse<'a, M> {
    /// Get the next chunk of text
    pub fn next_chunk(&mut self) -> Result<Option<String>, ChatSessionError> {
        if self.finished {
            return Ok(None);
        }

        let current_token = match self.next_token {
            Some(t) => t,
            None => return Ok(None),
        };

        // Check for EOS or max tokens
        if self.session.stop_tokens.contains(&current_token)
            || self.generated_count >= self.session.config.generation.max_tokens
        {
            self.finished = true;

            // Record assistant message
            let assistant_end = self.session.cached_tokens.len();
            self.session.messages.push(CachedMessage {
                role: MessageRole::Assistant,
                start_idx: self.assistant_start,
                end_idx: assistant_end,
            });

            return Ok(self
                .decoder
                .flush()
                .map_err(|e| ChatSessionError::TokenizerError(e.to_string()))?);
        }

        // Add token to cache
        self.session.cached_tokens.push(current_token);
        self.generated_count += 1;

        // Decode to text
        let text = self
            .decoder
            .step(current_token)
            .map_err(|e| ChatSessionError::TokenizerError(e.to_string()))?;

        // Check stop strings in accumulated decoded text
        if !self.session.stop_strings.is_empty() {
            if let Some(ref t) = text {
                self.decoded_buf.push_str(t);
            }
            if self.session.stop_strings.iter().any(|s| self.decoded_buf.contains(s.as_str())) {
                self.finished = true;
                let assistant_end = self.session.cached_tokens.len();
                self.session.messages.push(CachedMessage {
                    role: MessageRole::Assistant,
                    start_idx: self.assistant_start,
                    end_idx: assistant_end,
                });
                return Ok(None);
            }
        }

        // Get next token
        let pos = self.session.cached_tokens.len() - 1;
        let logits = self
            .session
            .model
            .forward(&[current_token], pos)
            .map_err(|e| ChatSessionError::ModelError(e.to_string()))?;

        self.next_token = Some(
            self.session
                .sampler
                .sample(&logits, &self.session.cached_tokens)
                .map_err(|e| ChatSessionError::SamplingError(e.to_string()))?,
        );

        Ok(text.or(Some(String::new())))
    }

    /// Check if generation is finished
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// Get number of tokens generated so far
    pub fn generated_count(&self) -> usize {
        self.generated_count
    }

    /// Consume remaining tokens and return full response
    pub fn collect(mut self) -> Result<String, ChatSessionError> {
        let mut full_response = String::new();
        while let Some(chunk) = self.next_chunk()? {
            full_response.push_str(&chunk);
        }
        Ok(full_response)
    }
}

/// Errors that can occur in ChatSession
#[derive(Debug, Clone)]
pub enum ChatSessionError {
    /// Error from the tokenizer
    TokenizerError(String),
    /// Error from the model
    ModelError(String),
    /// Error during sampling
    SamplingError(String),
}

impl std::fmt::Display for ChatSessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChatSessionError::TokenizerError(e) => write!(f, "Tokenizer error: {}", e),
            ChatSessionError::ModelError(e) => write!(f, "Model error: {}", e),
            ChatSessionError::SamplingError(e) => write!(f, "Sampling error: {}", e),
        }
    }
}

impl std::error::Error for ChatSessionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_builders() {
        let config = ChatSessionConfig::mistral()
            .with_context_length(8192)
            .with_keep_ratio(0.75);

        assert_eq!(config.context_length, 8192);
        assert_eq!(config.keep_ratio, 0.75);
    }
}
