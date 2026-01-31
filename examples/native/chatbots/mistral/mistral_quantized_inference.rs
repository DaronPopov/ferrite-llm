// Mistral 7B Quantized Inference (4-bit GGUF) with ChatSession
//
// Demonstrates proper multi-turn chat with incremental KV caching.
// Uses ChatSession to maintain context across conversation turns.
//
// The key insight: **never re-encode previously generated tokens**.
// ChatSession handles this automatically by:
// 1. Encoding each user turn incrementally (just new message + formatting)
// 2. Storing generated response tokens directly (never re-encode them)
// 3. KV cache position always equals `cached_tokens.len()`

use candle_core::{quantized::gguf_file, Device, Tensor};
use ferrite::{ChatSession, ChatSessionConfig, InferenceModel, Tokenizer};
use ferrite_examples::{cli, download_gguf, get_device};
use std::io::{self, Write};
use std::path::PathBuf;
use std::sync::Arc;

/// Wrapper for quantized Mistral model that implements InferenceModel trait
struct QuantizedMistral {
    model: candle_transformers::models::quantized_llama::ModelWeights,
    device: Device,
}

impl QuantizedMistral {
    fn new(gguf_path: &PathBuf, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
            gguf_content,
            &mut file,
            &device,
        )?;

        Ok(Self { model, device })
    }
}

impl InferenceModel for QuantizedMistral {
    fn forward(
        &mut self,
        tokens: &[u32],
        pos: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Ok(Tensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?);
        }

        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;
        Ok(logits)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len().saturating_sub(1)) {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }

        if let Some(&last_token) = tokens.last() {
            let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            Ok(logits)
        } else {
            Ok(Tensor::zeros((1, 32000), candle_core::DType::F32, &self.device)?)
        }
    }

    fn clear_cache(&mut self) {
        // Quantized models don't have an explicit cache clear method
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";
    let filename = "mistral-7b-instruct-v0.2.Q4_K_M.gguf";

    cli::print_banner("Mistral-7B-Q4", "4-bit quantized GGUF");

    let device = get_device()?;

    // Download GGUF file
    let gguf_path = download_gguf(model_id, filename).await?;

    // Load tokenizer from HF
    println!("[Init] Loading tokenizer...");
    let tokenizer = Arc::new(
        Tokenizer::from_pretrained("mistralai/Mistral-7B-Instruct-v0.2").await?,
    );
    println!("[Init] Vocab size: {}", tokenizer.vocab_size());

    println!("[Init] Loading quantized GGUF model...");
    let model = QuantizedMistral::new(&gguf_path, device)?;
    println!("[Init] Model ready! (4-bit quantized)");

    // Configure ChatSession
    let config = ChatSessionConfig::mistral()
        .with_context_length(32768)
        .with_generation(
            ferrite::GenerationConfig::default()
                .with_max_tokens(1024)
                .with_temperature(0.7)
                .with_top_p(0.9),
        );

    // Create ChatSession with system prompt
    let mut session = ChatSession::new(
        model,
        tokenizer,
        Some("You are a helpful assistant."),
        config,
    )?;

    cli::print_chat_session_prompt();

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.eq_ignore_ascii_case("clear") {
            session.clear();
            println!("[Cleared conversation history]\n");
            continue;
        }

        if input.eq_ignore_ascii_case("stats") {
            println!(
                "[Stats] Cache: {} tokens, {} messages, {} remaining capacity",
                session.token_count(),
                session.messages().len(),
                session.remaining_capacity()
            );
            println!();
            continue;
        }

        if input.is_empty() {
            continue;
        }

        print!("Mistral-Q4: ");
        io::stdout().flush()?;

        match session.user_turn(input) {
            Ok(_response) => {
                println!();
            }
            Err(e) => {
                println!("\nError: {}", e);
            }
        }
    }

    cli::print_goodbye();
    Ok(())
}
