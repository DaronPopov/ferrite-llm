// Llama-3 Speculative Decoding
//
// Target: Hermes-3-Llama-3.1-8B | Draft: Llama-3.2-1B
// Uses draft model to predict K tokens, verified by target model

use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use ferrite::{ChatMessage, Tokenizer, InferenceModel, SpeculativeInference, GenerationConfig};
use ferrite_examples::{cli, download_gguf, download_model, get_device};
use std::io::{self, Write};
use std::path::PathBuf;

pub struct Llama3Engine {
    model: ModelWeights,
    device: Device,
}

impl Llama3Engine {
    fn new(gguf_path: &PathBuf, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf_content, &mut file, &device)?;
        Ok(Self { model, device })
    }
}

impl InferenceModel for Llama3Engine {
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut current_pos = pos;
        let mut all_logits = Vec::new();

        for &token in tokens {
            let input = Tensor::new(&[[token]], &self.device)?;
            let logits = self.model.forward(&input, current_pos)?;
            all_logits.push(logits.squeeze(0)?);
            current_pos += 1;
        }

        if all_logits.is_empty() {
            return Err("No tokens provided".into());
        }

        Ok(Tensor::stack(&all_logits, 0)?)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len() - 1) {
            let input = Tensor::new(&[[token]], &self.device)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }
        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[[last_token]], &self.device)?;
        let logits = self.model.forward(&input, pos)?;
        Ok(logits.squeeze(0)?)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    cli::print_banner("Speculative Decoding", "Target: Hermes-3 8B | Draft: Llama-3.2 1B");

    let device = get_device()?;

    println!("[Init] Locating Target (Hermes-3-8B)...");
    let target_path = download_gguf(
        "NousResearch/Hermes-3-Llama-3.1-8B-GGUF",
        "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    ).await?;

    println!("[Init] Locating Draft (Llama-3.2-1B)...");
    let draft_path = download_gguf(
        "unsloth/Llama-3.2-1B-Instruct-GGUF",
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ).await?;

    let token_dir = download_model("NousResearch/Hermes-3-Llama-3.1-8B").await?;
    let tokenizer = Tokenizer::from_dir(&token_dir)?;

    println!("[Init] Loading Target to GPU and Draft to CPU...");
    let mut target = Llama3Engine::new(&target_path, device.clone())?;
    let mut draft = Llama3Engine::new(&draft_path, Device::Cpu)?;

    println!("\nSpeculative K=4 enabled (Greedy sampling).");

    cli::interactive_loop("You", "Hermes-3 (Speculative)", |input| {
        let messages = vec![ChatMessage::user(input)];
        let prompt = tokenizer
            .apply_chat_template(&messages, true)
            .unwrap_or_else(|_| {
                format!(
                    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
                    input
                )
            });

        let config = GenerationConfig::default()
            .with_max_tokens(512)
            .with_temperature(0.0);

        let mut stream = SpeculativeInference::new(
            &mut target,
            &mut draft,
            &tokenizer,
            &prompt,
            config,
            4, // Speculate 4 tokens at a time
        )?;

        while let Some(text) = stream.next()? {
            print!("{}", text);
            io::stdout().flush()?;
        }

        println!();
        stream.stats().print_summary();
        Ok(())
    })?;

    Ok(())
}
