// Mistral Inference Example
//
// Demonstrates Mistral-7B-Instruct inference using Candle with FP16.

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use ferrite::{ChatMessage, Tokenizer, StopCondition};
use ferrite_examples::{cli, download_model, get_device, load_safetensors, Precision, print_tps_stats};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Model configuration for Mistral-7B-Instruct-v0.2
fn mistral_7b_config() -> Config {
    Config {
        vocab_size: 32000,
        hidden_size: 4096,
        intermediate_size: 14336,
        num_hidden_layers: 32,
        num_attention_heads: 32,
        num_key_value_heads: 8,
        head_dim: Some(128),
        hidden_act: candle_nn::Activation::Silu,
        max_position_embeddings: 32768,
        rms_norm_eps: 1e-5,
        rope_theta: 1000000.0,
        sliding_window: Some(4096),
        use_flash_attn: cfg!(feature = "flash-attn"),
    }
}

struct TextGenerator {
    model: Mistral,
    device: Device,
    tokenizer: Tokenizer,
    precision: Precision,
}

impl TextGenerator {
    fn new(
        model_dir: &PathBuf,
        device: Device,
        config: Config,
        precision: Precision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[Init] Loading tokenizer...");
        let tokenizer = Tokenizer::from_dir(model_dir)?;
        println!("[Init] Vocab size: {}", tokenizer.vocab_size());

        let vb = load_safetensors(model_dir, precision.dtype(), &device)?;

        println!("[Init] Building model...");
        let model = Mistral::new(&config, vb)?;
        println!("[Init] Model ready!");

        Ok(Self {
            model,
            device,
            tokenizer,
            precision,
        })
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
        stop_sequences: &[&str],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt)?;
        let tokens: Vec<u32> = encoding.ids.clone();
        let prompt_len = tokens.len();

        println!("[Generate] Prompt: {} tokens", prompt_len);

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));
        let mut decoder = self.tokenizer.decode_stream(&tokens, true);
        let mut all_tokens = tokens.clone();
        let mut generated_text = String::new();

        // Build stop conditions
        let stop_conditions: Vec<StopCondition> = stop_sequences
            .iter()
            .map(|s| StopCondition::Text(s.to_string()))
            .collect();

        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Prefill
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;
        let logits = if self.precision.needs_f32_conversion() {
            logits.to_dtype(DType::F32)?
        } else {
            logits
        };

        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Ok(Some(text)) = decoder.step(next_token) {
            print!("{}", text);
            io::stdout().flush()?;
            generated_text.push_str(&text);
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(2);
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize;

        // Autoregressive decode
        for i in 0..max_tokens - 1 {
            if next_token == eos_id {
                break;
            }

            // Check stop sequences
            if stop_conditions.iter().any(|sc| sc.should_stop(next_token, &generated_text, generated_tokens)) {
                break;
            }

            let pos = prompt_len + i;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;
            let logits = if self.precision.needs_f32_conversion() {
                logits.to_dtype(DType::F32)?
            } else {
                logits
            };

            next_token = logits_processor.sample(&logits)?;

            if next_token == eos_id {
                break;
            }

            all_tokens.push(next_token);
            generated_tokens += 1;

            if let Ok(Some(text)) = decoder.step(next_token) {
                print!("{}", text);
                io::stdout().flush()?;
                generated_text.push_str(&text);

                // Check stop sequences after new text
                if stop_conditions.iter().any(|sc| sc.should_stop(next_token, &generated_text, generated_tokens)) {
                    break;
                }
            }
        }

        if let Ok(Some(text)) = decoder.flush() {
            print!("{}", text);
        }
        println!();

        print_tps_stats(generated_tokens, decode_start.elapsed());
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "mistralai/Mistral-7B-Instruct-v0.2";

    cli::print_banner("Mistral", model_id);

    let device = get_device()?;
    let precision = Precision::from_device(&device);

    let model_dir = download_model(model_id).await?;
    let config = mistral_7b_config();
    let mut generator = TextGenerator::new(&model_dir, device, config, precision)?;

    cli::interactive_loop("You", "Mistral", |input| {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(input),
        ];

        let prompt = generator
            .tokenizer
            .apply_chat_template(&messages, true)
            .unwrap_or_else(|_| format!("[INST] {} [/INST]", input));

        generator.generate(&prompt, 1024, 0.7, 0.9, &["</s>", "[/INST]"])
    })?;

    Ok(())
}
