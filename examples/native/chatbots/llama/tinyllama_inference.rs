// TinyLlama Inference Example
//
// Demonstrates LLM inference using Candle with FP16 for faster GPU execution.
// This example can be adapted for other Llama-based models.

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, Llama};
use ferrite::{ChatMessage, Tokenizer};
use ferrite_examples::{cli, download_model, get_device, load_safetensors, Precision, print_tps_stats};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Model configuration for TinyLlama-1.1B-Chat
fn tinyllama_config() -> Config {
    Config {
        hidden_size: 2048,
        intermediate_size: 5632,
        vocab_size: 32000,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        use_flash_attn: cfg!(feature = "flash-attn"),
        bos_token_id: Some(1),
        eos_token_id: None,
        rope_scaling: None,
        max_position_embeddings: 2048,
        tie_word_embeddings: false,
    }
}

struct TextGenerator {
    model: Llama,
    device: Device,
    tokenizer: Tokenizer,
    config: Config,
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
        let model = Llama::load(vb, &config)?;
        println!("[Init] Model ready!");

        Ok(Self {
            model,
            device,
            tokenizer,
            config,
            precision,
        })
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f64,
        top_p: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt)?;
        let tokens: Vec<u32> = encoding.ids.clone();
        let prompt_len = tokens.len();

        println!("[Generate] Prompt: {} tokens", prompt_len);

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));
        let mut cache = Cache::new(true, self.precision.dtype(), &self.config, &self.device)?;
        let mut decoder = self.tokenizer.decode_stream(&tokens, true);
        let mut all_tokens = tokens.clone();

        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Prefill
        let logits = self.model.forward(&input, 0, &mut cache)?;
        let logits = logits.squeeze(0)?;
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
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(2);
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize;

        // Autoregressive decode
        for _ in 0..max_tokens - 1 {
            if next_token == eos_id {
                break;
            }

            let pos = all_tokens.len() - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, pos, &mut cache)?;
            let logits = logits.squeeze(0)?;
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
    let model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0";

    cli::print_banner("TinyLlama", model_id);

    let device = get_device()?;
    let precision = Precision::from_device(&device);

    let model_dir = download_model(model_id).await?;
    let config = tinyllama_config();
    let mut generator = TextGenerator::new(&model_dir, device, config, precision)?;

    cli::interactive_loop("You", "TinyLlama", |input| {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(input),
        ];

        let prompt = generator
            .tokenizer
            .apply_chat_template(&messages, true)
            .unwrap_or_else(|_| format!("### Human: {}\n### Assistant:", input));

        generator.generate(&prompt, 128, 0.7, 0.9)
    })?;

    Ok(())
}
