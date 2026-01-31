// Phi-2 Inference Example
//
// Demonstrates Microsoft Phi-2 inference using Candle with FP16.
// Phi-2 is a 2.7B parameter model with strong reasoning capabilities.

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi::{Config, Model as Phi};
use ferrite::Tokenizer;
use ferrite_examples::{cli, download_model, get_device, load_safetensors, load_config_json, Precision, print_tps_stats};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

struct TextGenerator {
    model: Phi,
    device: Device,
    tokenizer: Tokenizer,
    precision: Precision,
}

impl TextGenerator {
    fn new(
        model_dir: &PathBuf,
        device: Device,
        precision: Precision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[Init] Loading tokenizer...");
        let tokenizer = Tokenizer::from_dir(model_dir)?;
        println!("[Init] Vocab size: {}", tokenizer.vocab_size());

        println!("[Init] Loading config...");
        let config: Config = load_config_json(model_dir)?;

        let vb = load_safetensors(model_dir, precision.dtype(), &device)?;

        println!("[Init] Building model...");
        let model = Phi::new(&config, vb)?;
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
    ) -> Result<(), Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt)?;
        let tokens: Vec<u32> = encoding.ids.clone();
        let prompt_len = tokens.len();

        println!("[Generate] Prompt: {} tokens", prompt_len);

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));
        let mut decoder = self.tokenizer.decode_stream(&tokens, true);
        let mut all_tokens = tokens.clone();

        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;

        // Prefill
        let logits = self.model.forward(&input)?;
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
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(50256);
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize;

        // Autoregressive decode
        // Note: Phi doesn't use KV cache in this basic example, so we reprocess all tokens
        for _ in 0..max_tokens - 1 {
            if next_token == eos_id {
                break;
            }

            let input = Tensor::new(all_tokens.as_slice(), &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input)?;
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
    let model_id = "microsoft/phi-2";

    cli::print_banner("Phi-2", model_id);

    let device = get_device()?;
    let precision = Precision::from_device(&device);

    let model_dir = download_model(model_id).await?;
    let mut generator = TextGenerator::new(&model_dir, device, precision)?;

    cli::interactive_loop("You", "Phi-2", |input| {
        // Phi-2 instruct format (Mistral-style)
        let prompt = format!("Instruct: {}\nOutput:", input);
        generator.generate(&prompt, 256, 0.7, 0.9)
    })?;

    Ok(())
}
