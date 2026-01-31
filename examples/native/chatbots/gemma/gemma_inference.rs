// Gemma Inference Example
//
// Demonstrates Google Gemma inference using Candle with FP16.
// Supports Gemma-2B and Gemma-7B models.

use candle_core::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma::{Config, Model as Gemma};
use ferrite::Tokenizer;
use ferrite_examples::{cli, download_model, get_device, load_safetensors, Precision, print_tps_stats};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Model configuration for Gemma-2B-it
fn gemma_2b_config() -> Config {
    Config {
        vocab_size: 256000,
        hidden_size: 2048,
        intermediate_size: 16384,
        num_hidden_layers: 18,
        num_attention_heads: 8,
        num_key_value_heads: 1,
        head_dim: 256,
        max_position_embeddings: 8192,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.0,
        hidden_activation: Some(candle_nn::Activation::GeluPytorchTanh),
        hidden_act: None,
        attention_bias: false,
    }
}

struct TextGenerator {
    model: Gemma,
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
        let model = Gemma::new(false, &config, vb)?;
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
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(1);
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize;

        // Autoregressive decode
        for i in 0..max_tokens - 1 {
            if next_token == eos_id {
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
    let model_id = "google/gemma-2b-it";

    cli::print_banner("Gemma", model_id);
    println!("[Note] Gemma requires accepting the license at https://huggingface.co/google/gemma-2b-it\n");

    let device = get_device()?;
    let precision = Precision::from_device(&device);

    let model_dir = download_model(model_id).await?;
    let config = gemma_2b_config();
    let mut generator = TextGenerator::new(&model_dir, device, config, precision)?;

    cli::interactive_loop("You", "Gemma", |input| {
        let prompt = format!("<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n", input);
        generator.generate(&prompt, 1024, 0.7, 0.9)
    })?;

    Ok(())
}
