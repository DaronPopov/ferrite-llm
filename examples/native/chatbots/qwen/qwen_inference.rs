// Qwen2 Inference Example (Refactored)
//
// Demonstrates Qwen2 inference using shared utilities to reduce duplication.
// Compare this with qwen_inference.rs to see the improvements.

use candle_core::{Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::qwen2::{Config, Model as Qwen2};
use ferrite::{ChatMessage, Tokenizer};
use ferrite_examples::{cli, download_model, get_device, load_safetensors, Precision};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

/// Model configuration for Qwen2-0.5B-Instruct
fn qwen2_0_5b_config() -> Config {
    Config {
        vocab_size: 151936,
        hidden_size: 896,
        intermediate_size: 4864,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        max_position_embeddings: 32768,
        sliding_window: 32768,
        max_window_layers: 21,
        tie_word_embeddings: true,
        rope_theta: 1000000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}

struct TextGenerator {
    model: Qwen2,
    device: Device,
    tokenizer: Tokenizer,
    precision: Precision,
}

impl TextGenerator {
    async fn new(
        model_id: &str,
        model_dir: &PathBuf,
        device: Device,
        config: Config,
        precision: Precision,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[Init] Loading tokenizer...");
        let tokenizer = Tokenizer::from_pretrained(model_id).await?;
        println!("[Init] Vocab size: {}", tokenizer.vocab_size());

        let vb = load_safetensors(model_dir, precision.dtype(), &device)?;

        println!("[Init] Building model...");
        let model = Qwen2::new(&config, vb)?;
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

        // Prefill
        let input = Tensor::new(tokens.as_slice(), &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, 0, None)?;
        let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;
        let logits = if self.precision.needs_f32_conversion() {
            logits.to_dtype(candle_core::DType::F32)?
        } else {
            logits
        };

        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Ok(Some(text)) = decoder.step(next_token) {
            print!("{}", text);
            io::stdout().flush()?;
        }

        // Qwen2 EOS tokens
        let eos_ids: Vec<u32> = vec![151643, 151644, 151645];

        let decode_start = Instant::now();
        let mut generated_tokens = 1usize;

        // Autoregressive decode
        for i in 0..max_tokens - 1 {
            if eos_ids.contains(&next_token) {
                break;
            }

            let pos = prompt_len + i;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;

            let logits = self.model.forward(&input, pos, None)?;
            let logits = logits.squeeze(0)?.get(logits.dim(0)? - 1)?;
            let logits = if self.precision.needs_f32_conversion() {
                logits.to_dtype(candle_core::DType::F32)?
            } else {
                logits
            };

            next_token = logits_processor.sample(&logits)?;

            if eos_ids.contains(&next_token) {
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

        // Print stats using utility
        ferrite_examples::print_tps_stats(generated_tokens, decode_start.elapsed());
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_id = "Qwen/Qwen2-0.5B-Instruct";

    cli::print_banner("Qwen2", model_id);

    let device = get_device()?;
    let precision = Precision::from_device(&device);

    let model_dir = download_model(model_id).await?;
    let config = qwen2_0_5b_config();
    let mut generator = TextGenerator::new(model_id, &model_dir, device, config, precision).await?;

    // Interactive loop using shared utility
    cli::interactive_loop("You", "Qwen2", |input| {
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(input),
        ];

        let prompt = match generator.tokenizer.apply_chat_template(&messages, true) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[Warning] Chat template failed: {}, using fallback", e);
                format!(
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
                    input
                )
            }
        };

        generator.generate(&prompt, 1024, 0.7, 0.9)
    })?;

    Ok(())
}
