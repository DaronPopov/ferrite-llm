// Mistral Inference Example
//
// Demonstrates Mistral-7B-Instruct inference using Candle with FP16.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::mistral::{Config, Model as Mistral};
use ferrite::{ChatMessage, Tokenizer, StopCondition};
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

#[derive(Clone, Copy)]
pub enum Precision {
    F32,
    F16,
    BF16,
}

impl Precision {
    fn dtype(&self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }
}

struct TextGenerator {
    model: Mistral,
    device: Device,
    tokenizer: Tokenizer,
    #[allow(dead_code)]
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

        println!("[Init] Loading model weights ({:?})...", precision.dtype());
        let safetensor_files: Vec<PathBuf> = std::fs::read_dir(model_dir)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|x| x == "safetensors")
                    .unwrap_or(false)
            })
            .map(|e| e.path())
            .collect();

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&safetensor_files, precision.dtype(), &device)?
        };

        println!("[Init] Building model...");
        let model = Mistral::new(&config, vb)?;

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
        stop_sequences: &[&str],
    ) -> Result<String, Box<dyn std::error::Error>> {
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

        let input = Tensor::new(tokens.as_slice(), &self.device)?;
        let input = input.unsqueeze(0)?;

        // Prefill
        let logits = self.model.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        let logits = logits.get(logits.dim(0)? - 1)?;
        let logits = if matches!(self.precision, Precision::F32) {
            logits
        } else {
            logits.to_dtype(DType::F32)?
        };

        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Ok(Some(text)) = decoder.step(next_token) {
            print!("{}", text);
            io::stdout().flush()?;
            generated_text.push_str(&text);
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(2);

        // Start timing for TPS calculation
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize; // Already generated one token above

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
            let input = Tensor::new(&[next_token], &self.device)?;
            let input = input.unsqueeze(0)?;

            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            let logits = logits.get(logits.dim(0)? - 1)?;
            let logits = if matches!(self.precision, Precision::F32) {
                logits
            } else {
                logits.to_dtype(DType::F32)?
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

        // Print TPS stats
        let elapsed = decode_start.elapsed();
        let tps = generated_tokens as f64 / elapsed.as_secs_f64();
        println!("[Stats] {} tokens in {:.2}s ({:.1} tok/s)",
                 generated_tokens, elapsed.as_secs_f64(), tps);

        self.tokenizer
            .decode(&all_tokens, true)
            .map_err(|e| e.into())
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     MISTRAL INFERENCE - Candle Backend                       ║");
    println!("║     Model: Mistral-7B-Instruct-v0.2                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    let precision = match &device {
        Device::Cuda(_) => {
            println!("[Init] Using CUDA with FP16");
            Precision::F16
        }
        Device::Cpu => {
            println!("[Init] Using CPU with FP32");
            Precision::F32
        }
        _ => Precision::F32,
    };

    let model_id = "mistralai/Mistral-7B-Instruct-v0.2";

    println!("[Download] Fetching from HuggingFace...");
    let model_dir = llm_tokenizer::hub::from_hf(model_id, false).await?;
    println!("[Download] Cached at: {:?}\n", model_dir);

    let config = mistral_7b_config();
    let mut generator = TextGenerator::new(&model_dir, device, config, precision)?;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  Ready! Type a prompt and press Enter. 'quit' to exit.");
    println!("════════════════════════════════════════════════════════════════\n");

    loop {
        print!("You: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
            break;
        }

        if input.is_empty() {
            continue;
        }

        // Mistral instruct format
        let messages = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user(input),
        ];

        let prompt = match generator.tokenizer.apply_chat_template(&messages, true) {
            Ok(p) => p,
            Err(_) => format!("[INST] {} [/INST]", input),
        };

        print!("Mistral: ");
        io::stdout().flush()?;

        if let Err(e) = generator.generate(&prompt, 1024, 0.7, 0.9, &["</s>", "[/INST]"]) {
            println!("Error: {}", e);
        }
    }

    println!("\nGoodbye!");
    Ok(())
}
