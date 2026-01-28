// Phi-2 Inference Example
//
// Demonstrates Microsoft Phi-2 inference using Candle with FP16.
// Phi-2 is a 2.7B parameter model with strong reasoning capabilities.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::phi::{Config, Model as Phi};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

use ferrite::Tokenizer;

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

        // Load config from model directory
        println!("[Init] Loading config...");
        let config_path = model_dir.join("config.json");
        let config: Config = serde_json::from_reader(std::fs::File::open(&config_path)?)?;

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
    ) -> Result<String, Box<dyn std::error::Error>> {
        let encoding = self.tokenizer.encode(prompt)?;
        let tokens: Vec<u32> = encoding.ids.clone();
        let prompt_len = tokens.len();

        println!("[Generate] Prompt: {} tokens", prompt_len);

        let mut logits_processor = LogitsProcessor::new(42, Some(temperature), Some(top_p));
        let mut decoder = self.tokenizer.decode_stream(&tokens, true);
        let mut all_tokens = tokens.clone();

        let input = Tensor::new(tokens.as_slice(), &self.device)?;
        let input = input.unsqueeze(0)?;

        // Prefill
        let logits = self.model.forward(&input)?;
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
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(50256);

        // Start timing for TPS calculation
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize; // Already generated one token above

        // Autoregressive decode
        for _ in 0..max_tokens - 1 {
            if next_token == eos_id {
                break;
            }

            let input = Tensor::new(all_tokens.as_slice(), &self.device)?;
            let input = input.unsqueeze(0)?;

            let logits = self.model.forward(&input)?;
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
    println!("║     PHI-2 INFERENCE - Candle Backend                         ║");
    println!("║     Model: Microsoft Phi-2 (2.7B)                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    // Phi-2 requires F32 due to RoPE implementation limitations
    let precision = Precision::F32;
    match &device {
        Device::Cuda(_) => println!("[Init] Using CUDA with F32 (Phi-2 requires F32 for RoPE)"),
        Device::Cpu => println!("[Init] Using CPU with F32"),
        _ => {}
    };

    let model_id = "microsoft/phi-2";

    println!("[Download] Fetching from HuggingFace...");
    let model_dir = llm_tokenizer::hub::from_hf(model_id, false).await?;
    println!("[Download] Cached at: {:?}\n", model_dir);

    let mut generator = TextGenerator::new(&model_dir, device, precision)?;

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  Ready! Type a prompt and press Enter. 'quit' to exit.");
    println!("  Phi-2 works best with code/reasoning prompts.");
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

        // Phi-2 prompt format (instruct style)
        let prompt = format!("Instruct: {}\nOutput:", input);

        print!("Phi-2: ");
        io::stdout().flush()?;

        if let Err(e) = generator.generate(&prompt, 256, 0.7, 0.9) {
            println!("Error: {}", e);
        }
    }

    println!("\nGoodbye!");
    Ok(())
}
