// Mistral 7B Quantized Inference (4-bit GGUF)
//
// Runs Mistral-7B with 4-bit quantization using GGUF format.
// Requires ~4-5GB VRAM instead of ~14GB for FP16.

use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::quantized_llama::ModelWeights;
use ferrite::{ChatMessage, Tokenizer};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;

struct QuantizedGenerator {
    model: ModelWeights,
    device: Device,
    tokenizer: Tokenizer,
}

impl QuantizedGenerator {
    fn new(
        gguf_path: &PathBuf,
        tokenizer_dir: &PathBuf,
        device: Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("[Init] Loading tokenizer...");
        let tokenizer = Tokenizer::from_dir(tokenizer_dir)?;
        println!("[Init] Vocab size: {}", tokenizer.vocab_size());

        println!("[Init] Loading quantized GGUF model...");
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf_content, &mut file, &device)?;

        println!("[Init] Model ready! (4-bit quantized)");

        Ok(Self {
            model,
            device,
            tokenizer,
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

        // Process prompt tokens one by one for quantized model
        let mut pos = 0;
        for &token in &tokens {
            // Input must be 2D: [batch_size, seq_len] = [1, 1]
            let input = Tensor::new(&[[token]], &self.device)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }

        // Get logits for last position
        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[[last_token]], &self.device)?;
        let logits = self.model.forward(&input, pos - 1)?;
        let logits = logits.squeeze(0)?;

        let mut next_token = logits_processor.sample(&logits)?;
        all_tokens.push(next_token);

        if let Ok(Some(text)) = decoder.step(next_token) {
            print!("{}", text);
            io::stdout().flush()?;
        }

        let eos_id = self.tokenizer.eos_token_id().unwrap_or(2);

        // Start timing for TPS calculation
        let decode_start = Instant::now();
        let mut generated_tokens = 1usize; // Already generated one token above

        // Autoregressive decode
        for _ in 0..max_tokens - 1 {
            if next_token == eos_id {
                break;
            }

            // Input must be 2D: [batch_size, seq_len] = [1, 1]
            let input = Tensor::new(&[[next_token]], &self.device)?;
            let logits = self.model.forward(&input, pos)?;
            let logits = logits.squeeze(0)?;
            pos += 1;

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

/// Download GGUF file from HuggingFace
async fn download_gguf(repo_id: &str, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {

    let cache_dir = dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("huggingface")
        .join("gguf");

    std::fs::create_dir_all(&cache_dir)?;

    let local_path = cache_dir.join(filename);

    if local_path.exists() {
        println!("[Download] Using cached: {:?}", local_path);
        return Ok(local_path);
    }

    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo_id, filename
    );

    println!("[Download] Fetching {} ...", filename);
    println!("[Download] URL: {}", url);

    let response = reqwest::get(&url).await?;

    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }

    let bytes = response.bytes().await?;
    std::fs::write(&local_path, &bytes)?;

    println!("[Download] Saved to: {:?}", local_path);
    Ok(local_path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     MISTRAL 7B QUANTIZED - 4-bit GGUF                        ║");
    println!("║     ~4GB VRAM instead of ~14GB                               ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    match &device {
        Device::Cuda(_) => println!("[Init] Using CUDA"),
        Device::Cpu => println!("[Init] Using CPU"),
        _ => {}
    }

    // Download quantized GGUF model (Q4_K_M is good quality/size balance)
    let gguf_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";
    let gguf_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf";

    println!("\n[Download] Getting quantized model...");
    let gguf_path = download_gguf(gguf_repo, gguf_file).await?;

    // Also need tokenizer from original model
    let tokenizer_repo = "mistralai/Mistral-7B-Instruct-v0.2";
    println!("[Download] Getting tokenizer...");
    let tokenizer_dir = llm_tokenizer::hub::from_hf(tokenizer_repo, false).await?;

    println!();

    let mut generator = QuantizedGenerator::new(&gguf_path, &tokenizer_dir, device)?;

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

        print!("Mistral-Q4: ");
        io::stdout().flush()?;

        if let Err(e) = generator.generate(&prompt, 1024, 0.7, 0.9) {
            println!("Error: {}", e);
        }
    }

    println!("\nGoodbye!");
    Ok(())
}
