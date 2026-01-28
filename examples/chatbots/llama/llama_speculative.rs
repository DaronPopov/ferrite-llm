// Llama-3 Speculative Decoding Example
//
// Target: Hermes-3-Llama-3.1-8B (Quantized 4-bit) - Open/Un-gated
// Scout: Llama-3.2-1B-Instruct (Quantized 4-bit)
//
// Both models share the EXACT SAME Llama-3 tokenizer (128k vocab),
// providing the highest possible speculative acceptance rate.

use candle_core::{Device, Tensor, quantized::gguf_file};
use candle_transformers::models::quantized_llama::ModelWeights;
use ferrite::{
    ChatMessage, Tokenizer, InferenceModel, SpeculativeInference,
    GenerationConfig
};
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
        let mut all_logits = Vec::new();
        let mut current_pos = pos;
        
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

async fn download_gguf(repo_id: &str, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".")).join("huggingface").join("gguf");
    std::fs::create_dir_all(&cache_dir)?;
    let local_path = cache_dir.join(filename);
    if local_path.exists() { return Ok(local_path); }
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    println!("[Download] Fetching {} ...", filename);
    let response = reqwest::get(&url).await?;
    let bytes = response.bytes().await?;
    std::fs::write(&local_path, &bytes)?;
    Ok(local_path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     LLAMA-3 ULTIMATE SPECULATIVE DECODING                    ║");
    println!("║     Target: Hermes-3 8B | Scout: Llama-3.2 1B                ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;

    // 1. Download/Locate Models
    println!("[Init] Locating Target (Hermes-3-8B)...");
    let target_path = download_gguf(
        "NousResearch/Hermes-3-Llama-3.1-8B-GGUF", 
        "Hermes-3-Llama-3.1-8B.Q4_K_M.gguf"
    ).await?;

    println!("[Init] Locating Scout (Llama-3.2-1B)...");
    let scout_path = download_gguf(
        "unsloth/Llama-3.2-1B-Instruct-GGUF", 
        "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    ).await?;

    // Tokenizer (Shared Llama-3 Vocab) - ONLY download tokenizer files
    let token_dir = llm_tokenizer::hub::from_hf("NousResearch/Hermes-3-Llama-3.1-8B", true).await?;
    let tokenizer = Tokenizer::from_dir(&token_dir)?;

    // 2. Init Engines
    println!("[Init] Loading Target to GPU and Scout to CPU...");
    let mut target = Llama3Engine::new(&target_path, device.clone())?;
    let mut scout = Llama3Engine::new(&scout_path, Device::Cpu)?;

    println!("\nReady! Speculative K=4 enabled (Greedy). Type 'quit' to exit.");
    
    loop {
        print!("\nYou: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input == "quit" || input == "exit" { break; }

        let messages = vec![ChatMessage::user(input)];
        let prompt = match tokenizer.apply_chat_template(&messages, true) {
            Ok(p) => p,
            Err(_) => format!("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", input),
        };
        
        let config = GenerationConfig::default()
            .with_max_tokens(512)
            .with_temperature(0.0);

        print!("Hermes-3 (Speculative): ");
        io::stdout().flush()?;

        let mut stream = SpeculativeInference::new(
            &mut target, 
            &mut scout, 
            &tokenizer, 
            &prompt, 
            config, 
            4 // Speculate 4 tokens at a time
        )?;
        
        while let Some(text) = stream.next()? {
            print!("{}", text);
            io::stdout().flush()?;
        }
        
        println!();
        stream.stats().print_summary();
    }

    Ok(())
}
