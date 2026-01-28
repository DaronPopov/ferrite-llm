// Mistral Streaming Inference Example (Quantized GGUF)
//
// Demonstrates the new Ferrite StreamingInference API with 4-bit quantization.

use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use ferrite::{
    ChatMessage, Tokenizer, InferenceModel, StreamingInference, 
    GenerationConfig
};
use std::io::{self, Write};
use std::path::PathBuf;

pub struct QuantizedMistralEngine {
    model: ModelWeights,
    device: Device,
}

impl QuantizedMistralEngine {
    fn new(gguf_path: &PathBuf, device: Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut file = std::fs::File::open(gguf_path)?;
        let gguf_content = gguf_file::Content::read(&mut file)?;
        let model = ModelWeights::from_gguf(gguf_content, &mut file, &device)?;
        Ok(Self { model, device })
    }
}

impl InferenceModel for QuantizedMistralEngine {
    fn forward(&mut self, token: u32, pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        let input = Tensor::new(&[[token]], &self.device)?;
        let logits = self.model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;
        Ok(logits)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len() - 1) {
            let input = Tensor::new(&[[token]], &self.device)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }
        
        // Return logits for the last token to sample the first next token
        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[[last_token]], &self.device)?;
        let logits = self.model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;
        Ok(logits)
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
        return Ok(local_path);
    }

    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    println!("[Download] Fetching {} ...", filename);
    let response = reqwest::get(&url).await?;
    if !response.status().is_success() {
        return Err(format!("Failed to download: HTTP {}", response.status()).into());
    }
    let bytes = response.bytes().await?;
    std::fs::write(&local_path, &bytes)?;
    Ok(local_path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     MISTRAL QUANTIZED STREAMING - 4-bit GGUF                 ║");
    println!("║     Using the new Ferrite StreamingInference API             ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    
    // Download quantized GGUF model
    let gguf_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";
    let gguf_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf";
    let gguf_path = download_gguf(gguf_repo, gguf_file).await?;

    // Download tokenizer
    let tokenizer_repo = "mistralai/Mistral-7B-Instruct-v0.2";
    let model_dir = llm_tokenizer::hub::from_hf(tokenizer_repo, false).await?;
    let tokenizer = Tokenizer::from_dir(&model_dir)?;
    
    let mut engine = QuantizedMistralEngine::new(&gguf_path, device)?;

    println!("\nReady! Enter a prompt:");
    
    loop {
        print!("\nYou: ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        if input == "quit" || input == "exit" { break; }

        let messages = vec![ChatMessage::user(input)];
        let prompt = tokenizer.apply_chat_template(&messages, true)?;
        
        let config = GenerationConfig::default()
            .with_max_tokens(512)
            .with_temperature(0.7);

        print!("Mistral-Q4: ");
        io::stdout().flush()?;

        let mut stream = StreamingInference::new(&mut engine, &tokenizer, &prompt, config)?;
        
        while let Some(text) = stream.next()? {
            print!("{}", text);
            io::stdout().flush()?;
        }
        
        println!();
        stream.stats().print_summary();
    }

    Ok(())
}
