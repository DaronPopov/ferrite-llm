// Mistral Stress Test (Quantized GGUF)
//
// Continuous inference for 3 minutes to verify memory stability and TPS consistency.

use candle_core::{quantized::gguf_file, Device, Tensor};
use candle_transformers::models::quantized_llama::ModelWeights;
use ferrite::{
    ChatMessage, Tokenizer, InferenceModel, StreamingInference, 
    GenerationConfig
};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::{Instant, Duration};

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
    fn forward(&mut self, tokens: &[u32], pos: usize) -> Result<Tensor, Box<dyn std::error::Error>> {
        if tokens.is_empty() {
            return Err("Cannot forward empty token sequence".into());
        }
        // Create 1D tensor and add batch dimension
        let input = Tensor::new(tokens, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;
        Ok(logits)
    }

    fn prefill(&mut self, tokens: &[u32]) -> Result<Tensor, Box<dyn std::error::Error>> {
        let mut pos = 0;
        for &token in tokens.iter().take(tokens.len() - 1) {
            let input = Tensor::new(&[token], &self.device)?.unsqueeze(0)?;
            let _ = self.model.forward(&input, pos)?;
            pos += 1;
        }

        let last_token = *tokens.last().unwrap();
        let input = Tensor::new(&[last_token], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input, pos)?;
        let logits = logits.squeeze(0)?;
        Ok(logits)
    }
}

async fn download_gguf(repo_id: &str, filename: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_dir = dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".")).join("huggingface").join("gguf");
    std::fs::create_dir_all(&cache_dir)?;
    let local_path = cache_dir.join(filename);
    if local_path.exists() { return Ok(local_path); }
    let url = format!("https://huggingface.co/{}/resolve/main/{}", repo_id, filename);
    let response = reqwest::get(&url).await?;
    let bytes = response.bytes().await?;
    std::fs::write(&local_path, &bytes)?;
    Ok(local_path)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║     MISTRAL STRESS TEST - Continuous Inference                ║");
    println!("║     Running for 3.5 minutes to verify stability              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    let device = Device::cuda_if_available(0)?;
    let gguf_repo = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF";
    let gguf_file = "mistral-7b-instruct-v0.2.Q4_K_M.gguf";
    let gguf_path = download_gguf(gguf_repo, gguf_file).await?;

    let tokenizer_repo = "mistralai/Mistral-7B-Instruct-v0.2";
    let model_dir = llm_tokenizer::hub::from_hf(tokenizer_repo, false).await?;
    let tokenizer = Tokenizer::from_dir(&model_dir)?;
    
    let mut engine = QuantizedMistralEngine::new(&gguf_path, device)?;

    let test_start = Instant::now();
    let duration = Duration::from_secs(210); // 3.5 minutes
    let mut iteration = 1;

    let prompts = [
        "Write a detailed story about a space explorer finding a lost civilization.",
        "Explain the concept of general relativity in simple terms for a child.",
        "Write a poem about the beauty of the Rust programming language and its safety.",
        "Compare and contrast binary search and linear search with examples.",
        "Describe a futuristic city where AI and humans live in perfect harmony."
    ];

    while test_start.elapsed() < duration {
        let elapsed = test_start.elapsed();
        let remaining = duration.saturating_sub(elapsed);
        
        println!("\n[ITERATION {}] - Time Remaining: {}:{:02}", 
                 iteration, remaining.as_secs() / 60, remaining.as_secs() % 60);
        
        let input = prompts[iteration % prompts.len()];
        println!("Prompt: \"{}\"", input);
        
        let messages = vec![ChatMessage::user(input)];
        let prompt = tokenizer.apply_chat_template(&messages, true)?;
        
        let config = GenerationConfig::default()
            .with_max_tokens(1024) // Large generations to stress the engine
            .with_temperature(0.7);

        print!("Response: ");
        io::stdout().flush()?;

        let mut stream = StreamingInference::new(&mut engine, &tokenizer, &prompt, config)?;
        
        while let Some(text) = stream.next()? {
            print!("{}", text);
            io::stdout().flush()?;
        }
        
        println!("\n");
        stream.stats().print_summary();
        
        iteration += 1;
    }

    println!("\n════════════════════════════════════════════════════════════════");
    println!("  STRESS TEST COMPLETE!");
    println!("  Total Duration: {:.2}s", test_start.elapsed().as_secs_f64());
    println!("  Total Iterations: {}", iteration - 1);
    println!("════════════════════════════════════════════════════════════════\n");

    Ok(())
}
