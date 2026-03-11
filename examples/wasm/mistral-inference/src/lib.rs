//! Mistral 7B Quantized Inference - WASM Module
//!
//! This demonstrates REAL neural inference running inside WASM!
//! Uses the ferrite runtime's trait signature engine to access
//! quantized Mistral 7B running on the host.

wit_bindgen::generate!({
    path: "../../../wit",
    world: "ferrite-guest",
});

use ferrite::inference::inference::{load_model, GenerationConfig};

struct MistralDemo;

impl Guest for MistralDemo {
    fn run() -> Result<(), String> {
        println!("🔥 Mistral 7B Quantized Inference - WASM Edition");
        println!("================================================\n");

        // Load the quantized Mistral model
        println!("📦 Loading Mistral 7B Q4 (4-bit quantized)...");
        println!("⏳ This may take a minute on first run (downloading ~4GB model)...\n");

        let hf_token = std::env::var("HF_TOKEN").ok();

        let model = load_model("mistral-7b-q4", hf_token.as_deref())?;

        println!("✅ Model loaded successfully!\n");

        // Configure generation
        let config = GenerationConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 200,
            seed: Some(42),
        };

        println!("💬 Entering Interactive Chat Mode");
        println!("💡 Type '/exit' or '/quit' to stop.\n");

        use std::io::{self, Write};

        loop {
            print!("👤 You: ");
            io::stdout().flush().map_err(|e| e.to_string())?;

            let mut input = String::new();
            io::stdin().read_line(&mut input).map_err(|e| e.to_string())?;
            let prompt = input.trim();

            if prompt.is_empty() {
                continue;
            }

            if prompt == "/exit" || prompt == "/quit" {
                break;
            }

            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            print!("🤖 Mistral: ");
            io::stdout().flush().map_err(|e| e.to_string())?;

            match model.start_generate_stream(prompt, config) {
                Ok(generation) => {
                    while let Some(chunk) = generation.next_chunk()? {
                        print!("{chunk}");
                        io::stdout().flush().map_err(|e| e.to_string())?;
                    }
                    println!();
                }
                Err(e) => {
                    println!("❌ Error: {}\n", e);
                }
            }
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        }

        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("✨ Chat session ended. See you next time!");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

        Ok(())
    }
}

export!(MistralDemo);
