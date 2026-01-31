//! Hello Inference - Example Ferrite WASM Module
//!
//! This is a simple example demonstrating how to write a neural inference
//! module that runs in the ferrite runtime.

// Generate the complete WIT bindings
wit_bindgen::generate!({
    path: "../../../wit",
    world: "ferrite-guest",
});

// Import the ferrite inference functions
use ferrite::inference::inference::{load_model, GenerationConfig};

struct Component;

// Export the run function as required by the ferrite-guest world
impl Guest for Component {
    fn run() -> Result<(), String> {
        // Load a model
        let model = load_model("qwen2-0.5b", None)?;

        // Configure generation
        let config = GenerationConfig {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 50,
            max_tokens: 100,
            seed: Some(42),
        };

        // Generate text
        let prompt = "What is Rust?";
        match model.generate(prompt, config) {
            Ok(_response) => {
                // Success
            }
            Err(e) => {
                return Err(format!("Generation failed: {}", e));
            }
        }

        Ok(())
    }
}

export!(Component);
