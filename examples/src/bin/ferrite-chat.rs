// Ferrite Chat - Unified LLM Inference CLI
//
// Usage:
//   ferrite-chat                    # Interactive model selection
//   ferrite-chat mistral            # Run Mistral
//   ferrite-chat --list             # List available models
//   ferrite-chat --login            # Set HuggingFace token
//   ferrite-chat --logout           # Remove stored token
//   ferrite-chat --quantized        # Use quantized model variant

use ferrite_examples::config::{self, ModelFamily};
use ferrite_examples::models::ModelInfo;
use std::io::{self, Write};
use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Handle special commands
    if args.len() > 1 {
        match args[1].as_str() {
            "--help" | "-h" => {
                print_help();
                return;
            }
            "--list" | "-l" => {
                list_models();
                return;
            }
            "--login" => {
                handle_login();
                return;
            }
            "--logout" => {
                handle_logout();
                return;
            }
            "--status" => {
                show_status();
                return;
            }
            _ => {}
        }
    }

    // Setup environment with stored token
    config::setup_env();

    // Parse model selection
    let quantized = args.iter().any(|a| a == "--quantized" || a == "-q");
    let model_arg = args.iter()
        .skip(1)
        .find(|a| !a.starts_with('-'));

    if let Some(model_name) = model_arg {
        run_model(model_name, quantized);
    } else {
        interactive_menu();
    }
}

fn print_help() {
    println!(r#"
Ferrite Chat - LLM Inference Platform

USAGE:
    ferrite-chat [OPTIONS] [MODEL]

OPTIONS:
    -h, --help       Show this help message
    -l, --list       List all available models
    -q, --quantized  Use quantized (4-bit) model variant if available
    --login          Set your HuggingFace token
    --logout         Remove stored HuggingFace token
    --status         Show current configuration status

MODELS:
    llama, tinyllama     TinyLlama 1.1B Chat
    mistral              Mistral 7B Instruct
    qwen, qwen2          Qwen2 0.5B/1.5B Instruct
    gemma                Google Gemma 2B (requires HF token)
    phi, phi2            Microsoft Phi-2

EXAMPLES:
    ferrite-chat                  # Interactive menu
    ferrite-chat mistral          # Run Mistral 7B
    ferrite-chat mistral -q       # Run Mistral 7B quantized (4-bit)
    ferrite-chat --login          # Setup HuggingFace token
"#);
}

fn list_models() {
    println!("\n Available Models\n");
    println!("{:<25} {:<10} {:<10} {}", "NAME", "SIZE", "QUANT", "DESCRIPTION");
    println!("{}", "─".repeat(80));

    for model in ModelInfo::all() {
        let quant = if model.quantized { "Q4" } else { "FP16" };
        println!("{:<25} {:<10} {:<10} {}",
                 model.name, model.size, quant, model.description);
    }
    println!();
}

fn handle_login() {
    println!("\n HuggingFace Token Setup\n");
    println!("Some models (Gemma, Llama) require accepting a license on HuggingFace.");
    println!("Get your token at: https://huggingface.co/settings/tokens\n");

    if let Some(existing) = config::get_token() {
        let masked = format!("{}...{}", &existing[..4], &existing[existing.len()-4..]);
        println!("Current token: {}", masked);
        print!("Replace existing token? [y/N]: ");
        io::stdout().flush().unwrap();

        let mut response = String::new();
        io::stdin().read_line(&mut response).unwrap();
        if !response.trim().eq_ignore_ascii_case("y") {
            println!("Keeping existing token.");
            return;
        }
    }

    match config::prompt_for_token() {
        Ok(token) if !token.is_empty() => {
            match config::save_token(&token) {
                Ok(_) => println!("\nToken saved to {:?}", config::token_path()),
                Err(e) => eprintln!("Failed to save token: {}", e),
            }
        }
        Ok(_) => println!("No token provided."),
        Err(e) => eprintln!("Error reading input: {}", e),
    }
}

fn handle_logout() {
    match config::delete_token() {
        Ok(_) => println!("Token removed."),
        Err(e) => eprintln!("Error removing token: {}", e),
    }
}

fn show_status() {
    println!("\n Ferrite Configuration\n");

    // Token status
    if let Some(token) = config::get_token() {
        let masked = format!("{}...{}", &token[..4.min(token.len())],
                            &token[token.len().saturating_sub(4)..]);
        println!("HuggingFace token: {} (stored)", masked);
    } else if std::env::var("HF_TOKEN").is_ok() {
        println!("HuggingFace token: (from environment)");
    } else {
        println!("HuggingFace token: Not configured");
        println!("  Run 'ferrite-chat --login' to set up");
    }

    println!("Config directory:  {:?}", config::config_dir());

    // CUDA status
    println!();
    let cuda_status = Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total")
        .arg("--format=csv,noheader")
        .output();

    match cuda_status {
        Ok(output) if output.status.success() => {
            let gpu_info = String::from_utf8_lossy(&output.stdout);
            println!("GPU: {}", gpu_info.trim());
        }
        _ => println!("GPU: Not detected (will use CPU)"),
    }
    println!();
}

fn interactive_menu() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║           Ferrite Chat - LLM Inference Platform              ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    println!("Select a model:\n");
    println!("  [1] TinyLlama 1.1B    - Fast, lightweight");
    println!("  [2] Mistral 7B        - High quality");
    println!("  [3] Mistral 7B Q4     - Quantized, ~4GB VRAM");
    println!("  [4] Qwen2 0.5B        - Tiny multilingual");
    println!("  [5] Gemma 2B          - Google (requires token)");
    println!("  [6] Phi-2             - Strong reasoning (~11GB)");
    println!("  [7] Phi-2 Q4          - Quantized, ~2GB VRAM");
    println!();
    println!("  [t] Setup HuggingFace token");
    println!("  [q] Quit");
    println!();

    print!("Choice: ");
    io::stdout().flush().unwrap();

    let mut choice = String::new();
    io::stdin().read_line(&mut choice).unwrap();

    match choice.trim() {
        "1" => run_model("llama", false),
        "2" => run_model("mistral", false),
        "3" => run_model("mistral", true),
        "4" => run_model("qwen", false),
        "5" => run_model("gemma", false),
        "6" => run_model("phi", false),
        "7" => run_model("phi", true),
        "t" | "T" => handle_login(),
        "q" | "Q" => println!("Goodbye!"),
        _ => eprintln!("Invalid choice"),
    }
}

fn run_model(name: &str, quantized: bool) {
    let family = match ModelFamily::from_str(name) {
        Some(f) => f,
        None => {
            eprintln!("Unknown model: {}", name);
            eprintln!("Run 'ferrite-chat --list' to see available models");
            return;
        }
    };

    // Check if token is needed and prompt if missing
    if family.requires_token() && config::get_token().is_none() && std::env::var("HF_TOKEN").is_err() {
        println!("\n{} requires a HuggingFace token.", family.display_name());
        println!("Get your token at: https://huggingface.co/settings/tokens\n");
        print!("Enter HuggingFace token (or press Enter to skip): ");
        io::stdout().flush().unwrap();

        let mut token = String::new();
        io::stdin().read_line(&mut token).unwrap();
        let token = token.trim();

        if !token.is_empty() {
            match config::save_token(token) {
                Ok(_) => {
                    println!("Token saved!\n");
                    config::setup_env();
                }
                Err(e) => eprintln!("Failed to save token: {}\n", e),
            }
        } else {
            println!("Continuing without token (model may fail to download)...\n");
        }
    }

    // Determine which binary to run
    let binary_name = match (family, quantized) {
        (ModelFamily::Llama, _) => "tinyllama_inference",
        (ModelFamily::Mistral, true) => "mistral_quantized_inference",
        (ModelFamily::Mistral, false) => "mistral_inference",
        (ModelFamily::Qwen, _) => "qwen_inference",
        (ModelFamily::Gemma, _) => "gemma_inference",
        (ModelFamily::Phi, true) => "phi_quantized_inference",
        (ModelFamily::Phi, false) => "phi_inference",
        (ModelFamily::Gpt, _) => "gpt_inference",
    };

    println!("\nLaunching {}...\n", family.display_name());

    // Find binary: check same directory as this executable, then PATH
    let binary_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join(binary_name)))
        .filter(|p| p.exists())
        .unwrap_or_else(|| std::path::PathBuf::from(binary_name));

    let status = Command::new(&binary_path).status();

    match status {
        Ok(s) if !s.success() => eprintln!("Model exited with error"),
        Err(e) => {
            eprintln!("Failed to launch model: {}", e);
            eprintln!("Binary path: {:?}", binary_path);
            eprintln!("\nTry running directly: {}", binary_name);
        }
        _ => {}
    }
}
