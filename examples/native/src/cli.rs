//! CLI utilities for interactive chatbot examples
//!
//! Provides common functions for user interaction, banners, and input handling.

use std::io::{self, Write};

/// Print a formatted banner for a model
pub fn print_banner(model_name: &str, model_id: &str) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  {:<58}  ║", format!("{} INFERENCE - Candle Backend", model_name.to_uppercase()));
    println!("║  {:<58}  ║", format!("Model: {}", model_id));
    println!("╚══════════════════════════════════════════════════════════════╝\n");
}

/// Print the ready prompt for interactive mode
pub fn print_ready_prompt() {
    println!("\n════════════════════════════════════════════════════════════════");
    println!("  Ready! Type a prompt and press Enter. 'quit' to exit.");
    println!("════════════════════════════════════════════════════════════════\n");
}

/// Read a line from stdin with a prompt
///
/// Returns None if user wants to quit, Some(input) otherwise
pub fn read_user_input(prompt: &str) -> Result<Option<String>, io::Error> {
    print!("{}: ", prompt);
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    // Check for quit commands
    if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") {
        return Ok(None);
    }

    // Skip empty input
    if input.is_empty() {
        return read_user_input(prompt);
    }

    Ok(Some(input.to_string()))
}

/// Print assistant response prompt
pub fn print_assistant_prompt(model_name: &str) {
    print!("{}: ", model_name);
    io::stdout().flush().ok();
}

/// Print goodbye message
pub fn print_goodbye() {
    println!("\nGoodbye!");
}

/// Run an interactive chat loop
///
/// Takes a closure that generates a response for each user input
pub fn interactive_loop<F>(
    user_prompt: &str,
    assistant_name: &str,
    mut generate_fn: F,
) -> Result<(), Box<dyn std::error::Error>>
where
    F: FnMut(&str) -> Result<(), Box<dyn std::error::Error>>,
{
    print_ready_prompt();

    loop {
        match read_user_input(user_prompt)? {
            Some(input) => {
                print_assistant_prompt(assistant_name);
                if let Err(e) = generate_fn(&input) {
                    println!("Error: {}", e);
                }
            }
            None => {
                print_goodbye();
                break;
            }
        }
    }

    Ok(())
}

/// Print ChatSession ready prompt with commands
pub fn print_chat_session_prompt() {
    println!("\n====================================================================");
    println!("  Ready! Type a message and press Enter. Commands:");
    println!("    'quit' or 'exit' - Exit the program");
    println!("    'clear'          - Reset conversation history");
    println!("    'stats'          - Show cache statistics");
    println!("====================================================================\n");
}
