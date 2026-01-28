// Configuration and HuggingFace token management

use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

const CONFIG_DIR: &str = ".ferrite";
const TOKEN_FILE: &str = "hf_token";

/// Get the config directory path
pub fn config_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(CONFIG_DIR)
}

/// Get the token file path
pub fn token_path() -> PathBuf {
    config_dir().join(TOKEN_FILE)
}

/// Read the stored HuggingFace token
pub fn get_token() -> Option<String> {
    fs::read_to_string(token_path())
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Save a HuggingFace token
pub fn save_token(token: &str) -> io::Result<()> {
    let dir = config_dir();
    fs::create_dir_all(&dir)?;

    let path = token_path();
    fs::write(&path, token.trim())?;

    // Set restrictive permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600))?;
    }

    Ok(())
}

/// Delete the stored token
pub fn delete_token() -> io::Result<()> {
    let path = token_path();
    if path.exists() {
        fs::remove_file(path)?;
    }
    Ok(())
}

/// Prompt user for token interactively
pub fn prompt_for_token() -> io::Result<String> {
    print!("Enter your HuggingFace token: ");
    io::stdout().flush()?;

    let mut token = String::new();
    io::stdin().read_line(&mut token)?;

    Ok(token.trim().to_string())
}

/// Set the HF_TOKEN environment variable if we have a stored token
pub fn setup_env() {
    if std::env::var("HF_TOKEN").is_err() {
        if let Some(token) = get_token() {
            std::env::set_var("HF_TOKEN", token);
        }
    }
}

/// Available model families
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelFamily {
    Llama,
    Mistral,
    Qwen,
    Gemma,
    Phi,
    Gpt,
}

impl ModelFamily {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "llama" | "tinyllama" => Some(Self::Llama),
            "mistral" => Some(Self::Mistral),
            "qwen" | "qwen2" => Some(Self::Qwen),
            "gemma" => Some(Self::Gemma),
            "phi" | "phi2" | "phi-2" => Some(Self::Phi),
            "gpt" | "gpt2" => Some(Self::Gpt),
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::Llama => "TinyLlama",
            Self::Mistral => "Mistral",
            Self::Qwen => "Qwen2",
            Self::Gemma => "Gemma",
            Self::Phi => "Phi-2",
            Self::Gpt => "GPT-2",
        }
    }

    pub fn default_model_id(&self) -> &'static str {
        match self {
            Self::Llama => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            Self::Mistral => "mistralai/Mistral-7B-Instruct-v0.2",
            Self::Qwen => "Qwen/Qwen2-0.5B-Instruct",
            Self::Gemma => "google/gemma-2b-it",
            Self::Phi => "microsoft/phi-2",
            Self::Gpt => "gpt2",
        }
    }

    pub fn requires_token(&self) -> bool {
        matches!(self, Self::Gemma | Self::Llama)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_family_from_str() {
        assert_eq!(ModelFamily::from_str("llama"), Some(ModelFamily::Llama));
        assert_eq!(ModelFamily::from_str("MISTRAL"), Some(ModelFamily::Mistral));
        assert_eq!(ModelFamily::from_str("unknown"), None);
    }
}
