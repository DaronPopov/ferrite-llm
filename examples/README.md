# Ferrite Inference Platform

A unified Rust-based inference platform for LLM chatbots and ML models.

## Quick Start

```bash
# Interactive model selection
cargo run --release --bin ferrite-chat

# Run a specific model
cargo run --release --bin ferrite-chat -- mistral

# Run quantized version (less VRAM)
cargo run --release --bin ferrite-chat -- mistral -q

# Setup HuggingFace token (required for some models)
cargo run --release --bin ferrite-chat -- --login
```

## Available Models

| Model | Size | VRAM | Description |
|-------|------|------|-------------|
| TinyLlama | 1.1B | ~3GB | Fast, lightweight chat model |
| Mistral 7B | 7B | ~14GB | High quality instruction model |
| Mistral 7B Q4 | 7B (4-bit) | ~4GB | Quantized, runs on most GPUs |
| Qwen2 0.5B | 0.5B | ~1GB | Tiny multilingual model |
| Gemma 2B | 2B | ~5GB | Google's efficient model |
| Phi-2 | 2.7B | ~6GB | Strong reasoning & code |

## Project Structure

```
examples/
├── chatbots/           # LLM inference examples
│   ├── llama/          # Llama family (TinyLlama)
│   ├── mistral/        # Mistral family (FP16 + quantized)
│   ├── qwen/           # Qwen family
│   ├── gemma/          # Google Gemma
│   ├── phi/            # Microsoft Phi
│   └── gpt/            # GPT-2 (Ferrite native)
├── vision/             # Image classification
├── embeddings/         # BERT embeddings
├── training/           # Training examples
└── src/
    ├── lib.rs          # Library root
    ├── config.rs       # HuggingFace token management
    └── models.rs       # Model registry
```

## HuggingFace Token

Some models require accepting a license on HuggingFace:
- **Gemma**: https://huggingface.co/google/gemma-2b-it
- **Llama variants**: Check model page for requirements

Setup your token:
```bash
cargo run --release --bin ferrite-chat -- --login
```

Or set the environment variable:
```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx
```

## Running Individual Models

```bash
# Llama
cargo run --release --bin tinyllama_inference

# Mistral
cargo run --release --bin mistral_inference
cargo run --release --bin mistral_quantized_inference

# Qwen
cargo run --release --bin qwen_inference

# Gemma
cargo run --release --bin gemma_inference

# Phi
cargo run --release --bin phi_inference

# GPT (Ferrite native backend)
cargo run --release --bin gpt_inference
```

## Performance Stats

All models display tokens-per-second (TPS) after each generation:
```
[Stats] 42 tokens in 3.21s (13.1 tok/s)
```
