# Ferrite Quick Start

Get up and running in 60 seconds! ⚡

## Installation

```bash
curl -sSL https://raw.githubusercontent.com/DaronPopov/ferrite/main/install.sh | bash
```

That's it! The installer handles everything automatically:
- Installs Rust if needed
- Builds Ferrite
- Adds to your PATH

## First Run

```bash
# Restart your terminal or:
source ~/.bashrc

# Launch the interactive launcher
ferrite-chat
```

## Usage

```bash
# Interactive mode
ferrite-chat

# Specific model
qwen_inference          # Smallest (0.5B)
mistral_inference       # Standard (7B)
gemma_inference         # Google Gemma (2B)
phi_inference           # Microsoft Phi (2.7B)

# Quantized (4-bit, uses less VRAM)
mistral_quantized_inference
```

## Your First Chat

```
$ ferrite-chat

Select a model:
  1. TinyLlama 1.1B
  2. Qwen2 0.5B (fastest)
  3. Mistral 7B
  ...

Choice: 2

[Loading Qwen2-0.5B...]
Ready!

You: Hello! What are you?
Qwen2: I'm Ferrite, a lightweight LLM inference engine written in pure Rust...
```

## Configuration

### HuggingFace Token (Optional)

Some models require authentication:

```bash
ferrite-chat --login
# Enter your token from: https://huggingface.co/settings/tokens
```

### Environment Variables

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxx    # HuggingFace token
export CUDA_VISIBLE_DEVICES=0       # Select GPU
```

## Tips

1. **Start small**: Try `qwen_inference` first (0.5B model, fastest)
2. **GPU recommended**: 4-bit models need ~4GB VRAM, FP16 needs ~14GB
3. **First run slow**: Models download automatically (~500MB-7GB)
4. **Quantized = faster**: Use `*_quantized_inference` for 4-bit models

## Next Steps

- Read [README.md](README.md) for architecture details
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for advanced setups
- See [CONTRIBUTING.md](CONTRIBUTING.md) to add new models

**Need help?** Open an issue: https://github.com/DaronPopov/ferrite/issues
