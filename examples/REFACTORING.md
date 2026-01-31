# Chatbot Examples Refactoring

## Summary

We've consolidated duplicate code across all chatbot examples by extracting common patterns into shared utilities.

## New Shared Utilities

### `examples/src/precision.rs`
- **Precision** enum with `F32`, `F16`, `BF16` variants
- `from_device()` - Auto-select optimal precision based on device
- `needs_f32_conversion()` - Check if dtype conversion needed for sampling

### `examples/src/cli.rs`
- `print_banner()` - Formatted model banner
- `print_ready_prompt()` - Interactive mode instructions
- `read_user_input()` - Read with quit detection
- `interactive_loop()` - Complete REPL loop with error handling
- `print_assistant_prompt()` - Model response prefix
- `print_goodbye()` - Exit message

### `examples/src/utils.rs`
- `get_device()` - CUDA detection with CPU fallback
- `download_model()` - HuggingFace Hub download
- `load_safetensors()` - Load model weights from directory
- `print_tps_stats()` - Tokens-per-second formatting
- **GenerationTimer** - Performance tracking helper

## Code Reduction

### Before Refactoring
```rust
// qwen_inference.rs - 296 lines
// Includes:
// - Precision enum (13 lines)
// - Device selection (8 lines)
// - Model loading (30 lines)
// - Interactive loop (40 lines)
// - TPS tracking (5 lines)
// - Banner printing (3 lines)
```

### After Refactoring
```rust
// qwen_inference_refactored.rs - 185 lines (-37%)
// Reuses:
// - Precision from shared module
// - Device selection: get_device()
// - Model loading: load_safetensors()
// - Interactive loop: cli::interactive_loop()
// - TPS tracking: print_tps_stats()
// - Banner: cli::print_banner()
```

**Result: 111 lines removed from a single example**

## Benefits

### Maintainability
- Bug fixes in one place benefit all examples
- Consistent behavior across all models
- Easier to understand example-specific logic

### Code Quality
- DRY (Don't Repeat Yourself) principle
- Single source of truth for common patterns
- Reduced test surface area

### Developer Experience
- Less boilerplate when adding new models
- Focus on model-specific logic only
- Clear separation of concerns

## Migration Guide

### Old Pattern
```rust
#[derive(Clone, Copy)]
pub enum Precision {
    F32, F16, BF16,
}

impl Precision {
    fn dtype(&self) -> DType { /* ... */ }
}

// ... 30 lines of model loading ...
// ... 40 lines of interactive loop ...
```

### New Pattern
```rust
use ferrite_examples::{
    cli, get_device, download_model,
    load_safetensors, Precision, print_tps_stats
};

// Model-specific logic only
fn model_config() -> Config { /* ... */ }

// Reuse shared utilities
let device = get_device()?;
let precision = Precision::from_device(&device);
let model_dir = download_model(model_id).await?;
let vb = load_safetensors(&model_dir, precision.dtype(), &device)?;

cli::interactive_loop("You", "ModelName", |input| {
    // Generation logic
})?;
```

## Next Steps

### Apply to All Examples
Refactor the remaining 9 chatbot examples:
- ✅ `qwen/qwen_inference.rs` - Example refactored
- ⬜ `mistral/mistral_inference.rs`
- ⬜ `mistral/mistral_quantized_inference.rs`
- ⬜ `mistral/mistral_streaming.rs`
- ⬜ `llama/tinyllama_inference.rs`
- ⬜ `gemma/gemma_inference.rs`
- ⬜ `phi/phi_inference.rs`
- ⬜ `phi/phi_quantized_inference.rs`

### Estimated Savings
- **Current total:** ~2,800 lines across 10 examples
- **After refactoring:** ~1,800 lines (-36%)
- **Removed duplication:** ~1,000 lines

### Additional Improvements
- Add shared quantization loading helper
- Create model config registry to reduce hardcoded configs
- Add common stop sequence handling
- Extract streaming utilities

## Testing

After refactoring, verify:
1. Each example compiles: `cargo build --release -p ferrite-examples`
2. Interactive mode works: Test user input/output
3. Model loading succeeds: Check weights load correctly
4. Generation works: Verify output quality unchanged
5. Stats accurate: Confirm TPS matches previous version

## Backwards Compatibility

The original example files remain unchanged. The refactored versions are:
- Side-by-side for comparison
- Drop-in replacements when ready
- Can coexist during migration period

Once validated, replace originals with refactored versions.
