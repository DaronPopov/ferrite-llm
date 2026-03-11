# PTX-Compute Examples

Examples demonstrating the high-level compute API.

## Running Examples

All examples require the PTX-OS library to be in the library path:

```bash
LD_LIBRARY_PATH=/path/to/ferriterc/ferrite-os/lib:$LD_LIBRARY_PATH \
  cargo run --release -p ptx-compute --example EXAMPLE_NAME
```

## Examples

### simple_matmul

Basic matrix multiplication demonstrating the clean API.

```bash
cargo run --release -p ptx-compute --example simple_matmul
```

**Features:**
- Clean, ergonomic API
- Automatic cuBLAS handle management
- RAII memory management

### tiled_matmul

Custom tiling configuration for optimal performance (like NVIDIA CuTe).

```bash
cargo run --release -p ptx-compute --example tiled_matmul
```

**Features:**
- Custom tile configuration
- Multi-level tiling (thread, block, grid)
- Memory hierarchy optimization
- Tile iterator for large tensors
- Auto-tuning suggestions

**What you'll learn:**
- How to configure block and thread tiles
- Memory usage calculation
- Grid dimension computation
- Optimal tiling strategies

## See Also

For benchmarks and stress tests, see `ptx-runtime/examples/`.
