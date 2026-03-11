# PTX-Kernels: Candle CUDA Kernels for PTX-OS

Optimized CUDA kernels from Candle, integrated with PTX-OS TLSF allocator through a thin guard/validation layer.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  User Application                                       │
│  (Your ML code, training loops, inference)              │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Safe API (safe_api.rs)                                 │
│  • unary::gelu(), relu(), silu(), etc.                  │
│  • binary::add(), mul(), sub(), div()                   │
│  • Type-safe, ergonomic wrappers                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Guard Layer (guards.rs)                                │
│  • GuardedBuffer - validates TLSF ownership             │
│  • KernelContext - stream management                    │
│  • Memory bounds checking                               │
│  • Automatic error handling                             │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  FFI Bindings (candle.rs)                               │
│  • Raw extern "C" function declarations                 │
│  • Direct kernel launch wrappers                        │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  C++ Kernel Launchers (launcher.cu)                     │
│  • Kernel launch configuration (blocks, threads)        │
│  • <<<>>> invocation syntax                             │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Candle CUDA Kernels (unary.cu, binary.cu, etc.)       │
│  • Optimized GPU kernels from Hugging Face Candle       │
│  • Element-wise operations, activations, reductions     │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  PTX-OS TLSF Allocator                                  │
│  • Two-Level Segregated Fit memory allocator           │
│  • O(1) allocation/deallocation                         │
│  • Minimal fragmentation                                │
│  • Stream-ordered operations                            │
└─────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Guard Layer (`guards.rs`)

**Purpose**: Ensures memory safety before kernel execution

**Features**:
- ✅ **GuardedBuffer**: Validates TLSF allocator ownership
- ✅ **Bounds checking**: Prevents buffer overruns
- ✅ **Type safety**: Ensures correct element counts
- ✅ **Error handling**: Clear, actionable error messages

**Example**:
```rust
// Create a guarded buffer (validates TLSF ownership)
let buffer = unsafe {
    GuardedBuffer::new(gpu_ptr, size_bytes, runtime.raw())?
};

// Automatic validation
buffer.validate_capacity::<f32>(count)?;  // Check capacity
buffer.revalidate()?;                      // Re-check TLSF ownership
```

### 2. Safe API (`safe_api.rs`)

**Purpose**: Ergonomic, type-safe kernel operations

**Features**:
- ✅ **Unary operations**: GELU, ReLU, SiLU, Sigmoid, Abs, Sqrt, Exp, Log, Tanh
- ✅ **Binary operations**: Add, Mul, Sub, Div
- ✅ **Automatic validation**: All parameters checked before launch
- ✅ **Error propagation**: Results instead of panics

**Example**:
```rust
use ptx_kernels::{unary, binary, GuardedBuffer, KernelContext};

let context = KernelContext::new(runtime.raw(), stream.raw());

// Execute GELU activation with automatic validation
unary::gelu(&input_buf, &output_buf, num_elements, &context)?;

// Chain operations
unary::relu(&output_buf, &temp_buf, num_elements, &context)?;
binary::add(&temp_buf, &input_buf, &output_buf, num_elements, &context)?;

// Sync and check for errors
context.sync()?;
```

### 3. FFI Bindings (`candle.rs`)

**Purpose**: Low-level kernel launch interface

**Example**:
```rust
extern "C" {
    pub fn candle_launch_ugelu_f32(
        numel: size_t,
        num_dims: size_t,
        info: *const size_t,
        inp: *const f32,
        out: *mut f32,
        stream: cudaStream_t,
    );
}
```

## Usage

### Basic Example

```rust
use ptx_runtime::PtxRuntime;
use ptx_kernels::{GuardedBuffer, KernelContext, unary};

// Initialize runtime
let runtime = PtxRuntime::new(0)?;
let stream = runtime.next_stream();

// Allocate memory via TLSF
let input_ptr = runtime.alloc_async(bytes, &stream)?;
let output_ptr = runtime.alloc_async(bytes, &stream)?;

// Create guarded buffers
let input = unsafe { GuardedBuffer::new(input_ptr, bytes, runtime.raw())? };
let output = unsafe { GuardedBuffer::new(output_ptr, bytes, runtime.raw())? };

// Create kernel context
let context = KernelContext::new(runtime.raw(), stream.raw());

// Execute kernel with full validation
unary::gelu(&input, &output, num_elements, &context)?;

// Sync
context.sync()?;
```

### Advanced Example - Kernel Chain

```rust
// Multi-stage computation with automatic validation
unary::exp(&input, &temp1, N, &ctx)?;          // exp(x)
unary::log(&temp1, &temp2, N, &ctx)?;          // log(exp(x)) = x
binary::mul(&temp2, &weights, &temp3, N, &ctx)?; // x * w
binary::add(&temp3, &bias, &output, N, &ctx)?;   // x * w + b
ctx.sync()?;
```

## Memory Model Integration

### TLSF Allocator Assumptions

The guard layer ensures:
1. **Pointer Ownership**: All pointers must be owned by PTX-OS TLSF
2. **Bounds Checking**: Buffer size >= required bytes
3. **Stream Ordering**: Operations respect stream dependencies
4. **Validation**: Automatic revalidation before kernel launch

### Validation Flow

```
┌────────────────┐
│ User creates   │
│ GuardedBuffer  │
└───────┬────────┘
        │
        ▼
┌────────────────────────────┐
│ Validate TLSF ownership    │
│ gpu_hot_owns_ptr()         │
└───────┬────────────────────┘
        │
        ▼
┌────────────────────────────┐
│ Check buffer capacity      │
│ size >= N * sizeof(T)      │
└───────┬────────────────────┘
        │
        ▼
┌────────────────────────────┐
│ Create operation guard     │
│ (UnaryOpGuard, BinaryOp)   │
└───────┬────────────────────┘
        │
        ▼
┌────────────────────────────┐
│ Launch kernel              │
│ (memory is guaranteed safe)│
└────────────────────────────┘
```

## Error Handling

All operations return `GuardResult<T>` with detailed errors:

```rust
pub enum GuardError {
    InvalidPointer { ptr: usize },
    BufferTooSmall { required: usize, available: usize },
    NullPointer { operation: &'static str },
    KernelLaunchFailed { kernel: &'static str, error_code: i32 },
    StreamSyncFailed { error_code: i32 },
}
```

**Example**:
```rust
match unary::gelu(&input, &output, N, &ctx) {
    Ok(()) => println!("Success!"),
    Err(GuardError::BufferTooSmall { required, available }) => {
        eprintln!("Buffer too small: need {} bytes, have {}", required, available);
    }
    Err(GuardError::InvalidPointer { ptr }) => {
        eprintln!("Pointer 0x{:x} not owned by TLSF allocator", ptr);
    }
    Err(e) => eprintln!("Error: {}", e),
}
```

## Building

```bash
# Build with default settings (sm_80)
cargo build --release

# Build for specific CUDA architecture
CUDA_ARCH=sm_86 cargo build --release

# Build with features
cargo build --release --features candle-kernels
```

## Examples

Run the safe API demo:
```bash
cargo run --example safe_api_demo --release
```

Expected output:
```
🛡️  Safe Candle Kernels API Demo
=================================

✓ PTX-OS runtime initialized
✓ Using stream ID: 0

📊 Configuration:
   Elements: 1048576
   Size: 4 MB

🔧 Allocating GPU memory...
🛡️  Creating guarded buffers...
   ✓ Input buffer:  0x7f8a00000000 (1048576 elements)
   ✓ Temp buffer:   0x7f8a00400000 (1048576 elements)
   ✓ Output buffer: 0x7f8a00800000 (1048576 elements)

🔥 Executing kernel chain:
   1. GELU activation...
      ✓ GELU completed
   2. ReLU activation...
      ✓ ReLU completed
   3. Element-wise multiplication...
      ✓ Multiplication completed
   4. Element-wise addition...
      ✓ Addition completed

✅ All operations completed successfully!
```

## Performance

Candle kernels are highly optimized:
- **Coalesced memory access** - Maximizes bandwidth
- **Warp-efficient** - Minimizes divergence
- **Register-optimized** - Reduces memory traffic
- **Stream-ordered** - Overlaps computation and memory

Benchmarks on A100 (sm_80):
- GELU (1M elements): ~0.05ms
- ReLU (1M elements): ~0.03ms
- Add (1M elements): ~0.03ms

## Future Extensions

- [ ] Reduction operations (sum, mean, max, min)
- [ ] Softmax/LogSoftmax
- [ ] Convolution operations
- [ ] Quantized operations (INT8, FP16)
- [ ] Strided tensor support
- [ ] Multi-dimensional operations

## License

PTX-Kernels: MIT OR Apache-2.0
Candle Kernels: MIT OR Apache-2.0 (Copyright © 2023-2025 Hugging Face, Inc.)
