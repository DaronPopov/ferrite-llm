//! Tensor operations — the verb vocabulary for ferrite-gpu-lang.
//!
//! Categories:
//! - binary: add, sub, mul, div, max, min, mod, scalar ops
//! - unary: exp, log, sqrt, sin, cos, tanh, erf, ...
//! - activation: relu, gelu, sigmoid, silu, ...
//! - comparison: eq, ne, lt, le, gt, ge
//! - logical: and, or, not, xor
//! - reduction: sum, mean, max, min, prod, argmax, argmin, var, std
//! - scan: cumsum, cumprod, cummax, cummin
//! - broadcast: broadcast binary ops with shape broadcasting
//! - transform: cat, stack, repeat, masked_fill
//! - pad: 2D constant padding
//! - casting: to_f32, to_f16, to_i32
//! - norm: L1, L2, normalize
//! - random: rand, randn
//! - gather, indexing, topk, argsort, where_cond
//! - softmax, matmul
//! - nn: layer_norm, rms_norm, dropout, embedding, batch_norm, attention
//! - loss: mse, cross_entropy, binary_cross_entropy
//! - conv: conv2d (im2col + GEMM)
//! - pool: max_pool2d, avg_pool2d, adaptive_avg_pool2d
//! - optim: SGD, Adam

pub mod binary;
pub mod unary;
pub mod activation;
pub mod comparison;
pub mod logical;
pub mod reduction;
pub mod scan;
pub mod broadcast;
pub mod transform;
pub mod pad;
pub mod casting;
pub mod norm;
pub mod random;
pub mod gather;
pub mod indexing;
pub mod topk;
pub mod argsort;
pub mod where_cond;
pub mod softmax;
pub mod matmul;
pub mod nn;
pub mod loss;
pub mod conv;
pub mod pool;
pub mod optim;

// Re-export enums
pub use binary::*;
pub use unary::*;
pub use activation::*;
pub use reduction::*;
pub use comparison::*;
pub use matmul::*;
pub use loss::Reduction;
