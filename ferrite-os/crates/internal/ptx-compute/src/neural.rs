//! Neural network operations.
//!
//! This module provides high-level APIs for neural network forward passes,
//! including linear layers, activation functions, and multi-layer networks.

use std::sync::Arc;
use ptx_runtime::{PtxRuntime, CublasHandle, GemmOp, Stream, Result, Error};

/// Activation function type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    /// ReLU activation: f(x) = max(0, x)
    ReLU,
    /// GELU activation (Gaussian Error Linear Unit)
    GELU,
    /// Softmax activation
    Softmax,
    /// No activation (identity)
    None,
}

/// A single neural network layer.
///
/// Performs: output = activation(input @ weights)
pub struct Layer {
    weights: usize,
    input_size: usize,
    output_size: usize,
    activation: Activation,
}

impl Layer {
    /// Create a new layer.
    ///
    /// # Arguments
    ///
    /// * `weights` - GPU pointer to weight matrix (input_size × output_size)
    /// * `input_size` - Size of input dimension
    /// * `output_size` - Size of output dimension
    /// * `activation` - Activation function to apply
    pub fn new(
        weights: usize,
        input_size: usize,
        output_size: usize,
        activation: Activation,
    ) -> Self {
        Self {
            weights,
            input_size,
            output_size,
            activation,
        }
    }

    /// Apply activation function.
    unsafe fn apply_activation(
        &self,
        input: *mut f32,
        output: *mut f32,
        batch_size: usize,
        stream: &Stream,
    ) {
        let size = batch_size * self.output_size;
        match self.activation {
            Activation::ReLU => {
                ptx_sys::ptx_tensor_relu_f32(
                    input,
                    output,
                    size,
                    stream.raw(),
                );
            }
            Activation::GELU => {
                ptx_sys::ptx_tensor_gelu_f32(
                    input,
                    output,
                    size,
                    stream.raw(),
                );
            }
            Activation::Softmax => {
                ptx_sys::ptx_tensor_softmax_f32(
                    input,
                    output,
                    batch_size,
                    self.output_size,
                    stream.raw(),
                );
            }
            Activation::None => {
                // No-op or memcpy if needed
            }
        }
    }
}

/// Multi-layer neural network.
///
/// Manages forward pass through multiple layers with automatic
/// buffer management and activation functions.
pub struct Network {
    handle: CublasHandle,
    #[allow(dead_code)]
    runtime: Arc<PtxRuntime>,
    layers: Vec<Layer>,
    #[allow(dead_code)]
    hidden_size: usize,
}

impl Network {
    /// Create a new network.
    ///
    /// # Arguments
    ///
    /// * `runtime` - PTX runtime
    /// * `hidden_size` - Size of hidden dimensions
    /// * `num_layers` - Number of layers
    pub fn new(
        runtime: &Arc<PtxRuntime>,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        Ok(Self {
            handle: CublasHandle::new()?,
            runtime: Arc::clone(runtime),
            layers: Vec::with_capacity(num_layers),
            hidden_size,
        })
    }

    /// Add a layer to the network.
    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Perform forward pass through all layers.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor (batch_size × input_size)
    /// * `output` - Output tensor (batch_size × output_size)
    /// * `temp_buffer` - Temporary buffer for intermediate activations
    /// * `batch_size` - Batch size
    /// * `stream` - CUDA stream to use
    ///
    /// # Safety
    ///
    /// All pointers must be valid GPU memory with correct sizes.
    pub unsafe fn forward(
        &self,
        input: *const f32,
        output: *mut f32,
        temp_buffer: *mut f32,
        batch_size: usize,
        stream: &Stream,
    ) -> Result<()> {
        if self.layers.is_empty() {
            return Err(Error::Internal {
                message: "Network has no layers".to_string(),
            });
        }

        self.handle.set_stream(stream)?;

        let mut current_input = input;
        let mut current_output = temp_buffer;

        // Forward through each layer
        for (i, layer) in self.layers.iter().enumerate() {
            let is_last = i == self.layers.len() - 1;
            let final_output = if is_last { output } else { current_output };

            // Linear transformation: output = input @ weights
            self.handle.sgemm(
                GemmOp::None,
                GemmOp::None,
                layer.output_size as i32,
                batch_size as i32,
                layer.input_size as i32,
                1.0,
                layer.weights as *const f32,
                layer.output_size as i32,
                current_input,
                layer.input_size as i32,
                0.0,
                final_output,
                layer.output_size as i32,
            )?;

            // Apply activation
            if layer.activation != Activation::None {
                layer.apply_activation(
                    final_output,
                    final_output,
                    batch_size,
                    stream,
                );
            }

            // Swap buffers for next layer
            current_input = final_output;
            if !is_last {
                // Alternate between temp_buffer and temp_buffer + offset
                current_output = if current_output == temp_buffer {
                    temp_buffer.add(batch_size * layer.output_size)
                } else {
                    temp_buffer
                };
            }
        }

        Ok(())
    }

    /// Calculate FLOPS for a forward pass.
    pub fn forward_flops(&self, batch_size: usize) -> f64 {
        let mut total = 0.0;
        for layer in &self.layers {
            // GEMM operations
            total += 2.0 * batch_size as f64 * layer.input_size as f64 * layer.output_size as f64;
        }
        total
    }
}

/// Standalone activation functions.
pub mod activations {
    use ptx_runtime::{Stream, Result};

    /// Apply ReLU activation in-place.
    pub unsafe fn relu_f32(data: *mut f32, size: usize, stream: &Stream) -> Result<()> {
        ptx_sys::ptx_tensor_relu_f32(data, data, size, stream.raw());
        Ok(())
    }

    /// Apply GELU activation.
    pub unsafe fn gelu_f32(
        input: *const f32,
        output: *mut f32,
        size: usize,
        stream: &Stream,
    ) -> Result<()> {
        ptx_sys::ptx_tensor_gelu_f32(input as *mut f32, output, size, stream.raw());
        Ok(())
    }

    /// Apply Softmax activation.
    pub unsafe fn softmax_f32(
        input: *const f32,
        output: *mut f32,
        batch_size: usize,
        feature_size: usize,
        stream: &Stream,
    ) -> Result<()> {
        ptx_sys::ptx_tensor_softmax_f32(
            input as *mut f32,
            output,
            batch_size,
            feature_size,
            stream.raw(),
        );
        Ok(())
    }
}
