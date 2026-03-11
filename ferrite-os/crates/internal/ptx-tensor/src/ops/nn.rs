//! Neural network operations — composed from existing tensor primitives.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error};

impl Tensor {
    /// Embedding lookup: select rows from a weight matrix by index.
    ///
    /// - weight: (vocab_size, embed_dim) F32
    /// - indices: (*) I32 tensor of token ids
    /// - Returns: (*, embed_dim)
    pub fn embedding(weight: &Tensor, indices: &Tensor) -> Result<Tensor> {
        weight.index_select(0, indices)
    }

    /// Layer normalization over the last `normalized_dims` dimensions.
    ///
    /// For a tensor of shape (*, D1, D2, ..., Dn) with normalized_dims = n,
    /// computes mean/var over the last n dims, normalizes, then applies
    /// optional affine: out = (x - mean) / sqrt(var + eps) * weight + bias.
    pub fn layer_norm(
        &self,
        normalized_dims: usize,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "layer_norm only supported for F32".to_string(),
            });
        }
        if normalized_dims == 0 || normalized_dims > self.ndim() {
            return Err(Error::Internal {
                message: format!(
                    "normalized_dims={} invalid for {}D tensor",
                    normalized_dims,
                    self.ndim()
                ),
            });
        }

        // Flatten the last normalized_dims into a single dim for mean/var computation.
        // Input shape: (batch..., D) where D = product of last normalized_dims.
        let ndim = self.ndim();
        let norm_size: usize = self.shape()[ndim - normalized_dims..].iter().product();
        let batch_size: usize = self.shape()[..ndim - normalized_dims].iter().product();
        let flat = self.reshape(&[batch_size, norm_size])?;

        // mean and var over last dim
        let mean = flat.mean_keepdim(-1)?;
        let centered = flat.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(-1)?;
        let inv_std = var.add_scalar(eps)?.rsqrt()?;
        let mut normed = centered.broadcast_mul(&inv_std)?;

        // Reshape back to original shape
        normed = normed.reshape(self.shape())?;

        // Optional affine
        if let Some(w) = weight {
            normed = normed.broadcast_mul(w)?;
        }
        if let Some(b) = bias {
            normed = normed.broadcast_add(b)?;
        }

        Ok(normed)
    }

    /// RMS normalization over the last `normalized_dims` dimensions.
    ///
    /// out = x / sqrt(mean(x^2) + eps) * weight (optional)
    pub fn rms_norm(
        &self,
        normalized_dims: usize,
        weight: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "rms_norm only supported for F32".to_string(),
            });
        }
        if normalized_dims == 0 || normalized_dims > self.ndim() {
            return Err(Error::Internal {
                message: format!(
                    "normalized_dims={} invalid for {}D tensor",
                    normalized_dims,
                    self.ndim()
                ),
            });
        }

        let ndim = self.ndim();
        let norm_size: usize = self.shape()[ndim - normalized_dims..].iter().product();
        let batch_size: usize = self.shape()[..ndim - normalized_dims].iter().product();
        let flat = self.reshape(&[batch_size, norm_size])?;

        let rms = flat.sqr()?.mean_keepdim(-1)?.add_scalar(eps)?.rsqrt()?;
        let mut normed = flat.broadcast_mul(&rms)?.reshape(self.shape())?;

        if let Some(w) = weight {
            normed = normed.broadcast_mul(w)?;
        }

        Ok(normed)
    }

    /// Dropout: randomly zero elements during training.
    ///
    /// Returns (output, mask) where mask is the U8 keep-mask (for backward).
    /// During inference (training=false), returns (self, None).
    pub fn dropout(&self, p: f32, training: bool) -> Result<(Tensor, Option<Tensor>)> {
        if !training || p == 0.0 {
            return Ok((self.clone(), None));
        }
        if p < 0.0 || p >= 1.0 {
            return Err(Error::Internal {
                message: format!("dropout probability must be in [0, 1), got {}", p),
            });
        }
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "dropout only supported for F32".to_string(),
            });
        }

        // Generate uniform random, compare >= p to get keep mask
        let rand = Tensor::rand(self.shape(), self.runtime())?;
        let threshold = Tensor::full(self.shape(), p, DType::F32, self.runtime())?;
        let mask = rand.ge(&threshold)?; // U8: 1 where keep, 0 where drop

        // Apply mask via where_cond, scale by 1/(1-p)
        let zeros = Tensor::zeros(self.shape(), DType::F32, self.runtime())?;
        let masked = self.where_cond(&mask, &zeros)?;
        let output = masked.div_scalar(1.0 - p)?;

        Ok((output, Some(mask)))
    }

    /// Batch normalization (inference mode).
    ///
    /// Input: (N, C, H, W) or (N, C).
    /// running_mean, running_var: (C,)
    /// weight, bias: (C,) optional affine parameters.
    pub fn batch_norm(
        &self,
        running_mean: &Tensor,
        running_var: &Tensor,
        weight: Option<&Tensor>,
        bias: Option<&Tensor>,
        eps: f32,
    ) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "batch_norm only supported for F32".to_string(),
            });
        }
        if self.ndim() < 2 {
            return Err(Error::Internal {
                message: format!("batch_norm requires at least 2D tensor, got {}D", self.ndim()),
            });
        }

        let c = self.shape()[1];

        // Build broadcast shape: (1, C, 1, 1, ...) matching input ndim
        let mut bcast_shape = vec![1usize; self.ndim()];
        bcast_shape[1] = c;

        let mean = running_mean.reshape(&bcast_shape)?;
        let var = running_var.reshape(&bcast_shape)?;

        let inv_std = var.add_scalar(eps)?.rsqrt()?;
        let mut out = self.broadcast_sub(&mean)?.broadcast_mul(&inv_std)?;

        if let Some(w) = weight {
            let w = w.reshape(&bcast_shape)?;
            out = out.broadcast_mul(&w)?;
        }
        if let Some(b) = bias {
            let b = b.reshape(&bcast_shape)?;
            out = out.broadcast_add(&b)?;
        }

        Ok(out)
    }

    /// Scaled dot-product attention.
    ///
    /// Q, K, V: (batch, heads, seq, dim)
    /// attn_mask: optional (batch, heads, seq_q, seq_k) or broadcastable
    /// Returns: (batch, heads, seq_q, dim)
    pub fn scaled_dot_product_attention(
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        dropout_p: f32,
        is_causal: bool,
    ) -> Result<Tensor> {
        if query.ndim() != 4 {
            return Err(Error::Internal {
                message: format!("SDPA requires 4D tensors (batch, heads, seq, dim), got {}D", query.ndim()),
            });
        }

        let batch = query.shape()[0];
        let heads = query.shape()[1];
        let seq_q = query.shape()[2];
        let dim = query.shape()[3];
        let seq_k = key.shape()[2];

        let scale = 1.0 / (dim as f32).sqrt();

        // Transpose K: (batch, heads, dim, seq_k) then make contiguous
        let kt = key.transpose(2, 3)?.contiguous()?;

        // Reshape to 3D for bmm: (batch*heads, seq, dim)
        let q_3d = query.reshape(&[batch * heads, seq_q, dim])?;
        let kt_3d = kt.reshape(&[batch * heads, dim, seq_k])?;

        // QK^T: (batch*heads, seq_q, seq_k)
        let mut attn = q_3d.bmm(&kt_3d)?;

        // Scale
        attn = attn.mul_scalar(scale)?;

        // Reshape back to 4D for mask ops
        attn = attn.reshape(&[batch, heads, seq_q, seq_k])?;

        // Causal mask: fill upper-right triangle with -inf
        if is_causal {
            let causal = Self::causal_mask(seq_q, seq_k, query.runtime())?;
            attn = attn.broadcast_add(&causal)?;
        }

        // Optional attention mask (additive)
        if let Some(mask) = attn_mask {
            attn = attn.broadcast_add(mask)?;
        }

        // Reshape to 3D for softmax along last dim
        attn = attn.reshape(&[batch * heads * seq_q, seq_k])?;
        attn = attn.softmax(-1)?;

        // Optional dropout
        if dropout_p > 0.0 {
            let (dropped, _mask) = attn.dropout(dropout_p, true)?;
            attn = dropped;
        }

        // Reshape for bmm: (batch*heads, seq_q, seq_k)
        attn = attn.reshape(&[batch * heads, seq_q, seq_k])?;

        // V: (batch*heads, seq_k, dim)
        let v_3d = value.reshape(&[batch * heads, seq_k, dim])?;

        // attn @ V: (batch*heads, seq_q, dim)
        let out = attn.bmm(&v_3d)?;

        // Reshape to 4D: (batch, heads, seq_q, dim)
        out.reshape(&[batch, heads, seq_q, dim])
    }

    /// Create a causal mask: 0 for allowed positions, -inf for masked.
    /// Shape: (1, 1, seq_q, seq_k) for broadcasting.
    fn causal_mask(
        seq_q: usize,
        seq_k: usize,
        runtime: &std::sync::Arc<ptx_runtime::PtxRuntime>,
    ) -> Result<Tensor> {
        // Build on CPU, upload to GPU
        let mut mask_data = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                // In causal attention, position i can only attend to positions <= i
                // Offset for when seq_k > seq_q (key has more positions)
                let key_pos = j as i64 - (seq_k as i64 - seq_q as i64);
                if key_pos > i as i64 {
                    mask_data[i * seq_k + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_slice(&mask_data, &[1, 1, seq_q, seq_k], DType::F32, runtime)?;
        Ok(mask)
    }
}
