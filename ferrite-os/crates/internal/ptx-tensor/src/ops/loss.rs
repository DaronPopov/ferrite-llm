//! Loss functions — composed from existing tensor primitives.

use crate::tensor::Tensor;
use crate::dtype::DType;
use ptx_runtime::{Result, Error};

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction — return per-element loss.
    None,
    /// Mean of all loss elements.
    Mean,
    /// Sum of all loss elements.
    Sum,
}

impl Tensor {
    /// Mean squared error loss: (input - target)^2, reduced.
    ///
    /// - input, target: same shape
    pub fn mse_loss(&self, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
        if self.shape() != target.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: target.shape().to_vec(),
            });
        }
        let diff = self.sub(target)?;
        let sq = diff.sqr()?;
        match reduction {
            Reduction::None => Ok(sq),
            Reduction::Mean => sq.mean_all(),
            Reduction::Sum => sq.sum_all(),
        }
    }

    /// Cross-entropy loss for classification.
    ///
    /// - self (input): (N, C) logits
    /// - target: (N,) I32 class indices in [0, C)
    pub fn cross_entropy_loss(&self, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "cross_entropy_loss only supported for F32 input".to_string(),
            });
        }
        if target.dtype() != DType::I32 {
            return Err(Error::NotSupported {
                message: "cross_entropy_loss requires I32 target".to_string(),
            });
        }
        if self.ndim() != 2 {
            return Err(Error::Internal {
                message: format!("cross_entropy_loss expects (N, C) input, got {:?}", self.shape()),
            });
        }
        if target.ndim() != 1 {
            return Err(Error::Internal {
                message: format!("cross_entropy_loss expects (N,) target, got {:?}", target.shape()),
            });
        }
        let n = self.shape()[0];
        if target.shape()[0] != n {
            return Err(Error::ShapeMismatch {
                expected: vec![n],
                actual: target.shape().to_vec(),
            });
        }

        // log_softmax along classes (dim -1)
        let log_probs = self.log_softmax(-1)?;

        // Gather the log-prob at the target class for each sample.
        // target: (N,) → unsqueeze to (N, 1)
        let target_2d = target.unsqueeze(1)?;
        // gather(dim=1, indices=target_2d): (N, 1)
        let nll = log_probs.gather(1, &target_2d)?;

        // nll is (N, 1), squeeze to (N,) and negate
        let nll = nll.flatten()?.neg()?;

        match reduction {
            Reduction::None => Ok(nll),
            Reduction::Mean => nll.mean_all(),
            Reduction::Sum => nll.sum_all(),
        }
    }

    /// Binary cross-entropy loss.
    ///
    /// - self (input): (N,) or (*) predicted probabilities in (0, 1)
    /// - target: same shape, values in {0, 1}
    ///
    /// BCE = -(target * log(input) + (1 - target) * log(1 - input))
    pub fn binary_cross_entropy_loss(
        &self,
        target: &Tensor,
        reduction: Reduction,
    ) -> Result<Tensor> {
        if self.shape() != target.shape() {
            return Err(Error::ShapeMismatch {
                expected: self.shape().to_vec(),
                actual: target.shape().to_vec(),
            });
        }
        if self.dtype() != DType::F32 {
            return Err(Error::NotSupported {
                message: "binary_cross_entropy_loss only supported for F32".to_string(),
            });
        }

        let eps = 1e-7f32;
        // Clamp input to avoid log(0)
        let clamped = self.clamp(eps, 1.0 - eps)?;

        // target * log(input)
        let term1 = clamped.log()?.mul(target)?;

        // (1 - target) * log(1 - input)
        let one_minus_input = clamped.neg()?.add_scalar(1.0)?;
        let one_minus_target = target.neg()?.add_scalar(1.0)?;
        let term2 = one_minus_input.log()?.mul(&one_minus_target)?;

        // -(term1 + term2)
        let loss = term1.add(&term2)?.neg()?;

        match reduction {
            Reduction::None => Ok(loss),
            Reduction::Mean => loss.mean_all(),
            Reduction::Sum => loss.sum_all(),
        }
    }
}
