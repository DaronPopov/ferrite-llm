//! Backward pass engine for automatic differentiation.


use ptx_tensor::Tensor;
use ptx_runtime::Result;

use crate::tape::{Tape, TensorId, TAPE};
use crate::Variable;

/// Run backward pass from a loss tensor.
///
/// This computes gradients for all tensors that require_grad and were involved
/// in computing the loss.
pub fn backward(loss_id: TensorId, loss_tensor: &Tensor) -> Result<()> {
    TAPE.with(|tape| {
        let mut tape = tape.lock();
        backward_impl(&mut tape, loss_id, loss_tensor, false)
    })
}

/// Run backward pass with graph creation for higher-order derivatives.
///
/// When `create_graph` is true, the gradient computations are recorded on
/// the tape via Variable ops, enabling second (and higher) derivatives.
pub fn backward_with_create_graph(loss_id: TensorId, loss_tensor: &Tensor) -> Result<()> {
    TAPE.with(|tape| {
        let mut tape = tape.lock();
        backward_impl(&mut tape, loss_id, loss_tensor, true)
    })
}

/// Internal backward implementation.
fn backward_impl(tape: &mut Tape, loss_id: TensorId, loss_tensor: &Tensor, create_graph: bool) -> Result<()> {
    // Initialize gradient of loss to ones (scalar 1 for scalar loss)
    let grad_loss = Tensor::ones(loss_tensor.shape(), loss_tensor.dtype(), loss_tensor.runtime())?;
    tape.set_grad(loss_id, grad_loss);

    // Collect node indices to process in reverse order
    let num_nodes = tape.nodes().len();

    if create_graph {
        // Higher-order gradient mode: wrap gradients as Variables so that
        // gradient computations are themselves recorded on the tape.
        backward_create_graph_impl(tape, num_nodes)?;
    } else {
        // Standard backward: compute gradients through raw tensor ops
        backward_standard_impl(tape, num_nodes)?;
    }

    Ok(())
}

/// Standard backward pass — gradient ops are NOT recorded.
fn backward_standard_impl(tape: &mut Tape, num_nodes: usize) -> Result<()> {
    for i in (0..num_nodes).rev() {
        let (output_id, inputs, saved_tensors) = {
            let node = &tape.nodes()[i];
            (node.output_id, node.inputs.clone(), node.saved_tensors.clone())
        };

        let grad_output = match tape.get_grad(output_id) {
            Some(grad) => grad.clone(),
            None => continue,
        };

        let input_grads = tape.nodes()[i].grad_fn.backward(&grad_output, &saved_tensors)?;

        for (input_id, maybe_grad) in inputs.iter().zip(input_grads.into_iter()) {
            if let Some(grad) = maybe_grad {
                tape.accumulate_grad(*input_id, grad)?;
            }
        }
    }

    Ok(())
}

/// Create-graph backward pass — gradient ops ARE recorded on the tape.
///
/// Wraps gradient tensors as `Variable`s (with requires_grad=true) and calls
/// `backward_create_graph` on each GradFn. The Variable ops record new nodes
/// on the tape, enabling higher-order derivatives.
fn backward_create_graph_impl(tape: &mut Tape, num_nodes: usize) -> Result<()> {
    for i in (0..num_nodes).rev() {
        let (output_id, inputs, saved_tensors) = {
            let node = &tape.nodes()[i];
            (node.output_id, node.inputs.clone(), node.saved_tensors.clone())
        };

        let grad_output_tensor = match tape.get_grad(output_id) {
            Some(grad) => grad.clone(),
            None => continue,
        };

        // Wrap grad_output as a Variable with requires_grad=true so that
        // operations on it (inside backward_create_graph) are recorded.
        let grad_output_var = Variable::param(grad_output_tensor);

        // Wrap saved tensors as leaf Variables (no grad needed for saved values,
        // but they participate in recorded ops).
        let saved_vars: Vec<Variable> = saved_tensors
            .iter()
            .map(|t| Variable::leaf(t.clone()))
            .collect();

        let input_grads = tape.nodes()[i]
            .grad_fn
            .backward_create_graph(&grad_output_var, &saved_vars)?;

        for (input_id, maybe_grad_var) in inputs.iter().zip(input_grads.into_iter()) {
            if let Some(grad_var) = maybe_grad_var {
                tape.accumulate_grad(*input_id, grad_var.into_tensor())?;
            }
        }
    }

    Ok(())
}

/// Run backward pass and return gradients for specific tensors.
pub fn backward_with_gradients(
    loss_id: TensorId,
    loss_tensor: &Tensor,
    tensor_ids: &[TensorId],
) -> Result<Vec<Option<Tensor>>> {
    backward(loss_id, loss_tensor)?;

    let grads = TAPE.with(|tape| {
        let tape = tape.lock();
        tensor_ids
            .iter()
            .map(|id| tape.get_grad(*id).cloned())
            .collect()
    });

    Ok(grads)
}

/// Compute gradients and return them without modifying .grad attributes.
pub fn grad(
    outputs: &[(&TensorId, &Tensor)],
    inputs: &[TensorId],
    _grad_outputs: Option<Vec<Tensor>>,
    retain_graph: bool,
    create_graph: bool,
) -> Result<Vec<Option<Tensor>>> {
    if outputs.len() != 1 {
        return Err(ptx_runtime::Error::NotSupported {
            message: "grad only supports single output for now".to_string(),
        });
    }

    let (loss_id, loss_tensor) = outputs[0];

    if create_graph {
        backward_with_create_graph(*loss_id, loss_tensor)?;
    } else {
        backward(*loss_id, loss_tensor)?;
    }

    let grads = TAPE.with(|tape| {
        let mut tape = tape.lock();
        let result: Vec<Option<Tensor>> = inputs
            .iter()
            .map(|id| tape.get_grad(*id).cloned())
            .collect();

        if !retain_graph {
            tape.clear();
        }

        result
    });

    Ok(grads)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_backward_simple() {
        // This would require a GPU, so skip in unit tests
    }
}
