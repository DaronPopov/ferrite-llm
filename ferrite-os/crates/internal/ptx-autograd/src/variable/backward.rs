use super::*;

impl Variable {
    /// Run backward pass from this variable (should be a scalar loss).
    pub fn backward(&self) -> Result<()> {
        if self.elem_count() != 1 {
            return Err(Error::Internal {
                message: "backward() can only be called on scalar tensors".to_string(),
            });
        }

        crate::backward::backward(self.id, &self.tensor)?;

        // Copy gradients to variable's grad field
        if let Some(grad) = get_grad(self.id) {
            self.set_grad(grad);
        }

        Ok(())
    }

    /// Compile the current tape into a CUDA graph using this variable as output.
    pub fn compile_graph(&self) -> Result<ptx_compiler::CompiledGraph> {
        crate::compiler::compile_from_tape(&[self.id], self.runtime())
    }
}
