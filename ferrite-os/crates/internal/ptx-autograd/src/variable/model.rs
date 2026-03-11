use super::*;

impl Variable {
    /// Create a new variable from a tensor.
    pub fn new(tensor: Tensor, requires_grad: bool) -> Self {
        let var = Self {
            tensor,
            id: TensorId::new(),
            requires_grad,
            grad: Arc::new(Mutex::new(None)),
        };
        register_tensor(var.id, &var.tensor);
        var
    }

    /// Create a variable that doesn't require gradients.
    pub fn leaf(tensor: Tensor) -> Self {
        Self::new(tensor, false)
    }

    /// Create a variable that requires gradients.
    pub fn param(tensor: Tensor) -> Self {
        Self::new(tensor, true)
    }

    /// Get the tensor ID.
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Get a reference to the underlying tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// Get the underlying tensor (consuming self).
    pub fn into_tensor(self) -> Tensor {
        self.tensor
    }

    /// Check if this variable requires gradient.
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Set whether this variable requires gradient.
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Get the accumulated gradient.
    pub fn grad(&self) -> Option<Tensor> {
        self.grad.lock().clone()
    }

    /// Set the gradient.
    pub fn set_grad(&self, grad: Tensor) {
        *self.grad.lock() = Some(grad);
    }

    /// Clear the gradient.
    pub fn zero_grad(&self) {
        *self.grad.lock() = None;
    }

    /// Detach from the computation graph (returns a new variable that doesn't require grad).
    pub fn detach(&self) -> Variable {
        Variable::new(self.tensor.clone(), false)
    }

    // ========================================================================
    // Tensor accessors (delegate to inner tensor)
    // ========================================================================

    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }

    pub fn elem_count(&self) -> usize {
        self.tensor.elem_count()
    }

    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    pub fn runtime(&self) -> &Arc<PtxRuntime> {
        self.tensor.runtime()
    }

    // ========================================================================
    // Factory methods
    // ========================================================================

    /// Create a zeros variable.
    pub fn zeros(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>, requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::zeros(shape, dtype, runtime)?;
        Ok(Self::new(tensor, requires_grad))
    }

    /// Create a ones variable.
    pub fn ones(shape: &[usize], dtype: DType, runtime: &Arc<PtxRuntime>, requires_grad: bool) -> Result<Self> {
        let tensor = Tensor::ones(shape, dtype, runtime)?;
        Ok(Self::new(tensor, requires_grad))
    }

    // ========================================================================
    // Helper for recording operations
    // ========================================================================

    fn record_unary<F>(
        &self,
        f: F,
        grad_fn: Box<dyn GradFn>,
        save_input: bool,
        op: OpCode,
        attrs: OpAttrs,
    ) -> Result<Variable>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        let output = f(&self.tensor)?;
        let output_var = Variable::new(output.clone(), self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            let saved = if save_input {
                smallvec![self.tensor.clone()]
            } else {
                smallvec![output]
            };

            record_op(TapeNode::new(
                output_var.id,
                op,
                attrs,
                grad_fn,
                SmallVec::from_slice(&[self.id]),
                saved,
            ));
        }

        Ok(output_var)
    }

    fn record_binary<F>(
        &self,
        other: &Variable,
        f: F,
        grad_fn: Box<dyn GradFn>,
        saved: SmallVec<[Tensor; 4]>,
        op: OpCode,
        attrs: OpAttrs,
    ) -> Result<Variable>
    where
        F: FnOnce(&Tensor, &Tensor) -> Result<Tensor>,
    {
        let output = f(&self.tensor, &other.tensor)?;
        let requires_grad = (self.requires_grad || other.requires_grad) && is_grad_enabled();
        let output_var = Variable::new(output, requires_grad);

        if requires_grad {
            record_op(TapeNode::new(
                output_var.id,
                op,
                attrs,
                grad_fn,
                SmallVec::from_slice(&[self.id, other.id]),
                saved,
            ));
        }

        Ok(output_var)
    }

    // ========================================================================
    // Binary Operations
    // ========================================================================

    pub fn add(&self, other: &Variable) -> Result<Variable> {
        self.record_binary(
            other,
            |a, b| a.add(b),
            Box::new(AddBackward),
            SmallVec::new(),
            OpCode::Add,
            OpAttrs::default(),
        )
    }

    pub fn sub(&self, other: &Variable) -> Result<Variable> {
        self.record_binary(
            other,
            |a, b| a.sub(b),
            Box::new(SubBackward),
            SmallVec::new(),
            OpCode::Sub,
            OpAttrs::default(),
        )
    }

    pub fn mul(&self, other: &Variable) -> Result<Variable> {
        self.record_binary(
            other,
            |a, b| a.mul(b),
            Box::new(MulBackward),
            smallvec![self.tensor.clone(), other.tensor.clone()],
            OpCode::Mul,
            OpAttrs::default(),
        )
    }

    pub fn div(&self, other: &Variable) -> Result<Variable> {
        self.record_binary(
            other,
            |a, b| a.div(b),
            Box::new(DivBackward),
            smallvec![self.tensor.clone(), other.tensor.clone()],
            OpCode::Div,
            OpAttrs::default(),
        )
    }

    // ========================================================================
    // Unary Operations
    // ========================================================================

    pub fn neg(&self) -> Result<Variable> {
        self.record_unary(
            |t| t.neg(),
            Box::new(NegBackward),
            false,
            OpCode::Neg,
            OpAttrs::default(),
        )
    }

    pub fn exp(&self) -> Result<Variable> {
        // Save output for backward
        let output = self.tensor.exp()?;
        let output_var = Variable::new(output.clone(), self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::Exp,
                OpAttrs::default(),
                Box::new(ExpBackward),
                SmallVec::from_slice(&[self.id]),
                smallvec![output],
            ));
        }

        Ok(output_var)
    }

    pub fn log(&self) -> Result<Variable> {
        self.record_unary(
            |t| t.log(),
            Box::new(LogBackward),
            true,
            OpCode::Log,
            OpAttrs::default(),
        )
    }

    pub fn sqrt(&self) -> Result<Variable> {
        let output = self.tensor.sqrt()?;
        let output_var = Variable::new(output.clone(), self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::Sqrt,
                OpAttrs::default(),
                Box::new(SqrtBackward),
                SmallVec::from_slice(&[self.id]),
                smallvec![output],
            ));
        }

        Ok(output_var)
    }

    pub fn tanh(&self) -> Result<Variable> {
        let output = self.tensor.tanh()?;
        let output_var = Variable::new(output.clone(), self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::Tanh,
                OpAttrs::default(),
                Box::new(TanhBackward),
                SmallVec::from_slice(&[self.id]),
                smallvec![output],
            ));
        }

        Ok(output_var)
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    pub fn relu(&self) -> Result<Variable> {
        self.record_unary(
            |t| t.relu(),
            Box::new(ReluBackward),
            true,
            OpCode::Relu,
            OpAttrs::default(),
        )
    }

    pub fn sigmoid(&self) -> Result<Variable> {
        let output = self.tensor.sigmoid()?;
        let output_var = Variable::new(output.clone(), self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::Sigmoid,
                OpAttrs::default(),
                Box::new(SigmoidBackward),
                SmallVec::from_slice(&[self.id]),
                smallvec![output],
            ));
        }

        Ok(output_var)
    }

    pub fn gelu(&self) -> Result<Variable> {
        self.record_unary(
            |t| t.gelu(),
            Box::new(GeluBackward),
            true,
            OpCode::Gelu,
            OpAttrs::default(),
        )
    }

    // ========================================================================
    // Reductions
    // ========================================================================

    pub fn sum(&self, dim: i32) -> Result<Variable> {
        let input_shape = self.shape().to_vec();
        let output = self.tensor.sum(dim)?;
        let output_var = Variable::new(output, self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::ReduceSum,
                OpAttrs::new().with_reduce_dim(dim).with_keepdim(false),
                Box::new(SumBackward {
                    input_shape,
                    dim,
                    keepdim: false,
                }),
                SmallVec::from_slice(&[self.id]),
                SmallVec::new(),
            ));
        }

        Ok(output_var)
    }

    pub fn mean(&self, dim: i32) -> Result<Variable> {
        let input_shape = self.shape().to_vec();
        let output = self.tensor.mean(dim)?;
        let output_var = Variable::new(output, self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::ReduceMean,
                OpAttrs::new().with_reduce_dim(dim).with_keepdim(false),
                Box::new(MeanBackward {
                    input_shape,
                    dim,
                    keepdim: false,
                }),
                SmallVec::from_slice(&[self.id]),
                SmallVec::new(),
            ));
        }

        Ok(output_var)
    }

    pub fn sum_all(&self) -> Result<Variable> {
        let flat = Variable::new(self.tensor.flatten()?, self.requires_grad);
        flat.sum(0)
    }

    pub fn mean_all(&self) -> Result<Variable> {
        let flat = Variable::new(self.tensor.flatten()?, self.requires_grad);
        flat.mean(0)
    }

    // ========================================================================
    // Softmax
    // ========================================================================

    pub fn softmax(&self, dim: i32) -> Result<Variable> {
        let output = self.tensor.softmax(dim)?;
        let output_var = Variable::new(output.clone(), self.requires_grad && is_grad_enabled());

        if self.requires_grad && is_grad_enabled() {
            record_op(TapeNode::new(
                output_var.id,
                OpCode::Softmax,
                OpAttrs::new().with_reduce_dim(dim),
                Box::new(SoftmaxBackward { dim }),
                SmallVec::from_slice(&[self.id]),
                smallvec![output],
            ));
        }

        Ok(output_var)
    }

    pub fn log_softmax(&self, dim: i32) -> Result<Variable> {
        // log_softmax = x - log(sum(exp(x)))
        // For now, compute as log(softmax(x))
        let softmax = self.softmax(dim)?;
        softmax.log()
    }
}
