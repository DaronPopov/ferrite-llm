use super::*;

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self {
            tensors: IndexMap::new(),
            nodes: IndexMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
            next_tensor_id: 0,
            next_node_id: 0,
        }
    }

    // ========================================================================
    // ID allocation
    // ========================================================================

    fn alloc_tensor_id(&mut self) -> TensorId {
        let id = TensorId::new(self.next_tensor_id);
        self.next_tensor_id += 1;
        id
    }

    fn alloc_node_id(&mut self) -> NodeId {
        let id = NodeId::new(self.next_node_id);
        self.next_node_id += 1;
        id
    }

    // ========================================================================
    // Graph construction
    // ========================================================================

    /// Add an input tensor to the graph.
    pub fn input(&mut self, shape: &[usize], dtype: DType) -> TensorId {
        let id = self.alloc_tensor_id();
        let meta = TensorMeta::input(id, shape, dtype);
        self.tensors.insert(id, meta);
        self.inputs.push(id);

        // Create input node
        let node_id = self.alloc_node_id();
        let node = Node::new(
            node_id,
            OpCode::Input,
            SmallVec::new(),
            id,
            OpAttrs::default(),
        );
        self.nodes.insert(node_id, node);

        id
    }

    /// Add a constant tensor to the graph.
    pub fn constant(&mut self, shape: &[usize], dtype: DType, value: Option<f32>) -> TensorId {
        let id = self.alloc_tensor_id();
        let meta = TensorMeta::constant(id, shape, dtype, value);
        self.tensors.insert(id, meta);

        // Create constant node
        let node_id = self.alloc_node_id();
        let node = Node::new(
            node_id,
            OpCode::Constant,
            SmallVec::new(),
            id,
            OpAttrs::default(),
        );
        self.nodes.insert(node_id, node);

        id
    }

    /// Mark a tensor as an output.
    pub fn mark_output(&mut self, tensor_id: TensorId) {
        if let Some(meta) = self.tensors.get_mut(&tensor_id) {
            meta.mark_output();
        }
        if !self.outputs.contains(&tensor_id) {
            self.outputs.push(tensor_id);
        }
    }

    /// Add a generic operation.
    fn add_op(
        &mut self,
        op: OpCode,
        inputs: &[TensorId],
        output_shape: &[usize],
        output_dtype: DType,
        attrs: OpAttrs,
    ) -> TensorId {
        let output_id = self.alloc_tensor_id();
        let output_meta = TensorMeta::new(output_id, output_shape, output_dtype);
        self.tensors.insert(output_id, output_meta);

        let node_id = self.alloc_node_id();
        let node = Node::new(
            node_id,
            op,
            SmallVec::from_slice(inputs),
            output_id,
            attrs,
        );
        self.nodes.insert(node_id, node);

        output_id
    }

    // ========================================================================
    // Binary operations
    // ========================================================================

    /// Element-wise addition.
    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let shape = self.tensors[&a].shape.clone();
        let dtype = self.tensors[&a].dtype;
        self.add_op(OpCode::Add, &[a, b], &shape, dtype, OpAttrs::default())
    }

    /// Element-wise subtraction.
    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let shape = self.tensors[&a].shape.clone();
        let dtype = self.tensors[&a].dtype;
        self.add_op(OpCode::Sub, &[a, b], &shape, dtype, OpAttrs::default())
    }

    /// Element-wise multiplication.
    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let shape = self.tensors[&a].shape.clone();
        let dtype = self.tensors[&a].dtype;
        self.add_op(OpCode::Mul, &[a, b], &shape, dtype, OpAttrs::default())
    }

    /// Element-wise division.
    pub fn div(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let shape = self.tensors[&a].shape.clone();
        let dtype = self.tensors[&a].dtype;
        self.add_op(OpCode::Div, &[a, b], &shape, dtype, OpAttrs::default())
    }

    /// Element-wise maximum.
    pub fn max(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let shape = self.tensors[&a].shape.clone();
        let dtype = self.tensors[&a].dtype;
        self.add_op(OpCode::Max, &[a, b], &shape, dtype, OpAttrs::default())
    }

    /// Element-wise minimum.
    pub fn min(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let shape = self.tensors[&a].shape.clone();
        let dtype = self.tensors[&a].dtype;
        self.add_op(OpCode::Min, &[a, b], &shape, dtype, OpAttrs::default())
    }

    // ========================================================================
    // Unary operations
    // ========================================================================

    fn unary_op(&mut self, op: OpCode, input: TensorId) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(op, &[input], &shape, dtype, OpAttrs::default())
    }

    pub fn neg(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Neg, input)
    }

    pub fn exp(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Exp, input)
    }

    pub fn log(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Log, input)
    }

    pub fn sqrt(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Sqrt, input)
    }

    pub fn abs(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Abs, input)
    }

    pub fn rsqrt(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Rsqrt, input)
    }

    pub fn sin(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Sin, input)
    }

    pub fn cos(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Cos, input)
    }

    pub fn tanh(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Tanh, input)
    }

    pub fn ceil(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Ceil, input)
    }

    pub fn floor(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Floor, input)
    }

    pub fn round(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Round, input)
    }

    pub fn sqr(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Sqr, input)
    }

    pub fn recip(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Recip, input)
    }

    pub fn copy(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Copy, input)
    }

    // ========================================================================
    // Activation functions
    // ========================================================================

    pub fn relu(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Relu, input)
    }

    pub fn relu6(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Relu6, input)
    }

    pub fn leaky_relu(&mut self, input: TensorId, alpha: f32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::LeakyRelu,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_scalar_a(alpha),
        )
    }

    pub fn elu(&mut self, input: TensorId, alpha: f32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::Elu,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_scalar_a(alpha),
        )
    }

    pub fn selu(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Selu, input)
    }

    pub fn gelu(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Gelu, input)
    }

    pub fn sigmoid(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Sigmoid, input)
    }

    pub fn silu(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Silu, input)
    }

    pub fn softplus(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Softplus, input)
    }

    pub fn mish(&mut self, input: TensorId) -> TensorId {
        self.unary_op(OpCode::Mish, input)
    }

    // ========================================================================
    // Reductions
    // ========================================================================

    pub fn reduce_sum(&mut self, input: TensorId, dim: i32, keepdim: bool) -> TensorId {
        let input_meta = &self.tensors[&input];
        let mut shape = input_meta.shape.clone();
        let dtype = input_meta.dtype;

        // Calculate output shape
        let ndim = shape.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
        }

        self.add_op(
            OpCode::ReduceSum,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim as i32).with_keepdim(keepdim),
        )
    }

    pub fn reduce_mean(&mut self, input: TensorId, dim: i32, keepdim: bool) -> TensorId {
        let input_meta = &self.tensors[&input];
        let mut shape = input_meta.shape.clone();
        let dtype = input_meta.dtype;

        let ndim = shape.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
        }

        self.add_op(
            OpCode::ReduceMean,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim as i32).with_keepdim(keepdim),
        )
    }

    pub fn reduce_max(&mut self, input: TensorId, dim: i32, keepdim: bool) -> TensorId {
        let input_meta = &self.tensors[&input];
        let mut shape = input_meta.shape.clone();
        let dtype = input_meta.dtype;

        let ndim = shape.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
        }

        self.add_op(
            OpCode::ReduceMax,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim as i32).with_keepdim(keepdim),
        )
    }

    pub fn reduce_min(&mut self, input: TensorId, dim: i32, keepdim: bool) -> TensorId {
        let input_meta = &self.tensors[&input];
        let mut shape = input_meta.shape.clone();
        let dtype = input_meta.dtype;

        let ndim = shape.len() as i32;
        let dim = if dim < 0 { ndim + dim } else { dim } as usize;
        if keepdim {
            shape[dim] = 1;
        } else {
            shape.remove(dim);
        }

        self.add_op(
            OpCode::ReduceMin,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim as i32).with_keepdim(keepdim),
        )
    }

    // ========================================================================
    // Gather/Scatter operations
    // ========================================================================

    pub fn gather(&mut self, input: TensorId, indices: TensorId, dim: i32) -> TensorId {
        let idx_shape = self.tensors[&indices].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::Gather,
            &[input, indices],
            &idx_shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim),
        )
    }

    /// Index-select — select slices along a dimension by index.
    /// Output shape: input shape with dim replaced by ids_dim_size.
    pub fn index_select(&mut self, input: TensorId, indices: TensorId, dim: i32) -> TensorId {
        let idx_shape = self.tensors[&indices].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::IndexSelect,
            &[input, indices],
            &idx_shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim),
        )
    }

    /// Argsort — returns uint32 indices that sort each row.
    /// Input viewed as [nrows, ncols]; ascending controls direction.
    pub fn argsort(&mut self, input: TensorId, dim: i32, ascending: bool) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        self.add_op(
            OpCode::Argsort,
            &[input],
            &shape,
            DType::U32,
            OpAttrs::new()
                .with_reduce_dim(dim)
                .with_scalar_a(if ascending { 1.0 } else { 0.0 }),
        )
    }

    /// Where — element-wise conditional: out[i] = cond[i] ? true_val[i] : false_val[i]
    pub fn where_cond(&mut self, cond: TensorId, true_val: TensorId, false_val: TensorId) -> TensorId {
        let shape = self.tensors[&true_val].shape.clone();
        let dtype = self.tensors[&true_val].dtype;
        self.add_op(
            OpCode::Where,
            &[cond, true_val, false_val],
            &shape,
            dtype,
            OpAttrs::default(),
        )
    }

    // ========================================================================
    // Scan/Prefix operations
    // ========================================================================

    pub fn cumsum(&mut self, input: TensorId, dim: i32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::CumSum,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim),
        )
    }

    // ========================================================================
    // Sort/Select operations
    // ========================================================================

    /// TopK — returns the values tensor (dim-th dimension replaced by k).
    pub fn topk(&mut self, input: TensorId, k: usize, dim: i32, largest: bool) -> TensorId {
        let mut shape = self.tensors[&input].shape.clone();
        let ndim = shape.len() as i32;
        let d = if dim < 0 { ndim + dim } else { dim } as usize;
        shape[d] = k;
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::TopK,
            &[input],
            &shape,
            dtype,
            OpAttrs::new()
                .with_reduce_dim(dim)
                .with_k(k)
                .with_scalar_a(if largest { 1.0 } else { 0.0 }),
        )
    }

    // ========================================================================
    // Softmax
    // ========================================================================

    pub fn softmax(&mut self, input: TensorId, dim: i32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::Softmax,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim),
        )
    }

    pub fn log_softmax(&mut self, input: TensorId, dim: i32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::LogSoftmax,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_reduce_dim(dim),
        )
    }

    // ========================================================================
    // Matrix operations
    // ========================================================================

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let a_meta = &self.tensors[&a];
        let b_meta = &self.tensors[&b];
        let dtype = a_meta.dtype;

        // A: (M, K), B: (K, N) -> (M, N)
        let m = a_meta.shape[0];
        let n = b_meta.shape[1];
        let shape: SmallVec<[usize; 8]> = SmallVec::from_slice(&[m, n]);

        self.add_op(OpCode::Matmul, &[a, b], &shape, dtype, OpAttrs::default())
    }

    pub fn bmm(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let a_meta = &self.tensors[&a];
        let b_meta = &self.tensors[&b];
        let dtype = a_meta.dtype;

        // A: (B, M, K), B: (B, K, N) -> (B, M, N)
        let batch = a_meta.shape[0];
        let m = a_meta.shape[1];
        let n = b_meta.shape[2];
        let shape: SmallVec<[usize; 8]> = SmallVec::from_slice(&[batch, m, n]);

        self.add_op(OpCode::BatchMatmul, &[a, b], &shape, dtype, OpAttrs::default())
    }

    // ========================================================================
    // Scalar/Transform operations
    // ========================================================================

    pub fn add_scalar(&mut self, input: TensorId, scalar: f32) -> TensorId {
        self.affine(input, 1.0, scalar)
    }

    pub fn sub_scalar(&mut self, input: TensorId, scalar: f32) -> TensorId {
        self.affine(input, 1.0, -scalar)
    }

    pub fn mul_scalar(&mut self, input: TensorId, scalar: f32) -> TensorId {
        self.affine(input, scalar, 0.0)
    }

    pub fn div_scalar(&mut self, input: TensorId, scalar: f32) -> TensorId {
        self.affine(input, 1.0 / scalar, 0.0)
    }

    pub fn affine(&mut self, input: TensorId, mul: f32, add: f32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::Affine,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_scalar_a(mul).with_scalar_b(add),
        )
    }

    pub fn clamp(&mut self, input: TensorId, min_val: f32, max_val: f32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::Clamp,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_scalar_a(min_val).with_scalar_b(max_val),
        )
    }

    pub fn powf(&mut self, input: TensorId, exp: f32) -> TensorId {
        let shape = self.tensors[&input].shape.clone();
        let dtype = self.tensors[&input].dtype;
        self.add_op(
            OpCode::Pow,
            &[input],
            &shape,
            dtype,
            OpAttrs::new().with_scalar_a(exp),
        )
    }

    pub fn fill(&mut self, shape: &[usize], dtype: DType, value: f32) -> TensorId {
        let output_id = self.alloc_tensor_id();
        let output_meta = TensorMeta::new(output_id, shape, dtype);
        self.tensors.insert(output_id, output_meta);

        let node_id = self.alloc_node_id();
        let node = Node::new(
            node_id,
            OpCode::Fill,
            SmallVec::new(),
            output_id,
            OpAttrs::new().with_scalar_a(value),
        );
        self.nodes.insert(node_id, node);

        output_id
    }
}
