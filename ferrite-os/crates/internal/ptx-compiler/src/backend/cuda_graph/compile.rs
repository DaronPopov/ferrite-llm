use super::*;

/// Compile the graph to a CUDA graph.
pub fn compile(
    graph: Graph,
    memory_plan: MemoryPlan,
    runtime: &Arc<PtxRuntime>,
) -> Result<CompiledGraph> {
    // Collect input/output metadata
    let input_metas: Vec<TensorMeta> = graph.inputs()
        .iter()
        .filter_map(|id| graph.tensor(*id).cloned())
        .collect();
    let output_metas: Vec<TensorMeta> = graph.outputs()
        .iter()
        .filter_map(|id| graph.tensor(*id).cloned())
        .collect();

    // Allocate buffers for non-input tensors
    let mut buffers = Vec::with_capacity(memory_plan.buffers.len());
    for buffer in &memory_plan.buffers {
        let ptr = Arc::new(runtime.alloc(buffer.size)?);
        buffers.push(ptr);
    }

    // Allocate input buffers (one per input tensor)
    let mut input_buffers: Vec<Arc<GpuPtr>> = Vec::with_capacity(input_metas.len());
    let mut input_buffer_map: HashMap<TensorId, usize> = HashMap::new();
    for (idx, meta) in input_metas.iter().enumerate() {
        let ptr = Arc::new(runtime.alloc(meta.size_bytes())?);
        input_buffers.push(ptr);
        input_buffer_map.insert(meta.id, idx);
    }

    // Begin CUDA graph capture
    let stream_id = 0;
    let capture = runtime.begin_capture(stream_id, "compiled_graph")?;

    // Get the stream for emitting operations
    let stream = runtime.stream(stream_id)?;

    // Emit operations in topological order
    for node_id in graph.topo_order() {
        let node = match graph.node(node_id) {
            Some(n) => n,
            None => continue,
        };

        // Skip input nodes (constants are emitted if possible)
        if node.is_input() {
            continue;
        }

        super::emit::emit_node(
            &graph,
            node,
            &memory_plan,
            &buffers,
            &input_buffers,
            &input_buffer_map,
            runtime,
            &stream,
        )?;
    }

    // End capture
    let cuda_graph = capture.end()?;

    Ok(CompiledGraph {
        graph_id: Some(cuda_graph.id()),
        memory_plan,
        buffers,
        input_buffers,
        input_ids: graph.inputs().to_vec(),
        input_metas,
        output_ids: graph.outputs().to_vec(),
        output_metas,
        runtime: Arc::clone(runtime),
    })
}
