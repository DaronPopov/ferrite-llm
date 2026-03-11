//! Compile recorded autograd tape into a PTX-OS computation graph.

use std::collections::HashMap;
use std::sync::Arc;

use ptx_compiler::{CompiledGraph, Graph, OpCode};
use ptx_runtime::{Error, PtxRuntime, Result};

use crate::tape::{TAPE, TensorId, TensorInfo};

fn ensure_input(
    graph: &mut Graph,
    map: &mut HashMap<TensorId, ptx_compiler::TensorId>,
    input_order: &mut Vec<TensorId>,
    info: &TensorInfo,
    id: TensorId,
) -> ptx_compiler::TensorId {
    if let Some(existing) = map.get(&id) {
        return *existing;
    }

    let gid = graph.input(&info.shape, info.dtype);
    map.insert(id, gid);
    input_order.push(id);
    gid
}

/// Compile the current thread-local tape into a CUDA graph.
///
/// `outputs` should contain the TensorIds you want to mark as graph outputs.
pub fn compile_from_tape(
    outputs: &[TensorId],
    runtime: &Arc<PtxRuntime>,
) -> Result<CompiledGraph> {
    let (compiled, _) = compile_from_tape_with_inputs(outputs, runtime)?;
    Ok(compiled)
}

/// Compile the tape and return the input order (TensorIds) required by `execute`.
pub fn compile_from_tape_with_inputs(
    outputs: &[TensorId],
    runtime: &Arc<PtxRuntime>,
) -> Result<(CompiledGraph, Vec<TensorId>)> {
    TAPE.with(|tape| {
        let tape = tape.lock();
        if tape.is_empty() {
            return Err(Error::Internal {
                message: "Tape is empty; nothing to compile".to_string(),
            });
        }

        let mut graph = Graph::new();
        let mut map: HashMap<TensorId, ptx_compiler::TensorId> = HashMap::new();
        let mut input_order: Vec<TensorId> = Vec::new();

        // Build graph from recorded ops in forward order
        for node in tape.nodes() {
            let mut input_ids: Vec<ptx_compiler::TensorId> = Vec::new();
            for input_id in &node.inputs {
                let info = tape.tensor_info(*input_id).ok_or_else(|| Error::Internal {
                    message: format!("Missing tensor metadata for input {:?}", input_id),
                })?;
                let gid = ensure_input(&mut graph, &mut map, &mut input_order, info, *input_id);
                input_ids.push(gid);
            }

            let out_id = match node.op {
                OpCode::Add => graph.add(input_ids[0], input_ids[1]),
                OpCode::Sub => graph.sub(input_ids[0], input_ids[1]),
                OpCode::Mul => graph.mul(input_ids[0], input_ids[1]),
                OpCode::Div => graph.div(input_ids[0], input_ids[1]),
                OpCode::Neg => graph.neg(input_ids[0]),
                OpCode::Exp => graph.exp(input_ids[0]),
                OpCode::Log => graph.log(input_ids[0]),
                OpCode::Sqrt => graph.sqrt(input_ids[0]),
                OpCode::Tanh => graph.tanh(input_ids[0]),
                OpCode::Relu => graph.relu(input_ids[0]),
                OpCode::Gelu => graph.gelu(input_ids[0]),
                OpCode::Sigmoid => graph.sigmoid(input_ids[0]),
                OpCode::Softmax => {
                    let dim = node.attrs.reduce_dim.ok_or_else(|| Error::Internal {
                        message: "Softmax missing dim".to_string(),
                    })?;
                    graph.softmax(input_ids[0], dim)
                }
                OpCode::ReduceSum => {
                    let dim = node.attrs.reduce_dim.ok_or_else(|| Error::Internal {
                        message: "ReduceSum missing dim".to_string(),
                    })?;
                    graph.reduce_sum(input_ids[0], dim, node.attrs.keepdim)
                }
                OpCode::ReduceMean => {
                    let dim = node.attrs.reduce_dim.ok_or_else(|| Error::Internal {
                        message: "ReduceMean missing dim".to_string(),
                    })?;
                    graph.reduce_mean(input_ids[0], dim, node.attrs.keepdim)
                }
                op => {
                    return Err(Error::NotSupported {
                        message: format!("Tape op {:?} not supported in compiler", op),
                    });
                }
            };

            map.insert(node.output_id, out_id);
        }

        // Mark outputs
        for output_id in outputs {
            let gid = if let Some(existing) = map.get(output_id) {
                *existing
            } else {
                let info = tape.tensor_info(*output_id).ok_or_else(|| Error::Internal {
                    message: format!("Missing tensor metadata for output {:?}", output_id),
                })?;
                ensure_input(&mut graph, &mut map, &mut input_order, info, *output_id)
            };
            graph.mark_output(gid);
        }

        let compiled = graph.compile(runtime)?;
        Ok((compiled, input_order))
    })
}
