//! Optimization passes for the computation graph.

pub mod dead_code;
pub mod constant_fold;
pub mod fusion;
pub mod memory;

use crate::ir::Graph;
use ptx_runtime::Result;

pub use memory::MemoryPlan;

/// Run all optimization passes on the graph.
pub fn optimize(mut graph: Graph) -> Result<Graph> {
    // Run passes in order
    graph = dead_code::eliminate_dead_code(graph)?;
    graph = constant_fold::fold_constants(graph)?;
    graph = dead_code::eliminate_dead_code(graph)?; // clean up after folding
    graph = fusion::fuse_elementwise(graph)?;

    Ok(graph)
}

/// Plan memory allocations.
pub fn plan_memory(graph: &Graph) -> Result<MemoryPlan> {
    memory::plan(graph)
}
