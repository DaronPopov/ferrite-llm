use super::*;

impl Graph {
    /// Compile the graph for execution.
    pub fn compile(self, runtime: &Arc<PtxRuntime>) -> Result<CompiledGraph> {
        // Run optimization passes
        let optimized = passes::optimize(self)?;

        // Plan memory
        let memory_plan = passes::plan_memory(&optimized)?;

        // Emit to backend
        crate::backend::cuda_graph::compile(optimized, memory_plan, runtime)
    }
}
