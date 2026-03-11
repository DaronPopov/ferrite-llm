use std::path::PathBuf;
use std::sync::Arc;

use ferrite_gpu_lang::jit::JitEngine;
use ferrite_gpu_lang::{CompiledProgram, GpuLangRuntime, HostTensor};
use ptx_runtime::PtxRuntime;

/// Result of executing a compiled program on the GPU.
pub struct ExecResult {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

pub struct ScriptRunner {
    engine: JitEngine,
    lang_runtime: GpuLangRuntime,
    last_compiled: Option<CompiledProgram>,
}

impl ScriptRunner {
    pub fn new(runtime: Arc<PtxRuntime>) -> Self {
        let lang_runtime = GpuLangRuntime::from_runtime(runtime);
        Self {
            engine: JitEngine::new(),
            lang_runtime,
            last_compiled: None,
        }
    }

    pub fn compile(&mut self, script: &str) -> Result<(), String> {
        match self.engine.compile(script) {
            Ok(program) => {
                self.last_compiled = Some(program.clone());
                Ok(())
            }
            Err(e) => Err(format!("{}", e)),
        }
    }

    pub fn execute_last(&self) -> Result<ExecResult, String> {
        let program = self
            .last_compiled
            .as_ref()
            .ok_or_else(|| "no compiled program".to_string())?;

        // Generate interesting input data — linear sweep [-2, 2]
        let input_shapes = program.input_shapes();
        let inputs: Vec<HostTensor> = input_shapes
            .iter()
            .map(|shape| {
                let numel: usize = shape.iter().product();
                let data: Vec<f32> = (0..numel)
                    .map(|i| (i as f32) / (numel as f32) * 4.0 - 2.0)
                    .collect();
                HostTensor::new(shape.clone(), data).map_err(|e| {
                    format!("failed to build input tensor for shape {:?}: {}", shape, e)
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        match self.lang_runtime.execute(program, &inputs) {
            Ok(output) => Ok(ExecResult {
                shape: output.shape().to_vec(),
                data: output.data().to_vec(),
            }),
            Err(e) => Err(format!("{}", e)),
        }
    }

    pub fn cache_len(&self) -> usize {
        self.engine.cache_len()
    }

    /// Current VRAM pool usage in bytes (for profiling snapshots).
    pub fn pool_used_bytes(&self) -> u64 {
        self.lang_runtime.runtime().tlsf_stats().allocated_bytes as u64
    }

    /// Inspect the last compiled program. Returns None if nothing was compiled.
    pub fn inspect_last(&self) -> Option<ProgramInfo> {
        self.last_compiled.as_ref().map(|p| ProgramInfo {
            node_count: p.node_count(),
            input_count: p.input_count(),
            input_shapes: p.input_shapes(),
            output_shape: p.output_shape().to_vec(),
            total_elements: p.total_elements(),
            op_summary: p.op_summary(),
        })
    }

    /// Clear the JIT compilation cache.
    pub fn clear_cache(&mut self) {
        self.engine.clear_cache();
        self.last_compiled = None;
    }

    /// Clear both memory and disk caches.
    #[allow(dead_code)]
    pub fn clear_all_caches(&mut self) {
        self.engine.clear_all();
        self.last_compiled = None;
    }

    /// Enable AOT disk cache at the given directory.
    pub fn enable_disk_cache(&mut self, dir: PathBuf) -> std::io::Result<()> {
        self.engine.enable_disk_cache(dir)
    }

    /// Number of entries in the disk cache.
    pub fn disk_cache_len(&self) -> usize {
        self.engine.disk_cache_len()
    }

    /// Number of disk cache hits (lifetime).
    pub fn disk_hits(&self) -> u64 {
        self.engine.disk_hits()
    }
}

/// Inspection snapshot of a compiled program.
pub struct ProgramInfo {
    pub node_count: usize,
    pub input_count: usize,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shape: Vec<usize>,
    pub total_elements: usize,
    pub op_summary: Vec<(&'static str, usize)>,
}
