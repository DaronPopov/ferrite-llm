/// Compile pipeline report — timing and counters for observability.
///
/// Returned by [`compile_script_report`] to give operators visibility
/// into where time is spent and what the compiler produced.

use std::time::Duration;

/// Compact report produced after a JIT compile.
#[derive(Debug, Clone)]
pub struct CompileReport {
    /// Time spent in the lexer.
    pub lex_time: Duration,
    /// Time spent in the parser.
    pub parse_time: Duration,
    /// Time spent in AST validation.
    pub validate_time: Duration,
    /// Time spent in the lowering pass (AST → HIR).
    pub lower_time: Duration,
    /// Time spent in constant folding.
    pub const_fold_time: Duration,
    /// Time spent in common subexpression elimination.
    pub cse_time: Duration,
    /// Time spent in the fusion optimizer.
    pub fuse_time: Duration,
    /// Time spent in dead code elimination.
    pub dce_time: Duration,
    /// Time spent in code generation (HIR → Program).
    pub codegen_time: Duration,
    /// Time spent in shape-checking / compile.
    pub compile_time: Duration,
    /// Total wall-clock time for the full pipeline.
    pub total_time: Duration,
    /// Number of nodes in the final compiled program.
    pub node_count: usize,
    /// Number of input tensors.
    pub input_count: usize,
    /// Output shape.
    pub output_shape: Vec<usize>,
    /// Whether fusion was enabled.
    pub fusion_enabled: bool,
    /// Whether the result came from cache (memory or disk).
    pub cache_hit: bool,
}

impl CompileReport {
    /// Create a zeroed report for cache hits.
    pub fn cache_hit_report(
        total_time: Duration,
        node_count: usize,
        input_count: usize,
        output_shape: Vec<usize>,
        fusion_enabled: bool,
    ) -> Self {
        Self {
            lex_time: Duration::ZERO,
            parse_time: Duration::ZERO,
            validate_time: Duration::ZERO,
            lower_time: Duration::ZERO,
            const_fold_time: Duration::ZERO,
            cse_time: Duration::ZERO,
            fuse_time: Duration::ZERO,
            dce_time: Duration::ZERO,
            codegen_time: Duration::ZERO,
            compile_time: Duration::ZERO,
            total_time,
            node_count,
            input_count,
            output_shape,
            fusion_enabled,
            cache_hit: true,
        }
    }
}

impl std::fmt::Display for CompileReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.cache_hit {
            write!(
                f,
                "compile: cache hit | total={:?} | nodes={} inputs={} out={:?}",
                self.total_time, self.node_count, self.input_count, self.output_shape
            )
        } else {
            write!(
                f,
                "compile: lex={:?} parse={:?} validate={:?} lower={:?} \
                 const_fold={:?} cse={:?} fuse={:?} dce={:?} codegen={:?} compile={:?} | \
                 total={:?} | nodes={} inputs={} out={:?} fusion={}",
                self.lex_time,
                self.parse_time,
                self.validate_time,
                self.lower_time,
                self.const_fold_time,
                self.cse_time,
                self.fuse_time,
                self.dce_time,
                self.codegen_time,
                self.compile_time,
                self.total_time,
                self.node_count,
                self.input_count,
                self.output_shape,
                self.fusion_enabled,
            )
        }
    }
}
