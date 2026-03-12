use super::ast::Script;
use super::hir::HirGraph;
use super::lower;
use super::Result;

/// Lower JIT AST into the HIR (typed intermediate representation).
pub fn lower_script(ast: Script) -> Result<HirGraph> {
    lower::lower(ast)
}
