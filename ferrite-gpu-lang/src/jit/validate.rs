/// AST validation pass — runs between parse and lower.
///
/// Catches structural errors that the parser's grammar permits but that
/// the lowering pass would reject in confusing ways.  Errors are
/// reported with stable error codes and remediation hints.

use super::ast::*;
use super::JitError;
use super::lexer::Span;

/// Validate a parsed script for structural correctness before lowering.
///
/// Returns `Ok(())` when the script is well-formed, or the first error
/// with a stable code and hint.
pub fn validate(script: &Script) -> Result<(), JitError> {
    let mut top_has_return = false;
    let mut defined_fns: Vec<&str> = Vec::new();

    for stmt in &script.stmts {
        match stmt {
            Stmt::FnDef { name, params, body, span } => {
                validate_fn_def(name, params, body, Some(*span))?;
                defined_fns.push(name);
            }
            Stmt::Tile { output, inputs, annotations, body, span } => {
                validate_tile(
                    output,
                    inputs,
                    annotations,
                    &TileAnnotations::default(),
                    body,
                    Some(*span),
                )?;
            }
            Stmt::Let { value, where_clauses, span, .. } => {
                validate_expr(value)?;
                validate_where_clauses(value, where_clauses, Some(*span))?;
            }
            Stmt::Return { .. } => {
                top_has_return = true;
            }
            Stmt::If { branches, else_body, span } => {
                validate_if(branches, else_body, Some(*span))?;
            }
            Stmt::For { start, end, body, span, .. } => {
                validate_for(*start, *end, body, Some(*span))?;
            }
        }
    }

    if !top_has_return {
        return Err(JitError::Validate {
            code: "V001",
            message: "script has no return statement".into(),
            hint: "add `return <variable>` as the last statement".into(),
            span: None,
        });
    }

    Ok(())
}

fn validate_fn_def(name: &str, params: &[String], body: &[Stmt], span: Option<Span>) -> Result<(), JitError> {
    if body.is_empty() {
        return Err(JitError::Validate {
            code: "V002",
            message: format!("function '{}' has an empty body", name),
            hint: "add at least one statement and a return to the function body".into(),
            span,
        });
    }

    // Check for duplicate parameter names
    let mut seen_params: Vec<&str> = Vec::new();
    for p in params {
        if seen_params.contains(&p.as_str()) {
            return Err(JitError::Validate {
                code: "V003",
                message: format!("function '{}' has duplicate parameter '{}'", name, p),
                hint: "use distinct parameter names".into(),
                span,
            });
        }
        seen_params.push(p);
    }

    // Check that the body ends with a return
    let has_return = body.iter().any(|s| matches!(s, Stmt::Return { .. }));
    if !has_return {
        return Err(JitError::Validate {
            code: "V004",
            message: format!("function '{}' has no return statement", name),
            hint: "add `return <variable>` at the end of the function body".into(),
            span,
        });
    }

    // Validate expressions in body
    for stmt in body {
        match stmt {
            Stmt::Let { value, where_clauses, span, .. } => {
                validate_expr(value)?;
                validate_where_clauses(value, where_clauses, Some(*span))?;
            }
            Stmt::FnDef { .. } => {
                return Err(JitError::Validate {
                    code: "V005",
                    message: "nested function definitions are not supported".into(),
                    hint: "move the inner function to the top level".into(),
                    span,
                });
            }
            Stmt::Tile { .. } => {
                return Err(JitError::Validate {
                    code: "V006",
                    message: "tile blocks are not supported inside function definitions".into(),
                    hint: "move the tile block outside the function".into(),
                    span,
                });
            }
            Stmt::Return { .. } => {}
            Stmt::If { branches, else_body, .. } => validate_if(branches, else_body, span)?,
            Stmt::For { start, end, body, .. } => validate_for(*start, *end, body, span)?,
        }
    }

    Ok(())
}

fn validate_tile(
    output: &str,
    inputs: &[String],
    annotations: &TileAnnotations,
    parent_annotations: &TileAnnotations,
    body: &[Stmt],
    span: Option<Span>,
) -> Result<(), JitError> {
    if inputs.is_empty() {
        return Err(JitError::Validate {
            code: "V007",
            message: "tile block has no input variables".into(),
            hint: "specify at least one input: `tile y over (x): ... end`".into(),
            span,
        });
    }

    if body.is_empty() {
        return Err(JitError::Validate {
            code: "V008",
            message: format!("tile block for '{}' has an empty body", output),
            hint: "add statements to compute the output".into(),
            span,
        });
    }

    // Check for duplicate input names
    let mut seen: Vec<&str> = Vec::new();
    for inp in inputs {
        if seen.contains(&inp.as_str()) {
            return Err(JitError::Validate {
                code: "V009",
                message: format!("tile block has duplicate input '{}'", inp),
                hint: "use distinct input names".into(),
                span,
            });
        }
        seen.push(inp);
    }

    // Output name must not collide with input names
    if inputs.iter().any(|i| i == output) {
        return Err(JitError::Validate {
            code: "V010",
            message: format!(
                "tile output '{}' shadows an input of the same name",
                output
            ),
            hint: "use a distinct name for the tile output".into(),
            span,
        });
    }

    let effective_annotations = annotations.inherit_from(parent_annotations);
    validate_tile_annotations(&effective_annotations, span)?;

    // Validate body statements
    for stmt in body {
        match stmt {
            Stmt::Let { value, where_clauses, span, .. } => {
                validate_expr(value)?;
                validate_where_clauses(value, where_clauses, Some(*span))?;
            }
            Stmt::FnDef { .. } => {
                return Err(JitError::Validate {
                    code: "V005",
                    message: "function definitions not allowed inside tile block".into(),
                    hint: "move the function to the top level".into(),
                    span,
                });
            }
            Stmt::Tile { output, inputs, annotations, body, span } => {
                validate_tile(
                    output,
                    inputs,
                    annotations,
                    &effective_annotations,
                    body,
                    Some(*span),
                )?;
            }
            Stmt::Return { .. } => {
                return Err(JitError::Validate {
                    code: "V012",
                    message: "return not allowed inside tile block".into(),
                    hint: "assign the result to the tile output variable instead".into(),
                    span,
                });
            }
            Stmt::If { branches, else_body, .. } => validate_if(branches, else_body, span)?,
            Stmt::For { start, end, body, .. } => validate_for(*start, *end, body, span)?,
        }
    }

    Ok(())
}

fn validate_if(branches: &[(SExpr, Vec<Stmt>)], else_body: &Option<Vec<Stmt>>, span: Option<Span>) -> Result<(), JitError> {
    if branches.is_empty() {
        return Err(JitError::Validate {
            code: "V015",
            message: "if statement has no condition branches".into(),
            hint: "if must have at least one condition: `if cond then ... end`".into(),
            span,
        });
    }
    for (cond, body) in branches {
        validate_expr(cond)?;
        for stmt in body {
            validate_stmt_inner(stmt, span)?;
        }
    }
    if let Some(eb) = else_body {
        for stmt in eb {
            validate_stmt_inner(stmt, span)?;
        }
    }
    Ok(())
}

fn validate_for(start: i64, end: i64, body: &[Stmt], span: Option<Span>) -> Result<(), JitError> {
    if start >= end {
        return Err(JitError::Validate {
            code: "V016",
            message: format!("for loop range is empty: {}..{}", start, end),
            hint: "start must be less than end".into(),
            span,
        });
    }
    if (end - start) > 1024 {
        return Err(JitError::Validate {
            code: "V017",
            message: format!("for loop range too large: {}..{} ({} iterations)", start, end, end - start),
            hint: "for loops are unrolled at compile time; max 1024 iterations".into(),
            span,
        });
    }
    for stmt in body {
        validate_stmt_inner(stmt, span)?;
    }
    Ok(())
}

fn validate_stmt_inner(stmt: &Stmt, parent_span: Option<Span>) -> Result<(), JitError> {
    match stmt {
        Stmt::Let { value, where_clauses, span, .. } => {
            validate_expr(value)?;
            validate_where_clauses(value, where_clauses, Some(*span).or(parent_span))
        }
        Stmt::Return { .. } => Ok(()),
        Stmt::FnDef { .. } => Err(JitError::Validate {
            code: "V005",
            message: "nested function definitions are not supported".into(),
            hint: "move the function to the top level".into(),
            span: parent_span,
        }),
        Stmt::Tile { output, inputs, annotations, body, span } => {
            validate_tile(
                output,
                inputs,
                annotations,
                &TileAnnotations::default(),
                body,
                Some(*span).or(parent_span),
            )
        }
        Stmt::If { branches, else_body, .. } => validate_if(branches, else_body, parent_span),
        Stmt::For { start, end, body, .. } => validate_for(*start, *end, body, parent_span),
    }
}

fn validate_where_clauses(value: &SExpr, where_clauses: &[SExpr], span: Option<Span>) -> Result<(), JitError> {
    if where_clauses.is_empty() {
        return Ok(());
    }
    let is_input_call = matches!(&value.node, Expr::Call { func, .. } if func == "input");
    if !is_input_call {
        return Err(JitError::Validate {
            code: "V029",
            message: "`where` clauses are only supported on input(...) bindings".into(),
            hint: "use: x = input([...], ...) where <constraint>".into(),
            span,
        });
    }
    for clause in where_clauses {
        validate_expr(clause)?;
    }
    Ok(())
}

fn validate_tile_annotations(ann: &TileAnnotations, span: Option<Span>) -> Result<(), JitError> {
    let positive_fields = [
        ("tile_m", ann.tile_m),
        ("tile_n", ann.tile_n),
        ("tile_k", ann.tile_k),
        ("unroll", ann.unroll),
        ("pipeline_stages", ann.pipeline_stages),
        ("replicas", ann.replicas),
    ];
    for (name, value) in positive_fields {
        if matches!(value, Some(0)) {
            return Err(JitError::Validate {
                code: "V018",
                message: format!("tile annotation '{}' must be > 0", name),
                hint: "use a positive integer value".into(),
                span,
            });
        }
    }

    let dist = ann.distribution.unwrap_or(TileDistribution::None);
    if ann.replicas.is_some() && matches!(dist, TileDistribution::None) {
        return Err(JitError::Validate {
            code: "V019",
            message: "tile annotation 'replicas' requires a distributed mode".into(),
            hint: "set dist=replicate|shard|reduce_scatter".into(),
            span,
        });
    }
    if ann.mesh_axis.is_some() && matches!(dist, TileDistribution::None) {
        return Err(JitError::Validate {
            code: "V020",
            message: "tile annotation 'mesh_axis' requires a distributed mode".into(),
            hint: "set dist=replicate|shard|reduce_scatter".into(),
            span,
        });
    }
    if !matches!(dist, TileDistribution::None) && ann.replicas.is_none() {
        return Err(JitError::Validate {
            code: "V021",
            message: "distributed tile annotations require 'replicas'".into(),
            hint: "add replicas=<N> to the tile annotations".into(),
            span,
        });
    }
    if !matches!(dist, TileDistribution::None) && ann.replicas.unwrap_or(0) < 2 {
        return Err(JitError::Validate {
            code: "V022",
            message: "distributed tile mode requires replicas >= 2".into(),
            hint: "set replicas to at least 2".into(),
            span,
        });
    }

    let collective = ann.collective.unwrap_or(TileCollective::None);
    if !matches!(collective, TileCollective::None) && matches!(dist, TileDistribution::None) {
        return Err(JitError::Validate {
            code: "V025",
            message: "tile annotation 'collective' requires distributed mode".into(),
            hint: "set dist=replicate|shard|reduce_scatter when collective is enabled".into(),
            span,
        });
    }
    if !matches!(collective, TileCollective::None) && ann.replicas.unwrap_or(0) < 2 {
        return Err(JitError::Validate {
            code: "V026",
            message: "collective tile mode requires replicas >= 2".into(),
            hint: "set replicas to at least 2 for collective operations".into(),
            span,
        });
    }

    if matches!(ann.accum, Some(TileAccum::Bf16)) && matches!(ann.precision, Some(TilePrecision::F32)) {
        return Err(JitError::Validate {
            code: "V027",
            message: "accum=bf16 conflicts with precision=f32".into(),
            hint: "use precision=f16|bf16 for accum=bf16, or set accum=f32".into(),
            span,
        });
    }

    let quant = ann.quant.unwrap_or(TileQuant::None);
    if !matches!(quant, TileQuant::None) {
        match ann.precision {
            Some(TilePrecision::F16) | Some(TilePrecision::Bf16) => {}
            Some(TilePrecision::F32) => {
                return Err(JitError::Validate {
                    code: "V023",
                    message: "quantized tile requires mixed precision (f16 or bf16)".into(),
                    hint: "set precision=f16 or precision=bf16 when quant is enabled".into(),
                    span,
                });
            }
            None => {
                return Err(JitError::Validate {
                    code: "V024",
                    message: "quantized tile requires explicit precision".into(),
                    hint: "add precision=f16 or precision=bf16".into(),
                    span,
                });
            }
        }
    }

    Ok(())
}

/// Validate an expression tree for structural correctness.
fn validate_expr(expr: &SExpr) -> Result<(), JitError> {
    match &expr.node {
        Expr::Shape(dims) => {
            if dims.is_empty() {
                return Err(JitError::Validate {
                    code: "V013",
                    message: "empty shape literal `[]`".into(),
                    hint: "shapes must have at least one dimension, e.g. `[4]`".into(),
                    span: Some(expr.span),
                });
            }
            for (i, d) in dims.iter().enumerate() {
                match d {
                    ShapeDim::Const(0) => {
                        return Err(JitError::Validate {
                            code: "V014",
                            message: format!(
                                "shape literal has zero at dimension {} — `{:?}`",
                                i, dims
                            ),
                            hint: "all shape dimensions must be >= 1".into(),
                            span: Some(expr.span),
                        });
                    }
                    ShapeDim::Const(_) => {}
                    ShapeDim::Symbol(name) => {
                        if name.is_empty() {
                            return Err(JitError::Validate {
                                code: "V028",
                                message: "shape symbol must be non-empty".into(),
                                hint: "use identifiers like B, T, or H".into(),
                                span: Some(expr.span),
                            });
                        }
                    }
                }
            }
        }
        Expr::Call { args, .. } => {
            for arg in args {
                match arg {
                    Arg::Positional(e) => validate_expr(e)?,
                    Arg::Named { value, .. } => validate_expr(value)?,
                }
            }
        }
        Expr::BinOp { left, right, .. } => {
            validate_expr(left)?;
            validate_expr(right)?;
        }
        Expr::Neg(inner) => validate_expr(inner)?,
        Expr::Cmp { left, right, .. } => {
            validate_expr(left)?;
            validate_expr(right)?;
        }
        Expr::Logic { left, right, .. } => {
            validate_expr(left)?;
            validate_expr(right)?;
        }
        Expr::LogicNot(inner) => validate_expr(inner)?,
        Expr::Var(_) | Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jit::lexer::tokenize;
    use crate::jit::parser;

    fn parse_script(src: &str) -> Script {
        let toks = tokenize(src).unwrap();
        parser::parse(toks).unwrap()
    }

    #[test]
    fn validate_ok_simple() {
        let script = parse_script("x = input([1, 4])\ny = relu(x)\nreturn y");
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_missing_return() {
        let script = parse_script("x = input([1, 4])\ny = relu(x)");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V001"));
    }

    #[test]
    fn validate_empty_fn_body() {
        // Parser allows fn with empty body; validation catches it.
        let script = parse_script("fn noop(): end\nx = input([1])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V002"));
    }

    #[test]
    fn validate_fn_missing_return() {
        let script = parse_script("fn bad(x): y = relu(x) end\nx = input([4])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V004"));
    }

    #[test]
    fn validate_duplicate_fn_params() {
        let script = parse_script("fn dup(x, x): return x end\na = input([4])\nreturn a");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V003"));
    }

    #[test]
    fn validate_empty_shape() {
        let script = parse_script("x = input([])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V013"));
    }

    #[test]
    fn validate_zero_dim_shape() {
        let script = parse_script("x = input([1, 0, 4])\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V014"));
    }

    #[test]
    fn validate_symbolic_shape_literal() {
        let script = parse_script("x = input([B, T, H], B=2, T=8, H=64)\nreturn x");
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_where_clause_only_on_input() {
        let script = parse_script("x = relu(input([4])) where true\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V029"));
    }

    #[test]
    fn validate_tile_no_inputs() {
        // Can't construct this with the parser (grammar requires at least one input),
        // but the validator handles it defensively.
    }

    #[test]
    fn validate_tile_empty_body() {
        let script = parse_script("x = input([4])\ntile y over x: end\nreturn y");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V008"));
    }

    #[test]
    fn validate_tile_output_shadows_input() {
        let script = parse_script("x = input([4])\ntile x over x: x = x * 2.0 end\nreturn x");
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V010"));
    }

    #[test]
    fn validate_ok_with_fn() {
        let script = parse_script(
            "fn activate(x):\n  h = relu(x)\n  return h\nend\na = input([4])\nb = activate(a)\nreturn b",
        );
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_ok_with_tile() {
        let script = parse_script(
            "x = input([4])\ntile y over (x):\n  y = x * 2.0 + 1.0\nend\nreturn y",
        );
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_ok_with_nested_tile() {
        let script = parse_script(
            "x = input([4])\ntile y over (x):\n  t = x * 2.0\n  tile z over (t):\n    z = t + 1.0\n  end\n  y = z * 3.0\nend\nreturn y",
        );
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_ok_with_annotated_tile() {
        let script = parse_script(
            "x = input([4])\ntile y over (x) with (tile_m=64, tile_n=64, precision=bf16, quant=nf4, dist=shard, replicas=2, mesh_axis=0, layout=blocked_32x8, accum=bf16, collective=all_reduce):\n  y = x * 2.0\nend\nreturn y",
        );
        assert!(validate(&script).is_ok());
    }

    #[test]
    fn validate_err_quant_requires_mixed_precision() {
        let script = parse_script(
            "x = input([4])\ntile y over (x) with (precision=f32, quant=int8):\n  y = x * 2.0\nend\nreturn y",
        );
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V023"));
    }

    #[test]
    fn validate_err_collective_requires_dist() {
        let script = parse_script(
            "x = input([4])\ntile y over (x) with (collective=all_reduce):\n  y = x * 2.0\nend\nreturn y",
        );
        let err = validate(&script).unwrap_err();
        assert!(err.to_string().contains("V025"));
    }

    #[test]
    fn validate_tile_annotation_inheritance() {
        let script = parse_script(
            "x = input([4])\ntile y over (x) with (dist=shard, replicas=2):\n  tile z over (x) with (collective=all_reduce):\n    z = x * 2.0\n  end\n  y = z\nend\nreturn y",
        );
        assert!(validate(&script).is_ok());
    }
}
