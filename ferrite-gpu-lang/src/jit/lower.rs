/// Lowers a parsed AST into a `HirGraph` (typed intermediate representation).
///
/// Builtin ops (`relu`, `add`, `input`, …) map 1:1 to `HirOp`
/// nodes.  User-defined functions are **inlined** — the function
/// body is spliced into the caller's graph with parameters bound to
/// the caller's `HirId`s.  No runtime call overhead.
///
/// Float literals and integer literals in expression position are
/// tracked as `Scalar(f64)` values until they meet a tensor operand,
/// at which point a `FillLike` op broadcasts them to the tensor's shape.

use std::collections::{HashMap, HashSet};

use super::ast::*;
use super::hir::*;
use super::lexer::Span;
use super::JitError;

/// Result of lowering an expression — either a concrete IR tensor or a
/// scalar constant that hasn't been materialised yet.
enum LowerResult {
    Tensor(HirId),
    Scalar(f64),
}

struct Ctx {
    graph: HirGraph,
    names: HashMap<String, HirId>,
    functions: HashMap<String, (Vec<String>, Vec<Stmt>)>,
}

impl Ctx {
    fn new() -> Self {
        Self {
            graph: HirGraph::new(),
            names: HashMap::new(),
            functions: HashMap::new(),
        }
    }

    // ── top-level ────────────────────────────────────────────────

    fn lower(mut self, script: Script) -> Result<HirGraph, JitError> {
        // First pass: collect function definitions
        for stmt in &script.stmts {
            if let Stmt::FnDef { name, params, body, .. } = stmt {
                self.functions
                    .insert(name.clone(), (params.clone(), body.clone()));
            }
        }

        // Second pass: process statements
        for stmt in &script.stmts {
            self.lower_stmt(stmt)?;
        }

        Ok(self.graph)
    }

    fn lower_stmt(&mut self, stmt: &Stmt) -> Result<(), JitError> {
        match stmt {
            Stmt::FnDef { .. } => {} // already collected
            Stmt::Let { name, value, where_clauses, .. } => {
                self.lower_let_stmt(name, value, where_clauses)?;
            }
            Stmt::Return { name, .. } => {
                let hid = self.resolve(name)?;
                self.graph.output = Some(hid);
            }
            Stmt::Tile { output, inputs, body, .. } => {
                self.lower_tile(output, inputs, body)?;
            }
            Stmt::If { branches, else_body, span } => {
                self.lower_if(branches, else_body, Some(*span))?;
            }
            Stmt::For { var, start, end, body, .. } => {
                self.lower_for(var, *start, *end, body)?;
            }
        }
        Ok(())
    }

    // ── tile lowering ───────────────────────────────────────────

    fn lower_tile(
        &mut self,
        output: &str,
        inputs: &[String],
        body: &[Stmt],
    ) -> Result<(), JitError> {
        // Verify all inputs exist
        for name in inputs {
            if !self.names.contains_key(name) {
                return Err(JitError::Lower(format!(
                    "tile: undefined input variable: {}",
                    name
                )));
            }
        }

        // Process body statements
        for stmt in body {
            match stmt {
                Stmt::Let { name, value, where_clauses, .. } => {
                    self.lower_let_stmt(name, value, where_clauses)?;
                }
                Stmt::FnDef { .. } => {
                    return Err(JitError::Lower(
                        "function definitions not allowed inside tile block".into(),
                    ));
                }
                Stmt::Return { .. } => {
                    return Err(JitError::Lower(
                        "return not allowed inside tile block".into(),
                    ));
                }
                Stmt::Tile { output, inputs, body, .. } => {
                    self.lower_tile(output, inputs, body)?;
                }
                Stmt::If { branches, else_body, span } => {
                    self.lower_if(branches, else_body, Some(*span))?;
                }
                Stmt::For { var, start, end, body, .. } => {
                    self.lower_for(var, *start, *end, body)?;
                }
            }
        }

        // Verify output was assigned
        if !self.names.contains_key(output) {
            return Err(JitError::Lower(format!(
                "tile: output variable '{}' was not assigned in body",
                output
            )));
        }

        Ok(())
    }

    // ── control flow lowering ──────────────────────────────────

    fn lower_if(
        &mut self,
        branches: &[(SExpr, Vec<Stmt>)],
        else_body: &Option<Vec<Stmt>>,
        _span: Option<Span>,
    ) -> Result<(), JitError> {
        // Both-branches-execute model: compute all branches, then Where
        // For now, we support if/else as:
        //   cond_mask = lower(cond)  -- must produce a tensor
        //   then_result = lower(then_body)  -- last assigned value
        //   else_result = lower(else_body)  -- last assigned value
        //   result = Where(cond_mask, then_result, else_result)
        //
        // For multiple elif branches, we chain Where ops.
        //
        // Limitations: Both branches must assign the same set of variables.
        // For now, we process branches for side effects (variable assignments)
        // and don't enforce strict matching — the GPU executes all branches.

        // Simple case: process all branches and else body as sequential code.
        // The if/elif conditions are available for future Where-based selection.
        for (_cond, body) in branches {
            for stmt in body {
                self.lower_stmt(stmt)?;
            }
        }
        if let Some(eb) = else_body {
            for stmt in eb {
                self.lower_stmt(stmt)?;
            }
        }
        Ok(())
    }

    fn lower_for(
        &mut self,
        var: &str,
        start: i64,
        end: i64,
        body: &[Stmt],
    ) -> Result<(), JitError> {
        // Unroll the loop: substitute loop variable as a constant each iteration
        for i in start..end {
            // Bind loop variable as a scalar constant
            // We can't directly store scalars as HirId, so we track via a sentinel.
            // For now, the loop variable is available for use in expressions
            // that will materialize it via FillLike when paired with a tensor.
            self.names.remove(var); // Remove any prior binding

            // Create a constant node for the iteration value
            let const_id = self.graph.push(
                HirOp::Constant { value: i as f64 },
                HirType::Scalar(ElemType::F32),
                None,
            );
            self.names.insert(var.to_string(), const_id);

            for stmt in body {
                self.lower_stmt(stmt)?;
            }
        }
        Ok(())
    }

    fn lower_let_stmt(&mut self, name: &str, value: &SExpr, where_clauses: &[SExpr]) -> Result<(), JitError> {
        if !where_clauses.is_empty() {
            let Expr::Call { func, args } = &value.node else {
                return Err(JitError::Lower(
                    "`where` clauses are only supported on input(...) bindings".into(),
                ));
            };
            if func != "input" {
                return Err(JitError::Lower(
                    "`where` clauses are only supported on input(...) bindings".into(),
                ));
            }
            let (_, bindings) = self.input_shape_with_bindings(args)?;
            for clause in where_clauses {
                let passed = self.eval_shape_constraint(clause, &bindings)?;
                if !passed {
                    return Err(JitError::Lower(format!(
                        "input shape contract failed: {:?}",
                        clause.node
                    )));
                }
            }
        }

        let hid = self.lower_expr_to_tensor(value)?;
        self.names.insert(name.to_string(), hid);
        Ok(())
    }

    fn eval_shape_constraint(&self, expr: &SExpr, bindings: &HashMap<String, i64>) -> Result<bool, JitError> {
        enum CV {
            Int(i64),
            Bool(bool),
        }
        fn eval(expr: &SExpr, bindings: &HashMap<String, i64>) -> Result<CV, JitError> {
            match &expr.node {
                Expr::Var(name) => {
                    let Some(v) = bindings.get(name) else {
                        return Err(JitError::Lower(format!(
                            "shape constraint references unknown symbol '{}'",
                            name
                        )));
                    };
                    Ok(CV::Int(*v))
                }
                Expr::Int(v) => Ok(CV::Int(*v)),
                Expr::Bool(v) => Ok(CV::Bool(*v)),
                Expr::Float(_) => Err(JitError::Lower(
                    "shape constraints only support integer and boolean literals".into(),
                )),
                Expr::Neg(inner) => match eval(inner, bindings)? {
                    CV::Int(v) => Ok(CV::Int(-v)),
                    CV::Bool(_) => Err(JitError::Lower(
                        "shape constraint negation expects integer expression".into(),
                    )),
                },
                Expr::BinOp { op, left, right } => {
                    let l = match eval(left, bindings)? {
                        CV::Int(v) => v,
                        CV::Bool(_) => {
                            return Err(JitError::Lower(
                                "shape constraint arithmetic expects integer operands".into(),
                            ))
                        }
                    };
                    let r = match eval(right, bindings)? {
                        CV::Int(v) => v,
                        CV::Bool(_) => {
                            return Err(JitError::Lower(
                                "shape constraint arithmetic expects integer operands".into(),
                            ))
                        }
                    };
                    let out = match op {
                        BinOper::Add => l.saturating_add(r),
                        BinOper::Sub => l.saturating_sub(r),
                        BinOper::Mul => l.saturating_mul(r),
                        BinOper::Div => {
                            if r == 0 {
                                return Err(JitError::Lower(
                                    "shape constraint division by zero".into(),
                                ));
                            }
                            l / r
                        }
                    };
                    Ok(CV::Int(out))
                }
                Expr::Cmp { op, left, right } => {
                    let l = match eval(left, bindings)? {
                        CV::Int(v) => v,
                        CV::Bool(_) => {
                            return Err(JitError::Lower(
                                "shape constraint comparisons expect integer operands".into(),
                            ))
                        }
                    };
                    let r = match eval(right, bindings)? {
                        CV::Int(v) => v,
                        CV::Bool(_) => {
                            return Err(JitError::Lower(
                                "shape constraint comparisons expect integer operands".into(),
                            ))
                        }
                    };
                    let out = match op {
                        CmpOp::Lt => l < r,
                        CmpOp::Gt => l > r,
                        CmpOp::Le => l <= r,
                        CmpOp::Ge => l >= r,
                        CmpOp::Eq => l == r,
                        CmpOp::Ne => l != r,
                    };
                    Ok(CV::Bool(out))
                }
                Expr::Logic { op, left, right } => {
                    let l = match eval(left, bindings)? {
                        CV::Bool(v) => v,
                        CV::Int(_) => {
                            return Err(JitError::Lower(
                                "shape constraint logical operators expect boolean operands".into(),
                            ))
                        }
                    };
                    let r = match eval(right, bindings)? {
                        CV::Bool(v) => v,
                        CV::Int(_) => {
                            return Err(JitError::Lower(
                                "shape constraint logical operators expect boolean operands".into(),
                            ))
                        }
                    };
                    Ok(CV::Bool(match op {
                        LogicOp::And => l && r,
                        LogicOp::Or => l || r,
                    }))
                }
                Expr::LogicNot(inner) => {
                    let v = match eval(inner, bindings)? {
                        CV::Bool(v) => v,
                        CV::Int(_) => {
                            return Err(JitError::Lower(
                                "shape constraint 'not' expects boolean operand".into(),
                            ))
                        }
                    };
                    Ok(CV::Bool(!v))
                }
                Expr::Call { .. } | Expr::Shape(_) => Err(JitError::Lower(
                    "shape constraints do not support function calls or shape literals".into(),
                )),
            }
        }

        match eval(expr, bindings)? {
            CV::Bool(v) => Ok(v),
            CV::Int(_) => Err(JitError::Lower(
                "shape constraint must evaluate to boolean".into(),
            )),
        }
    }

    // ── expressions ──────────────────────────────────────────────

    /// Lower an expression, requiring a tensor result.
    fn lower_expr_to_tensor(&mut self, expr: &SExpr) -> Result<HirId, JitError> {
        match self.lower_expr(expr)? {
            LowerResult::Tensor(hid) => Ok(hid),
            LowerResult::Scalar(_) => Err(JitError::Lower(
                "scalar literal cannot be used alone — it needs a tensor operand to infer shape"
                    .into(),
            )),
        }
    }

    fn lower_expr(&mut self, expr: &SExpr) -> Result<LowerResult, JitError> {
        let span = Some(expr.span);
        match &expr.node {
            Expr::Var(name) => {
                let hid = self.resolve(name)?;
                // Check if it's a constant node — if so, return as Scalar
                if let HirOp::Constant { value } = &self.graph.node(hid).op {
                    return Ok(LowerResult::Scalar(*value));
                }
                Ok(LowerResult::Tensor(hid))
            }
            Expr::Call { func, args } => {
                Ok(LowerResult::Tensor(self.lower_call(func, args, span)?))
            }
            Expr::Float(v) => Ok(LowerResult::Scalar(*v)),
            Expr::Int(v) => Ok(LowerResult::Scalar(*v as f64)),
            Expr::Bool(v) => Ok(LowerResult::Scalar(if *v { 1.0 } else { 0.0 })),
            Expr::Neg(inner) => {
                match self.lower_expr(inner)? {
                    LowerResult::Scalar(v) => Ok(LowerResult::Scalar(-v)),
                    LowerResult::Tensor(hid) => {
                        let ty = self.graph.type_of(hid).clone();
                        let neg = self.graph.push(HirOp::Neg(hid), ty, span);
                        Ok(LowerResult::Tensor(neg))
                    }
                }
            }
            Expr::BinOp { op, left, right } => {
                let lhs = self.lower_expr(left)?;
                let rhs = self.lower_expr(right)?;
                let (l, r) = self.resolve_binop_operands(lhs, rhs, span)?;
                let ty = self.graph.type_of(l).clone();
                let hir_op = match op {
                    BinOper::Add => HirOp::Add(l, r),
                    BinOper::Sub => HirOp::Sub(l, r),
                    BinOper::Mul => HirOp::Mul(l, r),
                    BinOper::Div => HirOp::Div(l, r),
                };
                let hid = self.graph.push(hir_op, ty, span);
                Ok(LowerResult::Tensor(hid))
            }
            Expr::Cmp { op, left, right } => {
                let lhs = self.lower_expr(left)?;
                let rhs = self.lower_expr(right)?;
                let (l, r) = self.resolve_binop_operands(lhs, rhs, span)?;
                let ty = self.graph.type_of(l).clone();
                let hir_op = match op {
                    CmpOp::Lt => HirOp::CmpLt(l, r),
                    CmpOp::Gt => HirOp::CmpGt(l, r),
                    CmpOp::Le => HirOp::CmpLe(l, r),
                    CmpOp::Ge => HirOp::CmpGe(l, r),
                    CmpOp::Eq => HirOp::CmpEq(l, r),
                    CmpOp::Ne => HirOp::CmpNe(l, r),
                };
                let hid = self.graph.push(hir_op, ty, span);
                Ok(LowerResult::Tensor(hid))
            }
            Expr::Logic { op, left, right } => {
                // Logic ops on masks: and = mul, or = max(a+b, 1.0) or a+b-a*b
                // Simple approach: and = mul, or = add then clamp
                let lhs = self.lower_expr(left)?;
                let rhs = self.lower_expr(right)?;
                let (l, r) = self.resolve_binop_operands(lhs, rhs, span)?;
                let ty = self.graph.type_of(l).clone();
                let hid = match op {
                    LogicOp::And => {
                        // mask_and = mul(a, b) — both must be 1.0
                        self.graph.push(HirOp::Mul(l, r), ty, span)
                    }
                    LogicOp::Or => {
                        // mask_or = a + b - a*b (ensures 0/1 output)
                        let sum = self.graph.push(HirOp::Add(l, r), ty.clone(), span);
                        let prod = self.graph.push(HirOp::Mul(l, r), ty.clone(), span);
                        self.graph.push(HirOp::Sub(sum, prod), ty, span)
                    }
                };
                Ok(LowerResult::Tensor(hid))
            }
            Expr::LogicNot(inner) => {
                // not(mask) = 1.0 - mask
                let inner_result = self.lower_expr(inner)?;
                match inner_result {
                    LowerResult::Scalar(v) => Ok(LowerResult::Scalar(if v == 0.0 { 1.0 } else { 0.0 })),
                    LowerResult::Tensor(hid) => {
                        let ty = self.graph.type_of(hid).clone();
                        let one = self.graph.push(
                            HirOp::FillLike { value: 1.0, like: hid },
                            ty.clone(),
                            span,
                        );
                        let not = self.graph.push(HirOp::Sub(one, hid), ty, span);
                        Ok(LowerResult::Tensor(not))
                    }
                }
            }
            Expr::Shape(_) => Err(JitError::Lower(
                "shape literal cannot be used as a standalone expression".into(),
            )),
        }
    }

    /// Resolve two LowerResult operands into (HirId, HirId),
    /// materializing scalars with FillLike when paired with a tensor.
    fn resolve_binop_operands(
        &mut self,
        lhs: LowerResult,
        rhs: LowerResult,
        span: Option<Span>,
    ) -> Result<(HirId, HirId), JitError> {
        match (lhs, rhs) {
            (LowerResult::Tensor(l), LowerResult::Tensor(r)) => Ok((l, r)),
            (LowerResult::Tensor(t), LowerResult::Scalar(s)) => {
                let ty = self.graph.type_of(t).clone();
                let fill = self.graph.push(HirOp::FillLike { value: s, like: t }, ty, span);
                Ok((t, fill))
            }
            (LowerResult::Scalar(s), LowerResult::Tensor(t)) => {
                let ty = self.graph.type_of(t).clone();
                let fill = self.graph.push(HirOp::FillLike { value: s, like: t }, ty, span);
                Ok((fill, t))
            }
            (LowerResult::Scalar(_), LowerResult::Scalar(_)) => Err(JitError::Lower(
                "cannot perform binary op on two scalars — no tensor to infer shape from".into(),
            )),
        }
    }

    fn lower_call(&mut self, func: &str, args: &[Arg], span: Option<Span>) -> Result<HirId, JitError> {
        match func {
            // ── builtins ─────────────────────────────────────────
            "input" => {
                let (shape, _) = self.input_shape_with_bindings(args)?;
                let ty = HirType::f32_tensor(shape.clone());
                let hid = self.graph.push(HirOp::Input { shape: shape.clone() }, ty, span);
                self.graph.inputs.push(hid);
                Ok(hid)
            }
            "relu" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Relu(x), ty, span))
            }
            "tanh" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Tanh(x), ty, span))
            }
            "sigmoid" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Sigmoid(x), ty, span))
            }
            "gelu" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Gelu(x), ty, span))
            }
            "silu" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Silu(x), ty, span))
            }
            "abs" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Abs(x), ty, span))
            }
            "sqrt" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Sqrt(x), ty, span))
            }
            "exp" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Exp(x), ty, span))
            }
            "log" => {
                let x = self.positional_expr(args, 0)?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Log(x), ty, span))
            }
            "add" => {
                let lhs = self.positional_expr(args, 0)?;
                let rhs = self.positional_expr(args, 1)?;
                let ty = self.graph.type_of(lhs).clone();
                Ok(self.graph.push(HirOp::Add(lhs, rhs), ty, span))
            }
            "mul" => {
                let lhs = self.positional_expr(args, 0)?;
                let rhs = self.positional_expr(args, 1)?;
                let ty = self.graph.type_of(lhs).clone();
                Ok(self.graph.push(HirOp::Mul(lhs, rhs), ty, span))
            }
            "cumsum" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::CumSum { input: x, dim: dim as usize }, ty, span))
            }
            "topk" => {
                let x = self.positional_expr(args, 0)?;
                let k = self.named_int(args, "k")?;
                let dim = self.named_int(args, "dim")?;
                let largest = self.named_bool(args, "largest").unwrap_or(true);
                // TopK output shape: input shape with dim replaced by k
                let mut shape = self.graph.shape_of(x);
                let d = dim as usize;
                if d < shape.len() {
                    shape[d] = k as usize;
                }
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::TopK { input: x, k: k as usize, dim: d, largest }, ty, span))
            }
            "where_cond" | "where" => {
                let cond = self.positional_expr(args, 0)?;
                let true_val = self.positional_expr(args, 1)?;
                let false_val = self.positional_expr(args, 2)?;
                let ty = self.graph.type_of(true_val).clone();
                Ok(self.graph.push(HirOp::Where { cond, true_val, false_val }, ty, span))
            }
            "gather" => {
                let input = self.positional_expr(args, 0)?;
                let indices = self.positional_expr(args, 1)?;
                let dim = self.named_int(args, "dim")?;
                // Output shape = indices shape
                let ty = self.graph.type_of(indices).clone();
                Ok(self.graph.push(HirOp::Gather { input, indices, dim: dim as usize }, ty, span))
            }
            "index_select" => {
                let input = self.positional_expr(args, 0)?;
                let indices = self.positional_expr(args, 1)?;
                let dim = self.named_int(args, "dim")?;
                // Output: input shape with dim replaced by indices length
                let mut shape = self.graph.shape_of(input);
                let idx_shape = self.graph.shape_of(indices);
                let d = dim as usize;
                if d < shape.len() {
                    // indices is 1D, its length replaces the dim
                    shape[d] = idx_shape.iter().product();
                }
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::IndexSelect { input, indices, dim: d }, ty, span))
            }
            "scatter_add" => {
                let input = self.positional_expr(args, 0)?;
                let indices = self.positional_expr(args, 1)?;
                let src = self.positional_expr(args, 2)?;
                let dim = self.named_int(args, "dim")?;
                // Output shape = input shape
                let ty = self.graph.type_of(input).clone();
                Ok(self.graph.push(HirOp::ScatterAdd { input, indices, src, dim: dim as usize }, ty, span))
            }
            "argsort" => {
                let input = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let ascending = self.named_bool(args, "ascending").unwrap_or(true);
                // Output shape = input shape (contains indices)
                let ty = self.graph.type_of(input).clone();
                Ok(self.graph.push(HirOp::Argsort { input, dim: dim as usize, ascending }, ty, span))
            }
            "sum" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let shape = self.reduced_shape(x, dim as usize);
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::ReduceSum { input: x, dim: dim as usize }, ty, span))
            }
            "mean" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let shape = self.reduced_shape(x, dim as usize);
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::ReduceMean { input: x, dim: dim as usize }, ty, span))
            }
            "max" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let shape = self.reduced_shape(x, dim as usize);
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::ReduceMax { input: x, dim: dim as usize }, ty, span))
            }
            "min" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let shape = self.reduced_shape(x, dim as usize);
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::ReduceMin { input: x, dim: dim as usize }, ty, span))
            }
            "argmax" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let shape = self.reduced_shape(x, dim as usize);
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::Argmax { input: x, dim: dim as usize }, ty, span))
            }
            "argmin" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                let shape = self.reduced_shape(x, dim as usize);
                let ty = HirType::f32_tensor(shape);
                Ok(self.graph.push(HirOp::Argmin { input: x, dim: dim as usize }, ty, span))
            }
            "softmax" => {
                let x = self.positional_expr(args, 0)?;
                let dim = self.named_int(args, "dim")?;
                // Softmax preserves shape
                let ty = self.graph.type_of(x).clone();
                Ok(self.graph.push(HirOp::Softmax { input: x, dim: dim as usize }, ty, span))
            }
            "matmul" => {
                let lhs = self.positional_expr(args, 0)?;
                let rhs = self.positional_expr(args, 1)?;
                // [M, K] x [K, N] -> [M, N]
                let l_shape = self.graph.shape_of(lhs);
                let r_shape = self.graph.shape_of(rhs);
                if l_shape.len() != 2 || r_shape.len() != 2 {
                    return Err(JitError::Lower(
                        "matmul requires 2D inputs".into(),
                    ));
                }
                let m = l_shape[0];
                let n = r_shape[1];
                let ty = HirType::f32_tensor(vec![m, n]);
                Ok(self.graph.push(HirOp::Matmul { lhs, rhs }, ty, span))
            }

            // ── user-defined function → inline ───────────────────
            name => {
                let (params, body) = self
                    .functions
                    .get(name)
                    .ok_or_else(|| JitError::Lower(format!("unknown function: {}", name)))?
                    .clone();
                self.inline_call(&params, &body, args)
            }
        }
    }

    /// Compute the shape after reducing a dimension.
    fn reduced_shape(&self, input: HirId, dim: usize) -> Vec<usize> {
        let shape = self.graph.shape_of(input);
        let mut result: Vec<usize> = shape.iter().enumerate()
            .filter(|(i, _)| *i != dim)
            .map(|(_, &d)| d)
            .collect();
        if result.is_empty() {
            result.push(1);
        }
        result
    }

    // ── function inlining ────────────────────────────────────────

    fn inline_call(
        &mut self,
        params: &[String],
        body: &[Stmt],
        args: &[Arg],
    ) -> Result<HirId, JitError> {
        // Resolve positional arguments to HirIds
        let mut arg_hids = Vec::with_capacity(args.len());
        for arg in args {
            match arg {
                Arg::Positional(expr) => arg_hids.push(self.lower_expr_to_tensor(expr)?),
                Arg::Named { .. } => {
                    return Err(JitError::Lower(
                        "named arguments not supported for user-defined functions".into(),
                    ));
                }
            }
        }

        if params.len() != arg_hids.len() {
            return Err(JitError::Lower(format!(
                "function expects {} arguments, got {}",
                params.len(),
                arg_hids.len(),
            )));
        }

        // Save outer scope, create function scope
        let outer = self.names.clone();
        for (param, hid) in params.iter().zip(&arg_hids) {
            self.names.insert(param.clone(), *hid);
        }

        // Process function body
        let mut return_hid = None;
        for stmt in body {
            match stmt {
                Stmt::Let { name, value, where_clauses, .. } => {
                    self.lower_let_stmt(name, value, where_clauses)?;
                }
                Stmt::Return { name, .. } => {
                    return_hid = Some(self.resolve(name)?);
                }
                Stmt::FnDef { .. } => {
                    return Err(JitError::Lower(
                        "nested function definitions are not supported".into(),
                    ));
                }
                Stmt::Tile { .. } => {
                    return Err(JitError::Lower(
                        "tile blocks not supported inside function definitions".into(),
                    ));
                }
                Stmt::If { branches, else_body, span } => {
                    self.lower_if(branches, else_body, Some(*span))?;
                }
                Stmt::For { var, start, end, body, .. } => {
                    self.lower_for(var, *start, *end, body)?;
                }
            }
        }

        // Restore outer scope
        self.names = outer;

        return_hid.ok_or_else(|| JitError::Lower("function body missing return statement".into()))
    }

    // ── argument helpers ─────────────────────────────────────────

    fn resolve(&self, name: &str) -> Result<HirId, JitError> {
        self.names
            .get(name)
            .copied()
            .ok_or_else(|| JitError::Lower(format!("undefined variable: {}", name)))
    }

    /// Lower a positional argument as an arbitrary expression to a HirId.
    fn positional_expr(&mut self, args: &[Arg], index: usize) -> Result<HirId, JitError> {
        match args.get(index) {
            Some(Arg::Positional(expr)) => self.lower_expr_to_tensor(expr),
            Some(other) => Err(JitError::Lower(format!(
                "expected positional argument at {}, got {:?}",
                index, other
            ))),
            None => Err(JitError::Lower(format!(
                "missing argument at position {}",
                index
            ))),
        }
    }

    fn positional_shape(&self, args: &[Arg], index: usize) -> Result<Vec<ShapeDim>, JitError> {
        match args.get(index) {
            Some(Arg::Positional(sexpr)) => match &sexpr.node {
                Expr::Shape(dims) => Ok(dims.clone()),
                other => Err(JitError::Lower(format!(
                    "expected shape at argument {}, got {:?}",
                    index, other
                ))),
            },
            Some(other) => Err(JitError::Lower(format!(
                "expected shape at argument {}, got {:?}",
                index, other
            ))),
            None => Err(JitError::Lower(format!(
                "missing shape argument at position {}",
                index
            ))),
        }
    }

    fn input_shape_with_bindings(&self, args: &[Arg]) -> Result<(Vec<usize>, HashMap<String, i64>), JitError> {
        let dims = self.positional_shape(args, 0)?;
        let mut named_bindings: HashMap<String, i64> = HashMap::new();
        for arg in args.iter().skip(1) {
            if let Arg::Named { name, value } = arg {
                let v = match &value.node {
                    Expr::Int(i) if *i > 0 => *i,
                    Expr::Int(i) => {
                        return Err(JitError::Lower(format!(
                            "shape symbol '{}' must be > 0, got {}",
                            name, i
                        )))
                    }
                    other => {
                        return Err(JitError::Lower(format!(
                            "shape symbol '{}' requires integer value, got {:?}",
                            name, other
                        )))
                    }
                };
                if named_bindings.insert(name.clone(), v).is_some() {
                    return Err(JitError::Lower(format!(
                        "duplicate shape symbol binding: {}",
                        name
                    )));
                }
            } else {
                return Err(JitError::Lower(
                    "input shape contract only supports one positional shape followed by named symbol bindings".into(),
                ));
            }
        }

        let mut used_symbols: HashSet<String> = HashSet::new();
        let mut out = Vec::with_capacity(dims.len());
        for dim in dims {
            match dim {
                ShapeDim::Const(v) => {
                    if v == 0 {
                        return Err(JitError::Lower(
                            "input shape dimensions must be >= 1".into(),
                        ));
                    }
                    out.push(v);
                }
                ShapeDim::Symbol(name) => {
                    used_symbols.insert(name.clone());
                    let Some(v) = named_bindings.get(&name) else {
                        return Err(JitError::Lower(format!(
                            "missing shape symbol binding '{}' for input shape",
                            name
                        )));
                    };
                    out.push(*v as usize);
                }
            }
        }

        for key in named_bindings.keys() {
            if !used_symbols.contains(key) {
                return Err(JitError::Lower(format!(
                    "shape symbol binding '{}' is not used by input shape",
                    key
                )));
            }
        }

        Ok((out, named_bindings))
    }

    fn named_int(&self, args: &[Arg], name: &str) -> Result<i64, JitError> {
        for arg in args {
            if let Arg::Named {
                name: n,
                value: sexpr,
            } = arg
            {
                if n == name {
                    if let Expr::Int(v) = &sexpr.node {
                        return Ok(*v);
                    }
                }
            }
        }
        Err(JitError::Lower(format!(
            "missing required named argument: {}",
            name
        )))
    }

    fn named_bool(&self, args: &[Arg], name: &str) -> Option<bool> {
        for arg in args {
            if let Arg::Named {
                name: n,
                value: sexpr,
            } = arg
            {
                if n == name {
                    if let Expr::Bool(v) = &sexpr.node {
                        return Some(*v);
                    }
                }
            }
        }
        None
    }
}

pub fn lower(script: Script) -> Result<HirGraph, JitError> {
    Ctx::new().lower(script)
}
