/// Recursive-descent parser for ferrite JIT scripts.
///
/// Grammar (informal):
/// ```text
/// script     := stmt*
/// stmt       := fn_def | tile_stmt | if_stmt | for_stmt | return_stmt | let_stmt
/// fn_def     := 'fn' IDENT '(' params ')' ':' stmt* 'end'
/// params     := (IDENT (',' IDENT)*)?
/// return_stmt:= 'return' IDENT
/// let_stmt   := IDENT '=' expr (where_clause)?
/// where_clause := 'where' expr (',' expr)*
/// tile_stmt  := 'tile' IDENT 'over' input_list ('with' tile_ann_list)? ':' stmt* 'end'
/// input_list := IDENT | '(' IDENT (',' IDENT)* ')'
/// tile_ann_list := '(' tile_ann (',' tile_ann)* ')' | tile_ann
/// tile_ann := IDENT '=' (IDENT | INT)
/// if_stmt    := 'if' expr 'then' stmt* ('elif' expr 'then' stmt*)* ('else' stmt*)? 'end'
/// for_stmt   := 'for' IDENT 'in' INT '..' INT ':' stmt* 'end'
///
/// expr       := logic_or
/// logic_or   := logic_and ('or' logic_and)*
/// logic_and  := logic_not ('and' logic_not)*
/// logic_not  := 'not' logic_not | comparison
/// comparison := add_expr (('<' | '>' | '<=' | '>=' | '==' | '!=') add_expr)?
/// add_expr   := mul_expr (('+' | '-') mul_expr)*
/// mul_expr   := unary_expr (('*' | '/') unary_expr)*
/// unary_expr := '-' unary_expr | atom
/// atom       := '(' expr ')' | call | shape | float | int | bool | var
/// call       := IDENT '(' args ')'
/// args       := (arg (',' arg)*)?
/// arg        := IDENT '=' expr   (named)
///             | expr              (positional)
/// shape      := '[' shape_dim (',' shape_dim)* ']'
/// shape_dim  := INT | IDENT
/// ```

use super::ast::*;
use super::lexer::{Span, Spanned as LexSpanned, Token};
use super::JitError;

pub struct Parser {
    tokens: Vec<LexSpanned>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<LexSpanned>) -> Self {
        Self { tokens, pos: 0 }
    }

    // ── public entry ─────────────────────────────────────────────

    pub fn parse(mut self) -> Result<Script, JitError> {
        let mut stmts = Vec::new();
        while !self.at_eof() {
            stmts.push(self.stmt()?);
        }
        Ok(Script { stmts })
    }

    // ── statements ───────────────────────────────────────────────

    fn stmt(&mut self) -> Result<Stmt, JitError> {
        match self.peek() {
            Token::Fn => self.fn_def(),
            Token::Tile => self.tile_stmt(),
            Token::If => self.if_stmt(),
            Token::For => self.for_stmt(),
            Token::Return => self.return_stmt(),
            Token::Ident(_) => self.let_stmt(),
            other => Err(self.err(format!("expected statement, got {:?}", other))),
        }
    }

    fn fn_def(&mut self) -> Result<Stmt, JitError> {
        let start_span = self.span();
        self.eat(&Token::Fn)?;
        let name = self.ident()?;
        self.eat(&Token::LParen)?;
        let params = self.params()?;
        self.eat(&Token::RParen)?;
        self.eat(&Token::Colon)?;

        let mut body = Vec::new();
        while !self.check(&Token::End) && !self.at_eof() {
            body.push(self.stmt()?);
        }
        let end_span = self.span();
        self.eat(&Token::End)?;

        Ok(Stmt::FnDef { name, params, body, span: start_span.merge(end_span) })
    }

    fn tile_stmt(&mut self) -> Result<Stmt, JitError> {
        let start_span = self.span();
        self.eat(&Token::Tile)?;
        let output = self.ident()?;
        self.eat(&Token::Over)?;

        // input_list: IDENT | '(' IDENT (',' IDENT)* ')'
        let inputs = if self.check(&Token::LParen) {
            self.advance(); // eat '('
            let mut names = Vec::new();
            if !self.check(&Token::RParen) {
                names.push(self.ident()?);
                while self.check(&Token::Comma) {
                    self.advance();
                    if self.check(&Token::RParen) {
                        break; // trailing comma
                    }
                    names.push(self.ident()?);
                }
            }
            self.eat(&Token::RParen)?;
            names
        } else {
            vec![self.ident()?]
        };

        let annotations = if self.check(&Token::With) {
            self.tile_annotations()?
        } else {
            TileAnnotations::default()
        };

        self.eat(&Token::Colon)?;

        let mut body = Vec::new();
        while !self.check(&Token::End) && !self.at_eof() {
            body.push(self.stmt()?);
        }
        let end_span = self.span();
        self.eat(&Token::End)?;

        Ok(Stmt::Tile { output, inputs, annotations, body, span: start_span.merge(end_span) })
    }

    fn tile_annotations(&mut self) -> Result<TileAnnotations, JitError> {
        self.eat(&Token::With)?;

        let mut ann = TileAnnotations::default();
        if self.check(&Token::LParen) {
            self.advance();
            if !self.check(&Token::RParen) {
                self.tile_annotation_pair(&mut ann)?;
                while self.check(&Token::Comma) {
                    self.advance();
                    if self.check(&Token::RParen) {
                        break;
                    }
                    self.tile_annotation_pair(&mut ann)?;
                }
            }
            self.eat(&Token::RParen)?;
        } else {
            self.tile_annotation_pair(&mut ann)?;
        }

        Ok(ann)
    }

    fn tile_annotation_pair(&mut self, ann: &mut TileAnnotations) -> Result<(), JitError> {
        let key = self.ident()?;
        self.eat(&Token::Eq)?;
        match key.as_str() {
            "tile_m" => ann.tile_m = Some(self.expect_usize()?),
            "tile_n" => ann.tile_n = Some(self.expect_usize()?),
            "tile_k" => ann.tile_k = Some(self.expect_usize()?),
            "unroll" => ann.unroll = Some(self.expect_usize()?),
            "pipeline_stages" => ann.pipeline_stages = Some(self.expect_usize()?),
            "replicas" => ann.replicas = Some(self.expect_usize()?),
            "mesh_axis" => ann.mesh_axis = Some(self.expect_usize()?),
            "precision" => {
                let raw = self.ident()?;
                ann.precision = Some(match raw.as_str() {
                    "f32" => TilePrecision::F32,
                    "f16" => TilePrecision::F16,
                    "bf16" => TilePrecision::Bf16,
                    _ => {
                        return Err(self.err(format!(
                            "invalid tile precision '{}' (expected f32|f16|bf16)",
                            raw
                        )))
                    }
                });
            }
            "quant" => {
                let raw = self.ident()?;
                ann.quant = Some(match raw.as_str() {
                    "none" => TileQuant::None,
                    "int8" => TileQuant::Int8,
                    "nf4" => TileQuant::Nf4,
                    _ => {
                        return Err(self.err(format!(
                            "invalid tile quant '{}' (expected none|int8|nf4)",
                            raw
                        )))
                    }
                });
            }
            "dist" => {
                let raw = self.ident()?;
                ann.distribution = Some(match raw.as_str() {
                    "none" => TileDistribution::None,
                    "replicate" => TileDistribution::Replicate,
                    "shard" => TileDistribution::Shard,
                    "reduce_scatter" => TileDistribution::ReduceScatter,
                    _ => {
                        return Err(self.err(format!(
                            "invalid tile dist '{}' (expected none|replicate|shard|reduce_scatter)",
                            raw
                        )))
                    }
                });
            }
            "layout" => {
                let raw = self.ident()?;
                ann.layout = Some(match raw.as_str() {
                    "row_major" => TileLayout::RowMajor,
                    "col_major" => TileLayout::ColMajor,
                    "blocked_32x8" => TileLayout::Blocked32x8,
                    "blocked_64x4" => TileLayout::Blocked64x4,
                    _ => {
                        return Err(self.err(format!(
                            "invalid tile layout '{}' (expected row_major|col_major|blocked_32x8|blocked_64x4)",
                            raw
                        )))
                    }
                });
            }
            "accum" => {
                let raw = self.ident()?;
                ann.accum = Some(match raw.as_str() {
                    "f32" => TileAccum::F32,
                    "bf16" => TileAccum::Bf16,
                    _ => {
                        return Err(self.err(format!(
                            "invalid tile accum '{}' (expected f32|bf16)",
                            raw
                        )))
                    }
                });
            }
            "collective" => {
                let raw = self.ident()?;
                ann.collective = Some(match raw.as_str() {
                    "none" => TileCollective::None,
                    "all_reduce" => TileCollective::AllReduce,
                    "reduce_scatter" => TileCollective::ReduceScatter,
                    "all_gather" => TileCollective::AllGather,
                    _ => {
                        return Err(self.err(format!(
                            "invalid tile collective '{}' (expected none|all_reduce|reduce_scatter|all_gather)",
                            raw
                        )))
                    }
                });
            }
            _ => {
                return Err(self.err(format!(
                    "unknown tile annotation key '{}' (supported: tile_m,tile_n,tile_k,unroll,pipeline_stages,precision,quant,dist,replicas,mesh_axis,layout,accum,collective)",
                    key
                )))
            }
        }
        Ok(())
    }

    fn if_stmt(&mut self) -> Result<Stmt, JitError> {
        let start_span = self.span();
        self.eat(&Token::If)?;
        let cond = self.expr()?;
        self.eat(&Token::Then)?;

        let mut body = Vec::new();
        while !self.check(&Token::Elif) && !self.check(&Token::Else)
            && !self.check(&Token::End) && !self.at_eof()
        {
            body.push(self.stmt()?);
        }
        let mut branches = vec![(cond, body)];

        // elif branches
        while self.check(&Token::Elif) {
            self.advance();
            let elif_cond = self.expr()?;
            self.eat(&Token::Then)?;
            let mut elif_body = Vec::new();
            while !self.check(&Token::Elif) && !self.check(&Token::Else)
                && !self.check(&Token::End) && !self.at_eof()
            {
                elif_body.push(self.stmt()?);
            }
            branches.push((elif_cond, elif_body));
        }

        // else branch
        let else_body = if self.check(&Token::Else) {
            self.advance();
            let mut eb = Vec::new();
            while !self.check(&Token::End) && !self.at_eof() {
                eb.push(self.stmt()?);
            }
            Some(eb)
        } else {
            None
        };

        let end_span = self.span();
        self.eat(&Token::End)?;

        Ok(Stmt::If { branches, else_body, span: start_span.merge(end_span) })
    }

    fn for_stmt(&mut self) -> Result<Stmt, JitError> {
        let start_span = self.span();
        self.eat(&Token::For)?;
        let var = self.ident()?;
        self.eat(&Token::In)?;
        let start = self.expect_int()?;
        self.eat(&Token::DotDot)?;
        let end = self.expect_int()?;
        self.eat(&Token::Colon)?;

        let mut body = Vec::new();
        while !self.check(&Token::End) && !self.at_eof() {
            body.push(self.stmt()?);
        }
        let end_span = self.span();
        self.eat(&Token::End)?;

        Ok(Stmt::For { var, start, end, body, span: start_span.merge(end_span) })
    }

    fn params(&mut self) -> Result<Vec<String>, JitError> {
        let mut out = Vec::new();
        if self.check(&Token::RParen) {
            return Ok(out);
        }
        out.push(self.ident()?);
        while self.check(&Token::Comma) {
            self.advance();
            out.push(self.ident()?);
        }
        Ok(out)
    }

    fn return_stmt(&mut self) -> Result<Stmt, JitError> {
        let span = self.span();
        self.advance(); // eat 'return'
        let name = self.ident()?;
        Ok(Stmt::Return { name, span })
    }

    fn let_stmt(&mut self) -> Result<Stmt, JitError> {
        let start_span = self.span();
        let name = self.ident()?;
        self.eat(&Token::Eq)?;
        let value = self.expr()?;

        let mut where_clauses = Vec::new();
        if self.is_ident_text("where") && !matches!(self.lookahead(1), Some(Token::Eq)) {
            self.advance(); // consume contextual `where`
            where_clauses.push(self.expr()?);
            while self.check(&Token::Comma) {
                self.advance();
                where_clauses.push(self.expr()?);
            }
        }

        let end_span = if let Some(last) = where_clauses.last() {
            value.span.merge(last.span)
        } else {
            value.span
        };

        Ok(Stmt::Let {
            name,
            value,
            where_clauses,
            span: start_span.merge(end_span),
        })
    }

    // ── expressions (precedence climbing) ─────────────────────────

    fn expr(&mut self) -> Result<SExpr, JitError> {
        self.logic_or()
    }

    fn logic_or(&mut self) -> Result<SExpr, JitError> {
        let mut left = self.logic_and()?;
        while self.check(&Token::Or) {
            self.advance();
            let right = self.logic_and()?;
            let span = left.span.merge(right.span);
            left = SExpr::new(
                Expr::Logic { op: LogicOp::Or, left: Box::new(left), right: Box::new(right) },
                span,
            );
        }
        Ok(left)
    }

    fn logic_and(&mut self) -> Result<SExpr, JitError> {
        let mut left = self.logic_not()?;
        while self.check(&Token::And) {
            self.advance();
            let right = self.logic_not()?;
            let span = left.span.merge(right.span);
            left = SExpr::new(
                Expr::Logic { op: LogicOp::And, left: Box::new(left), right: Box::new(right) },
                span,
            );
        }
        Ok(left)
    }

    fn logic_not(&mut self) -> Result<SExpr, JitError> {
        if self.check(&Token::Not) {
            let start_span = self.span();
            self.advance();
            let inner = self.logic_not()?;
            let span = start_span.merge(inner.span);
            Ok(SExpr::new(Expr::LogicNot(Box::new(inner)), span))
        } else {
            self.comparison()
        }
    }

    fn comparison(&mut self) -> Result<SExpr, JitError> {
        let left = self.add_expr()?;
        let op = match self.peek() {
            Token::Lt => CmpOp::Lt,
            Token::Gt => CmpOp::Gt,
            Token::Le => CmpOp::Le,
            Token::Ge => CmpOp::Ge,
            Token::EqEq => CmpOp::Eq,
            Token::Ne => CmpOp::Ne,
            _ => return Ok(left),
        };
        self.advance();
        let right = self.add_expr()?;
        let span = left.span.merge(right.span);
        Ok(SExpr::new(
            Expr::Cmp { op, left: Box::new(left), right: Box::new(right) },
            span,
        ))
    }

    fn add_expr(&mut self) -> Result<SExpr, JitError> {
        let mut left = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Token::Plus => BinOper::Add,
                Token::Minus => BinOper::Sub,
                _ => break,
            };
            self.advance();
            let right = self.mul_expr()?;
            let span = left.span.merge(right.span);
            left = SExpr::new(
                Expr::BinOp { op, left: Box::new(left), right: Box::new(right) },
                span,
            );
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> Result<SExpr, JitError> {
        let mut left = self.unary_expr()?;
        loop {
            let op = match self.peek() {
                Token::Star => BinOper::Mul,
                Token::Slash => BinOper::Div,
                _ => break,
            };
            self.advance();
            let right = self.unary_expr()?;
            let span = left.span.merge(right.span);
            left = SExpr::new(
                Expr::BinOp { op, left: Box::new(left), right: Box::new(right) },
                span,
            );
        }
        Ok(left)
    }

    fn unary_expr(&mut self) -> Result<SExpr, JitError> {
        if self.check(&Token::Minus) {
            let start_span = self.span();
            self.advance();
            let inner = self.unary_expr()?;
            let span = start_span.merge(inner.span);
            Ok(SExpr::new(Expr::Neg(Box::new(inner)), span))
        } else {
            self.atom()
        }
    }

    fn atom(&mut self) -> Result<SExpr, JitError> {
        let start_span = self.span();
        match self.peek() {
            Token::LParen => {
                self.advance(); // eat '('
                let inner = self.expr()?;
                self.eat(&Token::RParen)?;
                Ok(inner)
            }
            Token::LBracket => self.shape(),
            Token::Float(v) => {
                self.advance();
                Ok(SExpr::new(Expr::Float(v), start_span))
            }
            Token::Int(_) => self.int_lit(),
            Token::True => {
                self.advance();
                Ok(SExpr::new(Expr::Bool(true), start_span))
            }
            Token::False => {
                self.advance();
                Ok(SExpr::new(Expr::Bool(false), start_span))
            }
            Token::Ident(_) => {
                let name = self.ident()?;
                if self.check(&Token::LParen) {
                    self.call(name, start_span)
                } else {
                    Ok(SExpr::new(Expr::Var(name), start_span))
                }
            }
            other => Err(self.err(format!("expected expression, got {:?}", other))),
        }
    }

    fn call(&mut self, func: String, start_span: Span) -> Result<SExpr, JitError> {
        self.eat(&Token::LParen)?;
        let args = self.args()?;
        let end_span = self.span();
        self.eat(&Token::RParen)?;
        Ok(SExpr::new(Expr::Call { func, args }, start_span.merge(end_span)))
    }

    fn args(&mut self) -> Result<Vec<Arg>, JitError> {
        let mut out = Vec::new();
        if self.check(&Token::RParen) {
            return Ok(out);
        }
        out.push(self.arg()?);
        while self.check(&Token::Comma) {
            self.advance();
            out.push(self.arg()?);
        }
        Ok(out)
    }

    fn arg(&mut self) -> Result<Arg, JitError> {
        // Lookahead: IDENT followed by '=' → named argument
        if matches!(self.peek(), Token::Ident(_)) {
            if self.lookahead(1).map(|t| t == &Token::Eq).unwrap_or(false) {
                let name = self.ident()?;
                self.advance(); // eat '='
                let value = self.expr()?;
                return Ok(Arg::Named { name, value });
            }
        }
        let value = self.expr()?;
        Ok(Arg::Positional(value))
    }

    fn shape(&mut self) -> Result<SExpr, JitError> {
        let start_span = self.span();
        self.eat(&Token::LBracket)?;
        let mut dims = Vec::new();
        if !self.check(&Token::RBracket) {
            dims.push(self.shape_dim()?);
            while self.check(&Token::Comma) {
                self.advance();
                if self.check(&Token::RBracket) {
                    break; // trailing comma
                }
                dims.push(self.shape_dim()?);
            }
        }
        let end_span = self.span();
        self.eat(&Token::RBracket)?;
        Ok(SExpr::new(Expr::Shape(dims), start_span.merge(end_span)))
    }

    fn shape_dim(&mut self) -> Result<ShapeDim, JitError> {
        match self.peek() {
            Token::Int(v) => {
                self.advance();
                if v < 0 {
                    return Err(self.err(format!(
                        "shape dimensions must be non-negative, got {}",
                        v
                    )));
                }
                Ok(ShapeDim::Const(v as usize))
            }
            Token::Ident(name) => {
                self.advance();
                Ok(ShapeDim::Symbol(name))
            }
            other => Err(self.err(format!(
                "expected shape dimension (INT or IDENT), got {:?}",
                other
            ))),
        }
    }

    fn int_lit(&mut self) -> Result<SExpr, JitError> {
        let span = self.span();
        let v = self.expect_int()?;
        Ok(SExpr::new(Expr::Int(v), span))
    }

    // ── helpers ──────────────────────────────────────────────────

    fn peek(&self) -> Token {
        self.tokens
            .get(self.pos)
            .map(|s| s.token.clone())
            .unwrap_or(Token::Eof)
    }

    fn lookahead(&self, offset: usize) -> Option<&Token> {
        self.tokens.get(self.pos + offset).map(|s| &s.token)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn at_eof(&self) -> bool {
        self.pos >= self.tokens.len() || self.tokens[self.pos].token == Token::Eof
    }

    fn check(&self, token: &Token) -> bool {
        self.peek() == *token
    }

    fn is_ident_text(&self, text: &str) -> bool {
        matches!(self.peek(), Token::Ident(ref s) if s == text)
    }

    fn eat(&mut self, expected: &Token) -> Result<(), JitError> {
        let actual = self.peek();
        if actual == *expected {
            self.advance();
            Ok(())
        } else {
            Err(self.err(format!("expected {:?}, got {:?}", expected, actual)))
        }
    }

    fn ident(&mut self) -> Result<String, JitError> {
        match self.peek() {
            Token::Ident(name) => {
                self.advance();
                Ok(name)
            }
            other => Err(self.err(format!("expected identifier, got {:?}", other))),
        }
    }

    fn expect_int(&mut self) -> Result<i64, JitError> {
        match self.peek() {
            Token::Int(v) => {
                self.advance();
                Ok(v)
            }
            other => Err(self.err(format!("expected integer, got {:?}", other))),
        }
    }

    fn expect_usize(&mut self) -> Result<usize, JitError> {
        let v = self.expect_int()?;
        if v < 0 {
            return Err(self.err(format!("expected non-negative integer, got {}", v)));
        }
        Ok(v as usize)
    }

    fn span(&self) -> Span {
        self.tokens
            .get(self.pos)
            .map(|s| s.span)
            .unwrap_or(Span { line: 0, col: 0, offset: 0, len: 0 })
    }

    fn err(&self, message: String) -> JitError {
        let s = self.span();
        JitError::Parse {
            line: s.line,
            col: s.col,
            message,
        }
    }
}

pub fn parse(tokens: Vec<LexSpanned>) -> Result<Script, JitError> {
    Parser::new(tokens).parse()
}
