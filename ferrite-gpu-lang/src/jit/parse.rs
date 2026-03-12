use super::ast::Script;
use super::lexer;
use super::parser;
use super::Result;

/// Parse script text into the JIT AST.
pub fn parse_script(script: &str) -> Result<Script> {
    let tokens = lexer::tokenize(script)?;
    let ast = parser::parse(tokens)?;
    Ok(ast)
}
