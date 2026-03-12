/// Tokeniser for ferrite JIT scripts.
///
/// Produces a flat `Vec<Spanned>` from source text.  Whitespace and
/// `#`-comments are discarded.  Newlines are not significant — the
/// parser determines statement boundaries from token patterns.

use super::JitError;

// ── Tokens ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Ident(String),
    Int(i64),
    Float(f64),
    True,
    False,
    Eq,       // =
    EqEq,     // ==
    Ne,       // !=
    Lt,       // <
    Gt,       // >
    Le,       // <=
    Ge,       // >=
    LParen,   // (
    RParen,   // )
    LBracket, // [
    RBracket, // ]
    Comma,    // ,
    Colon,    // :
    Plus,     // +
    Minus,    // -
    Star,     // *
    Slash,    // /
    DotDot,   // ..
    Return,
    Fn,
    End,
    Tile,
    Over,
    With,
    If,
    Else,
    Elif,
    Then,
    For,
    While,
    In,
    Do,
    And,
    Or,
    Not,
    Eof,
}

#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub line: usize,
    pub col: usize,
    pub offset: usize,
    pub len: usize,
}

impl Span {
    /// Merge two spans into one covering both.
    pub fn merge(self, other: Span) -> Span {
        let start = self.offset.min(other.offset);
        let end = (self.offset + self.len).max(other.offset + other.len);
        Span {
            line: self.line,
            col: self.col,
            offset: start,
            len: end - start,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned {
    pub token: Token,
    pub span: Span,
}

// ── Lexer ────────────────────────────────────────────────────────

pub fn tokenize(input: &str) -> Result<Vec<Spanned>, JitError> {
    let mut tokens = Vec::new();
    let bytes = input.as_bytes();
    let mut pos = 0usize;
    let mut line = 1usize;
    let mut col = 1usize;

    while pos < bytes.len() {
        let b = bytes[pos];

        // Whitespace (space / tab / CR)
        if b == b' ' || b == b'\t' || b == b'\r' {
            pos += 1;
            col += 1;
            continue;
        }

        // Newline
        if b == b'\n' {
            pos += 1;
            line += 1;
            col = 1;
            continue;
        }

        // Line comment
        if b == b'#' {
            while pos < bytes.len() && bytes[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }

        let start_offset = pos;
        let span_line = line;
        let span_col = col;

        // Multi-character operators: ==, !=, <=, >=, ..
        if pos + 1 < bytes.len() {
            let two = [bytes[pos], bytes[pos + 1]];
            let tok = match &two {
                b"==" => Some(Token::EqEq),
                b"!=" => Some(Token::Ne),
                b"<=" => Some(Token::Le),
                b">=" => Some(Token::Ge),
                b".." => Some(Token::DotDot),
                _ => None,
            };
            if let Some(tok) = tok {
                tokens.push(Spanned {
                    token: tok,
                    span: Span { line: span_line, col: span_col, offset: start_offset, len: 2 },
                });
                pos += 2;
                col += 2;
                continue;
            }
        }

        // Single-character symbols
        let sym = match b {
            b'=' => Some(Token::Eq),
            b'<' => Some(Token::Lt),
            b'>' => Some(Token::Gt),
            b'(' => Some(Token::LParen),
            b')' => Some(Token::RParen),
            b'[' => Some(Token::LBracket),
            b']' => Some(Token::RBracket),
            b',' => Some(Token::Comma),
            b':' => Some(Token::Colon),
            b'+' => Some(Token::Plus),
            b'-' => Some(Token::Minus),
            b'*' => Some(Token::Star),
            b'/' => Some(Token::Slash),
            _ => None,
        };
        if let Some(tok) = sym {
            tokens.push(Spanned {
                token: tok,
                span: Span { line: span_line, col: span_col, offset: start_offset, len: 1 },
            });
            pos += 1;
            col += 1;
            continue;
        }

        // Number literal (integer or float)
        if b.is_ascii_digit() {
            let start = pos;
            while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                pos += 1;
            }
            let mut is_float = false;
            // Fractional part: '.' followed by digit (but not '..' range)
            if pos < bytes.len() && bytes[pos] == b'.'
                && pos + 1 < bytes.len() && bytes[pos + 1].is_ascii_digit()
            {
                is_float = true;
                pos += 1; // skip '.'
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            // Exponent part: 'e' or 'E', optional sign, digits
            if pos < bytes.len() && (bytes[pos] == b'e' || bytes[pos] == b'E') {
                is_float = true;
                pos += 1;
                if pos < bytes.len() && (bytes[pos] == b'+' || bytes[pos] == b'-') {
                    pos += 1;
                }
                if pos >= bytes.len() || !bytes[pos].is_ascii_digit() {
                    return Err(JitError::Lex {
                        line,
                        col,
                        message: "invalid number: expected digit after exponent".into(),
                    });
                }
                while pos < bytes.len() && bytes[pos].is_ascii_digit() {
                    pos += 1;
                }
            }
            let text = std::str::from_utf8(&bytes[start..pos]).unwrap();
            let token_len = pos - start;
            col += token_len;
            if is_float {
                let value = text.parse::<f64>().map_err(|_| JitError::Lex {
                    line,
                    col,
                    message: format!("invalid float: {}", text),
                })?;
                tokens.push(Spanned {
                    token: Token::Float(value),
                    span: Span { line: span_line, col: span_col, offset: start_offset, len: token_len },
                });
            } else {
                let value = text.parse::<i64>().map_err(|_| JitError::Lex {
                    line,
                    col,
                    message: format!("invalid integer: {}", text),
                })?;
                tokens.push(Spanned {
                    token: Token::Int(value),
                    span: Span { line: span_line, col: span_col, offset: start_offset, len: token_len },
                });
            }
            continue;
        }

        // Identifier / keyword
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = pos;
            while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
                pos += 1;
            }
            let text = std::str::from_utf8(&bytes[start..pos]).unwrap();
            let token = match text {
                "return" => Token::Return,
                "fn" => Token::Fn,
                "end" => Token::End,
                "true" => Token::True,
                "false" => Token::False,
                "tile" => Token::Tile,
                "over" => Token::Over,
                "with" => Token::With,
                "if" => Token::If,
                "else" => Token::Else,
                "elif" => Token::Elif,
                "then" => Token::Then,
                "for" => Token::For,
                "while" => Token::While,
                "in" => Token::In,
                "do" => Token::Do,
                "and" => Token::And,
                "or" => Token::Or,
                "not" => Token::Not,
                _ => Token::Ident(text.to_string()),
            };
            let token_len = pos - start;
            col += token_len;
            tokens.push(Spanned {
                token,
                span: Span { line: span_line, col: span_col, offset: start_offset, len: token_len },
            });
            continue;
        }

        return Err(JitError::Lex {
            line,
            col,
            message: format!("unexpected character: '{}'", b as char),
        });
    }

    tokens.push(Spanned {
        token: Token::Eof,
        span: Span { line, col, offset: pos, len: 0 },
    });
    Ok(tokens)
}
