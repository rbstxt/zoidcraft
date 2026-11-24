//! Tokenizer for the ZoidCraft deterministic JS-subset expression language.

use crate::ExprError;
use crate::ast::Span;

/// Single lexical token with source span and preceding-newline flag for ASI.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub line_before: bool,
}

/// Token kind enumeration.
#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Literals & identifiers
    Number(f64),
    String(String),
    Ident(String),

    // Template literal parts
    TemplateNoSub(String),
    TemplateHead(String),
    TemplateMiddle(String),
    TemplateTail(String),

    // Keywords
    Var,
    Let,
    Const,
    If,
    Else,
    For,
    In,
    While,
    Do,
    Switch,
    Case,
    Default,
    Break,
    Continue,
    Return,
    Throw,
    Try,
    Catch,
    Finally,
    Function,
    TypeOf,
    Void,
    Delete,
    True,
    False,
    Null,
    Undefined,
    New,

    // Forbidden sandbox keywords (tokenized explicitly for clear errors)
    Class,
    Import,
    Export,
    Async,
    Await,
    Yield,
    With,
    Debugger,
    This,
    Super,

    // Punctuation
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Semicolon,
    Colon,
    Dot,
    Ellipsis,
    Question,
    OptionalChain, // ?.
    Nullish,       // ??
    FatArrow,      // =>

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    StarStar, // **
    PlusPlus,
    MinusMinus,
    Eq,
    EqEq,
    EqEqEq,
    Bang,
    BangEq,
    BangEqEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Amp,
    AmpAmp,
    Pipe,
    PipePipe,
    Caret,
    Tilde,
    LtLt,
    GtGt,
    GtGtGt,
    PlusEq,
    MinusEq,
    StarEq,
    SlashEq,
    PercentEq,
    StarStarEq,

    Eof,
}

impl TokenKind {
    /// Returns the identifier text if this token may appear as a property key
    /// (identifiers and JS keywords are valid property names after `.` or in `{ key: val }`).
    pub fn as_property_name(&self) -> Option<&str> {
        match self {
            Self::Ident(s) => Some(s.as_str()),
            Self::Var => Some("var"),
            Self::Let => Some("let"),
            Self::Const => Some("const"),
            Self::If => Some("if"),
            Self::Else => Some("else"),
            Self::For => Some("for"),
            Self::In => Some("in"),
            Self::While => Some("while"),
            Self::Do => Some("do"),
            Self::Switch => Some("switch"),
            Self::Case => Some("case"),
            Self::Default => Some("default"),
            Self::Break => Some("break"),
            Self::Continue => Some("continue"),
            Self::Return => Some("return"),
            Self::Throw => Some("throw"),
            Self::Try => Some("try"),
            Self::Catch => Some("catch"),
            Self::Finally => Some("finally"),
            Self::Function => Some("function"),
            Self::TypeOf => Some("typeof"),
            Self::Void => Some("void"),
            Self::Delete => Some("delete"),
            Self::True => Some("true"),
            Self::False => Some("false"),
            Self::Null => Some("null"),
            Self::Undefined => Some("undefined"),
            Self::New => Some("new"),
            Self::Class => Some("class"),
            Self::Import => Some("import"),
            Self::Export => Some("export"),
            Self::Async => Some("async"),
            Self::Await => Some("await"),
            Self::Yield => Some("yield"),
            Self::With => Some("with"),
            Self::Debugger => Some("debugger"),
            Self::This => Some("this"),
            Self::Super => Some("super"),
            _ => None,
        }
    }
}

/// Lexer converting UTF-8 expression source into a stream of [`Token`]s.
pub struct Lexer<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
    template_brace_stack: Vec<usize>,
}

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
            template_brace_stack: Vec::new(),
        }
    }

    pub fn tokenize(mut self) -> Result<Vec<Token>, ExprError> {
        let mut tokens = Vec::new();
        loop {
            let line_before = self.skip_whitespace_and_comments()?;
            let start = self.pos;
            if self.pos >= self.bytes.len() {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span: Span::new(start, start),
                    line_before,
                });
                break;
            }
            let kind = self.next_token()?;
            let end = self.pos;
            tokens.push(Token {
                kind,
                span: Span::new(start, end),
                line_before,
            });
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn peek_at(&self, offset: usize) -> Option<u8> {
        self.bytes.get(self.pos + offset).copied()
    }

    fn bump(&mut self) -> Option<u8> {
        let b = self.bytes.get(self.pos).copied()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_whitespace_and_comments(&mut self) -> Result<bool, ExprError> {
        let mut saw_newline = false;
        while let Some(b) = self.peek() {
            if b == b'\n' || b == b'\r' {
                saw_newline = true;
                self.pos += 1;
            } else if b == b' ' || b == b'\t' || b == 0x0b || b == 0x0c {
                self.pos += 1;
            } else if b == b'/' && self.peek_at(1) == Some(b'/') {
                self.pos += 2;
                while let Some(c) = self.peek() {
                    if c == b'\n' || c == b'\r' {
                        break;
                    }
                    self.pos += 1;
                }
            } else if b == b'/' && self.peek_at(1) == Some(b'*') {
                let start = self.pos;
                self.pos += 2;
                let mut closed = false;
                while self.pos + 1 < self.bytes.len() {
                    if self.bytes[self.pos] == b'\n' || self.bytes[self.pos] == b'\r' {
                        saw_newline = true;
                    }
                    if self.bytes[self.pos] == b'*' && self.bytes[self.pos + 1] == b'/' {
                        self.pos += 2;
                        closed = true;
                        break;
                    }
                    self.pos += 1;
                }
                if !closed {
                    return Err(ExprError::SyntaxError {
                        offset: start,
                        message: "unterminated block comment".to_string(),
                    });
                }
            } else {
                break;
            }
        }
        Ok(saw_newline)
    }

    fn next_token(&mut self) -> Result<TokenKind, ExprError> {
        let start = self.pos;
        let b = self.bytes[self.pos];

        // String literal
        if b == b'"' || b == b'\'' {
            return self.lex_string(b);
        }

        // Template literal start
        if b == b'`' {
            self.pos += 1;
            return self.lex_template_chunk(true);
        }

        // Closing brace: may resume an outer template literal
        if b == b'}' {
            self.pos += 1;
            if let Some(depth) = self.template_brace_stack.last_mut() {
                if *depth == 0 {
                    self.template_brace_stack.pop();
                    return self.lex_template_chunk(false);
                }
                *depth -= 1;
            }
            return Ok(TokenKind::RBrace);
        }

        if b == b'{' {
            self.pos += 1;
            if let Some(depth) = self.template_brace_stack.last_mut() {
                *depth += 1;
            }
            return Ok(TokenKind::LBrace);
        }

        // Number literal (digit or `.digit`)
        if b.is_ascii_digit() || (b == b'.' && self.peek_at(1).is_some_and(|c| c.is_ascii_digit()))
        {
            return self.lex_number();
        }

        // Identifier or keyword (ASCII or UTF-8 alphabetic)
        if is_ident_start_byte(b) || b >= 0x80 {
            return self.lex_ident();
        }

        // Operators & punctuation
        self.pos += 1;
        match b {
            b'(' => Ok(TokenKind::LParen),
            b')' => Ok(TokenKind::RParen),
            b'[' => Ok(TokenKind::LBracket),
            b']' => Ok(TokenKind::RBracket),
            b',' => Ok(TokenKind::Comma),
            b';' => Ok(TokenKind::Semicolon),
            b':' => Ok(TokenKind::Colon),
            b'~' => Ok(TokenKind::Tilde),
            b'^' => Ok(TokenKind::Caret),
            b'.' => {
                if self.peek() == Some(b'.') && self.peek_at(1) == Some(b'.') {
                    self.pos += 2;
                    Ok(TokenKind::Ellipsis)
                } else {
                    Ok(TokenKind::Dot)
                }
            }
            b'?' => {
                if self.peek() == Some(b'?') {
                    self.pos += 1;
                    Ok(TokenKind::Nullish)
                } else if self.peek() == Some(b'.')
                    && !self.peek_at(1).is_some_and(|c| c.is_ascii_digit())
                {
                    self.pos += 1;
                    Ok(TokenKind::OptionalChain)
                } else {
                    Ok(TokenKind::Question)
                }
            }
            b'+' => match self.peek() {
                Some(b'+') => {
                    self.pos += 1;
                    Ok(TokenKind::PlusPlus)
                }
                Some(b'=') => {
                    self.pos += 1;
                    Ok(TokenKind::PlusEq)
                }
                _ => Ok(TokenKind::Plus),
            },
            b'-' => match self.peek() {
                Some(b'-') => {
                    self.pos += 1;
                    Ok(TokenKind::MinusMinus)
                }
                Some(b'=') => {
                    self.pos += 1;
                    Ok(TokenKind::MinusEq)
                }
                _ => Ok(TokenKind::Minus),
            },
            b'*' => match self.peek() {
                Some(b'*') => {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        Ok(TokenKind::StarStarEq)
                    } else {
                        Ok(TokenKind::StarStar)
                    }
                }
                Some(b'=') => {
                    self.pos += 1;
                    Ok(TokenKind::StarEq)
                }
                _ => Ok(TokenKind::Star),
            },
            b'/' => {
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Ok(TokenKind::SlashEq)
                } else {
                    Ok(TokenKind::Slash)
                }
            }
            b'%' => {
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    Ok(TokenKind::PercentEq)
                } else {
                    Ok(TokenKind::Percent)
                }
            }
            b'=' => match self.peek() {
                Some(b'>') => {
                    self.pos += 1;
                    Ok(TokenKind::FatArrow)
                }
                Some(b'=') => {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        Ok(TokenKind::EqEqEq)
                    } else {
                        Ok(TokenKind::EqEq)
                    }
                }
                _ => Ok(TokenKind::Eq),
            },
            b'!' => {
                if self.peek() == Some(b'=') {
                    self.pos += 1;
                    if self.peek() == Some(b'=') {
                        self.pos += 1;
                        Ok(TokenKind::BangEqEq)
                    } else {
                        Ok(TokenKind::BangEq)
                    }
                } else {
                    Ok(TokenKind::Bang)
                }
            }
            b'<' => match self.peek() {
                Some(b'=') => {
                    self.pos += 1;
                    Ok(TokenKind::LtEq)
                }
                Some(b'<') => {
                    self.pos += 1;
                    Ok(TokenKind::LtLt)
                }
                _ => Ok(TokenKind::Lt),
            },
            b'>' => match self.peek() {
                Some(b'=') => {
                    self.pos += 1;
                    Ok(TokenKind::GtEq)
                }
                Some(b'>') => {
                    self.pos += 1;
                    if self.peek() == Some(b'>') {
                        self.pos += 1;
                        Ok(TokenKind::GtGtGt)
                    } else {
                        Ok(TokenKind::GtGt)
                    }
                }
                _ => Ok(TokenKind::Gt),
            },
            b'&' => {
                if self.peek() == Some(b'&') {
                    self.pos += 1;
                    Ok(TokenKind::AmpAmp)
                } else {
                    Ok(TokenKind::Amp)
                }
            }
            b'|' => {
                if self.peek() == Some(b'|') {
                    self.pos += 1;
                    Ok(TokenKind::PipePipe)
                } else {
                    Ok(TokenKind::Pipe)
                }
            }
            other => Err(ExprError::SyntaxError {
                offset: start,
                message: format!("unexpected character '{}'", other as char),
            }),
        }
    }

    fn lex_string(&mut self, quote: u8) -> Result<TokenKind, ExprError> {
        let start = self.pos;
        self.pos += 1; // skip opening quote
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == quote {
                self.pos += 1;
                return Ok(TokenKind::String(out));
            }
            if b == b'\n' || b == b'\r' {
                return Err(ExprError::SyntaxError {
                    offset: start,
                    message: "unterminated string literal".to_string(),
                });
            }
            if b == b'\\' {
                self.pos += 1;
                let ch = self.lex_escape(start)?;
                out.push(ch);
            } else {
                let ch = self.next_utf8_char(start)?;
                out.push(ch);
            }
        }
        Err(ExprError::SyntaxError {
            offset: start,
            message: "unterminated string literal".to_string(),
        })
    }

    fn lex_template_chunk(&mut self, is_start: bool) -> Result<TokenKind, ExprError> {
        let start = self.pos;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            if b == b'`' {
                self.pos += 1;
                return if is_start {
                    Ok(TokenKind::TemplateNoSub(out))
                } else {
                    Ok(TokenKind::TemplateTail(out))
                };
            }
            if b == b'$' && self.peek_at(1) == Some(b'{') {
                self.pos += 2;
                self.template_brace_stack.push(0);
                return if is_start {
                    Ok(TokenKind::TemplateHead(out))
                } else {
                    Ok(TokenKind::TemplateMiddle(out))
                };
            }
            if b == b'\\' {
                self.pos += 1;
                let ch = self.lex_escape(start)?;
                out.push(ch);
            } else {
                let ch = self.next_utf8_char(start)?;
                out.push(ch);
            }
        }
        Err(ExprError::SyntaxError {
            offset: start,
            message: "unterminated template literal".to_string(),
        })
    }

    fn lex_escape(&mut self, start: usize) -> Result<char, ExprError> {
        let Some(esc) = self.bump() else {
            return Err(ExprError::SyntaxError {
                offset: start,
                message: "unterminated escape sequence".to_string(),
            });
        };
        match esc {
            b'n' => Ok('\n'),
            b'r' => Ok('\r'),
            b't' => Ok('\t'),
            b'b' => Ok('\u{0008}'),
            b'f' => Ok('\u{000C}'),
            b'0' => Ok('\0'),
            b'\\' => Ok('\\'),
            b'\'' => Ok('\''),
            b'"' => Ok('"'),
            b'`' => Ok('`'),
            b'$' => Ok('$'),
            b'x' => {
                let code = self.read_hex_digits(2, start)?;
                char::from_u32(code).ok_or_else(|| ExprError::SyntaxError {
                    offset: start,
                    message: "invalid hex escape".to_string(),
                })
            }
            b'u' => {
                if self.peek() == Some(b'{') {
                    self.pos += 1;
                    let hex_start = self.pos;
                    while self.peek().is_some_and(|c| c != b'}') {
                        self.pos += 1;
                    }
                    let hex = &self.src[hex_start..self.pos];
                    if self.bump() != Some(b'}') || hex.is_empty() {
                        return Err(ExprError::SyntaxError {
                            offset: start,
                            message: "invalid unicode escape".to_string(),
                        });
                    }
                    let code =
                        u32::from_str_radix(hex, 16).map_err(|_| ExprError::SyntaxError {
                            offset: start,
                            message: "invalid unicode escape".to_string(),
                        })?;
                    char::from_u32(code).ok_or_else(|| ExprError::SyntaxError {
                        offset: start,
                        message: "invalid unicode scalar value".to_string(),
                    })
                } else {
                    let code = self.read_hex_digits(4, start)?;
                    char::from_u32(code).ok_or_else(|| ExprError::SyntaxError {
                        offset: start,
                        message: "invalid unicode escape".to_string(),
                    })
                }
            }
            other => Ok(other as char),
        }
    }

    fn read_hex_digits(&mut self, count: usize, start: usize) -> Result<u32, ExprError> {
        if self.pos + count > self.bytes.len() {
            return Err(ExprError::SyntaxError {
                offset: start,
                message: "incomplete hex escape".to_string(),
            });
        }
        let slice = &self.src[self.pos..self.pos + count];
        self.pos += count;
        u32::from_str_radix(slice, 16).map_err(|_| ExprError::SyntaxError {
            offset: start,
            message: "invalid hex escape".to_string(),
        })
    }

    fn next_utf8_char(&mut self, start: usize) -> Result<char, ExprError> {
        let rest = &self.src[self.pos..];
        let Some(ch) = rest.chars().next() else {
            return Err(ExprError::SyntaxError {
                offset: start,
                message: "unexpected end of input".to_string(),
            });
        };
        self.pos += ch.len_utf8();
        Ok(ch)
    }

    fn lex_number(&mut self) -> Result<TokenKind, ExprError> {
        let start = self.pos;

        // Hex / binary / octal integer literals
        if self.bytes[self.pos] == b'0'
            && let Some(prefix) = self.peek_at(1)
        {
            let radix = match prefix {
                b'x' | b'X' => Some(16),
                b'b' | b'B' => Some(2),
                b'o' | b'O' => Some(8),
                _ => None,
            };
            if let Some(radix) = radix {
                self.pos += 2;
                let mut digits = String::new();
                while let Some(c) = self.peek() {
                    if c == b'_' {
                        self.pos += 1;
                        continue;
                    }
                    if (c as char).is_digit(radix) {
                        digits.push(c as char);
                        self.pos += 1;
                    } else {
                        break;
                    }
                }
                if digits.is_empty() {
                    return Err(ExprError::SyntaxError {
                        offset: start,
                        message: "expected digits after base prefix".to_string(),
                    });
                }
                let val =
                    u64::from_str_radix(&digits, radix).map_err(|_| ExprError::SyntaxError {
                        offset: start,
                        message: "integer literal overflow".to_string(),
                    })?;
                return Ok(TokenKind::Number(val as f64));
            }
        }

        let mut text = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                text.push(c as char);
                self.pos += 1;
            } else if c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }

        if self.peek() == Some(b'.') && self.peek_at(1) != Some(b'.') {
            text.push('.');
            self.pos += 1;
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c as char);
                    self.pos += 1;
                } else if c == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
        }

        if matches!(self.peek(), Some(b'e' | b'E')) {
            text.push('e');
            self.pos += 1;
            if let Some(sign @ (b'+' | b'-')) = self.peek() {
                text.push(sign as char);
                self.pos += 1;
            }
            let exp_start = text.len();
            while let Some(c) = self.peek() {
                if c.is_ascii_digit() {
                    text.push(c as char);
                    self.pos += 1;
                } else if c == b'_' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if text.len() == exp_start {
                return Err(ExprError::SyntaxError {
                    offset: start,
                    message: "expected exponent digits".to_string(),
                });
            }
        }

        let val = text.parse::<f64>().map_err(|_| ExprError::SyntaxError {
            offset: start,
            message: format!("invalid number literal '{text}'"),
        })?;
        Ok(TokenKind::Number(val))
    }

    fn lex_ident(&mut self) -> Result<TokenKind, ExprError> {
        let start = self.pos;
        let first = self.next_utf8_char(start)?;
        if !is_ident_start_char(first) {
            return Err(ExprError::SyntaxError {
                offset: start,
                message: format!("unexpected character '{first}'"),
            });
        }
        while self.pos < self.bytes.len() {
            let rest = &self.src[self.pos..];
            let Some(ch) = rest.chars().next() else {
                break;
            };
            if is_ident_continue_char(ch) {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        let ident = &self.src[start..self.pos];
        let kind = match ident {
            "var" => TokenKind::Var,
            "let" => TokenKind::Let,
            "const" => TokenKind::Const,
            "if" => TokenKind::If,
            "else" => TokenKind::Else,
            "for" => TokenKind::For,
            "in" => TokenKind::In,
            "while" => TokenKind::While,
            "do" => TokenKind::Do,
            "switch" => TokenKind::Switch,
            "case" => TokenKind::Case,
            "default" => TokenKind::Default,
            "break" => TokenKind::Break,
            "continue" => TokenKind::Continue,
            "return" => TokenKind::Return,
            "throw" => TokenKind::Throw,
            "try" => TokenKind::Try,
            "catch" => TokenKind::Catch,
            "finally" => TokenKind::Finally,
            "function" => TokenKind::Function,
            "typeof" => TokenKind::TypeOf,
            "void" => TokenKind::Void,
            "delete" => TokenKind::Delete,
            "true" => TokenKind::True,
            "false" => TokenKind::False,
            "null" => TokenKind::Null,
            "undefined" => TokenKind::Undefined,
            "new" => TokenKind::New,
            "class" => TokenKind::Class,
            "import" => TokenKind::Import,
            "export" => TokenKind::Export,
            "async" => TokenKind::Async,
            "await" => TokenKind::Await,
            "yield" => TokenKind::Yield,
            "with" => TokenKind::With,
            "debugger" => TokenKind::Debugger,
            "this" => TokenKind::This,
            "super" => TokenKind::Super,
            _ => TokenKind::Ident(ident.to_string()),
        };
        Ok(kind)
    }
}

fn is_ident_start_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b == b'$'
}

fn is_ident_start_char(ch: char) -> bool {
    ch.is_alphabetic() || ch == '_' || ch == '$'
}

fn is_ident_continue_char(ch: char) -> bool {
    ch.is_alphanumeric() || ch == '_' || ch == '$'
}
