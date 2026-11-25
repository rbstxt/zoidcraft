use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum Token {
    Ident(String),
    Number(String),
    Punct(String),
    Preprocessor(String),
    Newline,
}

pub(super) fn tokenize(src: &str) -> Vec<Token> {
    let bytes = src.as_bytes();
    let len = bytes.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    let mut at_line_start = true;

    while i < len {
        let b = bytes[i];

        if b == b'\r' {
            i += 1;
            continue;
        }
        if b == b'\n' {
            tokens.push(Token::Newline);
            i += 1;
            at_line_start = true;
            continue;
        }
        if b == b' ' || b == b'\t' {
            i += 1;
            continue;
        }

        // Line comment
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            while i < len && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }

        // Block comment
        if b == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            i += 2;
            while i + 1 < len && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                if bytes[i] == b'\n' {
                    tokens.push(Token::Newline);
                    at_line_start = true;
                }
                i += 1;
            }
            if i + 1 < len {
                i += 2;
            }
            continue;
        }

        // Preprocessor line (#define, #ifdef, #if, #elif, #else, #endif, #undef)
        if b == b'#' && at_line_start {
            let mut directive = String::new();
            let mut in_block_comment = false;
            loop {
                let seg_start = i;
                let mut hit_line_comment = false;
                while i < len && bytes[i] != b'\n' {
                    if in_block_comment {
                        if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                            in_block_comment = false;
                            i += 2;
                            continue;
                        }
                        i += 1;
                        continue;
                    }
                    if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                        in_block_comment = true;
                        i += 2;
                        continue;
                    }
                    if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                        hit_line_comment = true;
                        break;
                    }
                    i += 1;
                }
                let raw_seg = &src[seg_start..i];
                if hit_line_comment {
                    while i < len && bytes[i] != b'\n' {
                        i += 1;
                    }
                    directive.push_str(raw_seg.trim_end());
                    break;
                }
                let trimmed_seg = raw_seg.trim_end_matches([' ', '\t', '\r']);
                if !in_block_comment && let Some(continued) = trimmed_seg.strip_suffix('\\') {
                    directive.push_str(continued);
                    if i < len && bytes[i] == b'\n' {
                        i += 1;
                    } else {
                        break;
                    }
                } else {
                    directive.push_str(trimmed_seg);
                    if in_block_comment && i < len && bytes[i] == b'\n' {
                        directive.push(' ');
                        i += 1;
                    } else {
                        break;
                    }
                }
            }
            tokens.push(Token::Preprocessor(directive));
            at_line_start = false;
            continue;
        }

        at_line_start = false;

        // Identifier
        if b.is_ascii_alphabetic() || b == b'_' {
            let start = i;
            i += 1;
            while i < len && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            tokens.push(Token::Ident(src[start..i].to_string()));
            continue;
        }

        // Number (decimal, hex, float starting with digit or `.digit`)
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let start = i;
            if b == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
                i += 2;
                while i < len && bytes[i].is_ascii_hexdigit() {
                    i += 1;
                }
                if i < len && (bytes[i] == b'u' || bytes[i] == b'U') {
                    i += 1;
                }
            } else {
                while i < len && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i < len && bytes[i] == b'.' {
                    i += 1;
                    while i < len && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < len && (bytes[i] == b'e' || bytes[i] == b'E') {
                    i += 1;
                    if i < len && (bytes[i] == b'+' || bytes[i] == b'-') {
                        i += 1;
                    }
                    while i < len && bytes[i].is_ascii_digit() {
                        i += 1;
                    }
                }
                if i < len
                    && (bytes[i] == b'u'
                        || bytes[i] == b'U'
                        || bytes[i] == b'f'
                        || bytes[i] == b'F')
                {
                    i += 1;
                }
            }
            tokens.push(Token::Number(src[start..i].to_string()));
            continue;
        }

        // Multi-character operators
        if i + 2 < len {
            let tri = &src[i..i + 3];
            if tri == "<<=" || tri == ">>=" {
                tokens.push(Token::Punct(tri.to_string()));
                i += 3;
                continue;
            }
        }
        if i + 1 < len {
            let pair = &src[i..i + 2];
            if matches!(
                pair,
                "<=" | ">="
                    | "=="
                    | "!="
                    | "&&"
                    | "||"
                    | "^^"
                    | "+="
                    | "-="
                    | "*="
                    | "/="
                    | "%="
                    | "++"
                    | "--"
                    | "<<"
                    | ">>"
                    | "&="
                    | "|="
                    | "^="
            ) {
                tokens.push(Token::Punct(pair.to_string()));
                i += 2;
                continue;
            }
        }

        // Single-character punctuation
        let ch = src[i..].chars().next().unwrap_or(' ');
        tokens.push(Token::Punct(ch.to_string()));
        i += ch.len_utf8();
    }

    tokens
}

// ---------------------------------------------------------------------------
// Stage 3: Reject Unsupported Legacy WebGL 1 Built-ins

pub(super) fn reject_unsupported_features(tokens: &[Token]) -> Result<(), ShaderError> {
    const FORBIDDEN_IDENTS: &[&str] = &["gl_FragDepthEXT", "gl_FragData"];

    for tok in tokens {
        if let Token::Ident(name) = tok
            && FORBIDDEN_IDENTS.contains(&name.as_str())
        {
            return Err(ShaderError::UnsupportedFeature(name.clone()));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4: Extract Top-Level Uniforms, Varyings & Strip Precision Statements
