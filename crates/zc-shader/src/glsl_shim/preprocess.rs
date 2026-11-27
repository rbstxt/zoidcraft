use super::*;

#[derive(Debug, Clone)]
struct MacroDef {
    params: Option<Vec<String>>,
    body: String,
}

#[derive(Debug, Clone, Copy)]
struct CondFrame {
    parent_active: bool,
    this_active: bool,
    any_taken: bool,
    seen_else: bool,
}

struct PpState {
    macros: HashMap<String, MacroDef>,
    int_defines: HashMap<String, u32>,
    included: HashSet<String>,
    cond_stack: Vec<CondFrame>,
    glsl_version: Option<u32>,
    extensions: BTreeMap<String, ExtensionBehavior>,
    budget: usize,
}

impl PpState {
    pub(super) fn new(options: &CompileOptions) -> Self {
        let mut macros = HashMap::new();
        let mut int_defines = HashMap::new();
        for (k, v) in [
            ("GL_ES", "1"),
            ("GL_FRAGMENT_PRECISION_HIGH", "1"),
            ("GL_OES_standard_derivatives", "1"),
            ("GL_EXT_shader_texture_lod", "1"),
            ("__VERSION__", "100"),
        ] {
            macros.insert(
                k.to_string(),
                MacroDef {
                    params: None,
                    body: v.to_string(),
                },
            );
            if let Some(n) = parse_int_literal_i64(v).and_then(|x| u32::try_from(x).ok()) {
                int_defines.insert(k.to_string(), n);
            }
        }
        for (k, v) in &options.defines {
            let norm_key = fix_property_name(k);
            let trimmed = v.trim();
            macros.insert(
                norm_key.clone(),
                MacroDef {
                    params: None,
                    body: trimmed.to_string(),
                },
            );
            if let Some(n) = parse_int_literal_i64(trimmed).and_then(|x| u32::try_from(x).ok()) {
                int_defines.insert(norm_key, n);
            }
        }
        Self {
            macros,
            int_defines,
            included: HashSet::new(),
            cond_stack: Vec::new(),
            glsl_version: None,
            extensions: BTreeMap::new(),
            budget: 0,
        }
    }

    fn is_active(&self) -> bool {
        self.cond_stack.last().is_none_or(|f| f.this_active)
    }

    pub(super) fn charge(&mut self, bytes: usize) -> Result<(), ShaderError> {
        self.budget = self.budget.saturating_add(bytes);
        if self.budget > MAX_GLSL_SOURCE_BYTES * 2 {
            return Err(ShaderError::GlslParse(format!(
                "preprocessed shader exceeded {} byte expansion budget",
                MAX_GLSL_SOURCE_BYTES
            )));
        }
        Ok(())
    }
}

fn lookup_include<'a>(name: &str, options: &'a CompileOptions) -> Option<&'a str> {
    if let Some(custom) = options.extra_includes.get(name) {
        return Some(custom.as_str());
    }
    match name {
        "zoidium/temporal" => Some(INCLUDE_TEMPORAL),
        "zoidium/gradient" => Some(INCLUDE_GRADIENT),
        "zoidium/curve" => Some(INCLUDE_CURVE),
        "zoidium/text" => Some(INCLUDE_TEXT),
        "zoidium/hash" => Some(INCLUDE_HASH),
        "zoidium/noise" => Some(INCLUDE_NOISE),
        "zoidium/color" => Some(INCLUDE_COLOR),
        "zoidium/sdf" => Some(INCLUDE_SDF),
        _ => None,
    }
}

/// Strips `/* ... */` and `// ...` comments cleanly across physical lines (scanning the
/// remainder of a line after `*/` for additional `/*` openings or directives, preserving
/// `// #include ...` lines), then joins trailing-backslash (`\`) line continuations.
fn strip_comments_and_join_continuations(source: &str) -> Vec<String> {
    let mut stripped_lines: Vec<String> = Vec::new();
    let mut in_block_comment = false;

    for raw_line in source.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        let bytes = line.as_bytes();
        let len = bytes.len();
        let mut out = String::with_capacity(len);
        let mut i = 0;

        while i < len {
            if in_block_comment {
                if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                    in_block_comment = false;
                    out.push(' ');
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }

            if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
                in_block_comment = true;
                out.push(' ');
                i += 2;
                continue;
            }

            if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
                // Preserve Panzoid/Zoidium `// #include <...>` directives when nothing precedes `//`.
                if out.trim().is_empty() {
                    let after_slashes = line[i + 2..].trim_start();
                    if let Some(after_hash) = after_slashes.strip_prefix('#') {
                        let after_hash_trimmed = after_hash.trim_start();
                        if let Some(after_inc) = after_hash_trimmed.strip_prefix("include")
                            && after_inc
                                .chars()
                                .next()
                                .is_some_and(|c| c.is_whitespace() || c == '<' || c == '"')
                        {
                            // Strip any trailing comments inside the `// #include ...` line itself.
                            let cleaned = strip_inline_comments(after_slashes);
                            out.push_str(&cleaned);
                        }
                    }
                }
                break;
            }

            let ch = line[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        }

        stripped_lines.push(out);
    }

    // Join lines ending with `\` continuation.
    let mut logical: Vec<String> = Vec::with_capacity(stripped_lines.len());
    let mut acc: Option<String> = None;
    for line in stripped_lines {
        let trimmed_end = line.trim_end_matches([' ', '\t']);
        if let Some(continued) = trimmed_end.strip_suffix('\\') {
            match &mut acc {
                Some(buf) => {
                    buf.push_str(continued);
                    buf.push(' ');
                }
                None => {
                    let mut buf = String::from(continued);
                    buf.push(' ');
                    acc = Some(buf);
                }
            }
        } else if let Some(mut buf) = acc.take() {
            buf.push_str(&line);
            logical.push(buf);
        } else {
            logical.push(line);
        }
    }
    if let Some(buf) = acc {
        logical.push(buf);
    }

    logical
}

fn strip_inline_comments(s: &str) -> String {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;
    let mut in_block = false;
    while i < len {
        if in_block {
            if bytes[i] == b'*' && i + 1 < len && bytes[i + 1] == b'/' {
                in_block = false;
                out.push(' ');
                i += 2;
            } else {
                i += 1;
            }
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'*' {
            in_block = true;
            out.push(' ');
            i += 2;
            continue;
        }
        if bytes[i] == b'/' && i + 1 < len && bytes[i + 1] == b'/' {
            break;
        }
        let ch = s[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }
    out
}

fn parse_include_arg(rest: &str) -> Option<&str> {
    let arg = rest.trim();
    if let Some(inner) = arg.strip_prefix('<').and_then(|s| s.strip_suffix('>')) {
        Some(inner.trim())
    } else if let Some(inner) = arg.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        Some(inner.trim())
    } else {
        None
    }
}

pub(super) fn is_ident_start_byte(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_'
}

pub(super) fn is_ident_continue_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn scan_number_literal_len(bytes: &[u8], start: usize) -> usize {
    let len = bytes.len();
    let mut i = start;
    if bytes[i] == b'0' && i + 1 < len && (bytes[i + 1] == b'x' || bytes[i + 1] == b'X') {
        i += 2;
        while i < len && bytes[i].is_ascii_hexdigit() {
            i += 1;
        }
        if i < len && (bytes[i] == b'u' || bytes[i] == b'U') {
            i += 1;
        }
        return i - start;
    }
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
    if i < len && matches!(bytes[i], b'u' | b'U' | b'f' | b'F') {
        i += 1;
    }
    i - start
}

fn parse_macro_call_args(text: &str, open_paren_idx: usize) -> Option<(Vec<String>, usize)> {
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut args = Vec::new();
    let mut arg_start = open_paren_idx + 1;
    let mut depth = 1_usize;
    let mut i = open_paren_idx + 1;

    while i < len {
        match bytes[i] {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    let tail = &text[arg_start..i];
                    if !args.is_empty() || !tail.trim().is_empty() {
                        args.push(tail.trim().to_string());
                    }
                    return Some((args, i + 1));
                }
            }
            b',' if depth == 1 => {
                args.push(text[arg_start..i].trim().to_string());
                arg_start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    None
}

fn substitute_macro_params(body: &str, param_map: &HashMap<&str, &str>) -> String {
    if param_map.is_empty() {
        return body.to_string();
    }
    let bytes = body.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let nlen = scan_number_literal_len(bytes, i);
            out.push_str(&body[i..i + nlen]);
            i += nlen;
            continue;
        }
        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &body[start..i];
            if let Some(replacement) = param_map.get(ident) {
                if out
                    .as_bytes()
                    .last()
                    .is_some_and(|&pb| is_ident_continue_byte(pb))
                    && replacement
                        .as_bytes()
                        .first()
                        .is_some_and(|&nb| is_ident_continue_byte(nb))
                {
                    out.push(' ');
                }
                out.push_str(replacement);
            } else {
                out.push_str(ident);
            }
            continue;
        }
        let ch = body[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn expand_macros_in_text(
    text: &str,
    macros: &HashMap<String, MacroDef>,
    hidden: &HashSet<String>,
    depth: usize,
    budget: &mut usize,
) -> Result<String, ShaderError> {
    if depth > MAX_MACRO_EXPANSION_DEPTH {
        return Err(ShaderError::GlslParse(format!(
            "macro expansion nesting depth exceeds {MAX_MACRO_EXPANSION_DEPTH} limit"
        )));
    }
    let bytes = text.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_digit() || (b == b'.' && i + 1 < len && bytes[i + 1].is_ascii_digit()) {
            let nlen = scan_number_literal_len(bytes, i);
            out.push_str(&text[i..i + nlen]);
            i += nlen;
            continue;
        }

        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &text[start..i];

            if !hidden.contains(ident)
                && let Some(def) = macros.get(ident)
            {
                match &def.params {
                    None => {
                        *budget = budget.saturating_add(def.body.len().max(1));
                        if *budget > MAX_GLSL_SOURCE_BYTES * 2 {
                            return Err(ShaderError::GlslParse(format!(
                                "macro expansion exceeded {} byte budget",
                                MAX_GLSL_SOURCE_BYTES
                            )));
                        }
                        let mut next_hidden = hidden.clone();
                        next_hidden.insert(ident.to_string());
                        let expanded = expand_macros_in_text(
                            &def.body,
                            macros,
                            &next_hidden,
                            depth + 1,
                            budget,
                        )?;
                        if out
                            .as_bytes()
                            .last()
                            .is_some_and(|&pb| is_ident_continue_byte(pb))
                            && expanded
                                .as_bytes()
                                .first()
                                .is_some_and(|&nb| is_ident_continue_byte(nb))
                        {
                            out.push(' ');
                        }
                        out.push_str(&expanded);
                        continue;
                    }
                    Some(params) => {
                        let mut look = i;
                        while look < len && matches!(bytes[look], b' ' | b'\t' | b'\n' | b'\r') {
                            look += 1;
                        }
                        if look < len
                            && bytes[look] == b'('
                            && let Some((raw_args, after_call)) = parse_macro_call_args(text, look)
                            && raw_args.len() == params.len()
                        {
                            let mut expanded_args = Vec::with_capacity(raw_args.len());
                            for raw_arg in &raw_args {
                                expanded_args.push(expand_macros_in_text(
                                    raw_arg,
                                    macros,
                                    hidden,
                                    depth + 1,
                                    budget,
                                )?);
                            }
                            let param_map: HashMap<&str, &str> = params
                                .iter()
                                .zip(expanded_args.iter())
                                .map(|(p, a)| (p.as_str(), a.as_str()))
                                .collect();
                            let substituted = substitute_macro_params(&def.body, &param_map);
                            *budget = budget.saturating_add(substituted.len().max(1));
                            if *budget > MAX_GLSL_SOURCE_BYTES * 2 {
                                return Err(ShaderError::GlslParse(format!(
                                    "macro expansion exceeded {} byte budget",
                                    MAX_GLSL_SOURCE_BYTES
                                )));
                            }
                            let mut next_hidden = hidden.clone();
                            next_hidden.insert(ident.to_string());
                            let expanded = expand_macros_in_text(
                                &substituted,
                                macros,
                                &next_hidden,
                                depth + 1,
                                budget,
                            )?;
                            if out
                                .as_bytes()
                                .last()
                                .is_some_and(|&pb| is_ident_continue_byte(pb))
                                && expanded
                                    .as_bytes()
                                    .first()
                                    .is_some_and(|&nb| is_ident_continue_byte(nb))
                            {
                                out.push(' ');
                            }
                            out.push_str(&expanded);
                            i = after_call;
                            continue;
                        }
                    }
                }
            }

            out.push_str(ident);
            continue;
        }

        let ch = text[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    Ok(out)
}

fn replace_defined_operators(expr: &str, macros: &HashMap<String, MacroDef>) -> String {
    let bytes = expr.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len);
    let mut i = 0;

    while i < len {
        let b = bytes[i];
        if b.is_ascii_digit() {
            let nlen = scan_number_literal_len(bytes, i);
            out.push_str(&expr[i..i + nlen]);
            i += nlen;
            continue;
        }
        if is_ident_start_byte(b) {
            let start = i;
            i += 1;
            while i < len && is_ident_continue_byte(bytes[i]) {
                i += 1;
            }
            let ident = &expr[start..i];
            if ident == "defined" {
                let mut j = i;
                while j < len && matches!(bytes[j], b' ' | b'\t') {
                    j += 1;
                }
                let has_paren = j < len && bytes[j] == b'(';
                if has_paren {
                    j += 1;
                    while j < len && matches!(bytes[j], b' ' | b'\t') {
                        j += 1;
                    }
                }
                if j < len && is_ident_start_byte(bytes[j]) {
                    let name_start = j;
                    j += 1;
                    while j < len && is_ident_continue_byte(bytes[j]) {
                        j += 1;
                    }
                    let macro_name = &expr[name_start..j];
                    if has_paren {
                        while j < len && matches!(bytes[j], b' ' | b'\t') {
                            j += 1;
                        }
                        if j < len && bytes[j] == b')' {
                            j += 1;
                        }
                    }
                    out.push_str(if macros.contains_key(macro_name) {
                        "1"
                    } else {
                        "0"
                    });
                    i = j;
                    continue;
                }
            }
            out.push_str(ident);
            continue;
        }
        let ch = expr[i..].chars().next().unwrap_or(' ');
        out.push(ch);
        i += ch.len_utf8();
    }

    out
}

fn parse_int_literal_i64(raw: &str) -> Option<i64> {
    let s = raw.trim().trim_end_matches(['u', 'U']);
    if s.is_empty() {
        return None;
    }
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        i64::from_str_radix(hex, 16).ok()
    } else if s.len() > 1 && s.starts_with('0') && s.bytes().all(|b| (b'0'..=b'7').contains(&b)) {
        i64::from_str_radix(&s[1..], 8).ok()
    } else {
        s.parse::<i64>().ok()
    }
}

struct ConstIntParser<'a> {
    tokens: &'a [Token],
    pos: usize,
    int_defines: Option<&'a HashMap<String, u32>>,
    unknown_ident_as_zero: bool,
    eval_enabled: bool,
}

impl<'a> ConstIntParser<'a> {
    fn peek(&self) -> Option<&'a Token> {
        let mut p = self.pos;
        while matches!(self.tokens.get(p), Some(Token::Newline)) {
            p += 1;
        }
        self.tokens.get(p)
    }

    fn advance(&mut self) -> Option<&'a Token> {
        while matches!(self.tokens.get(self.pos), Some(Token::Newline)) {
            self.pos += 1;
        }
        let tok = self.tokens.get(self.pos);
        if tok.is_some() {
            self.pos += 1;
        }
        tok
    }

    fn consume_punct(&mut self, expected: &str) -> bool {
        if matches!(self.peek(), Some(Token::Punct(p)) if p == expected) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> Result<i64, ShaderError> {
        self.parse_ternary()
    }

    fn parse_ternary(&mut self) -> Result<i64, ShaderError> {
        let cond = self.parse_logical_or()?;
        if self.consume_punct("?") {
            let prev = self.eval_enabled;
            self.eval_enabled = prev && cond != 0;
            let then_val = self.parse_expr()?;
            self.eval_enabled = prev;
            if !self.consume_punct(":") {
                return Err(ShaderError::GlslParse(
                    "expected ':' in conditional constant expression".to_string(),
                ));
            }
            self.eval_enabled = prev && cond == 0;
            let else_val = self.parse_ternary()?;
            self.eval_enabled = prev;
            Ok(if cond != 0 { then_val } else { else_val })
        } else {
            Ok(cond)
        }
    }

    fn parse_logical_or(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_logical_and()?;
        while self.consume_punct("||") {
            let prev = self.eval_enabled;
            self.eval_enabled = prev && lhs == 0;
            let rhs = self.parse_logical_and()?;
            self.eval_enabled = prev;
            lhs = i64::from(lhs != 0 || rhs != 0);
        }
        Ok(lhs)
    }

    fn parse_logical_and(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_bit_or()?;
        while self.consume_punct("&&") {
            let prev = self.eval_enabled;
            self.eval_enabled = prev && lhs != 0;
            let rhs = self.parse_bit_or()?;
            self.eval_enabled = prev;
            lhs = i64::from(lhs != 0 && rhs != 0);
        }
        Ok(lhs)
    }

    fn parse_bit_or(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_bit_xor()?;
        while self.consume_punct("|") {
            let rhs = self.parse_bit_xor()?;
            lhs |= rhs;
        }
        Ok(lhs)
    }

    fn parse_bit_xor(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_bit_and()?;
        while self.consume_punct("^") {
            let rhs = self.parse_bit_and()?;
            lhs ^= rhs;
        }
        Ok(lhs)
    }

    fn parse_bit_and(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_equality()?;
        while self.consume_punct("&") {
            let rhs = self.parse_equality()?;
            lhs &= rhs;
        }
        Ok(lhs)
    }

    fn parse_equality(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_relational()?;
        loop {
            if self.consume_punct("==") {
                let rhs = self.parse_relational()?;
                lhs = i64::from(lhs == rhs);
            } else if self.consume_punct("!=") {
                let rhs = self.parse_relational()?;
                lhs = i64::from(lhs != rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_relational(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_shift()?;
        loop {
            if self.consume_punct("<=") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs <= rhs);
            } else if self.consume_punct(">=") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs >= rhs);
            } else if self.consume_punct("<") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs < rhs);
            } else if self.consume_punct(">") {
                let rhs = self.parse_shift()?;
                lhs = i64::from(lhs > rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_shift(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_additive()?;
        loop {
            if self.consume_punct("<<") {
                let rhs = self.parse_additive()?;
                lhs = lhs.wrapping_shl((rhs.clamp(0, 63)) as u32);
            } else if self.consume_punct(">>") {
                let rhs = self.parse_additive()?;
                lhs = lhs.wrapping_shr((rhs.clamp(0, 63)) as u32);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_multiplicative()?;
        loop {
            if self.consume_punct("+") {
                let rhs = self.parse_multiplicative()?;
                lhs = lhs.wrapping_add(rhs);
            } else if self.consume_punct("-") {
                let rhs = self.parse_multiplicative()?;
                lhs = lhs.wrapping_sub(rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_multiplicative(&mut self) -> Result<i64, ShaderError> {
        let mut lhs = self.parse_unary()?;
        loop {
            if self.consume_punct("*") {
                let rhs = self.parse_unary()?;
                lhs = lhs.wrapping_mul(rhs);
            } else if self.consume_punct("/") {
                let rhs = self.parse_unary()?;
                if rhs == 0 {
                    if !self.eval_enabled {
                        lhs = 0;
                        continue;
                    }
                    return Err(ShaderError::GlslParse(
                        "division by zero in constant integer expression".to_string(),
                    ));
                }
                lhs = lhs.wrapping_div(rhs);
            } else if self.consume_punct("%") {
                let rhs = self.parse_unary()?;
                if rhs == 0 {
                    if !self.eval_enabled {
                        lhs = 0;
                        continue;
                    }
                    return Err(ShaderError::GlslParse(
                        "modulo by zero in constant integer expression".to_string(),
                    ));
                }
                lhs = lhs.wrapping_rem(rhs);
            } else {
                break;
            }
        }
        Ok(lhs)
    }

    fn parse_unary(&mut self) -> Result<i64, ShaderError> {
        if self.consume_punct("+") {
            return self.parse_unary();
        }
        if self.consume_punct("-") {
            return Ok(self.parse_unary()?.wrapping_neg());
        }
        if self.consume_punct("!") {
            return Ok(i64::from(self.parse_unary()? == 0));
        }
        if self.consume_punct("~") {
            return Ok(!self.parse_unary()?);
        }
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<i64, ShaderError> {
        if self.consume_punct("(") {
            let val = self.parse_expr()?;
            if !self.consume_punct(")") {
                return Err(ShaderError::GlslParse(
                    "expected ')' in constant integer expression".to_string(),
                ));
            }
            return Ok(val);
        }

        match self.advance() {
            Some(Token::Number(n)) => parse_int_literal_i64(n).ok_or_else(|| {
                ShaderError::GlslParse(format!("invalid integer literal `{n}` in expression"))
            }),
            Some(Token::Ident(name)) => {
                if let Some(map) = self.int_defines
                    && let Some(&v) = map.get(name)
                {
                    return Ok(i64::from(v));
                }
                if self.unknown_ident_as_zero {
                    Ok(0)
                } else {
                    Err(ShaderError::GlslParse(format!(
                        "unknown identifier `{name}` in constant integer expression"
                    )))
                }
            }
            _ => Err(ShaderError::GlslParse(
                "invalid token in constant integer expression".to_string(),
            )),
        }
    }
}

fn eval_pp_expr(
    expr: &str,
    macros: &HashMap<String, MacroDef>,
    budget: &mut usize,
) -> Result<i64, ShaderError> {
    let with_defined = replace_defined_operators(expr, macros);
    let expanded = expand_macros_in_text(&with_defined, macros, &HashSet::new(), 0, budget)?;
    let tokens = tokenize(&expanded);
    let mut parser = ConstIntParser {
        tokens: &tokens,
        pos: 0,
        int_defines: None,
        unknown_ident_as_zero: true,
        eval_enabled: true,
    };
    let val = parser.parse_expr()?;
    if parser.peek().is_some() {
        return Err(ShaderError::GlslParse(format!(
            "unexpected trailing tokens in preprocessor expression `{}`",
            expr.trim()
        )));
    }
    Ok(val)
}

pub(super) fn eval_const_int_tokens(
    tokens: &[Token],
    int_defines: &HashMap<String, u32>,
) -> Option<i64> {
    let mut parser = ConstIntParser {
        tokens,
        pos: 0,
        int_defines: Some(int_defines),
        unknown_ident_as_zero: false,
        eval_enabled: true,
    };
    let val = parser.parse_expr().ok()?;
    if parser.peek().is_some() {
        return None;
    }
    Some(val)
}

pub(super) struct PreprocessedSource {
    pub(super) source: String,
    pub(super) included: HashSet<String>,
    pub(super) int_defines: HashMap<String, u32>,
    pub(super) glsl_version: Option<u32>,
    pub(super) extensions: BTreeMap<String, ExtensionBehavior>,
}

fn parse_define_directive(rest: &str) -> Option<(String, MacroDef)> {
    let rest = rest.trim_start();
    let bytes = rest.as_bytes();
    if bytes.is_empty() || !is_ident_start_byte(bytes[0]) {
        return None;
    }
    let mut name_end = 1;
    while name_end < bytes.len() && is_ident_continue_byte(bytes[name_end]) {
        name_end += 1;
    }
    let name = rest[..name_end].to_string();
    let after_name = &rest[name_end..];

    // Function-like macro ONLY when `(` immediately follows macro name with no whitespace.
    if let Some(after_open) = after_name.strip_prefix('(') {
        let close_idx = after_open.find(')')?;
        let params_str = &after_open[..close_idx];
        let mut params = Vec::new();
        for p in params_str.split(',') {
            let pt = p.trim();
            if !pt.is_empty() {
                params.push(pt.to_string());
            }
        }
        let body = after_open[close_idx + 1..].trim().to_string();
        Some((
            name,
            MacroDef {
                params: Some(params),
                body,
            },
        ))
    } else {
        let body = after_name.trim().to_string();
        Some((name, MacroDef { params: None, body }))
    }
}

pub(super) fn preprocess_source(
    source: &str,
    options: &CompileOptions,
) -> Result<PreprocessedSource, ShaderError> {
    let mut state = PpState::new(options);
    let mut out = String::with_capacity(source.len());
    preprocess_lines_into(source, options, &mut state, 0, &mut out)?;
    if !state.cond_stack.is_empty() {
        return Err(ShaderError::GlslParse(
            "unterminated #if/#ifdef/#ifndef directive".to_string(),
        ));
    }
    Ok(PreprocessedSource {
        source: out,
        included: state.included,
        int_defines: state.int_defines,
        glsl_version: state.glsl_version,
        extensions: state.extensions,
    })
}

fn flush_active_chunk(
    chunk: &mut String,
    state: &mut PpState,
    out: &mut String,
) -> Result<(), ShaderError> {
    if chunk.is_empty() {
        return Ok(());
    }
    let expanded =
        expand_macros_in_text(chunk, &state.macros, &HashSet::new(), 0, &mut state.budget)?;
    state.charge(expanded.len())?;
    out.push_str(&expanded);
    chunk.clear();
    Ok(())
}

fn preprocess_lines_into(
    source: &str,
    options: &CompileOptions,
    state: &mut PpState,
    depth: usize,
    out: &mut String,
) -> Result<(), ShaderError> {
    let logical_lines = strip_comments_and_join_continuations(source);
    let mut active_chunk = String::new();
    let base_cond_depth = state.cond_stack.len();

    for line in logical_lines {
        let trimmed = line.trim();
        if let Some(after_hash) = trimmed.strip_prefix('#') {
            flush_active_chunk(&mut active_chunk, state, out)?;

            let directive_line = after_hash.trim_start();
            let dir_end = directive_line
                .bytes()
                .take_while(|b| b.is_ascii_alphabetic())
                .count();
            let dir_name = &directive_line[..dir_end];
            let rest = &directive_line[dir_end..];

            // Conditional directives are evaluated even when inside an inactive branch.
            match dir_name {
                "ifdef" => {
                    let parent = state.is_active();
                    let name = rest.split_whitespace().next().unwrap_or("");
                    let cond = parent && !name.is_empty() && state.macros.contains_key(name);
                    state.cond_stack.push(CondFrame {
                        parent_active: parent,
                        this_active: cond,
                        any_taken: cond,
                        seen_else: false,
                    });
                    continue;
                }
                "ifndef" => {
                    let parent = state.is_active();
                    let name = rest.split_whitespace().next().unwrap_or("");
                    let cond = parent && !name.is_empty() && !state.macros.contains_key(name);
                    state.cond_stack.push(CondFrame {
                        parent_active: parent,
                        this_active: cond,
                        any_taken: cond,
                        seen_else: false,
                    });
                    continue;
                }
                "if" => {
                    let parent = state.is_active();
                    let cond = if parent {
                        eval_pp_expr(rest, &state.macros, &mut state.budget)? != 0
                    } else {
                        false
                    };
                    state.cond_stack.push(CondFrame {
                        parent_active: parent,
                        this_active: cond,
                        any_taken: cond,
                        seen_else: false,
                    });
                    continue;
                }
                "elif" => {
                    let Some(frame) = state.cond_stack.last_mut() else {
                        return Err(ShaderError::GlslParse("unexpected #elif".to_string()));
                    };
                    if frame.seen_else {
                        return Err(ShaderError::GlslParse("#elif after #else".to_string()));
                    }
                    if !frame.parent_active || frame.any_taken {
                        frame.this_active = false;
                    } else {
                        let cond = eval_pp_expr(rest, &state.macros, &mut state.budget)? != 0;
                        frame.this_active = cond;
                        if cond {
                            frame.any_taken = true;
                        }
                    }
                    continue;
                }
                "else" => {
                    let Some(frame) = state.cond_stack.last_mut() else {
                        return Err(ShaderError::GlslParse("unexpected #else".to_string()));
                    };
                    if frame.seen_else {
                        return Err(ShaderError::GlslParse("duplicate #else".to_string()));
                    }
                    frame.seen_else = true;
                    frame.this_active = frame.parent_active && !frame.any_taken;
                    if frame.this_active {
                        frame.any_taken = true;
                    }
                    continue;
                }
                "endif" => {
                    if state.cond_stack.len() <= base_cond_depth {
                        return Err(ShaderError::GlslParse("unexpected #endif".to_string()));
                    }
                    state.cond_stack.pop();
                    continue;
                }
                _ => {}
            }

            if !state.is_active() {
                continue;
            }

            match dir_name {
                "include" => {
                    let Some(inc_name) = parse_include_arg(rest) else {
                        return Err(ShaderError::GlslParse(format!(
                            "malformed #include directive: {trimmed}"
                        )));
                    };
                    if depth >= MAX_INCLUDE_DEPTH {
                        return Err(ShaderError::IncludeDepthExceeded(inc_name.to_string()));
                    }
                    let snippet = lookup_include(inc_name, options)
                        .ok_or_else(|| ShaderError::UnknownInclude(inc_name.to_string()))?;
                    if state.included.insert(inc_name.to_string()) {
                        preprocess_lines_into(snippet, options, state, depth + 1, out)?;
                        out.push('\n');
                    }
                }
                "version" => {
                    let mut parts = rest.split_whitespace();
                    let Some(ver_str) = parts.next() else {
                        return Err(ShaderError::GlslParse(
                            "expected version number after #version".to_string(),
                        ));
                    };
                    let Ok(ver) = ver_str.parse::<u32>() else {
                        return Err(ShaderError::GlslParse(format!(
                            "invalid #version number `{ver_str}`"
                        )));
                    };
                    if ver == 0 {
                        return Err(ShaderError::GlslParse(
                            "invalid #version number `0`".to_string(),
                        ));
                    }
                    if let Some(profile) = parts.next()
                        && !matches!(profile, "es" | "core" | "compatibility")
                    {
                        return Err(ShaderError::GlslParse(format!(
                            "invalid #version profile `{profile}`"
                        )));
                    }
                    if parts.next().is_some() {
                        return Err(ShaderError::GlslParse(format!(
                            "unexpected trailing tokens in #version directive: {trimmed}"
                        )));
                    }
                    state.glsl_version = Some(ver);
                    state.macros.insert(
                        "__VERSION__".to_string(),
                        MacroDef {
                            params: None,
                            body: ver.to_string(),
                        },
                    );
                    state.int_defines.insert("__VERSION__".to_string(), ver);
                }
                "pragma" | "line" | "" => {}
                "extension" => {
                    let Some((ext_raw, behavior_raw)) = rest.split_once(':') else {
                        return Err(ShaderError::GlslParse(format!(
                            "malformed #extension directive (missing ':'): {trimmed}"
                        )));
                    };
                    let ext_name = ext_raw.trim();
                    let behavior_str = behavior_raw.trim();
                    if ext_name.is_empty() {
                        return Err(ShaderError::GlslParse(format!(
                            "missing extension name in #extension directive: {trimmed}"
                        )));
                    }
                    let behavior = match behavior_str {
                        "enable" => ExtensionBehavior::Enable,
                        "require" => ExtensionBehavior::Require,
                        "warn" => ExtensionBehavior::Warn,
                        "disable" => ExtensionBehavior::Disable,
                        _ => {
                            return Err(ShaderError::GlslParse(format!(
                                "invalid #extension behavior `{behavior_str}` for `{ext_name}`"
                            )));
                        }
                    };
                    if ext_name == "all" {
                        match behavior {
                            ExtensionBehavior::Enable | ExtensionBehavior::Require => {
                                return Err(ShaderError::GlslParse(
                                    "#extension all only supports `warn` or `disable`".to_string(),
                                ));
                            }
                            ExtensionBehavior::Disable => {
                                for v in state.extensions.values_mut() {
                                    *v = ExtensionBehavior::Disable;
                                }
                                state
                                    .extensions
                                    .insert("all".to_string(), ExtensionBehavior::Disable);
                            }
                            ExtensionBehavior::Warn => {
                                state
                                    .extensions
                                    .insert("all".to_string(), ExtensionBehavior::Warn);
                            }
                        }
                        continue;
                    }
                    let is_supported = matches!(
                        ext_name,
                        "GL_OES_standard_derivatives" | "GL_EXT_shader_texture_lod"
                    );
                    if !is_supported
                        && matches!(
                            behavior,
                            ExtensionBehavior::Enable | ExtensionBehavior::Require
                        )
                    {
                        return Err(ShaderError::UnsupportedExtension(ext_name.to_string()));
                    }
                    state.extensions.insert(ext_name.to_string(), behavior);
                }
                "define" => {
                    if let Some((name, def)) = parse_define_directive(rest) {
                        if def.params.is_none() {
                            if let Ok(val) =
                                eval_pp_expr(&def.body, &state.macros, &mut state.budget)
                                && let Ok(u) = u32::try_from(val)
                            {
                                state.int_defines.insert(name.clone(), u);
                            } else {
                                state.int_defines.remove(&name);
                            }
                        } else {
                            state.int_defines.remove(&name);
                        }
                        state.macros.insert(name, def);
                    }
                }
                "undef" => {
                    if let Some(name) = rest.split_whitespace().next() {
                        state.macros.remove(name);
                        state.int_defines.remove(name);
                    }
                }
                "error" => {
                    return Err(ShaderError::GlslParse(format!("#error {}", rest.trim())));
                }
                _ => {
                    // Ignore unknown preprocessor directives in active branches.
                }
            }
            continue;
        }

        if !state.is_active() {
            continue;
        }

        active_chunk.push_str(&line);
        active_chunk.push('\n');
    }

    flush_active_chunk(&mut active_chunk, state, out)?;
    if state.cond_stack.len() != base_cond_depth {
        return Err(ShaderError::GlslParse(
            "unterminated #if/#ifdef/#ifndef directive".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod preprocess_tests {
    use super::*;

    #[test]
    fn handles_version_and_extension_directives_cleanly() {
        let src = r#"
            #version 300 es
            #extension GL_OES_standard_derivatives : enable
            #extension GL_EXT_shader_texture_lod : warn
            #extension GL_EXT_frag_depth : disable
            precision highp float;
            void main() {
                #if __VERSION__ >= 300
                    gl_FragColor = vec4(1.0, 0.5, 0.25, 1.0);
                #else
                    gl_FragColor = vec4(0.0);
                #endif
            }
        "#;
        let compiled = translate_glsl_es100(src).expect("#version 300 es + extensions compile");
        assert_eq!(compiled.glsl_version, Some(300));
        assert_eq!(
            compiled.extensions.get("GL_OES_standard_derivatives"),
            Some(&ExtensionBehavior::Enable)
        );
        assert_eq!(
            compiled.extensions.get("GL_EXT_shader_texture_lod"),
            Some(&ExtensionBehavior::Warn)
        );
        assert_eq!(
            compiled.extensions.get("GL_EXT_frag_depth"),
            Some(&ExtensionBehavior::Disable)
        );
        assert!(
            compiled
                .glsl450_source
                .contains("vec4(1.0, 0.5, 0.25, 1.0)")
        );

        // `#extension all : enable` must be rejected per GLSL spec.
        let bad_all = "#extension all : enable\nvoid main() { gl_FragColor = vec4(1.0); }";
        assert!(translate_glsl_es100(bad_all).is_err());

        // Malformed `#version` must be rejected.
        let bad_ver = "#version bogus\nvoid main() { gl_FragColor = vec4(1.0); }";
        assert!(translate_glsl_es100(bad_ver).is_err());
    }
}

// ---------------------------------------------------------------------------
// Stage 2: GLSL Lexer (comments stripped cleanly)
