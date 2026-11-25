use super::*;

/// `None` means all letters are selected; `Some(set)` holds 0-based selected indices.
pub type LetterSelection = Option<BTreeSet<usize>>;

fn is_selection_sep(c: char) -> bool {
    matches!(
        c,
        ',' | '.' | '\u{3002}' | '\u{FF0C}' | '\u{FF0E}' | ';' | '|' | '+'
    ) || c.is_whitespace()
}

fn add_token_selection(picked: &mut BTreeSet<usize>, token: &str, total: usize) {
    if token.is_empty() {
        return;
    }
    let parts: Vec<&str> = token
        .split(['-', '~', '\u{FF5E}'])
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() == 2
        && parts[0].bytes().all(|b| b.is_ascii_digit())
        && parts[1].bytes().all(|b| b.is_ascii_digit())
    {
        let Ok(mut from) = parts[0].parse::<usize>() else {
            return;
        };
        let Ok(mut to) = parts[1].parse::<usize>() else {
            return;
        };
        if from > to {
            std::mem::swap(&mut from, &mut to);
        }
        for n in from..=to {
            if n >= 1 && n <= total {
                picked.insert(n - 1);
            }
        }
        return;
    }
    if token.bytes().all(|b| b.is_ascii_digit())
        && let Ok(n) = token.parse::<usize>()
        && n >= 1
        && n <= total
    {
        picked.insert(n - 1);
    }
}

pub fn resolve_selection(
    value: &str,
    count: usize,
    sections: &BTreeMap<String, Vec<String>>,
) -> LetterSelection {
    if count == 0 {
        return Some(BTreeSet::new());
    }
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    let mut picked = BTreeSet::new();
    for token in trimmed.split(is_selection_sep) {
        if token.is_empty() {
            continue;
        }
        if let Some((kind_raw, idx_raw)) = token.split_once('-')
            && !kind_raw.is_empty()
            && kind_raw.bytes().all(|b| b.is_ascii_alphabetic())
            && !idx_raw.is_empty()
            && idx_raw.bytes().all(|b| b.is_ascii_digit())
            && let Ok(n_1based) = idx_raw.parse::<usize>()
            && n_1based >= 1
        {
            let kind = kind_raw.to_ascii_lowercase();
            if let Some(src) = sections.get(&kind).and_then(|v| v.get(n_1based - 1)) {
                let empty_sections = BTreeMap::new();
                match resolve_selection(src, count, &empty_sections) {
                    None => {
                        for i in 0..count {
                            picked.insert(i);
                        }
                    }
                    Some(sub) => picked.extend(sub),
                }
            }
            continue;
        }
        add_token_selection(&mut picked, token, count);
    }
    Some(picked)
}

pub fn parse_letter_selection(value: &str, count: usize) -> LetterSelection {
    resolve_selection(value, count, &BTreeMap::new())
}

pub fn is_letter_selected(selection: &LetterSelection, index: usize) -> bool {
    match selection {
        None => true,
        Some(set) => set.contains(&index),
    }
}

pub fn strength_weight(value: f64) -> f64 {
    let v = if value.is_finite() { value } else { 0.0 };
    v.clamp(0.0, 100.0) / 100.0
}

pub fn field_edge(d: f64) -> f64 {
    if d <= 0.75 {
        return 1.0;
    }
    if d >= 1.0 {
        return 0.0;
    }
    let u = (d - 0.75) / 0.25;
    1.0 - u * u * (3.0 - 2.0 * u)
}

/// Field falloff weight (`0` = infinite, `1` = linear X ramp, `2` = box, `3` = sphere).
pub fn field_weight(kind: f64, point: [f64; 3], position: [f64; 3], scale: [f64; 3]) -> f64 {
    let ty = if kind.is_finite() {
        js_round(kind) as i32
    } else {
        0
    };
    if ty <= 0 {
        return 1.0;
    }
    let s = [
        scale[0].abs().max(0.01),
        scale[1].abs().max(0.01),
        scale[2].abs().max(0.01),
    ];
    if ty == 2 {
        let dx = (point[0] - position[0]).abs() / (s[0] / 2.0);
        let dy = (point[1] - position[1]).abs() / (s[1] / 2.0);
        let dz = (point[2] - position[2]).abs() / (s[2] / 2.0);
        return field_edge(dx.max(dy).max(dz));
    }
    if ty == 3 {
        let dx = (point[0] - position[0]) / (s[0] / 2.0);
        let dy = (point[1] - position[1]) / (s[1] / 2.0);
        let dz = (point[2] - position[2]) / (s[2] / 2.0);
        return field_edge((dx * dx + dy * dy + dz * dz).sqrt());
    }
    let lo = position[0] - s[0] / 2.0;
    ((point[0] - lo) / s[0]).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Formula Micro-Language (`expresstext-formula.js`)
