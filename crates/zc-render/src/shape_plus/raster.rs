//! Vector text layout, procedural render-command compiler, and antialiased scanline rasterizer.

use super::*;

// ---------------------------------------------------------------------------
// Live Vector Text & Text Animator (`text`, `textAnimator`)
// ---------------------------------------------------------------------------

/// Built-in stroke-based vector font producing genuine closed 2D outlines for all
/// printable ASCII characters (`0x21..=0x7E`), plus fallback box for other glyphs.
/// Coordinates are normalized in `[0..0.60, 0..1.0]` (baseline at `y = 0`, cap height `y = 0.72`).
pub(crate) fn vector_glyph_strokes(ch: char) -> Vec<Vec<[f64; 2]>> {
    match ch {
        'A' => vec![
            vec![[0.05, 0.0], [0.28, 0.72], [0.51, 0.0]],
            vec![[0.14, 0.25], [0.42, 0.25]],
        ],
        'B' => vec![vec![
            [0.08, 0.0],
            [0.08, 0.72],
            [0.36, 0.72],
            [0.48, 0.63],
            [0.48, 0.45],
            [0.36, 0.37],
            [0.08, 0.37],
            [0.38, 0.37],
            [0.50, 0.27],
            [0.50, 0.09],
            [0.38, 0.0],
            [0.08, 0.0],
        ]],
        'C' => vec![vec![
            [0.49, 0.60],
            [0.39, 0.72],
            [0.20, 0.72],
            [0.08, 0.56],
            [0.08, 0.16],
            [0.20, 0.0],
            [0.39, 0.0],
            [0.49, 0.12],
        ]],
        'D' => vec![vec![
            [0.08, 0.0],
            [0.08, 0.72],
            [0.33, 0.72],
            [0.49, 0.56],
            [0.49, 0.16],
            [0.33, 0.0],
            [0.08, 0.0],
        ]],
        'E' => vec![
            vec![[0.48, 0.72], [0.08, 0.72], [0.08, 0.0], [0.48, 0.0]],
            vec![[0.08, 0.37], [0.40, 0.37]],
        ],
        'F' => vec![
            vec![[0.48, 0.72], [0.08, 0.72], [0.08, 0.0]],
            vec![[0.08, 0.37], [0.40, 0.37]],
        ],
        'G' => vec![vec![
            [0.49, 0.60],
            [0.39, 0.72],
            [0.20, 0.72],
            [0.08, 0.56],
            [0.08, 0.16],
            [0.20, 0.0],
            [0.42, 0.0],
            [0.49, 0.12],
            [0.49, 0.34],
            [0.30, 0.34],
        ]],
        'H' => vec![
            vec![[0.08, 0.0], [0.08, 0.72]],
            vec![[0.48, 0.0], [0.48, 0.72]],
            vec![[0.08, 0.37], [0.48, 0.37]],
        ],
        'I' => vec![
            vec![[0.14, 0.72], [0.42, 0.72]],
            vec![[0.28, 0.72], [0.28, 0.0]],
            vec![[0.14, 0.0], [0.42, 0.0]],
        ],
        'J' => vec![vec![
            [0.44, 0.72],
            [0.44, 0.16],
            [0.34, 0.0],
            [0.18, 0.0],
            [0.08, 0.12],
        ]],
        'K' => vec![
            vec![[0.08, 0.0], [0.08, 0.72]],
            vec![[0.48, 0.72], [0.08, 0.34], [0.49, 0.0]],
        ],
        'L' => vec![vec![[0.08, 0.72], [0.08, 0.0], [0.48, 0.0]]],
        'M' => vec![vec![
            [0.06, 0.0],
            [0.06, 0.72],
            [0.28, 0.28],
            [0.50, 0.72],
            [0.50, 0.0],
        ]],
        'N' => vec![vec![[0.08, 0.0], [0.08, 0.72], [0.48, 0.0], [0.48, 0.72]]],
        'O' => vec![vec![
            [0.28, 0.72],
            [0.44, 0.62],
            [0.50, 0.36],
            [0.44, 0.10],
            [0.28, 0.0],
            [0.12, 0.10],
            [0.06, 0.36],
            [0.12, 0.62],
            [0.28, 0.72],
        ]],
        'P' => vec![vec![
            [0.08, 0.0],
            [0.08, 0.72],
            [0.36, 0.72],
            [0.49, 0.62],
            [0.49, 0.44],
            [0.36, 0.34],
            [0.08, 0.34],
        ]],
        'Q' => vec![
            vec![
                [0.28, 0.72],
                [0.44, 0.62],
                [0.50, 0.36],
                [0.44, 0.10],
                [0.28, 0.0],
                [0.12, 0.10],
                [0.06, 0.36],
                [0.12, 0.62],
                [0.28, 0.72],
            ],
            vec![[0.34, 0.18], [0.52, -0.04]],
        ],
        'R' => vec![
            vec![
                [0.08, 0.0],
                [0.08, 0.72],
                [0.36, 0.72],
                [0.49, 0.62],
                [0.49, 0.44],
                [0.36, 0.35],
                [0.08, 0.35],
            ],
            vec![[0.30, 0.35], [0.50, 0.0]],
        ],
        'S' => vec![vec![
            [0.48, 0.62],
            [0.36, 0.72],
            [0.18, 0.72],
            [0.08, 0.61],
            [0.08, 0.48],
            [0.20, 0.39],
            [0.38, 0.33],
            [0.49, 0.22],
            [0.49, 0.10],
            [0.36, 0.0],
            [0.18, 0.0],
            [0.07, 0.11],
        ]],
        'T' => vec![
            vec![[0.04, 0.72], [0.52, 0.72]],
            vec![[0.28, 0.72], [0.28, 0.0]],
        ],
        'U' => vec![vec![
            [0.08, 0.72],
            [0.08, 0.18],
            [0.18, 0.0],
            [0.38, 0.0],
            [0.48, 0.18],
            [0.48, 0.72],
        ]],
        'V' => vec![vec![[0.06, 0.72], [0.28, 0.0], [0.50, 0.72]]],
        'W' => vec![vec![
            [0.04, 0.72],
            [0.15, 0.0],
            [0.28, 0.44],
            [0.41, 0.0],
            [0.52, 0.72],
        ]],
        'X' => vec![
            vec![[0.07, 0.72], [0.49, 0.0]],
            vec![[0.49, 0.72], [0.07, 0.0]],
        ],
        'Y' => vec![
            vec![[0.06, 0.72], [0.28, 0.36], [0.50, 0.72]],
            vec![[0.28, 0.36], [0.28, 0.0]],
        ],
        'Z' => vec![vec![[0.08, 0.72], [0.48, 0.72], [0.08, 0.0], [0.48, 0.0]]],
        'a'..='z' => {
            // Distinct lowercase proportions (x-height ~0.50) using clean strokes.
            let upper = ch.to_ascii_uppercase();
            vector_glyph_strokes(upper)
                .into_iter()
                .map(|stroke| {
                    stroke
                        .into_iter()
                        .map(|[x, y]| [x * 0.92, y * 0.72])
                        .collect()
                })
                .collect()
        }
        '0' => vec![
            vec![
                [0.28, 0.72],
                [0.44, 0.62],
                [0.48, 0.36],
                [0.44, 0.10],
                [0.28, 0.0],
                [0.12, 0.10],
                [0.08, 0.36],
                [0.12, 0.62],
                [0.28, 0.72],
            ],
            vec![[0.16, 0.16], [0.40, 0.56]],
        ],
        '1' => vec![
            vec![[0.16, 0.58], [0.28, 0.72], [0.28, 0.0]],
            vec![[0.14, 0.0], [0.42, 0.0]],
        ],
        '2' => vec![vec![
            [0.08, 0.58],
            [0.20, 0.72],
            [0.38, 0.72],
            [0.48, 0.58],
            [0.48, 0.44],
            [0.08, 0.0],
            [0.49, 0.0],
        ]],
        '3' => vec![vec![
            [0.09, 0.62],
            [0.22, 0.72],
            [0.38, 0.72],
            [0.48, 0.60],
            [0.48, 0.46],
            [0.36, 0.37],
            [0.22, 0.37],
            [0.38, 0.37],
            [0.49, 0.27],
            [0.49, 0.11],
            [0.38, 0.0],
            [0.20, 0.0],
            [0.08, 0.11],
        ]],
        '4' => vec![vec![[0.40, 0.0], [0.40, 0.72], [0.07, 0.24], [0.50, 0.24]]],
        '5' => vec![vec![
            [0.47, 0.72],
            [0.10, 0.72],
            [0.10, 0.40],
            [0.36, 0.40],
            [0.49, 0.30],
            [0.49, 0.11],
            [0.36, 0.0],
            [0.18, 0.0],
            [0.08, 0.10],
        ]],
        '6' => vec![vec![
            [0.44, 0.66],
            [0.32, 0.72],
            [0.16, 0.62],
            [0.08, 0.38],
            [0.08, 0.14],
            [0.20, 0.0],
            [0.38, 0.0],
            [0.49, 0.12],
            [0.49, 0.28],
            [0.38, 0.39],
            [0.18, 0.39],
            [0.08, 0.28],
        ]],
        '7' => vec![vec![[0.08, 0.72], [0.49, 0.72], [0.22, 0.0]]],
        '8' => vec![
            vec![
                [0.28, 0.72],
                [0.44, 0.63],
                [0.44, 0.46],
                [0.28, 0.38],
                [0.12, 0.46],
                [0.12, 0.63],
                [0.28, 0.72],
            ],
            vec![
                [0.28, 0.38],
                [0.47, 0.28],
                [0.47, 0.10],
                [0.28, 0.0],
                [0.09, 0.10],
                [0.09, 0.28],
                [0.28, 0.38],
            ],
        ],
        '9' => vec![vec![
            [0.12, 0.06],
            [0.24, 0.0],
            [0.40, 0.10],
            [0.48, 0.34],
            [0.48, 0.58],
            [0.36, 0.72],
            [0.18, 0.72],
            [0.08, 0.60],
            [0.08, 0.44],
            [0.18, 0.34],
            [0.38, 0.34],
            [0.48, 0.44],
        ]],
        '!' => vec![
            vec![[0.28, 0.72], [0.28, 0.22]],
            vec![[0.28, 0.06], [0.28, 0.02]],
        ],
        '.' => vec![vec![[0.28, 0.06], [0.28, 0.02]]],
        ',' => vec![vec![[0.28, 0.08], [0.22, -0.06]]],
        '-' => vec![vec![[0.10, 0.36], [0.46, 0.36]]],
        '+' => vec![
            vec![[0.10, 0.36], [0.46, 0.36]],
            vec![[0.28, 0.54], [0.28, 0.18]],
        ],
        ':' => vec![
            vec![[0.28, 0.48], [0.28, 0.44]],
            vec![[0.28, 0.12], [0.28, 0.08]],
        ],
        '?' => vec![
            vec![
                [0.10, 0.58],
                [0.22, 0.72],
                [0.38, 0.72],
                [0.48, 0.60],
                [0.48, 0.48],
                [0.28, 0.34],
                [0.28, 0.22],
            ],
            vec![[0.28, 0.06], [0.28, 0.02]],
        ],
        '/' => vec![vec![[0.10, 0.0], [0.46, 0.72]]],
        '(' => vec![vec![
            [0.38, 0.74],
            [0.22, 0.56],
            [0.18, 0.36],
            [0.22, 0.16],
            [0.38, -0.02],
        ]],
        ')' => vec![vec![
            [0.18, 0.74],
            [0.34, 0.56],
            [0.38, 0.36],
            [0.34, 0.16],
            [0.18, -0.02],
        ]],
        _ => vec![vec![
            [0.08, 0.0],
            [0.48, 0.0],
            [0.48, 0.72],
            [0.08, 0.72],
            [0.08, 0.0],
        ]],
    }
}

pub(crate) fn glyph_closed_contours(
    ch: char,
    scale_x: f64,
    scale_y: f64,
    ox: f64,
    oy: f64,
) -> Vec<ShapePath> {
    if ch.is_whitespace() {
        return Vec::new();
    }
    let strokes = vector_glyph_strokes(ch);
    let thickness = (scale_x.min(scale_y) * 0.11).max(1.0);
    let style = StrokeStyle {
        width: thickness,
        cap: StrokeCap::Round,
        join: StrokeJoin::Round,
        miter_limit: 4.0,
        dash_pairs: 0,
        dashes: [0.0; 3],
        gaps: [0.0; 3],
        dash_offset: 0.0,
        start_length: 0.0,
        end_length: 0.0,
        start_width: 100.0,
        end_width: 100.0,
        start_ease: 0.0,
        end_ease: 0.0,
        wave_amount: 0.0,
        wave_units_length: false,
        wavelength: 100.0,
        wave_cycles: 1.0,
        wave_phase: 0.0,
    };
    let raw_paths: Vec<ShapePath> = strokes
        .into_iter()
        .map(|pts| ShapePath {
            points: pts
                .into_iter()
                .map(|[x, y]| [ox + x * scale_x, oy + y * scale_y])
                .collect(),
            vertices: None,
            closed: false,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect();
    expand_stroke_polygons(&raw_paths, &style, 0.25)
        .into_iter()
        .map(|points| ShapePath {
            points,
            vertices: None,
            closed: true,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect()
}

#[derive(Debug, Clone)]
pub(crate) struct TextGlyphInstance {
    pub(crate) ch: char,
    pub(crate) line: usize,
    pub(crate) word: Option<usize>,
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) advance: f64,
    pub(crate) scale_x: f64,
    pub(crate) scale_y: f64,
    pub(crate) paths: Vec<ShapePath>,
}

pub(crate) fn layout_shape_plus_text(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> (Vec<TextGlyphInstance>, usize) {
    let get = |k: &str| props.and_then(|p| p.get(k));
    let mut source = eval_prop_str(get("sourceText"), frame, "Text");
    let caps = eval_prop_str(get("caps"), frame, "normal").to_ascii_lowercase();
    if caps == "all" {
        source = source.to_uppercase();
    }
    let font_size = eval_prop_f64(get("fontSize"), frame, 100.0, eval_expr).max(0.0);
    if font_size <= 0.0 || source.is_empty() {
        return (Vec::new(), 0);
    }
    let tracking = eval_prop_f64(get("tracking"), frame, 0.0, eval_expr);
    let leading = eval_prop_f64(get("leading"), frame, font_size * 1.2, eval_expr);
    let baseline_shift = eval_prop_f64(get("baselineShift"), frame, 0.0, eval_expr);
    let h_scale = eval_prop_f64(get("horizontalScale"), frame, 100.0, eval_expr) / 100.0;
    let v_scale = eval_prop_f64(get("verticalScale"), frame, 100.0, eval_expr) / 100.0;
    let origin = eval_prop_vec2(get("position"), frame, [0.0, 0.0], eval_expr);
    let alignment = eval_prop_str(get("alignment"), frame, "left").to_ascii_lowercase();
    let text_mode = eval_prop_str(get("textMode"), frame, "point").to_ascii_lowercase();
    let box_size = eval_prop_vec2(get("boxSize"), frame, [600.0, 400.0], eval_expr);
    let first_line_indent = eval_prop_f64(get("firstLineIndent"), frame, 0.0, eval_expr);
    let space_before = eval_prop_f64(get("spaceBefore"), frame, 0.0, eval_expr);
    let space_after = eval_prop_f64(get("spaceAfter"), frame, 0.0, eval_expr);

    let sx = font_size * h_scale;
    let sy = font_size * v_scale;
    let char_adv = |c: char| -> f64 {
        let factor = if c == ' ' {
            0.38
        } else if matches!(c, 'I' | 'i' | 'l' | '!' | '.' | ',' | ':') {
            0.44
        } else {
            0.62
        };
        factor * sx + tracking * font_size / 1000.0
    };

    // Split paragraphs on '\n', and word-wrap in box mode.
    let mut lines: Vec<(String, bool)> = Vec::new();
    for para in source.split('\n') {
        if text_mode == "box" && box_size[0] > 0.0 {
            let max_w = box_size[0];
            let mut cur_line = String::new();
            let mut cur_w = first_line_indent;
            let mut first = true;
            for word in para.split(' ') {
                let word_w: f64 = word.chars().map(char_adv).sum();
                let sp_w = if cur_line.is_empty() {
                    0.0
                } else {
                    char_adv(' ')
                };
                if !cur_line.is_empty() && cur_w + sp_w + word_w > max_w {
                    lines.push((cur_line, first));
                    cur_line = word.to_owned();
                    cur_w = word_w;
                    first = false;
                } else {
                    if !cur_line.is_empty() {
                        cur_line.push(' ');
                        cur_w += sp_w;
                    }
                    cur_line.push_str(word);
                    cur_w += word_w;
                }
            }
            lines.push((cur_line, first));
        } else {
            lines.push((para.to_owned(), true));
        }
    }

    let line_count = lines.len();
    let mut glyphs = Vec::new();
    let mut pen_y = origin[1] + baseline_shift;
    let mut next_word_id = 0usize;

    for (line_idx, (line_str, is_para_start)) in lines.iter().enumerate() {
        if line_idx > 0 {
            pen_y -= leading;
            if *is_para_start {
                pen_y -= space_before + space_after;
            }
        }
        let line_w: f64 = line_str.chars().map(char_adv).sum();
        let align_x = match alignment.as_str() {
            "center" | "justifycenter" => -line_w * 0.5,
            "right" | "justifyright" => -line_w,
            _ => 0.0,
        };
        let indent = if *is_para_start {
            first_line_indent
        } else {
            0.0
        };
        let mut pen_x = origin[0] + align_x + indent;
        let mut in_word = false;
        let mut cur_word = 0usize;

        for ch in line_str.chars() {
            let word_opt = if ch.is_whitespace() {
                in_word = false;
                None
            } else {
                if !in_word {
                    in_word = true;
                    cur_word = next_word_id;
                    next_word_id += 1;
                }
                Some(cur_word)
            };
            let adv = char_adv(ch);
            let (gx, gy, gch) = if caps == "small" && ch.is_ascii_lowercase() {
                (sx * 0.8, sy * 0.8, ch.to_ascii_uppercase())
            } else {
                (sx, sy, ch)
            };
            let paths = glyph_closed_contours(gch, gx, gy, pen_x, pen_y);
            glyphs.push(TextGlyphInstance {
                ch,
                line: line_idx,
                word: word_opt,
                x: pen_x,
                y: pen_y,
                advance: adv,
                scale_x: gx,
                scale_y: gy,
                paths,
            });
            pen_x += adv;
        }
    }

    (glyphs, line_count)
}

pub(crate) fn selector_hash_seed(seed: f64, salt: i32) -> f64 {
    let mut h = ((seed.floor() as i32).wrapping_mul(374_761_393))
        .wrapping_add(salt.wrapping_mul(668_265_263));
    h = (h ^ ((h as u32) >> 13) as i32).wrapping_mul(1_274_126_177);
    h ^= ((h as u32) >> 16) as i32;
    f64::from(h as u32) / 4_294_967_296.0
}

pub(crate) fn selector_random_perm(count: usize, seed: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        let r = selector_hash_seed(seed * 1000.0 + (i as f64) * 131.0 + 101.0 * 17.0, 7);
        let j = ((r * (i + 1) as f64).floor() as usize).min(i);
        order.swap(i, j);
    }
    order
}

pub(crate) fn selector_ease_low(value: f64, percent: f64) -> f64 {
    let v = value.clamp(0.0, 1.0);
    let amt = percent / 100.0;
    if amt >= 0.0 {
        v.powf(1.0 + amt)
    } else {
        v.powf(1.0 / (1.0 - amt))
    }
}

pub(crate) fn selector_ease_high(value: f64, percent: f64) -> f64 {
    let v = value.clamp(0.0, 1.0);
    let amt = percent / 100.0;
    if amt >= 0.0 {
        1.0 - (1.0 - v).powf(1.0 + amt)
    } else {
        1.0 - (1.0 - v).powf(1.0 / (1.0 - amt))
    }
}

pub(crate) fn selector_shape_value(shape: &str, i: usize, n: usize, start: f64, end: f64) -> f64 {
    if n == 0 || (start - end).abs() < 1e-12 {
        return 0.0;
    }
    let nf = n as f64;
    if shape == "square" {
        let a = start.max(i as f64 / nf);
        let b = end.min((i + 1) as f64 / nf);
        return ((b - a) * nf).clamp(0.0, 1.0);
    }
    let c = (i as f64 + 0.5) / nf;
    if shape == "rampUp" {
        if c <= start {
            return 0.0;
        }
        if c >= end {
            return 1.0;
        }
        return (c - start) / (end - start);
    }
    if shape == "rampDown" {
        if c <= start {
            return 1.0;
        }
        if c >= end {
            return 0.0;
        }
        return (end - c) / (end - start);
    }
    if c <= start || c >= end {
        return 0.0;
    }
    let t = (c - start) / (end - start);
    match shape {
        "triangle" => {
            if t < 0.5 {
                t * 2.0
            } else {
                (1.0 - t) * 2.0
            }
        }
        "round" => (1.0 - (t * 2.0 - 1.0).powi(2)).max(0.0).sqrt(),
        _ => 0.5 - 0.5 * (t * TAU).cos(),
    }
}

pub(crate) fn selector_combine(mode: &str, a: f64, b: f64) -> f64 {
    let r = match mode {
        "subtract" => a - b,
        "intersect" => a * b,
        "min" => a.min(b),
        "max" => a.max(b),
        "difference" => (a - b).abs(),
        _ => a + b,
    };
    r.clamp(-1.0, 1.0)
}

pub(crate) fn compute_text_units(
    glyphs: &[TextGlyphInstance],
    line_count: usize,
    based_on: &str,
) -> (usize, Vec<isize>) {
    match based_on {
        "lines" => (line_count, glyphs.iter().map(|g| g.line as isize).collect()),
        "words" => {
            let mut max_w = 0usize;
            let map: Vec<isize> = glyphs
                .iter()
                .map(|g| match g.word {
                    Some(w) => {
                        max_w = max_w.max(w + 1);
                        w as isize
                    }
                    None => -1,
                })
                .collect();
            (max_w, map)
        }
        "noSpaces" => {
            let mut count = 0isize;
            let map: Vec<isize> = glyphs
                .iter()
                .map(|g| {
                    if g.ch.is_whitespace() {
                        -1
                    } else {
                        let id = count;
                        count += 1;
                        id
                    }
                })
                .collect();
            (count as usize, map)
        }
        _ => (glyphs.len(), (0..glyphs.len() as isize).collect()),
    }
}

pub(crate) fn shift_codepoint(ch: char, shift: i32) -> char {
    if shift == 0 {
        return ch;
    }
    let code = ch as u32;
    for (base, size) in [(97_u32, 26_i32), (65_u32, 26_i32), (48_u32, 10_i32)] {
        if code >= base && code < base + size as u32 {
            let shifted = ((code - base) as i32 + shift).rem_euclid(size) as u32;
            return char::from_u32(base + shifted).unwrap_or(ch);
        }
    }
    ch
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct CharPaintOverride {
    pub(crate) opacity: f64,
    pub(crate) fill_color: Option<[f64; 3]>,
}

pub(crate) fn animator_prop_enabled(
    props: &Map<String, Value>,
    key: &str,
    has_any_flag: bool,
) -> bool {
    let flag_key = format!("enabled_{key}");
    if let Some(v) = props.get(&flag_key) {
        return v
            .as_bool()
            .unwrap_or_else(|| v.as_f64().unwrap_or(0.0) != 0.0);
    }
    if has_any_flag {
        return false;
    }
    props.contains_key(key) || (key == "fillColor" && props.contains_key("color"))
}

pub(crate) fn evaluate_text_with_animators(
    item: &Value,
    animators: &[&Value],
    frame: f64,
    fps: f64,
    eval_expr: ExprEval<'_>,
) -> Vec<(Vec<ShapePath>, Mat2D, Option<CharPaintOverride>)> {
    let props = item.get("properties").and_then(Value::as_object);
    let font_size = eval_prop_f64(
        props.and_then(|p| p.get("fontSize")),
        frame,
        100.0,
        eval_expr,
    );
    let (mut glyphs, line_count) = layout_shape_plus_text(props, frame, eval_expr);
    if glyphs.is_empty() {
        return Vec::new();
    }

    let active_animators: Vec<&Map<String, Value>> = animators
        .iter()
        .filter_map(|a| {
            let ap = a.get("properties").and_then(Value::as_object)?;
            if !eval_prop_enabled(Some(ap), frame, eval_expr) {
                return None;
            }
            Some(ap)
        })
        .collect();

    if active_animators.is_empty() {
        let all_paths: Vec<ShapePath> = glyphs.into_iter().flat_map(|g| g.paths).collect();
        return vec![(all_paths, MAT_IDENTITY, None)];
    }

    let time = if fps > 0.0 { frame / fps } else { 0.0 };
    let n = glyphs.len();
    struct AnimWeights {
        s: Vec<f64>,
        vx: Vec<f64>,
        vy: Vec<f64>,
        has_any_flag: bool,
    }

    let mut all_weights = Vec::with_capacity(active_animators.len());
    for ap in &active_animators {
        let get = |k: &str| ap.get(k);
        let has_any_flag = ap.keys().any(|k| k.starts_with("enabled_"));
        let based_on = eval_prop_str(get("selectorBasedOn"), frame, "characters");
        let (unit_count, unit_map) = compute_text_units(&glyphs, line_count, &based_on);
        let units_mode = eval_prop_str(get("selectorUnits"), frame, "percentage");
        let start_raw = eval_prop_f64(get("selectorStart"), frame, 0.0, eval_expr);
        let end_raw = eval_prop_f64(get("selectorEnd"), frame, 100.0, eval_expr);
        let off_raw = eval_prop_f64(get("selectorOffset"), frame, 0.0, eval_expr);
        let (start, end) = if units_mode == "index" && unit_count > 0 {
            let nf = unit_count as f64;
            let raw_e = if (end_raw - 100.0).abs() < 1e-9
                && !ap
                    .get("selectorEnd")
                    .is_some_and(|v| v.get("keyframes").is_some())
            {
                nf
            } else {
                end_raw
            };
            ((start_raw + off_raw) / nf, (raw_e + off_raw) / nf)
        } else {
            ((start_raw + off_raw) / 100.0, (end_raw + off_raw) / 100.0)
        };
        let from = start.min(end);
        let to = start.max(end);
        let shape = eval_prop_str(get("selectorShape"), frame, "square");
        let ease_low = eval_prop_f64(get("selectorEaseLow"), frame, 0.0, eval_expr);
        let ease_high = eval_prop_f64(get("selectorEaseHigh"), frame, 0.0, eval_expr);
        let amount = eval_prop_f64(get("selectorAmount"), frame, 100.0, eval_expr) / 100.0;
        let randomize = eval_prop_f64(get("selectorRandomize"), frame, 0.0, eval_expr) != 0.0;
        let rand_seed = eval_prop_f64(get("selectorRandomSeed"), frame, 0.0, eval_expr);
        let perm = if randomize {
            Some(selector_random_perm(unit_count, rand_seed))
        } else {
            None
        };
        let range_vals: Vec<f64> = (0..unit_count)
            .map(|i| {
                let ui = perm.as_ref().map_or(i, |p| p[i]);
                let raw = selector_shape_value(&shape, ui, unit_count, from, to);
                selector_ease_high(selector_ease_low(raw, ease_low), ease_high) * amount
            })
            .collect();

        let sel_mode = eval_prop_str(get("selectorMode"), frame, "add");
        let wiggly_on = eval_prop_f64(get("wigglyOn"), frame, 0.0, eval_expr) != 0.0;
        let wiggly_mode = eval_prop_str(get("wigglyMode"), frame, "add");
        let wiggly_max = eval_prop_f64(get("wigglyMax"), frame, 100.0, eval_expr) / 100.0;
        let wiggly_min = eval_prop_f64(get("wigglyMin"), frame, -100.0, eval_expr) / 100.0;
        let wiggly_based = eval_prop_str(get("wigglyBasedOn"), frame, "characters");
        let locked = eval_prop_f64(get("lockDimensions"), frame, 0.0, eval_expr) != 0.0;
        let wp = WiggleParams {
            time,
            frequency: eval_prop_f64(get("wigglesPerSecond"), frame, 2.0, eval_expr),
            correlation: eval_prop_f64(get("correlation"), frame, 50.0, eval_expr),
            temporal_phase: eval_prop_f64(get("temporalPhase"), frame, 0.0, eval_expr),
            spatial_phase: eval_prop_f64(get("spatialPhase"), frame, 0.0, eval_expr),
            seed: eval_prop_f64(get("wigglySeed"), frame, 0.0, eval_expr).round() as i32,
        };
        let wiggly_map = if wiggly_on && wiggly_based != based_on {
            compute_text_units(&glyphs, line_count, &wiggly_based).1
        } else {
            unit_map.clone()
        };

        let mut s = vec![0.0; n];
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        for i in 0..n {
            let u = unit_map[i];
            let r = if u < 0 {
                0.0
            } else {
                selector_combine(&sel_mode, 0.0, range_vals[u as usize])
            };
            if !wiggly_on {
                s[i] = r;
                vx[i] = r;
                vy[i] = r;
            } else {
                let uw = wiggly_map[i];
                let sample_w = |comp: i32| -> f64 {
                    if uw < 0 {
                        0.0
                    } else {
                        let nv = sample_wiggle_noise(wp, uw as f64, comp, 0, 0);
                        wiggly_min + (nv + 1.0) * 0.5 * (wiggly_max - wiggly_min)
                    }
                };
                s[i] = selector_combine(&wiggly_mode, r, sample_w(0));
                if locked {
                    vx[i] = s[i];
                    vy[i] = s[i];
                } else {
                    vx[i] = selector_combine(&wiggly_mode, r, sample_w(1));
                    vy[i] = selector_combine(&wiggly_mode, r, sample_w(2));
                }
            }
        }
        all_weights.push(AnimWeights {
            s,
            vx,
            vy,
            has_any_flag,
        });
    }

    // Apply charOffset and tracking shifts first.
    let mut line_carry: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, glyph) in glyphs.iter_mut().enumerate() {
        let mut shift = 0i32;
        let mut track_step = 0.0;
        for (a_idx, ap) in active_animators.iter().enumerate() {
            let w = &all_weights[a_idx];
            if animator_prop_enabled(ap, "charOffset", w.has_any_flag) {
                let co = eval_prop_f64(ap.get("charOffset"), frame, 0.0, eval_expr);
                shift += (co * w.s[i]).round() as i32;
            }
            if animator_prop_enabled(ap, "tracking", w.has_any_flag) {
                let tr = eval_prop_f64(ap.get("tracking"), frame, 0.0, eval_expr);
                track_step += tr * w.s[i] * font_size / 1000.0;
            }
        }
        let carry = line_carry.entry(glyph.line).or_insert(0.0);
        if *carry != 0.0 || shift != 0 {
            glyph.x += *carry;
            if shift != 0 {
                glyph.ch = shift_codepoint(glyph.ch, shift);
            }
            glyph.paths =
                glyph_closed_contours(glyph.ch, glyph.scale_x, glyph.scale_y, glyph.x, glyph.y);
        }
        *carry += track_step;
    }

    // Build per-glyph composed transform matrix & optional opacity/fillColor override.
    let mut out = Vec::with_capacity(n);
    for (i, glyph) in glyphs.into_iter().enumerate() {
        if glyph.paths.is_empty() {
            continue;
        }
        let mut mat = MAT_IDENTITY;
        let mut opacity = 1.0_f64;
        let mut fill_color: Option<[f64; 3]> = None;
        let ref_pt = [glyph.x + glyph.advance * 0.5, glyph.y];

        for (a_idx, ap) in active_animators.iter().enumerate() {
            let w = &all_weights[a_idx];
            let sx_w = w.vx[i];
            let sy_w = w.vy[i];
            let sc_w = w.s[i];
            let anchor_off = if animator_prop_enabled(ap, "anchor", w.has_any_flag) {
                let a = eval_prop_vec2(ap.get("anchor"), frame, [0.0, 0.0], eval_expr);
                [a[0] * sx_w, a[1] * sy_w]
            } else {
                [0.0, 0.0]
            };
            let pivot = [ref_pt[0] + anchor_off[0], ref_pt[1] + anchor_off[1]];
            let pos_off = if animator_prop_enabled(ap, "position", w.has_any_flag) {
                let p = eval_prop_vec2(ap.get("position"), frame, [0.0, 0.0], eval_expr);
                [p[0] * sx_w, p[1] * sy_w]
            } else {
                [0.0, 0.0]
            };
            let scale = if animator_prop_enabled(ap, "scale", w.has_any_flag) {
                let s = eval_prop_vec2(ap.get("scale"), frame, [100.0, 100.0], eval_expr);
                [
                    1.0 + (s[0] / 100.0 - 1.0) * sx_w,
                    1.0 + (s[1] / 100.0 - 1.0) * sy_w,
                ]
            } else {
                [1.0, 1.0]
            };
            let rot = if animator_prop_enabled(ap, "rotation", w.has_any_flag) {
                eval_prop_f64(ap.get("rotation"), frame, 0.0, eval_expr) * sc_w
            } else {
                0.0
            };
            let skew = if animator_prop_enabled(ap, "skew", w.has_any_flag) {
                eval_prop_f64(ap.get("skew"), frame, 0.0, eval_expr) * sc_w
            } else {
                0.0
            };
            let skew_axis = if animator_prop_enabled(ap, "skewAxis", w.has_any_flag) {
                eval_prop_f64(ap.get("skewAxis"), frame, 0.0, eval_expr) * sc_w
            } else {
                0.0
            };
            if animator_prop_enabled(ap, "opacity", w.has_any_flag) {
                let target_op = eval_prop_f64(ap.get("opacity"), frame, 1.0, eval_expr);
                opacity *= (1.0 + (target_op - 1.0) * sc_w).clamp(0.0, 1.0);
            }
            if animator_prop_enabled(ap, "fillColor", w.has_any_flag) && sc_w.abs() > 1e-6 {
                let col_val = ap.get("fillColor").or_else(|| ap.get("color"));
                let target = eval_prop_color_rgb(col_val, frame, [1.0, 1.0, 1.0], eval_expr);
                let base = fill_color.unwrap_or([1.0, 1.0, 1.0]);
                let f = sc_w.clamp(0.0, 1.0);
                fill_color = Some([
                    lerp_f64(base[0], target[0], f),
                    lerp_f64(base[1], target[1], f),
                    lerp_f64(base[2], target[2], f),
                ]);
            }

            let level = group_matrix(
                pivot,
                [pivot[0] + pos_off[0], pivot[1] + pos_off[1]],
                scale,
                rot,
                skew,
                skew_axis,
            );
            mat = mat_mul(level, mat);
        }

        let ov = if (opacity - 1.0).abs() > 1e-6 || fill_color.is_some() {
            Some(CharPaintOverride {
                opacity,
                fill_color,
            })
        } else {
            None
        };
        out.push((glyph.paths, mat, ov));
    }
    out
}

// ---------------------------------------------------------------------------
// Procedural Compiler (matching Zoidium `compiler-procedural.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct PathView {
    pub(crate) base_idx: usize,
    pub(crate) matrix: Mat2D,
    pub(crate) wiggle_stack: Vec<(Mat2D, WiggleTransformConfig)>,
    pub(crate) char_override: Option<CharPaintOverride>,
}

impl PathView {
    pub(crate) fn matrix_for_copy(&self, copy: i32) -> Mat2D {
        if self.wiggle_stack.is_empty() {
            return self.matrix;
        }
        let mut m = self.matrix;
        for &(post_mat, wcfg) in &self.wiggle_stack {
            let wm = eval_wiggle_transform_matrix(wcfg, copy);
            m = mat_mul(post_mat, mat_mul(wm, m));
        }
        m
    }
}

#[derive(Debug, Clone)]
pub(crate) enum PaintKind {
    Fill {
        color: [f64; 3],
        even_odd: bool,
    },
    GradientFill {
        gradient: GradientPaint,
        even_odd: bool,
    },
    Stroke {
        color: [f64; 3],
        style: StrokeStyle,
    },
    GradientStroke {
        gradient: GradientPaint,
        style: StrokeStyle,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum ScopeCommand {
    Group {
        matrix: Mat2D,
        wiggle: Option<WiggleTransformConfig>,
        opacity: f64,
        commands: Vec<ScopeCommand>,
    },
    Paint {
        views: Vec<PathView>,
        kind: PaintKind,
        opacity: f64,
        tolerance: f64,
    },
}

#[derive(Debug, Clone)]
pub(crate) enum RenderCommand {
    Isolation {
        opacity: f64,
        commands: Vec<RenderCommand>,
    },
    SolidFill {
        polygons: Vec<Vec<[f64; 2]>>,
        color: [f64; 3],
        opacity: f64,
        even_odd: bool,
    },
    GradientFill {
        polygons: Vec<Vec<[f64; 2]>>,
        gradient: GradientPaint,
        opacity: f64,
        even_odd: bool,
    },
}

impl RenderCommand {
    pub(crate) fn transformed(self, m: Mat2D) -> Self {
        if mat_is_identity(m) {
            return self;
        }
        match self {
            Self::Isolation { opacity, commands } => Self::Isolation {
                opacity,
                commands: commands.into_iter().map(|c| c.transformed(m)).collect(),
            },
            Self::SolidFill {
                polygons,
                color,
                opacity,
                even_odd,
            } => Self::SolidFill {
                polygons: polygons
                    .into_iter()
                    .map(|p| p.into_iter().map(|pt| mat_transform_pt(pt, m)).collect())
                    .collect(),
                color,
                opacity,
                even_odd,
            },
            Self::GradientFill {
                polygons,
                mut gradient,
                opacity,
                even_odd,
            } => {
                gradient.matrix = mat_mul(m, gradient.matrix);
                Self::GradientFill {
                    polygons: polygons
                        .into_iter()
                        .map(|p| p.into_iter().map(|pt| mat_transform_pt(pt, m)).collect())
                        .collect(),
                    gradient,
                    opacity,
                    even_odd,
                }
            }
        }
    }
}

pub(crate) struct RepeaterCopy {
    pub(crate) matrix: Mat2D,
    pub(crate) opacity: f64,
}

pub(crate) fn repeater_copies(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> Vec<RepeaterCopy> {
    let get = |k: &str| props.and_then(|p| p.get(k));
    let count = eval_prop_f64(get("copies"), frame, 3.0, eval_expr).clamp(0.0, MAX_REPEATER_COPIES);
    let n = count.ceil() as usize;
    let offset = eval_prop_f64(get("offset"), frame, 0.0, eval_expr);
    let anchor = eval_prop_vec2(get("anchor"), frame, [0.0, 0.0], eval_expr);
    let pos = eval_prop_vec2(get("position"), frame, [100.0, 0.0], eval_expr);
    let scale = eval_prop_vec2(get("scale"), frame, [1.0, 1.0], eval_expr);
    let rot = eval_prop_f64(get("rotation"), frame, 0.0, eval_expr);
    let start_op = eval_prop_f64(get("startOpacity"), frame, 1.0, eval_expr).clamp(0.0, 1.0);
    let end_op = eval_prop_f64(get("endOpacity"), frame, 1.0, eval_expr).clamp(0.0, 1.0);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let exponent = i as f64 + offset;
        let power = |s: f64| -> f64 {
            if exponent == 0.0 {
                1.0
            } else if s < 0.0 && exponent.fract() != 0.0 {
                s.abs().powf(exponent)
            } else if s == 0.0 && exponent < 0.0 {
                0.0
            } else {
                s.powf(exponent)
            }
        };
        let m = group_matrix(
            anchor,
            [anchor[0] + pos[0] * exponent, anchor[1] + pos[1] * exponent],
            [power(scale[0]), power(scale[1])],
            rot * exponent,
            0.0,
            0.0,
        );
        if !m.iter().all(|v| v.is_finite()) {
            continue;
        }
        let frac = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let op = lerp_f64(start_op, end_op, frac).clamp(0.0, 1.0) * (count - i as f64).min(1.0);
        out.push(RepeaterCopy {
            matrix: m,
            opacity: op.clamp(0.0, 1.0),
        });
    }
    out
}

pub(crate) struct Compiler<'a> {
    pub(crate) frame: f64,
    pub(crate) fps: f64,
    pub(crate) eval_expr: ExprEval<'a>,
    pub(crate) bases: Vec<Vec<ShapePath>>,
}

impl<'a> Compiler<'a> {
    pub(crate) fn alloc_base(&mut self, paths: Vec<ShapePath>) -> usize {
        let idx = self.bases.len();
        self.bases.push(paths);
        idx
    }

    pub(crate) fn view_paths(&self, view: &PathView) -> Vec<ShapePath> {
        let m = view.matrix_for_copy(0);
        self.bases[view.base_idx]
            .iter()
            .map(|p| p.transformed(m))
            .collect()
    }

    pub(crate) fn below_animators<'b>(&self, list: &'b [Value], index: usize) -> Vec<&'b Value> {
        list[index + 1..]
            .iter()
            .filter(|item| {
                item.get("kind").and_then(Value::as_str) == Some("textAnimator")
                    && eval_prop_enabled(
                        item.get("properties").and_then(Value::as_object),
                        self.frame,
                        self.eval_expr,
                    )
            })
            .collect()
    }

    pub(crate) fn compile_scope(
        &mut self,
        list: &[Value],
        tolerance: f64,
        depth: usize,
        outer_animators: &[&Value],
    ) -> (Vec<PathView>, Vec<ScopeCommand>) {
        if depth > MAX_SHAPE_PLUS_DEPTH {
            return (Vec::new(), Vec::new());
        }
        let mut views: Vec<PathView> = Vec::new();
        let mut commands: Vec<ScopeCommand> = Vec::new();

        // Precompute downstream repeater scale factor for adaptive curve tolerance.
        let mut future_scales = vec![1.0_f64; list.len()];
        let mut future_scale = 1.0_f64;
        for i in (0..list.len()).rev() {
            future_scales[i] = future_scale;
            let item = &list[i];
            if item.get("kind").and_then(Value::as_str) == Some("repeater") {
                let props = item.get("properties").and_then(Value::as_object);
                if eval_prop_enabled(props, self.frame, self.eval_expr) {
                    let copies = repeater_copies(props, self.frame, self.eval_expr);
                    let max_s = copies
                        .iter()
                        .map(|c| mat_max_scale(c.matrix))
                        .fold(1.0_f64, f64::max);
                    future_scale = (future_scale * max_s).min(1024.0);
                }
            }
        }

        for (index, item) in list.iter().enumerate() {
            let Some(kind) = item.get("kind").and_then(Value::as_str) else {
                continue;
            };
            let props = item.get("properties").and_then(Value::as_object);
            if !eval_prop_enabled(props, self.frame, self.eval_expr) {
                continue;
            }
            let item_tol = (tolerance / future_scales[index]).max(1e-3);
            let get = |k: &str| props.and_then(|p| p.get(k));

            match kind {
                "textAnimator" => {}
                "text" => {
                    let mut anims = self.below_animators(list, index);
                    anims.extend_from_slice(outer_animators);
                    let glyph_groups = evaluate_text_with_animators(
                        item,
                        &anims,
                        self.frame,
                        self.fps,
                        self.eval_expr,
                    );
                    for (paths, mat, ov) in glyph_groups {
                        if paths.is_empty() {
                            continue;
                        }
                        let base_idx = self.alloc_base(paths);
                        views.push(PathView {
                            base_idx,
                            matrix: mat,
                            wiggle_stack: Vec::new(),
                            char_override: ov,
                        });
                    }
                }
                "group" => {
                    let anchor =
                        eval_prop_vec2(get("anchor"), self.frame, [0.0, 0.0], self.eval_expr);
                    let pos =
                        eval_prop_vec2(get("position"), self.frame, [0.0, 0.0], self.eval_expr);
                    let scale =
                        eval_prop_vec2(get("scale"), self.frame, [1.0, 1.0], self.eval_expr);
                    let rot = eval_prop_f64(get("rotation"), self.frame, 0.0, self.eval_expr);
                    let skew = eval_prop_f64(get("skew"), self.frame, 0.0, self.eval_expr);
                    let skew_axis = eval_prop_f64(get("skewAxis"), self.frame, 0.0, self.eval_expr);
                    let opacity = eval_prop_f64(get("opacity"), self.frame, 1.0, self.eval_expr)
                        .clamp(0.0, 1.0);
                    let m = group_matrix(anchor, pos, scale, rot, skew, skew_axis);
                    let g_scale = mat_max_scale(m);
                    let mut next_outer = self.below_animators(list, index);
                    next_outer.extend_from_slice(outer_animators);
                    let children = item
                        .get("contents")
                        .and_then(Value::as_array)
                        .map_or(&[][..], Vec::as_slice);
                    let (child_views, child_cmds) =
                        self.compile_scope(children, item_tol / g_scale, depth + 1, &next_outer);

                    // Propagate nested group paths UP to parent scope!
                    for mut cv in child_views {
                        if cv.wiggle_stack.is_empty() {
                            cv.matrix = mat_mul(m, cv.matrix);
                        } else if let Some(last) = cv.wiggle_stack.last_mut() {
                            last.0 = mat_mul(m, last.0);
                        }
                        views.push(cv);
                    }
                    commands.insert(
                        0,
                        ScopeCommand::Group {
                            matrix: m,
                            wiggle: None,
                            opacity,
                            commands: child_cmds,
                        },
                    );
                }
                "fill" | "stroke" | "gradientFill" | "gradientStroke" => {
                    let opacity = eval_prop_f64(get("opacity"), self.frame, 1.0, self.eval_expr)
                        .clamp(0.0, 1.0);
                    let composite_above = eval_prop_str(get("composite"), self.frame, "below")
                        .eq_ignore_ascii_case("above");
                    let even_odd = matches!(
                        eval_prop_str(get("fillRule"), self.frame, "nonzero")
                            .to_ascii_lowercase()
                            .as_str(),
                        "even-odd" | "evenodd"
                    );
                    let paint_kind = match kind {
                        "fill" => PaintKind::Fill {
                            color: eval_prop_color_rgb(
                                get("color"),
                                self.frame,
                                [1.0, 1.0, 1.0],
                                self.eval_expr,
                            ),
                            even_odd,
                        },
                        "gradientFill" => PaintKind::GradientFill {
                            gradient: eval_gradient_paint(props, self.frame, self.eval_expr),
                            even_odd,
                        },
                        "stroke" => PaintKind::Stroke {
                            color: eval_prop_color_rgb(
                                get("color"),
                                self.frame,
                                [1.0, 1.0, 1.0],
                                self.eval_expr,
                            ),
                            style: StrokeStyle::from_props(props, self.frame, self.eval_expr),
                        },
                        _ => PaintKind::GradientStroke {
                            gradient: eval_gradient_paint(props, self.frame, self.eval_expr),
                            style: StrokeStyle::from_props(props, self.frame, self.eval_expr),
                        },
                    };
                    let paint = ScopeCommand::Paint {
                        views: views.clone(),
                        kind: paint_kind,
                        opacity,
                        tolerance: item_tol,
                    };
                    if composite_above {
                        commands.push(paint);
                    } else {
                        commands.insert(0, paint);
                    }
                }
                "offsetPaths" | "mergePaths" => {
                    let all_paths: Vec<ShapePath> =
                        views.iter().flat_map(|v| self.view_paths(v)).collect();
                    let changed = if kind == "offsetPaths" {
                        let amount = eval_prop_f64(get("amount"), self.frame, 10.0, self.eval_expr);
                        let join = match eval_prop_str(get("join"), self.frame, "miter")
                            .to_ascii_lowercase()
                            .as_str()
                        {
                            "round" => StrokeJoin::Round,
                            "bevel" => StrokeJoin::Bevel,
                            _ => StrokeJoin::Miter,
                        };
                        let miter_limit =
                            eval_prop_f64(get("miterLimit"), self.frame, 4.0, self.eval_expr);
                        apply_offset_paths(&all_paths, amount, join, miter_limit, item_tol)
                    } else {
                        let mode =
                            eval_prop_str(get("mode"), self.frame, "add").to_ascii_lowercase();
                        apply_merge_paths(&all_paths, &mode)
                    };
                    for v in &views {
                        self.bases[v.base_idx].clear();
                    }
                    let new_base = self.alloc_base(changed);
                    let retarget_ok = !self.bases[new_base].is_empty();
                    fn retarget_cmds(cmds: &mut [ScopeCommand], m: Mat2D, new_base: usize) {
                        for cmd in cmds.iter_mut() {
                            match cmd {
                                ScopeCommand::Paint { views, .. } => {
                                    if let Some(inv) = mat_inv(m) {
                                        *views = vec![PathView {
                                            base_idx: new_base,
                                            matrix: inv,
                                            wiggle_stack: Vec::new(),
                                            char_override: None,
                                        }];
                                    }
                                }
                                ScopeCommand::Group {
                                    matrix, commands, ..
                                } => {
                                    retarget_cmds(commands, mat_mul(m, *matrix), new_base);
                                }
                            }
                        }
                    }
                    retarget_cmds(&mut commands, MAT_IDENTITY, new_base);
                    views = if retarget_ok {
                        vec![PathView {
                            base_idx: new_base,
                            matrix: MAT_IDENTITY,
                            wiggle_stack: Vec::new(),
                            char_override: None,
                        }]
                    } else {
                        Vec::new()
                    };
                }
                "wiggleTransform" => {
                    let wp = WiggleParams {
                        time: if self.fps > 0.0 {
                            self.frame / self.fps
                        } else {
                            0.0
                        },
                        frequency: eval_prop_f64(
                            get("frequency").or_else(|| get("wigglesPerSecond")),
                            self.frame,
                            1.0,
                            self.eval_expr,
                        ),
                        correlation: eval_prop_f64(
                            get("correlation"),
                            self.frame,
                            50.0,
                            self.eval_expr,
                        ),
                        temporal_phase: eval_prop_f64(
                            get("temporalPhase"),
                            self.frame,
                            0.0,
                            self.eval_expr,
                        ),
                        spatial_phase: eval_prop_f64(
                            get("spatialPhase"),
                            self.frame,
                            0.0,
                            self.eval_expr,
                        ),
                        seed: eval_prop_f64(get("seed"), self.frame, 1.0, self.eval_expr).round()
                            as i32,
                    };
                    let wcfg = WiggleTransformConfig {
                        wp,
                        anchor: eval_prop_vec2(
                            get("anchor"),
                            self.frame,
                            [0.0, 0.0],
                            self.eval_expr,
                        ),
                        position: eval_prop_vec2(
                            get("position"),
                            self.frame,
                            [20.0, 20.0],
                            self.eval_expr,
                        ),
                        scale: eval_prop_vec2(get("scale"), self.frame, [0.0, 0.0], self.eval_expr),
                        rotation: eval_prop_f64(get("rotation"), self.frame, 0.0, self.eval_expr),
                    };
                    for v in &mut views {
                        v.wiggle_stack.push((MAT_IDENTITY, wcfg));
                    }
                    let wm0 = eval_wiggle_transform_matrix(wcfg, 0);
                    let prev_cmds = std::mem::take(&mut commands);
                    commands.push(ScopeCommand::Group {
                        matrix: wm0,
                        wiggle: Some(wcfg),
                        opacity: 1.0,
                        commands: prev_cmds,
                    });
                }
                "trim" | "roundCorners" | "twist" | "zigZag" | "puckerBloat" | "wigglePaths" => {
                    let tagged: Vec<(usize, ShapePath)> = views
                        .iter()
                        .enumerate()
                        .flat_map(|(vi, v)| self.view_paths(v).into_iter().map(move |p| (vi, p)))
                        .collect();
                    let changed: Vec<(usize, ShapePath)> = match kind {
                        "trim" => {
                            let start =
                                eval_prop_f64(get("start"), self.frame, 0.0, self.eval_expr);
                            let end = eval_prop_f64(get("end"), self.frame, 100.0, self.eval_expr);
                            let offset =
                                eval_prop_f64(get("offset"), self.frame, 0.0, self.eval_expr);
                            let indiv = eval_prop_str(get("mode"), self.frame, "simultaneous")
                                .eq_ignore_ascii_case("individual");
                            apply_trim(&tagged, start, end, offset, indiv)
                        }
                        "roundCorners" => {
                            let radius =
                                eval_prop_f64(get("radius"), self.frame, 10.0, self.eval_expr);
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_round_corners(&p, radius, item_tol)))
                                .collect()
                        }
                        "twist" => {
                            let angle =
                                eval_prop_f64(get("angle"), self.frame, 0.0, self.eval_expr);
                            let center = eval_prop_vec2(
                                get("center"),
                                self.frame,
                                [0.0, 0.0],
                                self.eval_expr,
                            );
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_twist(&p, angle, center, item_tol)))
                                .collect()
                        }
                        "zigZag" => {
                            let size = eval_prop_f64(get("size"), self.frame, 10.0, self.eval_expr);
                            let ridges =
                                eval_prop_f64(get("ridges"), self.frame, 3.0, self.eval_expr);
                            let sm = eval_prop_str(get("points"), self.frame, "corner")
                                .eq_ignore_ascii_case("smooth");
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_zig_zag(&p, size, ridges, sm, item_tol)))
                                .collect()
                        }
                        "puckerBloat" => {
                            let amount =
                                eval_prop_f64(get("amount"), self.frame, 0.0, self.eval_expr);
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_pucker_bloat(&p, amount)))
                                .collect()
                        }
                        _ => {
                            let size = eval_prop_f64(get("size"), self.frame, 10.0, self.eval_expr);
                            let detail =
                                eval_prop_f64(get("detail"), self.frame, 5.0, self.eval_expr);
                            let sm = eval_prop_str(get("points"), self.frame, "corner")
                                .eq_ignore_ascii_case("smooth");
                            let wp = WiggleParams {
                                time: if self.fps > 0.0 {
                                    self.frame / self.fps
                                } else {
                                    0.0
                                },
                                frequency: eval_prop_f64(
                                    get("frequency").or_else(|| get("wigglesPerSecond")),
                                    self.frame,
                                    1.0,
                                    self.eval_expr,
                                ),
                                correlation: eval_prop_f64(
                                    get("correlation"),
                                    self.frame,
                                    50.0,
                                    self.eval_expr,
                                ),
                                temporal_phase: eval_prop_f64(
                                    get("temporalPhase"),
                                    self.frame,
                                    0.0,
                                    self.eval_expr,
                                ),
                                spatial_phase: eval_prop_f64(
                                    get("spatialPhase"),
                                    self.frame,
                                    0.0,
                                    self.eval_expr,
                                ),
                                seed: eval_prop_f64(get("seed"), self.frame, 1.0, self.eval_expr)
                                    .round() as i32,
                            };
                            tagged
                                .into_iter()
                                .map(|(o, p)| {
                                    (o, apply_wiggle_paths(&p, size, detail, sm, wp, item_tol))
                                })
                                .collect()
                        }
                    };

                    for (vi, view) in views.iter().enumerate() {
                        let m = view.matrix_for_copy(0);
                        let Some(inv) = mat_inv(m) else {
                            continue;
                        };
                        let updated: Vec<ShapePath> = changed
                            .iter()
                            .filter(|(o, _)| *o == vi)
                            .map(|(_, p)| p.transformed(inv))
                            .collect();
                        self.bases[view.base_idx] = updated;
                    }
                }
                "repeater" => {
                    let orig_views = std::mem::take(&mut views);
                    let orig_cmds = std::mem::take(&mut commands);
                    let copies = repeater_copies(props, self.frame, self.eval_expr);
                    let composite_above = eval_prop_str(get("composite"), self.frame, "below")
                        .eq_ignore_ascii_case("above");

                    for (copy_idx, copy) in copies.into_iter().enumerate() {
                        let ci = copy_idx as i32;
                        let mut base_remap: HashMap<usize, usize> = HashMap::new();
                        let clone_view = |v: &PathView,
                                          bases: &mut Vec<Vec<ShapePath>>,
                                          remap: &mut HashMap<usize, usize>|
                         -> PathView {
                            let new_base = *remap.entry(v.base_idx).or_insert_with(|| {
                                let cloned = bases[v.base_idx].clone();
                                let idx = bases.len();
                                bases.push(cloned);
                                idx
                            });
                            PathView {
                                base_idx: new_base,
                                matrix: v.matrix_for_copy(ci),
                                wiggle_stack: Vec::new(),
                                char_override: v.char_override,
                            }
                        };

                        fn clone_cmd(
                            cmd: &ScopeCommand,
                            ci: i32,
                            bases: &mut Vec<Vec<ShapePath>>,
                            remap: &mut HashMap<usize, usize>,
                        ) -> ScopeCommand {
                            match cmd {
                                ScopeCommand::Paint {
                                    views,
                                    kind,
                                    opacity,
                                    tolerance,
                                } => ScopeCommand::Paint {
                                    views: views
                                        .iter()
                                        .map(|v| {
                                            let nb =
                                                *remap.entry(v.base_idx).or_insert_with(|| {
                                                    let cloned = bases[v.base_idx].clone();
                                                    let idx = bases.len();
                                                    bases.push(cloned);
                                                    idx
                                                });
                                            PathView {
                                                base_idx: nb,
                                                matrix: v.matrix_for_copy(ci),
                                                wiggle_stack: Vec::new(),
                                                char_override: v.char_override,
                                            }
                                        })
                                        .collect(),
                                    kind: kind.clone(),
                                    opacity: *opacity,
                                    tolerance: *tolerance,
                                },
                                ScopeCommand::Group {
                                    matrix,
                                    wiggle,
                                    opacity,
                                    commands,
                                } => ScopeCommand::Group {
                                    matrix: wiggle
                                        .map_or(*matrix, |wc| eval_wiggle_transform_matrix(wc, ci)),
                                    wiggle: None,
                                    opacity: *opacity,
                                    commands: commands
                                        .iter()
                                        .map(|c| clone_cmd(c, ci, bases, remap))
                                        .collect(),
                                },
                            }
                        }

                        for ov in &orig_views {
                            let mut cv = clone_view(ov, &mut self.bases, &mut base_remap);
                            cv.matrix = mat_mul(copy.matrix, cv.matrix);
                            views.push(cv);
                        }
                        let cloned_cmds: Vec<ScopeCommand> = orig_cmds
                            .iter()
                            .map(|c| clone_cmd(c, ci, &mut self.bases, &mut base_remap))
                            .collect();
                        let grp = ScopeCommand::Group {
                            matrix: copy.matrix,
                            wiggle: None,
                            opacity: copy.opacity,
                            commands: cloned_cmds,
                        };
                        if composite_above {
                            commands.push(grp);
                        } else {
                            commands.insert(0, grp);
                        }
                    }
                }
                "rectangle" | "ellipse" | "polystar" | "path" => {
                    if let Some(p) =
                        build_primitive_path(item, self.frame, item_tol, self.eval_expr)
                    {
                        let base_idx = self.alloc_base(vec![p]);
                        views.push(PathView {
                            base_idx,
                            matrix: MAT_IDENTITY,
                            wiggle_stack: Vec::new(),
                            char_override: None,
                        });
                    }
                }
                _ => {}
            }
        }

        (views, commands)
    }

    pub(crate) fn lower_command(&self, cmd: &ScopeCommand) -> Vec<RenderCommand> {
        match cmd {
            ScopeCommand::Group {
                matrix,
                opacity,
                commands,
                ..
            } => {
                let lowered: Vec<RenderCommand> = commands
                    .iter()
                    .flat_map(|c| self.lower_command(c))
                    .map(|rc| rc.transformed(*matrix))
                    .collect();
                if lowered.is_empty() || *opacity <= 0.0 {
                    return Vec::new();
                }
                if *opacity < 1.0 - 1e-6 {
                    vec![RenderCommand::Isolation {
                        opacity: *opacity,
                        commands: lowered,
                    }]
                } else {
                    lowered
                }
            }
            ScopeCommand::Paint {
                views,
                kind,
                opacity,
                tolerance,
            } => {
                if views.is_empty() || *opacity <= 0.0 {
                    return Vec::new();
                }
                let has_char_ov = views.iter().any(|v| v.char_override.is_some());
                if has_char_ov {
                    let mut out = Vec::new();
                    for v in views {
                        let paths = self.view_paths(v);
                        if paths.is_empty() {
                            continue;
                        }
                        let ov = v.char_override;
                        let eff_op = opacity * ov.map_or(1.0, |o| o.opacity);
                        if eff_op <= 0.0 {
                            continue;
                        }
                        out.extend(self.lower_paint_paths(&paths, kind, eff_op, *tolerance, ov));
                    }
                    return out;
                }
                let paths: Vec<ShapePath> = views.iter().flat_map(|v| self.view_paths(v)).collect();
                if paths.is_empty() {
                    return Vec::new();
                }
                self.lower_paint_paths(&paths, kind, *opacity, *tolerance, None)
            }
        }
    }

    pub(crate) fn lower_paint_paths(
        &self,
        paths: &[ShapePath],
        kind: &PaintKind,
        opacity: f64,
        tolerance: f64,
        ov: Option<CharPaintOverride>,
    ) -> Vec<RenderCommand> {
        match kind {
            PaintKind::Fill { color, even_odd } => {
                let polygons: Vec<Vec<[f64; 2]>> = paths
                    .iter()
                    .filter(|p| p.points.len() >= 3)
                    .map(|p| p.points.clone())
                    .collect();
                if polygons.is_empty() {
                    return Vec::new();
                }
                let eff_color = ov.and_then(|o| o.fill_color).unwrap_or(*color);
                vec![RenderCommand::SolidFill {
                    polygons,
                    color: eff_color,
                    opacity,
                    even_odd: *even_odd,
                }]
            }
            PaintKind::GradientFill { gradient, even_odd } => {
                let polygons: Vec<Vec<[f64; 2]>> = paths
                    .iter()
                    .filter(|p| p.points.len() >= 3)
                    .map(|p| p.points.clone())
                    .collect();
                if polygons.is_empty() {
                    return Vec::new();
                }
                vec![RenderCommand::GradientFill {
                    polygons,
                    gradient: gradient.clone(),
                    opacity,
                    even_odd: *even_odd,
                }]
            }
            PaintKind::Stroke { color, style } => {
                let polygons = expand_stroke_polygons(paths, style, tolerance);
                if polygons.is_empty() {
                    return Vec::new();
                }
                let eff_color = ov.and_then(|o| o.fill_color).unwrap_or(*color);
                vec![RenderCommand::SolidFill {
                    polygons,
                    color: eff_color,
                    opacity,
                    even_odd: false,
                }]
            }
            PaintKind::GradientStroke { gradient, style } => {
                let polygons = expand_stroke_polygons(paths, style, tolerance);
                if polygons.is_empty() {
                    return Vec::new();
                }
                vec![RenderCommand::GradientFill {
                    polygons,
                    gradient: gradient.clone(),
                    opacity,
                    even_odd: false,
                }]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2D Antialiased Scanline Rasterizer (4x Vertical SS + Exact Horizontal Coverage)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) struct Edge {
    pub(crate) y_min: f64,
    pub(crate) y_max: f64,
    pub(crate) x_at_ymin: f64,
    pub(crate) dx_dy: f64,
    pub(crate) dir: i32,
}

pub(crate) fn build_scanline_edges(polygons: &[Vec<[f64; 2]>]) -> (Vec<Edge>, f64, f64) {
    let mut edges = Vec::new();
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for poly in polygons {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if (a[1] - b[1]).abs() < 1e-9 {
                continue;
            }
            let (top, bot, dir) = if a[1] < b[1] { (a, b, 1) } else { (b, a, -1) };
            min_y = min_y.min(top[1]);
            max_y = max_y.max(bot[1]);
            edges.push(Edge {
                y_min: top[1],
                y_max: bot[1],
                x_at_ymin: top[0],
                dx_dy: (bot[0] - top[0]) / (bot[1] - top[1]),
                dir,
            });
        }
    }
    (edges, min_y, max_y)
}

/// Computes exact horizontal span coverage averaged across 4 vertical sub-rows per scanline
/// and calls `blend_row(y, x0, &row_cov[x0..x1])` for every row with non-zero coverage.
pub(crate) fn rasterize_polygons_aa<F>(
    polygons: &[Vec<[f64; 2]>],
    width: usize,
    height: usize,
    even_odd: bool,
    mut blend_row: F,
) where
    F: FnMut(usize, usize, &[f32]),
{
    if width == 0 || height == 0 || polygons.is_empty() {
        return;
    }
    let (mut edges, min_y, max_y) = build_scanline_edges(polygons);
    if edges.is_empty() || !min_y.is_finite() || !max_y.is_finite() {
        return;
    }
    let y0 = (min_y.floor() as isize).clamp(0, height as isize) as usize;
    let y1 = (max_y.ceil() as isize).clamp(0, height as isize) as usize;
    if y0 >= y1 {
        return;
    }

    // Sort edges by y_min so each scanline maintains only active in-range edges.
    edges.sort_by(|a, b| a.y_min.total_cmp(&b.y_min));

    let mut row_cov = vec![0.0_f32; width];
    let mut active_edges: Vec<Edge> = Vec::new();
    let mut next_edge = 0usize;
    let mut crossings: Vec<(f64, i32)> = Vec::new();
    const SUB_SAMPLES: usize = 4;
    const SUB_WEIGHT: f32 = 1.0 / (SUB_SAMPLES as f32);
    const SUB_MIN_OFF: f64 = 0.5 / (SUB_SAMPLES as f64);
    const SUB_MAX_OFF: f64 = (SUB_SAMPLES as f64 - 0.5) / (SUB_SAMPLES as f64);
    let wf = width as f64;

    for y in y0..y1 {
        let sy_min = y as f64 + SUB_MIN_OFF;
        let sy_max = y as f64 + SUB_MAX_OFF;

        // Evict edges that end before this scanline's first sub-sample.
        active_edges.retain(|e| e.y_max > sy_min);
        // Activate edges whose y_min starts on or before this scanline's last sub-sample.
        while next_edge < edges.len() {
            let e = edges[next_edge];
            if e.y_min > sy_max {
                break;
            }
            if e.y_max > sy_min {
                active_edges.push(e);
            }
            next_edge += 1;
        }
        if active_edges.len() < 2 {
            continue;
        }

        let mut row_min_px = width;
        let mut row_max_px = 0usize;
        for sub in 0..SUB_SAMPLES {
            let ys = y as f64 + (sub as f64 + 0.5) / (SUB_SAMPLES as f64);
            crossings.clear();
            for e in &active_edges {
                if ys >= e.y_min && ys < e.y_max {
                    let x = e.x_at_ymin + (ys - e.y_min) * e.dx_dy;
                    crossings.push((x, e.dir));
                }
            }
            if crossings.len() < 2 {
                continue;
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut winding = 0i32;
            let mut inside = false;
            let mut span_start = 0.0_f64;

            for &(x, dir) in &crossings {
                let prev_in = inside;
                if even_odd {
                    inside = !inside;
                } else {
                    winding += dir;
                    inside = winding != 0;
                }
                if !prev_in && inside {
                    span_start = x;
                } else if prev_in && !inside {
                    let a = span_start.clamp(0.0, wf);
                    let b = x.clamp(0.0, wf);
                    if b > a {
                        let px0 = (a.floor() as usize).min(width - 1);
                        let px1 = ((b.ceil() as usize).saturating_sub(1)).min(width - 1);
                        row_min_px = row_min_px.min(px0);
                        row_max_px = row_max_px.max(px1 + 1);
                        if px0 == px1 {
                            row_cov[px0] += ((b - a) as f32) * SUB_WEIGHT;
                        } else {
                            row_cov[px0] += (((px0 + 1) as f64 - a) as f32) * SUB_WEIGHT;
                            for cov in &mut row_cov[(px0 + 1)..px1] {
                                *cov += SUB_WEIGHT;
                            }
                            row_cov[px1] += ((b - px1 as f64) as f32) * SUB_WEIGHT;
                        }
                    }
                }
            }
        }
        if row_min_px < row_max_px {
            blend_row(y, row_min_px, &row_cov[row_min_px..row_max_px]);
            row_cov[row_min_px..row_max_px].fill(0.0);
        }
    }
}

#[inline]
pub(crate) fn composite_premul_over(dst: &mut [f32; 4], src: [f32; 4]) {
    let inv = 1.0 - src[3];
    dst[0] = src[0] + dst[0] * inv;
    dst[1] = src[1] + dst[1] * inv;
    dst[2] = src[2] + dst[2] * inv;
    dst[3] = src[3] + dst[3] * inv;
}

pub(crate) fn execute_render_commands(
    commands: &[RenderCommand],
    display: Mat2D,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    for cmd in commands {
        match cmd {
            RenderCommand::Isolation { opacity, commands } => {
                let op = (*opacity as f32).clamp(0.0, 1.0);
                if op <= 0.0 || commands.is_empty() {
                    continue;
                }
                let mut local = vec![[0.0_f32; 4]; width * height];
                execute_render_commands(commands, display, width, height, &mut local);
                for (dst, src) in fb.iter_mut().zip(local.iter()) {
                    if src[3] > 0.0 {
                        let scaled = [src[0] * op, src[1] * op, src[2] * op, src[3] * op];
                        composite_premul_over(dst, scaled);
                    }
                }
            }
            RenderCommand::SolidFill {
                polygons,
                color,
                opacity,
                even_odd,
            } => {
                let op = (*opacity as f32).clamp(0.0, 1.0);
                if op <= 0.0 {
                    continue;
                }
                let px_polys: Vec<Vec<[f64; 2]>> = polygons
                    .iter()
                    .map(|p| p.iter().map(|&pt| mat_transform_pt(pt, display)).collect())
                    .collect();
                let cr = color[0] as f32;
                let cg = color[1] as f32;
                let cb = color[2] as f32;
                rasterize_polygons_aa(&px_polys, width, height, *even_odd, |y, x0, cov| {
                    let row_start = y * width + x0;
                    let row = &mut fb[row_start..row_start + cov.len()];
                    for (px, &c) in row.iter_mut().zip(cov.iter()) {
                        if c > 0.0 {
                            let a = (c * op).clamp(0.0, 1.0);
                            composite_premul_over(px, [cr * a, cg * a, cb * a, a]);
                        }
                    }
                });
            }
            RenderCommand::GradientFill {
                polygons,
                gradient,
                opacity,
                even_odd,
            } => {
                let op = (*opacity as f32).clamp(0.0, 1.0);
                if op <= 0.0 {
                    continue;
                }
                let total_mat = mat_mul(display, gradient.matrix);
                let Some(inv_total) = mat_inv(total_mat) else {
                    continue;
                };
                let px_polys: Vec<Vec<[f64; 2]>> = polygons
                    .iter()
                    .map(|p| p.iter().map(|&pt| mat_transform_pt(pt, display)).collect())
                    .collect();
                rasterize_polygons_aa(&px_polys, width, height, *even_odd, |y, x0, cov| {
                    let py = y as f64 + 0.5;
                    let row_start = y * width + x0;
                    let row = &mut fb[row_start..row_start + cov.len()];
                    for (dx, (px, &c)) in row.iter_mut().zip(cov.iter()).enumerate() {
                        if c > 0.0 {
                            let x = x0 + dx;
                            let sampled = gradient.sample_premul([x as f64 + 0.5, py], inv_total);
                            let mult = c * op;
                            let src = [
                                (sampled[0] as f32) * mult,
                                (sampled[1] as f32) * mult,
                                (sampled[2] as f32) * mult,
                                ((sampled[3] as f32) * mult).clamp(0.0, 1.0),
                            ];
                            composite_premul_over(px, src);
                        }
                    }
                });
            }
        }
    }
}

pub(crate) fn fb_to_premul_rgba8(fb: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(fb.len() * 4);
    for px in fb {
        let a = px[3].clamp(0.0, 1.0);
        let r = px[0].clamp(0.0, a);
        let g = px[1].clamp(0.0, a);
        let b = px[2].clamp(0.0, a);
        out.push((r * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
        out.push((g * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
        out.push((b * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
        out.push((a * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
    }
    out
}

/// Compiles and rasterizes a full Shape+ JSON payload into a premultiplied RGBA8 buffer
/// of length `width * height * 4`.
///
/// Sequence space has `(0, 0)` at the center of the frame with `+x` right and `+y` up
/// (matching Zoidium `viewport-mapping.js` and `zc-render`'s 2D layer convention).
#[must_use]
pub fn rasterize_shape_plus(
    payload: &Value,
    width: u32,
    height: u32,
    seq_res: [u32; 2],
    frame: f64,
    fps: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let Some(contents) = payload.get("contents").and_then(Value::as_array) else {
        return vec![0_u8; w * h * 4];
    };

    let seq_w = seq_res[0].max(1) as f64;
    let seq_h = seq_res[1].max(1) as f64;
    let sx = (w as f64) / seq_w;
    let sy = (h as f64) / seq_h;
    let pixel_scale = sx.max(sy).max(1e-6);
    let tolerance = (DEFAULT_TOLERANCE / pixel_scale).clamp(1e-3, 1.0);
    // Display matrix: sequence (center origin, y up) -> pixel (top-left origin, y down).
    let display: Mat2D = [sx, 0.0, 0.0, -sy, (w as f64) * 0.5, (h as f64) * 0.5];

    let mut compiler = Compiler {
        frame: if frame.is_finite() { frame } else { 0.0 },
        fps: if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        },
        eval_expr,
        bases: Vec::new(),
    };
    let (_views, scope_cmds) = compiler.compile_scope(contents, tolerance, 0, &[]);
    let render_cmds: Vec<RenderCommand> = scope_cmds
        .iter()
        .flat_map(|c| compiler.lower_command(c))
        .collect();

    let mut fb = vec![[0.0_f32; 4]; w * h];
    execute_render_commands(&render_cmds, display, w, h, &mut fb);
    fb_to_premul_rgba8(&fb)
}
