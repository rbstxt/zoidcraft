use super::*;

pub const FORMULA_DEG: f64 = PI / 180.0;

pub const DELAY_EVEN_SPAN: f64 = 2.5;

pub const DELAY_BLEND_CATCH: f64 = 0.25;

pub const DELAY_SPRING_STIFF: f64 = 0.08;

pub const DELAY_SPRING_DAMP: f64 = 0.9;

pub const DELAY_SPRING_DAMP_MIN: f64 = 0.5;

pub const DELAY_SIM_WINDOW: usize = 7200;

pub const REPEATER_MAX_COPIES: usize = 64;

pub const VORONOI_MAX_CHUNKS: usize = 64;

/// Property value stored in an ExpressText+ effector layer.
#[derive(Debug, Clone, PartialEq)]
pub enum EtValue {
    Number(f64),
    Vec3([f64; 3]),
    Text(String),
}

impl EtValue {
    pub fn as_number(&self, fallback: f64) -> f64 {
        match self {
            Self::Number(n) if n.is_finite() => *n,
            Self::Text(s) => {
                let t = s.trim();
                if t.is_empty() {
                    0.0
                } else {
                    t.parse::<f64>()
                        .ok()
                        .filter(|v| v.is_finite())
                        .unwrap_or(fallback)
                }
            }
            _ => fallback,
        }
    }

    pub fn as_vec3(&self, fallback: [f64; 3]) -> [f64; 3] {
        match self {
            Self::Vec3(v) => [
                if v[0].is_finite() { v[0] } else { fallback[0] },
                if v[1].is_finite() { v[1] } else { fallback[1] },
                if v[2].is_finite() { v[2] } else { fallback[2] },
            ],
            _ => fallback,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Property map for one ExpressText+ layer (either static values or frame-sampled callbacks).
pub trait EtPropReader {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue>;

    /// Returns `true` if `key` is guaranteed constant across frames (enabling hoisting
    /// across Delay simulation loops). Defaults to `false` for dynamic readers.
    fn is_static(&self, _key: &str) -> bool {
        false
    }
}

impl EtPropReader for BTreeMap<String, EtValue> {
    fn read(&self, key: &str, _frame: f64) -> Option<EtValue> {
        self.get(key).cloned()
    }

    fn is_static(&self, _key: &str) -> bool {
        true
    }
}

pub(super) fn read_num(props: &dyn EtPropReader, key: &str, frame: f64, fallback: f64) -> f64 {
    props
        .read(key, frame)
        .map_or(fallback, |v| v.as_number(fallback))
}

pub(super) fn read_vec3(
    props: &dyn EtPropReader,
    key: &str,
    frame: f64,
    fallback: [f64; 3],
) -> [f64; 3] {
    props
        .read(key, frame)
        .map_or(fallback, |v| v.as_vec3(fallback))
}

pub(super) fn read_str(props: &dyn EtPropReader, key: &str, frame: f64, fallback: &str) -> String {
    props
        .read(key, frame)
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_else(|| fallback.to_string())
}

// ---------------------------------------------------------------------------
// Seeded PRNG, 3D Lattice / Fractal Value Noise & Stagger Ranks
// (`expresstext-random.js`)
