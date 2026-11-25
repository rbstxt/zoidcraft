use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EffectorKind {
    Shake,
    Plain,
    Random,
    Formula,
    Twist,
    Wrap,
    Orbit,
    Spline,
    Delay,
    Repeater,
    Voronoi,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct EffectorPropKeys {
    pub(super) letters: &'static str,
    pub(super) delay_order: &'static str,
    pub(super) seed: &'static str,
    pub(super) delay: &'static str,
    pub(super) strength: &'static str,
    pub(super) field: &'static str,
    pub(super) field_position: &'static str,
    pub(super) field_scale: &'static str,
}

macro_rules! effector_prop_keys {
    ($prefix:literal) => {
        EffectorPropKeys {
            letters: concat!($prefix, "Letters"),
            delay_order: concat!($prefix, "DelayOrder"),
            seed: concat!($prefix, "Seed"),
            delay: concat!($prefix, "Delay"),
            strength: concat!($prefix, "Strength"),
            field: concat!($prefix, "Field"),
            field_position: concat!($prefix, "FieldPosition"),
            field_scale: concat!($prefix, "FieldScale"),
        }
    };
}

impl EffectorKind {
    pub(super) const fn prop_keys(self) -> EffectorPropKeys {
        match self {
            Self::Shake => effector_prop_keys!("shake"),
            Self::Plain => effector_prop_keys!("plain"),
            Self::Random => effector_prop_keys!("random"),
            Self::Formula => effector_prop_keys!("formula"),
            Self::Twist => effector_prop_keys!("twist"),
            Self::Wrap => effector_prop_keys!("wrap"),
            Self::Orbit => effector_prop_keys!("orbit"),
            Self::Spline => effector_prop_keys!("spline"),
            Self::Delay => effector_prop_keys!("delay"),
            Self::Repeater => effector_prop_keys!("repeater"),
            Self::Voronoi => effector_prop_keys!("voronoi"),
        }
    }

    /// Canonical Zoidium panel order for all 11 ExpressText+ effectors.
    pub const ALL: [Self; 11] = [
        Self::Shake,
        Self::Plain,
        Self::Random,
        Self::Formula,
        Self::Delay,
        Self::Spline,
        Self::Orbit,
        Self::Repeater,
        Self::Twist,
        Self::Wrap,
        Self::Voronoi,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Shake => "shake",
            Self::Plain => "plain",
            Self::Random => "random",
            Self::Formula => "formula",
            Self::Twist => "twist",
            Self::Wrap => "wrap",
            Self::Orbit => "orbit",
            Self::Spline => "spline",
            Self::Delay => "delay",
            Self::Repeater => "repeater",
            Self::Voronoi => "voronoi",
        }
    }

    /// Parses a short effector name (`"repeater"`), display name (`"Repeater"`),
    /// or full Zoidium object type (`"zoidium:expresstext-plus/repeater"`).
    pub fn from_name(raw: &str) -> Option<Self> {
        let trimmed = raw.trim();
        let tail = trimmed.rsplit('/').next().unwrap_or(trimmed);
        let head = tail.split_whitespace().next().unwrap_or(tail);
        match head.to_ascii_lowercase().as_str() {
            "shake" => Some(Self::Shake),
            "plain" => Some(Self::Plain),
            "random" => Some(Self::Random),
            "formula" => Some(Self::Formula),
            "twist" => Some(Self::Twist),
            "wrap" => Some(Self::Wrap),
            "orbit" => Some(Self::Orbit),
            "spline" => Some(Self::Spline),
            "delay" => Some(Self::Delay),
            "repeater" => Some(Self::Repeater),
            "voronoi" => Some(Self::Voronoi),
            _ => None,
        }
    }

    /// Infers the effector kind from a prefixed property key (`"repeaterCopies"` -> `Repeater`),
    /// matching Zoidium `expresstext-sections.js:kindOfKey`.
    pub fn from_property_key(key: &str) -> Option<Self> {
        for kind in Self::ALL {
            let prefix = kind.as_str();
            if key == prefix {
                return Some(kind);
            }
            if let Some(rest) = key.strip_prefix(prefix)
                && rest.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
            {
                return Some(kind);
            }
        }
        None
    }
}

/// Container letter numbering scope (`expresstext-constants.js` / `expresstext-layout.js`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LettersMode {
    /// Each 3D text child gets its own independent `0..len` letter run.
    #[default]
    PerText = 0,
    /// All 3D text children under the ExpressText+ container share one connected letter run.
    AcrossTexts = 1,
}

impl LettersMode {
    /// Matches Zoidium `lettersScopeOf`: explicit `0` is `PerText`, everything else fails
    /// closed to `AcrossTexts`.
    pub fn from_raw(value: f64) -> Self {
        if value.is_finite() && js_round(value) as i32 == 0 {
            Self::PerText
        } else {
            Self::AcrossTexts
        }
    }
}

/// Builds the `(LetterList, self_start_global_index)` for `self_text_index` given the letter
/// counts of every sibling text object under an ExpressText+ container.
pub fn build_letter_run_for_text(
    mode: LettersMode,
    text_letter_counts: &[usize],
    self_text_index: usize,
) -> (LetterList, usize) {
    let self_count = text_letter_counts
        .get(self_text_index)
        .copied()
        .unwrap_or(0);
    match mode {
        LettersMode::PerText => (
            LetterList {
                total: self_count,
                order: (0..self_count).collect(),
            },
            0,
        ),
        LettersMode::AcrossTexts => {
            let total: usize = text_letter_counts.iter().copied().sum();
            let self_start: usize = text_letter_counts
                .iter()
                .take(self_text_index)
                .copied()
                .sum();
            (
                LetterList {
                    total,
                    order: (0..total).collect(),
                },
                self_start,
            )
        }
    }
}

/// One effector layer in an ExpressText+ stack.
pub struct EffectorLayer<'a> {
    pub kind: EffectorKind,
    pub props: &'a dyn EtPropReader,
}

/// Ordering metadata for the letters in a text run.
#[derive(Debug, Clone)]
pub struct LetterList {
    pub total: usize,
    pub order: Vec<usize>,
}

/// Rigid transform descriptor for layout<->world conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TextTransform {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl Default for TextTransform {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// Auxiliary line metadata shared across letters.
#[derive(Debug, Clone)]
pub struct EvalAux {
    pub rate: f64,
    pub extent: Option<(f64, f64)>,
    pub bases: Option<Vec<[f64; 3]>>,
    pub self_world: Option<TextTransform>,
}

impl Default for EvalAux {
    fn default() -> Self {
        Self {
            rate: 30.0,
            extent: None,
            bases: None,
            self_world: None,
        }
    }
}

/// Per-letter evaluated effector output (`position` delta, `rotation` delta radians, `scale` multiplier).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterControls {
    pub position: [f64; 3],
    pub rotation: [f64; 3],
    pub scale: [f64; 3],
}

impl Default for LetterControls {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}
