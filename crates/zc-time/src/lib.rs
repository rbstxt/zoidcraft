//! Temporal primitives for ZoidCraft: frame rates, sub-frame timestamps,
//! high-precision integer ticks, CM3 timecode formatting/parsing, and audio sample timing.

use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign};

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// High-precision tick rate: `254,016,000,000` ticks per second (`2^12 * 3^4 * 5^6 * 7^2`).
///
/// Divides losslessly by standard integer frame rates (24, 25, 30, 48, 50, 60, 120, etc.)
/// as well as NTSC rational frame rates (`24000/1001`, `30000/1001`, `60000/1001`) and
/// common audio sample rates (`48_000` Hz, `44_100` Hz).
pub const TICKS_PER_SECOND: i64 = 254_016_000_000;

/// Minimum allowed sequence frame rate in frames per second.
pub const MIN_FPS: u32 = 1;

/// Maximum allowed sequence frame rate in frames per second.
pub const MAX_FPS: u32 = 120;

/// Default CM3 / ZoidCraft sequence frame rate in frames per second.
pub const DEFAULT_FPS: u32 = 30;

/// Default CM3 / ZoidCraft audio sample rate in Hertz.
pub const DEFAULT_AUDIO_SAMPLE_RATE: u32 = 48_000;

/// Standard stereo channel count for ZoidCraft audio output.
pub const AUDIO_CHANNELS: usize = 2;

/// Errors returned when validating time-related values or parsing timecodes.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum TimeError {
    /// Frame rate is outside the supported `1..=120` range.
    #[error("invalid frame rate {value}: expected integer in {MIN_FPS}..={MAX_FPS}")]
    InvalidFrameRate { value: u32 },

    /// Audio sample rate is zero.
    #[error("invalid audio sample rate {value}: must be greater than zero")]
    InvalidSampleRate { value: u32 },

    /// Timecode string could not be parsed for the requested format.
    #[error("invalid timecode '{input}': {reason}")]
    InvalidTimecode { input: String, reason: &'static str },
}

/// Integer sequence frame rate constrained to `1..=120` fps (default `30`).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct FrameRate(u32);

impl FrameRate {
    /// Default CM3 sequence frame rate (`30` fps).
    pub const DEFAULT: Self = Self(DEFAULT_FPS);

    /// Common 24 fps cinema rate.
    pub const FPS_24: Self = Self(24);

    /// Common 30 fps rate.
    pub const FPS_30: Self = Self(30);

    /// Common 60 fps rate.
    pub const FPS_60: Self = Self(60);

    /// Maximum 120 fps rate.
    pub const FPS_120: Self = Self(120);

    /// Creates a validated frame rate in `1..=120`.
    pub fn new(fps: u32) -> Result<Self, TimeError> {
        if (MIN_FPS..=MAX_FPS).contains(&fps) {
            Ok(Self(fps))
        } else {
            Err(TimeError::InvalidFrameRate { value: fps })
        }
    }

    /// Creates a frame rate by clamping `fps` into `1..=120`.
    pub const fn new_clamped(fps: u32) -> Self {
        if fps < MIN_FPS {
            Self(MIN_FPS)
        } else if fps > MAX_FPS {
            Self(MAX_FPS)
        } else {
            Self(fps)
        }
    }

    /// Returns the integer frames per second.
    pub const fn fps(self) -> u32 {
        self.0
    }

    /// Returns the frame rate as `f64`.
    pub const fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Returns the duration of one frame in seconds (`1.0 / fps`).
    pub fn frame_duration_secs(self) -> f64 {
        1.0 / self.as_f64()
    }

    /// Returns the number of ticks per frame (rounded to nearest tick for non-divisors).
    pub const fn ticks_per_frame(self) -> i64 {
        let den = self.0 as i128;
        let num = TICKS_PER_SECOND as i128;
        ((num + den / 2) / den) as i64
    }

    /// Returns `true` if `TICKS_PER_SECOND` is evenly divisible by this frame rate.
    pub const fn is_exact_tick_rate(self) -> bool {
        (TICKS_PER_SECOND % self.0 as i64) == 0
    }
}

impl Default for FrameRate {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<u32> for FrameRate {
    type Error = TimeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<FrameRate> for u32 {
    fn from(rate: FrameRate) -> Self {
        rate.0
    }
}

impl fmt::Display for FrameRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} fps", self.0)
    }
}

/// Exact floating-point frame position (supports sub-frame motion blur and keyframe evaluation).
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Frame(pub f64);

impl Frame {
    /// Zero frame (`0.0`).
    pub const ZERO: Self = Self(0.0);

    /// Creates a `Frame` from an `f64`.
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    /// Creates an exact whole `Frame` from an `i64`.
    pub const fn from_i64(frame: i64) -> Self {
        Self(frame as f64)
    }

    /// Returns the underlying `f64` value.
    pub const fn as_f64(self) -> f64 {
        self.0
    }

    /// Returns the floored integer frame index.
    pub fn floor_i64(self) -> i64 {
        self.0.floor() as i64
    }

    /// Returns the rounded integer frame index.
    pub fn round_i64(self) -> i64 {
        self.0.round() as i64
    }

    /// Returns the ceiling integer frame index.
    pub fn ceil_i64(self) -> i64 {
        self.0.ceil() as i64
    }

    /// Returns the truncated integer frame index.
    pub fn trunc_i64(self) -> i64 {
        self.0.trunc() as i64
    }

    /// Clamps negative or non-finite frame positions to `0.0`.
    pub fn clamp_non_negative(self) -> Self {
        if self.0.is_finite() && self.0 > 0.0 {
            self
        } else {
            Self::ZERO
        }
    }

    /// Clamps this frame into `[min, max]`.
    pub fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    /// Converts this frame position to seconds at the given frame rate.
    pub fn to_seconds(self, rate: FrameRate) -> f64 {
        self.0 / rate.as_f64()
    }

    /// Converts a time in seconds to a `Frame` at the given frame rate.
    pub fn from_seconds(seconds: f64, rate: FrameRate) -> Self {
        Self(seconds * rate.as_f64())
    }

    /// Converts this frame to high-precision integer `Tick`s.
    pub fn to_tick(self, rate: FrameRate) -> Tick {
        Tick::from_frame(self, rate)
    }

    /// Converts high-precision integer `Tick`s to a `Frame`.
    pub fn from_tick(tick: Tick, rate: FrameRate) -> Self {
        tick.to_frame(rate)
    }

    /// Converts a sequence frame to a clip-local frame (`self - clip_start`).
    pub fn to_local(self, clip_start: Self) -> Self {
        Self(self.0 - clip_start.0)
    }

    /// Computes CM3 multisample motion blur sub-frame offset:
    /// `f_local(s) = frame - clip_start + (sample_index / sample_count) * shutter`.
    pub fn motion_blur_subframe(
        self,
        clip_start: Self,
        sample_index: u32,
        sample_count: u32,
        shutter: f64,
    ) -> Self {
        let n = sample_count.max(1) as f64;
        Self((self.0 - clip_start.0) + (f64::from(sample_index) / n) * shutter)
    }
}

impl From<f64> for Frame {
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<i64> for Frame {
    fn from(value: i64) -> Self {
        Self(value as f64)
    }
}

impl From<u32> for Frame {
    fn from(value: u32) -> Self {
        Self(f64::from(value))
    }
}

impl Add for Frame {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Frame {
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Frame {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Frame {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Neg for Frame {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(-self.0)
    }
}

impl Mul<f64> for Frame {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self::Output {
        Self(self.0 * rhs)
    }
}

impl Div<f64> for Frame {
    type Output = Self;

    fn div(self, rhs: f64) -> Self::Output {
        Self(self.0 / rhs)
    }
}

impl fmt::Display for Frame {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// High-precision integer timestamp measured in `1 / 254_016_000_000` second units.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct Tick(pub i64);

impl Tick {
    /// Zero ticks.
    pub const ZERO: Self = Self(0);

    /// Number of ticks in one second (`254,016,000,000`).
    pub const PER_SECOND: i64 = TICKS_PER_SECOND;

    /// Number of ticks in one millisecond (`254,016,000`).
    pub const PER_MILLISECOND: i64 = TICKS_PER_SECOND / 1_000;

    /// Creates a `Tick` from a raw tick count.
    pub const fn new(ticks: i64) -> Self {
        Self(ticks)
    }

    /// Returns the raw tick count.
    pub const fn as_i64(self) -> i64 {
        self.0
    }

    /// Clamps negative ticks to `0`.
    pub const fn clamp_non_negative(self) -> Self {
        if self.0 < 0 { Self::ZERO } else { self }
    }

    /// Converts seconds to the nearest integer `Tick`.
    pub fn from_seconds(seconds: f64) -> Self {
        if !seconds.is_finite() {
            return Self::ZERO;
        }
        Self((seconds * TICKS_PER_SECOND as f64).round() as i64)
    }

    /// Converts ticks to seconds.
    pub fn to_seconds(self) -> f64 {
        self.0 as f64 / TICKS_PER_SECOND as f64
    }

    /// Converts whole milliseconds to exact ticks.
    pub const fn from_millis(millis: i64) -> Self {
        Self(millis.saturating_mul(Self::PER_MILLISECOND))
    }

    /// Converts ticks to whole milliseconds (truncating toward zero).
    pub const fn to_millis(self) -> i64 {
        self.0 / Self::PER_MILLISECOND
    }

    /// Converts an exact whole frame count at `rate` into ticks using 128-bit integer arithmetic.
    pub const fn from_whole_frame(frame: i64, rate: FrameRate) -> Self {
        let num = (frame as i128) * (TICKS_PER_SECOND as i128);
        let den = rate.fps() as i128;
        let ticks = div_round_i128(num, den);
        if ticks > i64::MAX as i128 {
            Self(i64::MAX)
        } else if ticks < i64::MIN as i128 {
            Self(i64::MIN)
        } else {
            Self(ticks as i64)
        }
    }

    /// Converts a whole frame count at a rational frame rate `fps_num / fps_den`
    /// (for example NTSC `24000 / 1001`, `30000 / 1001`, or `60000 / 1001`) into ticks.
    ///
    /// Returns `None` if `fps_num == 0`, `fps_den == 0`, or the result overflows `i64`.
    pub fn from_rational_frame(frame: i64, fps_num: u32, fps_den: u32) -> Option<Self> {
        if fps_num == 0 || fps_den == 0 {
            return None;
        }
        let num = i128::from(frame)
            .checked_mul(i128::from(TICKS_PER_SECOND))?
            .checked_mul(i128::from(fps_den))?;
        let ticks = checked_div_round_i128(num, i128::from(fps_num))?;
        i64::try_from(ticks).ok().map(Self)
    }

    /// Converts a potentially fractional `Frame` at `rate` into ticks while preserving
    /// full 64-bit integer precision on the whole-frame part.
    pub fn from_frame(frame: Frame, rate: FrameRate) -> Self {
        if !frame.0.is_finite() {
            return Self::ZERO;
        }
        let whole = frame.0.trunc();
        let frac = frame.0.fract();
        let whole_ticks = Self::from_whole_frame(whole as i64, rate).0;
        let frac_ticks = (frac * (TICKS_PER_SECOND as f64 / rate.as_f64())).round() as i64;
        Self(whole_ticks.saturating_add(frac_ticks))
    }

    /// Converts ticks to a `Frame` at `rate` with split whole/remainder precision.
    pub fn to_frame(self, rate: FrameRate) -> Frame {
        let scaled = i128::from(self.0) * i128::from(rate.fps());
        let tps = i128::from(TICKS_PER_SECOND);
        let whole = scaled / tps;
        let rem = scaled % tps;
        Frame(whole as f64 + (rem as f64) / (TICKS_PER_SECOND as f64))
    }

    /// Converts ticks to the nearest whole frame integer at `rate`.
    pub const fn to_whole_frame_round(self, rate: FrameRate) -> i64 {
        let scaled = (self.0 as i128) * (rate.fps() as i128);
        div_round_i128(scaled, TICKS_PER_SECOND as i128) as i64
    }

    /// Converts ticks to the nearest whole frame integer at a rational frame rate `fps_num / fps_den`.
    pub fn to_rational_frame_round(self, fps_num: u32, fps_den: u32) -> Option<i64> {
        if fps_num == 0 || fps_den == 0 {
            return None;
        }
        let num = i128::from(self.0) * i128::from(fps_num);
        let den = i128::from(TICKS_PER_SECOND) * i128::from(fps_den);
        i64::try_from(div_round_i128(num, den)).ok()
    }
}

impl Add for Tick {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_add(rhs.0))
    }
}

impl AddAssign for Tick {
    fn add_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_add(rhs.0);
    }
}

impl Sub for Tick {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.saturating_sub(rhs.0))
    }
}

impl SubAssign for Tick {
    fn sub_assign(&mut self, rhs: Self) {
        self.0 = self.0.saturating_sub(rhs.0);
    }
}

impl Neg for Tick {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self(self.0.saturating_neg())
    }
}

impl Mul<i64> for Tick {
    type Output = Self;

    fn mul(self, rhs: i64) -> Self::Output {
        Self(self.0.saturating_mul(rhs))
    }
}

impl Div<i64> for Tick {
    type Output = Self;

    fn div(self, rhs: i64) -> Self::Output {
        Self(self.0.saturating_div(rhs))
    }
}

impl fmt::Display for Tick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}t", self.0)
    }
}

const fn div_round_i128(num: i128, den: i128) -> i128 {
    let half = den / 2;
    if num >= 0 {
        (num + half) / den
    } else {
        (num - half) / den
    }
}

fn checked_div_round_i128(num: i128, den: i128) -> Option<i128> {
    if den <= 0 {
        return None;
    }
    let half = den / 2;
    let rounded = if num >= 0 {
        num.checked_add(half)?
    } else {
        num.checked_sub(half)?
    };
    Some(rounded / den)
}

/// CM3 timeline timecode display format (cycled modulo 3 on right-click).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeFormat {
    /// Mode `0`: `HH:MM:SS.mmm` (hours, minutes, seconds, milliseconds).
    HmsMillis,
    /// Mode `1`: `HH:MM:SS:FF` (SMPTE non-drop-frame timecode).
    Smpte,
    /// Mode `2`: integer frame number (CM3 default).
    #[default]
    Frames,
}

impl TimeFormat {
    /// Cycles to the next CM3 display format modulo 3:
    /// `HmsMillis (0) -> Smpte (1) -> Frames (2) -> HmsMillis (0)`.
    pub const fn next(self) -> Self {
        match self {
            Self::HmsMillis => Self::Smpte,
            Self::Smpte => Self::Frames,
            Self::Frames => Self::HmsMillis,
        }
    }

    /// Converts a CM3 format index (`0`, `1`, or `2`) to a `TimeFormat`.
    pub const fn from_index(index: u8) -> Option<Self> {
        match index {
            0 => Some(Self::HmsMillis),
            1 => Some(Self::Smpte),
            2 => Some(Self::Frames),
            _ => None,
        }
    }

    /// Returns the CM3 numeric format index (`0`, `1`, or `2`).
    pub const fn to_index(self) -> u8 {
        match self {
            Self::HmsMillis => 0,
            Self::Smpte => 1,
            Self::Frames => 2,
        }
    }

    /// Formats `frame` at `rate` according to this timecode format.
    ///
    /// Negative or non-finite frame values are clamped to `0`.
    pub fn format_frame(self, frame: Frame, rate: FrameRate) -> String {
        let clamped = frame.clamp_non_negative().as_f64();
        match self {
            Self::HmsMillis => {
                let total_ms = ((clamped / rate.as_f64()) * 1_000.0).round() as u64;
                let ms = total_ms % 1_000;
                let total_secs = total_ms / 1_000;
                let secs = total_secs % 60;
                let total_mins = total_secs / 60;
                let mins = total_mins % 60;
                let hours = total_mins / 60;
                format!("{hours:02}:{mins:02}:{secs:02}.{ms:03}")
            }
            Self::Smpte => {
                let fps = u64::from(rate.fps());
                let total_frames = (clamped + 1e-9).floor() as u64;
                let ff = total_frames % fps;
                let total_secs = total_frames / fps;
                let secs = total_secs % 60;
                let total_mins = total_secs / 60;
                let mins = total_mins % 60;
                let hours = total_mins / 60;
                format!("{hours:02}:{mins:02}:{secs:02}:{ff:02}")
            }
            Self::Frames => {
                let total_frames = (clamped + 1e-9).floor() as u64;
                total_frames.to_string()
            }
        }
    }

    /// Formats a `Tick` timestamp at `rate` according to this timecode format.
    pub fn format_tick(self, tick: Tick, rate: FrameRate) -> String {
        self.format_frame(tick.to_frame(rate), rate)
    }

    /// Parses a timecode string strictly matching this format into a `Frame` at `rate`.
    pub fn parse(self, input: &str, rate: FrameRate) -> Result<Frame, TimeError> {
        let trimmed = input.trim();
        if trimmed.is_empty() {
            return Err(TimeError::InvalidTimecode {
                input: input.to_string(),
                reason: "empty timecode",
            });
        }
        match self {
            Self::HmsMillis => parse_hms_millis(trimmed, rate),
            Self::Smpte => parse_smpte(trimmed, rate),
            Self::Frames => parse_frames(trimmed),
        }
    }

    /// Auto-detects the format of `input` and parses it into `(Frame, TimeFormat)`.
    pub fn parse_auto(input: &str, rate: FrameRate) -> Result<(Frame, Self), TimeError> {
        let trimmed = input.trim();
        if trimmed.contains('.') {
            Ok((Self::HmsMillis.parse(trimmed, rate)?, Self::HmsMillis))
        } else if trimmed.contains(':') {
            Ok((Self::Smpte.parse(trimmed, rate)?, Self::Smpte))
        } else {
            Ok((Self::Frames.parse(trimmed, rate)?, Self::Frames))
        }
    }
}

fn parse_unsigned_digits(
    part: &str,
    full_input: &str,
    reason: &'static str,
) -> Result<u64, TimeError> {
    if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TimeError::InvalidTimecode {
            input: full_input.to_string(),
            reason,
        });
    }
    part.parse::<u64>().map_err(|_| TimeError::InvalidTimecode {
        input: full_input.to_string(),
        reason,
    })
}

fn parse_hms_millis(input: &str, rate: FrameRate) -> Result<Frame, TimeError> {
    let (hms, ms_str) = input
        .split_once('.')
        .ok_or_else(|| TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "expected HH:MM:SS.mmm format",
        })?;
    if ms_str.is_empty() || ms_str.len() > 3 || !ms_str.bytes().all(|b| b.is_ascii_digit()) {
        return Err(TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "milliseconds must be 1 to 3 digits",
        });
    }
    let parts: Vec<&str> = hms.split(':').collect();
    if parts.len() != 3 {
        return Err(TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "expected HH:MM:SS.mmm format",
        });
    }
    let hours = parse_unsigned_digits(parts[0], input, "invalid hours")?;
    let mins = parse_unsigned_digits(parts[1], input, "invalid minutes")?;
    let secs = parse_unsigned_digits(parts[2], input, "invalid seconds")?;
    if mins >= 60 || secs >= 60 {
        return Err(TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "minutes and seconds must be < 60",
        });
    }
    let raw_ms = parse_unsigned_digits(ms_str, input, "invalid milliseconds")?;
    let ms = match ms_str.len() {
        1 => raw_ms * 100,
        2 => raw_ms * 10,
        _ => raw_ms,
    };
    let whole_secs = hours
        .checked_mul(3_600)
        .and_then(|v| v.checked_add(mins * 60 + secs))
        .ok_or_else(|| TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "timecode overflow",
        })?;
    let total_secs = whole_secs as f64 + (ms as f64) / 1_000.0;
    Ok(Frame::from_seconds(total_secs, rate))
}

fn parse_smpte(input: &str, rate: FrameRate) -> Result<Frame, TimeError> {
    let parts: Vec<&str> = input.split(':').collect();
    if parts.len() != 4 {
        return Err(TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "expected HH:MM:SS:FF format",
        });
    }
    let hours = parse_unsigned_digits(parts[0], input, "invalid hours")?;
    let mins = parse_unsigned_digits(parts[1], input, "invalid minutes")?;
    let secs = parse_unsigned_digits(parts[2], input, "invalid seconds")?;
    let frames = parse_unsigned_digits(parts[3], input, "invalid frame field")?;
    if mins >= 60 || secs >= 60 {
        return Err(TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "minutes and seconds must be < 60",
        });
    }
    let fps = u64::from(rate.fps());
    if frames >= fps {
        return Err(TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "frame field must be < frame rate",
        });
    }
    let total_frames = hours
        .checked_mul(3_600)
        .and_then(|v| v.checked_add(mins * 60 + secs))
        .and_then(|v| v.checked_mul(fps))
        .and_then(|v| v.checked_add(frames))
        .ok_or_else(|| TimeError::InvalidTimecode {
            input: input.to_string(),
            reason: "timecode overflow",
        })?;
    Ok(Frame::new(total_frames as f64))
}

fn parse_frames(input: &str) -> Result<Frame, TimeError> {
    let n = parse_unsigned_digits(input, input, "expected non-negative integer frame number")?;
    Ok(Frame::new(n as f64))
}

/// Audio sample rate in Hertz (default `48,000 Hz` per CM3 / ZoidCraft specification).
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "u32", into = "u32")]
pub struct AudioSampleRate(u32);

impl AudioSampleRate {
    /// Default CM3 / ZoidCraft audio sample rate (`48,000 Hz`).
    pub const DEFAULT: Self = Self(DEFAULT_AUDIO_SAMPLE_RATE);

    /// Creates a validated non-zero audio sample rate.
    pub fn new(hz: u32) -> Result<Self, TimeError> {
        if hz > 0 {
            Ok(Self(hz))
        } else {
            Err(TimeError::InvalidSampleRate { value: hz })
        }
    }

    /// Returns the sample rate in Hertz.
    pub const fn hz(self) -> u32 {
        self.0
    }

    /// Returns the sample rate as `f64`.
    pub const fn as_f64(self) -> f64 {
        self.0 as f64
    }

    /// Returns the duration of one audio sample in seconds.
    pub fn sample_duration_secs(self) -> f64 {
        1.0 / self.as_f64()
    }

    /// Computes total audio samples for a sequence/export duration using the CM3 formula:
    /// `floor((length / rate) * sample_rate)`.
    ///
    /// Negative or non-finite lengths return `0`.
    pub fn total_samples(self, length_frames: Frame, rate: FrameRate) -> u64 {
        let len = length_frames.as_f64();
        if !len.is_finite() || len <= 0.0 {
            return 0;
        }
        ((len / rate.as_f64()) * self.as_f64()).floor() as u64
    }

    /// Computes total audio samples for a duration in seconds (`floor(seconds * sample_rate)`).
    pub fn samples_for_seconds(self, seconds: f64) -> u64 {
        if !seconds.is_finite() || seconds <= 0.0 {
            return 0;
        }
        (seconds * self.as_f64()).floor() as u64
    }

    /// Converts an audio sample index to seconds.
    pub fn seconds_for_sample(self, sample_index: u64) -> f64 {
        sample_index as f64 / self.as_f64()
    }

    /// Converts an audio sample index to a `Frame` at `rate`.
    pub fn frame_at_sample(self, sample_index: u64, rate: FrameRate) -> Frame {
        Frame::from_seconds(self.seconds_for_sample(sample_index), rate)
    }
}

impl Default for AudioSampleRate {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl TryFrom<u32> for AudioSampleRate {
    type Error = TimeError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<AudioSampleRate> for u32 {
    fn from(rate: AudioSampleRate) -> Self {
        rate.0
    }
}

impl fmt::Display for AudioSampleRate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} Hz", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, eps: f64) {
        assert!(
            (a - b).abs() <= eps,
            "expected {a} to be within {eps} of {b} (diff = {})",
            (a - b).abs()
        );
    }

    #[test]
    fn frame_rate_bounds_and_defaults() {
        assert_eq!(FrameRate::default().fps(), 30);
        assert_close(
            FrameRate::default().frame_duration_secs(),
            1.0 / 30.0,
            1e-12,
        );
        assert!(FrameRate::new(0).is_err());
        assert!(FrameRate::new(121).is_err());
        assert_eq!(FrameRate::new(1).unwrap().fps(), 1);
        assert_eq!(FrameRate::new(120).unwrap().fps(), 120);
        assert_eq!(FrameRate::new_clamped(0).fps(), 1);
        assert_eq!(FrameRate::new_clamped(999).fps(), 120);

        for fps in [24, 25, 30, 48, 50, 60, 120] {
            let rate = FrameRate::new(fps).unwrap();
            assert!(
                rate.is_exact_tick_rate(),
                "{fps} fps should divide TICKS_PER_SECOND exactly"
            );
            assert_eq!(rate.ticks_per_frame() * i64::from(fps), TICKS_PER_SECOND);
        }
    }

    #[test]
    fn tick_round_trips_integer_and_ntsc_rates() {
        for fps in [24, 30, 60, 120] {
            let rate = FrameRate::new(fps).unwrap();
            for frame_idx in [0_i64, 1, 15, 29, 30, 1_000, 108_000] {
                let tick = Tick::from_whole_frame(frame_idx, rate);
                assert_eq!(tick.to_whole_frame_round(rate), frame_idx);
                let back = tick.to_frame(rate);
                assert_close(back.as_f64(), frame_idx as f64, 1e-12);
                assert_eq!(Tick::from_frame(back, rate), tick);
            }
        }

        // Lossless NTSC 23.976, 29.97, 59.94 fps checks.
        for (num, den) in [(24_000_u32, 1_001_u32), (30_000, 1_001), (60_000, 1_001)] {
            assert_eq!(TICKS_PER_SECOND % i64::from(num), 0);
            for frame_idx in [0_i64, 1, 100, 1_001, 86_400] {
                let tick = Tick::from_rational_frame(frame_idx, num, den).unwrap();
                assert_eq!(tick.to_rational_frame_round(num, den), Some(frame_idx));
            }
        }

        // Seconds and milliseconds conversions.
        let half_sec = Tick::from_seconds(0.5);
        assert_eq!(half_sec.as_i64(), TICKS_PER_SECOND / 2);
        assert_close(half_sec.to_seconds(), 0.5, 1e-12);
        assert_eq!(Tick::from_millis(250).to_millis(), 250);
        assert_eq!(Tick::new(-50).clamp_non_negative(), Tick::ZERO);
    }

    #[test]
    fn frame_seconds_and_motion_blur_subframe() {
        let rate = FrameRate::FPS_30;
        let f = Frame::from_seconds(2.5, rate);
        assert_close(f.as_f64(), 75.0, 1e-12);
        assert_close(f.to_seconds(rate), 2.5, 1e-12);

        // Spec 03 §1.3: f_local(s) = f - clip.start + (s / N) * S
        let sub = Frame::new(100.0).motion_blur_subframe(Frame::new(40.0), 4, 16, 0.5);
        assert_close(sub.as_f64(), 60.125, 1e-12);

        assert_eq!(Frame::new(-12.5).clamp_non_negative(), Frame::ZERO);
        assert_eq!(Frame::new(f64::NAN).clamp_non_negative(), Frame::ZERO);
    }

    #[test]
    fn timecode_formatting_and_parsing_all_three_formats() {
        assert_eq!(TimeFormat::default(), TimeFormat::Frames);
        assert_eq!(TimeFormat::HmsMillis.next(), TimeFormat::Smpte);
        assert_eq!(TimeFormat::Smpte.next(), TimeFormat::Frames);
        assert_eq!(TimeFormat::Frames.next(), TimeFormat::HmsMillis);
        for idx in 0..=2 {
            let fmt = TimeFormat::from_index(idx).unwrap();
            assert_eq!(fmt.to_index(), idx);
        }
        assert_eq!(TimeFormat::from_index(3), None);

        let r30 = FrameRate::FPS_30;
        // Zero and negative clamp.
        for f in [
            Frame::ZERO,
            Frame::new(-42.0),
            Frame::new(f64::NEG_INFINITY),
        ] {
            assert_eq!(TimeFormat::HmsMillis.format_frame(f, r30), "00:00:00.000");
            assert_eq!(TimeFormat::Smpte.format_frame(f, r30), "00:00:00:00");
            assert_eq!(TimeFormat::Frames.format_frame(f, r30), "0");
        }

        // 1h 01m 01s + 15 frames at 30 fps = 109_845 frames = 3661.500 s.
        let f = Frame::from_i64(3_661 * 30 + 15);
        let hms = TimeFormat::HmsMillis.format_frame(f, r30);
        let smpte = TimeFormat::Smpte.format_frame(f, r30);
        let frames = TimeFormat::Frames.format_frame(f, r30);
        assert_eq!(hms, "01:01:01.500");
        assert_eq!(smpte, "01:01:01:15");
        assert_eq!(frames, "109845");

        assert_close(
            TimeFormat::HmsMillis.parse(&hms, r30).unwrap().as_f64(),
            f.as_f64(),
            1e-9,
        );
        assert_close(
            TimeFormat::Smpte.parse(&smpte, r30).unwrap().as_f64(),
            f.as_f64(),
            1e-9,
        );
        assert_close(
            TimeFormat::Frames.parse(&frames, r30).unwrap().as_f64(),
            f.as_f64(),
            1e-9,
        );

        assert_eq!(
            TimeFormat::parse_auto(&hms, r30).unwrap().1,
            TimeFormat::HmsMillis
        );
        assert_eq!(
            TimeFormat::parse_auto(&smpte, r30).unwrap().1,
            TimeFormat::Smpte
        );
        assert_eq!(
            TimeFormat::parse_auto(&frames, r30).unwrap().1,
            TimeFormat::Frames
        );
    }

    #[test]
    fn timecode_smpte_edge_cases_across_fps() {
        for fps in [24_u32, 30, 60, 120] {
            let rate = FrameRate::new(fps).unwrap();
            let last_in_sec = Frame::from_i64(i64::from(fps - 1));
            let formatted = TimeFormat::Smpte.format_frame(last_in_sec, rate);
            let parsed = TimeFormat::Smpte.parse(&formatted, rate).unwrap();
            assert_close(parsed.as_f64(), (fps - 1) as f64, 1e-9);

            let one_sec = Frame::from_i64(i64::from(fps));
            assert_eq!(TimeFormat::Smpte.format_frame(one_sec, rate), "00:00:01:00");

            // Frame field == fps must fail.
            let invalid = format!("00:00:00:{fps:02}");
            assert!(TimeFormat::Smpte.parse(&invalid, rate).is_err());
        }

        // Invalid timecode strings.
        let r30 = FrameRate::FPS_30;
        assert!(TimeFormat::HmsMillis.parse("00:60:00.000", r30).is_err());
        assert!(TimeFormat::HmsMillis.parse("00:00:60.000", r30).is_err());
        assert!(TimeFormat::HmsMillis.parse("00:00:00.1234", r30).is_err());
        assert!(TimeFormat::Smpte.parse("00:00:60:00", r30).is_err());
        assert!(TimeFormat::Frames.parse("-1", r30).is_err());
        assert!(TimeFormat::Frames.parse("+5", r30).is_err());
        assert!(TimeFormat::Frames.parse("", r30).is_err());
    }

    #[test]
    fn audio_sample_rate_total_samples() {
        let sr = AudioSampleRate::default();
        assert_eq!(sr.hz(), 48_000);
        assert!(AudioSampleRate::new(0).is_err());

        // Exact 1 second at 30 fps -> 48_000 samples.
        assert_eq!(
            sr.total_samples(Frame::from_i64(30), FrameRate::FPS_30),
            48_000
        );
        // Fractional check: floor((1 / 24) * 48000) = 2000.
        assert_eq!(
            sr.total_samples(Frame::from_i64(1), FrameRate::FPS_24),
            2_000
        );
        // floor((1 / 60) * 48000) = 800.
        assert_eq!(sr.total_samples(Frame::from_i64(1), FrameRate::FPS_60), 800);
        // floor((1 / 120) * 48000) = 400.
        assert_eq!(
            sr.total_samples(Frame::from_i64(1), FrameRate::FPS_120),
            400
        );
        // floor((1 / 7) * 48000) = 6857.
        assert_eq!(
            sr.total_samples(Frame::from_i64(1), FrameRate::new(7).unwrap()),
            6_857
        );
        // Edge cases: 0 and negative.
        assert_eq!(sr.total_samples(Frame::ZERO, FrameRate::FPS_30), 0);
        assert_eq!(sr.total_samples(Frame::new(-10.0), FrameRate::FPS_30), 0);
    }

    #[test]
    fn overflow_guards_for_timecode_parsing_and_tick_math() {
        // Bug 1: huge hours must return Err instead of panicking or wrapping.
        let r120 = FrameRate::FPS_120;
        assert!(
            TimeFormat::HmsMillis
                .parse("9999999999999999:00:00.000", r120)
                .is_err()
        );
        assert!(
            TimeFormat::Smpte
                .parse("9999999999999999:00:00:00", r120)
                .is_err()
        );
        assert!(
            TimeFormat::Smpte
                .parse("42700796466921:00:00:00", r120)
                .is_err()
        );

        // Bug 2: i128 overflow in Tick::from_rational_frame returns None.
        assert_eq!(Tick::from_rational_frame(i64::MAX, 1, u32::MAX), None);
        assert_eq!(Tick::from_rational_frame(i64::MIN, 1, u32::MAX), None);
        assert_eq!(Tick::from_rational_frame(i64::MAX, 24_000, 1_001), None);

        // Bug 3: Tick(i64::MIN) / -1 saturates instead of panicking.
        assert_eq!(Tick::new(i64::MIN) / -1, Tick::new(i64::MAX));
    }
}
