//! Export container, codec, quality settings, progress/report structs, and [`FrameProvider`] trait.

use std::path::PathBuf;

use crate::ExportError;
use crate::padding::even_padded_dimensions;

/// Maximum single-axis export resolution in pixels (8K UHD per Spec 04 §4.1).
pub const MAX_EXPORT_DIMENSION: u32 = 7_680;
/// Maximum raw RGBA8 frame buffer byte size (`7680 * 7680 * 4` = 235 MiB, capped at 256 MiB).
pub const MAX_EXPORT_FRAME_BYTES: usize = 256 * 1024 * 1024;
/// Minimum allowed export frame rate in fps (Spec 04 §4.1).
pub const MIN_EXPORT_FPS: f64 = 1.0;
/// Maximum allowed export frame rate in fps (Spec 04 §4.1).
pub const MAX_EXPORT_FPS: f64 = 240.0;
/// Maximum frames allowed in a single export job (5 hours at 60 fps).
pub const MAX_EXPORT_FRAMES: u32 = 1_080_000;
/// Minimum supported audio export sample rate in Hz.
pub const MIN_EXPORT_SAMPLE_RATE: u32 = 8_000;
/// Maximum supported audio export sample rate in Hz.
pub const MAX_EXPORT_SAMPLE_RATE: u32 = 192_000;

/// Output container format for an export job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportContainer {
    /// MPEG-4 Part 14 (`.mp4`).
    Mp4,
    /// WebM (`.webm`).
    Webm,
    /// Matroska (`.mkv`).
    Mkv,
    /// Directory of RGBA8 PNG frames (`frame_000001.png`, ...), exported directly without FFmpeg.
    PngSequence,
}

/// Video codec selection for FFmpeg-backed containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    /// H.264 / AVC (`libx264`).
    H264,
    /// H.265 / HEVC (`libx265`).
    H265,
    /// VP9 (`libvpx-vp9`, supports `yuva420p` alpha).
    Vp9,
    /// VP8 (`libvpx`).
    Vp8,
    /// AV1 (`libsvtav1`).
    Av1,
}

/// Audio codec selection for FFmpeg-backed containers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioCodec {
    /// AAC-LC (`aac`).
    Aac,
    /// Opus (`libopus`).
    Opus,
    /// Uncompressed 16-bit PCM (`pcm_s16le`).
    Pcm,
    /// Omit audio track (`-an`).
    None,
}

/// Video rate-control mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityMode {
    /// Constant Rate Factor (`0..=51` for H.264/H.265, `4..=63` for VP8, `0..=63` for VP9/AV1).
    Crf(u8),
    /// Target average video bitrate in kilobits per second.
    BitrateKbps(u32),
}

/// Complete configuration for an export job.
#[derive(Debug, Clone, PartialEq)]
pub struct ExportSettings {
    /// Logical rendered frame width in pixels.
    pub width: u32,
    /// Logical rendered frame height in pixels.
    pub height: u32,
    /// Output frame rate in frames per second.
    pub fps: f64,
    /// Inclusive starting frame index.
    pub start_frame: u32,
    /// Exclusive ending frame index (`total_frames = end_frame - start_frame`).
    pub end_frame: u32,
    /// Output container format.
    pub container: ExportContainer,
    /// Video codec (ignored for [`ExportContainer::PngSequence`]).
    pub video_codec: VideoCodec,
    /// Audio codec (ignored for [`ExportContainer::PngSequence`]).
    pub audio_codec: AudioCodec,
    /// Video quality or bitrate control.
    pub quality: QualityMode,
    /// Lossy audio bitrate in kbps (e.g. `192`; Opus is clamped to `6..=510`). Ignored for PCM/None.
    pub audio_bitrate_kbps: u32,
    /// Audio sample rate in Hz (typically `48_000` or `44_100`).
    pub sample_rate: u32,
    /// Whether to preserve transparent alpha (`yuva420p` for VP9, RGBA8 for PNG sequence).
    pub transparent_alpha: bool,
    /// Output video file path (or destination directory for [`ExportContainer::PngSequence`]).
    pub output_path: PathBuf,
    /// Optional explicit path to the `ffmpeg` executable (defaults to `ffmpeg` on `PATH`).
    pub ffmpeg_binary: Option<PathBuf>,
}

impl ExportSettings {
    /// Returns the total number of frames to export (`end_frame - start_frame`).
    #[must_use]
    pub const fn total_frames(&self) -> u32 {
        self.end_frame.saturating_sub(self.start_frame)
    }

    /// Returns `true` when the selected container/codec requires even width and height (4:2:0).
    #[must_use]
    pub const fn requires_even_dimensions(&self) -> bool {
        !matches!(self.container, ExportContainer::PngSequence)
            && matches!(
                self.video_codec,
                VideoCodec::H264
                    | VideoCodec::H265
                    | VideoCodec::Vp9
                    | VideoCodec::Vp8
                    | VideoCodec::Av1
            )
    }

    /// Returns the encoded output dimensions after odd-dimension 1px right/bottom padding.
    #[must_use]
    pub const fn padded_dimensions(&self) -> (u32, u32) {
        if self.requires_even_dimensions() {
            even_padded_dimensions(self.width, self.height)
        } else {
            (self.width, self.height)
        }
    }

    /// Validates settings for internal consistency, resource bounds, and container/codec compatibility.
    pub fn validate(&self) -> Result<(), ExportError> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_EXPORT_DIMENSION
            || self.height > MAX_EXPORT_DIMENSION
        {
            return Err(ExportError::InvalidSettings(format!(
                "width ({}) and height ({}) must be in 1..={MAX_EXPORT_DIMENSION}",
                self.width, self.height
            )));
        }
        let (enc_w, enc_h) = self.padded_dimensions();
        let padded_bytes = (enc_w as usize)
            .checked_mul(enc_h as usize)
            .and_then(|px| px.checked_mul(4))
            .ok_or_else(|| {
                ExportError::InvalidSettings("export frame byte size overflow".to_string())
            })?;
        if padded_bytes > MAX_EXPORT_FRAME_BYTES {
            return Err(ExportError::InvalidSettings(format!(
                "export frame size ({padded_bytes} bytes) exceeds limit ({MAX_EXPORT_FRAME_BYTES} bytes)"
            )));
        }
        if !self.fps.is_finite() || self.fps < MIN_EXPORT_FPS || self.fps > MAX_EXPORT_FPS {
            return Err(ExportError::InvalidSettings(format!(
                "fps ({}) must be finite and in {MIN_EXPORT_FPS}..={MAX_EXPORT_FPS}",
                self.fps
            )));
        }
        if self.end_frame <= self.start_frame {
            return Err(ExportError::InvalidSettings(format!(
                "end_frame ({}) must be greater than start_frame ({})",
                self.end_frame, self.start_frame
            )));
        }
        let total_frames = self.total_frames();
        if total_frames > MAX_EXPORT_FRAMES {
            return Err(ExportError::InvalidSettings(format!(
                "total export frames ({total_frames}) exceeds limit ({MAX_EXPORT_FRAMES})"
            )));
        }
        if self.output_path.as_os_str().is_empty() {
            return Err(ExportError::InvalidSettings(
                "output_path must not be empty".to_string(),
            ));
        }

        if matches!(self.container, ExportContainer::PngSequence) {
            return Ok(());
        }

        if let QualityMode::BitrateKbps(kbps) = self.quality
            && kbps == 0
        {
            return Err(ExportError::InvalidSettings(
                "video bitrate must be greater than zero".to_string(),
            ));
        }

        if self.audio_codec != AudioCodec::None
            && !(MIN_EXPORT_SAMPLE_RATE..=MAX_EXPORT_SAMPLE_RATE).contains(&self.sample_rate)
        {
            return Err(ExportError::InvalidSettings(format!(
                "sample_rate ({}) must be in {MIN_EXPORT_SAMPLE_RATE}..={MAX_EXPORT_SAMPLE_RATE} Hz when audio is enabled",
                self.sample_rate
            )));
        }

        if matches!(self.audio_codec, AudioCodec::Aac | AudioCodec::Opus)
            && self.audio_bitrate_kbps == 0
        {
            return Err(ExportError::InvalidSettings(
                "audio_bitrate_kbps must be greater than zero for lossy audio codecs".to_string(),
            ));
        }

        if self.transparent_alpha && self.video_codec != VideoCodec::Vp9 {
            return Err(ExportError::InvalidSettings(format!(
                "transparent_alpha is only supported for PngSequence or VP9 (WebM/MKV), not {:?}",
                self.video_codec
            )));
        }

        match self.container {
            ExportContainer::Webm => {
                if !matches!(
                    self.video_codec,
                    VideoCodec::Vp8 | VideoCodec::Vp9 | VideoCodec::Av1
                ) {
                    return Err(ExportError::InvalidSettings(format!(
                        "WebM container does not support video codec {:?}",
                        self.video_codec
                    )));
                }
                if !matches!(self.audio_codec, AudioCodec::Opus | AudioCodec::None) {
                    return Err(ExportError::InvalidSettings(format!(
                        "WebM container requires Opus or None audio, not {:?}",
                        self.audio_codec
                    )));
                }
            }
            ExportContainer::Mp4 => {
                if self.transparent_alpha {
                    return Err(ExportError::InvalidSettings(
                        "MP4 container does not support transparent_alpha".to_string(),
                    ));
                }
                if self.video_codec == VideoCodec::Vp8 {
                    return Err(ExportError::InvalidSettings(
                        "MP4 container does not support VP8 video".to_string(),
                    ));
                }
                if self.audio_codec == AudioCodec::Pcm {
                    return Err(ExportError::InvalidSettings(
                        "MP4 container does not support raw PCM audio".to_string(),
                    ));
                }
            }
            ExportContainer::Mkv | ExportContainer::PngSequence => {}
        }

        Ok(())
    }
}

/// Progress snapshot reported after each exported frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExportProgress {
    /// 1-based count of frames exported so far.
    pub completed_frames: u32,
    /// Total frames scheduled for export.
    pub total_frames: u32,
    /// Timeline frame index that just completed.
    pub frame_index: u32,
}

/// Summary returned after a successful export.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExportReport {
    /// Number of frames written.
    pub frames_exported: u32,
    /// Final encoded width (including any 1px odd-dimension padding).
    pub output_width: u32,
    /// Final encoded height (including any 1px odd-dimension padding).
    pub output_height: u32,
    /// Number of stereo audio sample frames muxed (`0` if silent/video-only).
    pub audio_samples_written: usize,
    /// Path to the output file or PNG sequence directory.
    pub output_path: PathBuf,
}

/// Supplies tightly packed top-down RGBA8 frames into a caller-reused buffer.
pub trait FrameProvider {
    /// Renders `frame_index` into `rgba_out` (already sized to `width * height * 4`).
    fn render_frame(&mut self, frame_index: u32, rgba_out: &mut Vec<u8>)
    -> Result<(), ExportError>;
}

impl<F> FrameProvider for F
where
    F: FnMut(u32, &mut Vec<u8>) -> Result<(), ExportError>,
{
    fn render_frame(
        &mut self,
        frame_index: u32,
        rgba_out: &mut Vec<u8>,
    ) -> Result<(), ExportError> {
        self(frame_index, rgba_out)
    }
}
