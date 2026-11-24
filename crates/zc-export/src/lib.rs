//! Deadlock-free, zero-leak video and image-sequence export pipeline for ZoidCraft.
//!
//! Supports:
//! - FFmpeg subprocess encoding (MP4/H.264/H.265/AV1, WebM/VP9/VP8 + alpha, MKV/AV1/PCM/Opus)
//! - Odd-dimension 1px right/bottom black padding for 4:2:0 codecs without scaling
//! - RAII temporary stereo WAV audio pre-staging (preventing dual-pipe deadlocks and temp leaks)
//! - Bounded stderr draining on a scoped thread (preventing stderr pipe stalls and memory leaks)
//! - Cooperative `Arc<AtomicBool>` cancellation with guaranteed child reaping (zero zombies)
//! - Direct PNG sequence export via the `image` crate (no FFmpeg required)

#![forbid(unsafe_code)]

pub mod exporter;
pub mod ffmpeg;
pub mod padding;
pub mod settings;
pub mod temp_audio;

pub use exporter::{export_video, export_video_with_progress};
pub use ffmpeg::{BoundedStderrBuffer, build_ffmpeg_args, probe_ffmpeg};
pub use padding::{even_padded_dimensions, pad_rgba_frame_into};
pub use settings::{
    AudioCodec, ExportContainer, ExportProgress, ExportReport, ExportSettings, FrameProvider,
    QualityMode, VideoCodec,
};
pub use temp_audio::{TempFileGuard, write_temp_stereo_wav_f32};

use thiserror::Error;

/// Errors returned by the export pipeline.
#[derive(Debug, Error)]
pub enum ExportError {
    /// Export settings failed validation.
    #[error("invalid export settings: {0}")]
    InvalidSettings(String),

    /// FFmpeg binary was not found or could not be executed.
    #[error("ffmpeg binary not found: {0}")]
    FfmpegNotFound(String),

    /// Export was cancelled via the cancellation token.
    #[error("export cancelled")]
    Cancelled,

    /// Frame callback returned a buffer whose byte length did not match `width * height * 4`.
    #[error("frame {frame_index} had invalid RGBA byte size: expected {expected}, got {actual}")]
    InvalidFrameSize {
        frame_index: u32,
        expected: usize,
        actual: usize,
    },

    /// Frame callback reported a rendering error.
    #[error("frame render error: {0}")]
    Render(String),

    /// FFmpeg exited with a non-zero status code.
    #[error("ffmpeg failed (exit code {exit_code:?}): {stderr}")]
    FfmpegFailed {
        exit_code: Option<i32>,
        stderr: String,
    },

    /// PNG encoding failed.
    #[error("image encoding error: {0}")]
    ImageEncode(String),

    /// Filesystem or pipe I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
