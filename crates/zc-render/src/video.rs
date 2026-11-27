//! Video metadata and straight-alpha RGBA8 frame decoding.
//!
//! Native builds use persistent FFmpeg streams. Browser builds read frames from
//! the host's WebCodecs cache.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(not(target_arch = "wasm32"))]
pub use native::{
    VideoDecoderPool, VideoStreamSession, default_managed_tools_dir, derive_sibling_ffprobe_path,
    ffprobe_binary_available, probe_video_metadata, refresh_media_tool_cache,
    resolve_ffmpeg_binary_path, resolve_ffprobe_binary_path, set_custom_media_tool_paths,
};
#[cfg(target_arch = "wasm32")]
pub use web::{
    VideoDecoderPool, VideoStreamSession, ffprobe_binary_available, probe_video_metadata,
    register_web_video_metadata, resolve_ffmpeg_binary_path, resolve_ffprobe_binary_path,
    web_video_frame_is_exact, zc_webcodecs_get_video_frame,
};

/// Maximum number of simultaneously active persistent `ffmpeg` decoder sessions (LRU evicted).
pub const MAX_ACTIVE_VIDEO_SESSIONS: usize = 8;

/// Maximum number of decoded RGBA frames prefetched ahead per active video stream.
pub const MAX_PREFETCH_FRAMES: usize = 16;

/// Maximum byte budget for prefetched RGBA frames per active video stream (64 MiB).
pub const MAX_PREFETCH_BYTES: usize = 64 * 1024 * 1024;

/// Probed container / stream metadata for a video asset.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoMetadata {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration_secs: f64,
    pub has_audio: bool,
}

/// Decoded straight-alpha RGBA8 video frame produced by [`VideoDecoderPool`].
#[derive(Debug, Clone)]
pub struct DecodedVideoImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Arc<Vec<u8>>,
}

/// Cache key identifying a persistent video decoder stream at a specific decode resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VideoStreamKey {
    pub asset_hash: u64,
    pub dec_w: u32,
    pub dec_h: u32,
}

/// Computes target decode dimensions preserving aspect ratio, rounded to even sizes (`>= 2`),
/// never upscaling past native `(src_w, src_h)` and never exceeding `max_gpu_dim`.
#[must_use]
pub fn compute_decode_dimensions(
    src_w: u32,
    src_h: u32,
    target_hint: Option<(u32, u32)>,
    max_gpu_dim: u32,
) -> (u32, u32) {
    if src_w == 0 || src_h == 0 {
        return (2, 2);
    }
    let max_dim = max_gpu_dim.max(2);
    let (limit_w, limit_h) = match target_hint {
        Some((hw, hh)) if hw > 0 && hh > 0 => (hw.min(max_dim), hh.min(max_dim)),
        _ => (max_dim, max_dim),
    };
    let scale = (f64::from(limit_w) / f64::from(src_w))
        .min(f64::from(limit_h) / f64::from(src_h))
        .min(1.0);

    let upper_w = src_w.min(max_dim);
    let upper_h = src_h.min(max_dim);

    let round_even = |raw: f64, upper: u32| -> u32 {
        let r = raw.round().max(2.0) as u32;
        let even = ((r + 1) & !1).max(2);
        if even > upper {
            (upper & !1).max(2)
        } else {
            even
        }
    };

    (
        round_even(f64::from(src_w) * scale, upper_w),
        round_even(f64::from(src_h) * scale, upper_h),
    )
}

/// Returns `true` if any pixel in an RGBA8 buffer has alpha `< 255`.
#[must_use]
pub fn has_non_opaque_alpha(pixels: &[u8]) -> bool {
    pixels.as_chunks::<4>().0.iter().any(|px| px[3] != 255)
}

/// Fast integer premultiply in-place for RGBA8 buffers.
///
/// Skips opaque pixels (`a == 255`) and computes exact rounded `((c * a + 127) / 255)` for
/// translucent pixels.
pub fn fast_premultiply_rgba8(pixels: &mut [u8]) {
    for px in pixels.as_chunks_mut::<4>().0 {
        let a = px[3];
        if a == 255 {
            continue;
        }
        if a == 0 {
            px[0] = 0;
            px[1] = 0;
            px[2] = 0;
            continue;
        }
        let a16 = u16::from(a);
        px[0] = ((u16::from(px[0]) * a16 + 127) / 255) as u8;
        px[1] = ((u16::from(px[1]) * a16 + 127) / 255) as u8;
        px[2] = ((u16::from(px[2]) * a16 + 127) / 255) as u8;
    }
}

/// Clamps a requested timestamp to the valid playable range of `meta`.
#[must_use]
pub fn clamp_video_time(time_secs: f64, meta: &VideoMetadata) -> f64 {
    let t = if time_secs.is_finite() {
        time_secs.max(0.0)
    } else {
        0.0
    };
    if meta.duration_secs > 0.0 {
        let frame_dur = 1.0 / meta.fps.clamp(1.0, 240.0);
        t.min((meta.duration_secs - frame_dur * 0.5).max(0.0))
    } else {
        t
    }
}
