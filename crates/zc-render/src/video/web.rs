use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use js_sys::{Function, Reflect, Uint8Array};
use wasm_bindgen::JsCast;

use super::{
    DecodedVideoImage, MAX_ACTIVE_VIDEO_SESSIONS, VideoMetadata, VideoStreamKey, clamp_video_time,
    compute_decode_dimensions,
};
use crate::effects::fnv1a_hash64;

thread_local! {
    static VIDEO_METADATA: RefCell<HashMap<u64, VideoMetadata>> = RefCell::default();
}

/// Registers metadata on the browser thread that runs the renderer. Use
/// [`crate::web_video_asset_hash`] for project assets.
pub fn register_web_video_metadata(asset_hash: u64, meta: VideoMetadata) {
    VIDEO_METADATA.with_borrow_mut(|metadata| metadata.insert(asset_hash, meta));
}

fn registered_metadata(asset_hash: u64) -> Option<VideoMetadata> {
    VIDEO_METADATA.with_borrow(|metadata| metadata.get(&asset_hash).cloned())
}

/// Reads a cached straight-alpha RGBA8 frame from `globalThis.__zcWebCodecs`.
/// The host's `getVideoFrame` method is synchronous and returns `null` or
/// `undefined` while the frame is unavailable.
#[must_use]
pub fn zc_webcodecs_get_video_frame(
    asset_hash_hex: &str,
    time_secs: f64,
    dec_w: u32,
    dec_h: u32,
) -> Option<Uint8Array> {
    let bridge = Reflect::get(&js_sys::global(), &"__zcWebCodecs".into()).ok()?;
    let get_frame = Reflect::get(&bridge, &"getVideoFrame".into())
        .ok()?
        .dyn_into::<Function>()
        .ok()?;
    get_frame
        .call4(
            &bridge,
            &asset_hash_hex.into(),
            &time_secs.into(),
            &dec_w.into(),
            &dec_h.into(),
        )
        .ok()?
        .dyn_into::<Uint8Array>()
        .ok()
}

fn decode_frame_bytes(
    key: VideoStreamKey,
    meta: &VideoMetadata,
    time_secs: f64,
) -> Option<Vec<u8>> {
    let expected_len = key.dec_w.checked_mul(key.dec_h)?.checked_mul(4)?;
    let rgba = zc_webcodecs_get_video_frame(
        &format!("{:016x}", key.asset_hash),
        clamp_video_time(time_secs, meta),
        key.dec_w,
        key.dec_h,
    )?;
    (rgba.length() == expected_len).then(|| rgba.to_vec())
}

/// Looks up registered metadata for standalone media identified by the FNV-1a
/// 64-bit hash of the path's UTF-8 bytes.
#[must_use]
pub fn probe_video_metadata(path: &Path) -> Option<VideoMetadata> {
    registered_metadata(fnv1a_hash64(path.to_str()?.as_bytes()))
}

/// Returns the default FFmpeg command name. Browser builds cannot execute it.
#[must_use]
pub fn resolve_ffmpeg_binary_path() -> PathBuf {
    PathBuf::from("ffmpeg")
}

/// Returns the default FFprobe command name. Browser builds cannot execute it.
#[must_use]
pub fn resolve_ffprobe_binary_path() -> PathBuf {
    PathBuf::from("ffprobe")
}

/// FFprobe is unavailable in the browser.
#[must_use]
pub fn ffprobe_binary_available() -> bool {
    false
}

/// A standalone video stream backed by the browser's WebCodecs cache.
pub struct VideoStreamSession {
    key: VideoStreamKey,
    meta: VideoMetadata,
}

impl VideoStreamSession {
    /// Uses the FNV-1a 64-bit hash of the path's UTF-8 bytes as the asset key.
    #[must_use]
    pub fn new(input_path: PathBuf, meta: VideoMetadata, dec_w: u32, dec_h: u32) -> Self {
        Self {
            key: VideoStreamKey {
                asset_hash: fnv1a_hash64(input_path.to_string_lossy().as_bytes()),
                dec_w: dec_w.max(2),
                dec_h: dec_h.max(2),
            },
            meta,
        }
    }

    /// Reads the cached straight-alpha RGBA8 frame for the requested time.
    pub fn request_frame(&mut self, target_secs: f64) -> Option<Arc<Vec<u8>>> {
        decode_frame_bytes(self.key, &self.meta, target_secs).map(Arc::new)
    }
}

/// Tracks recently used WebCodecs streams. The host owns the decoder and frame cache.
#[derive(Default)]
pub struct VideoDecoderPool {
    sessions: VecDeque<VideoStreamKey>,
}

impl VideoDecoderPool {
    /// Creates an empty decoder pool.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of recently used decoder streams tracked by this pool.
    #[must_use]
    pub fn active_session_count(&self) -> usize {
        self.sessions.len()
    }

    /// Clears this pool's stream history. Registered metadata is shared by all pools.
    pub fn clear(&mut self) {
        self.sessions.clear();
    }

    /// Returns metadata registered for this asset. Hosts may register metadata asynchronously.
    pub fn probe_asset(
        &mut self,
        asset_hash: u64,
        _url: Option<&str>,
        _filename: &str,
        _bytes: Option<&[u8]>,
    ) -> Option<VideoMetadata> {
        registered_metadata(asset_hash)
    }

    /// Reads a cached straight-alpha RGBA8 frame at the requested decode resolution.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_frame(
        &mut self,
        asset_hash: u64,
        url: Option<&str>,
        filename: &str,
        bytes: Option<&[u8]>,
        time_secs: f64,
        target_hint: Option<(u32, u32)>,
        max_gpu_dim: u32,
    ) -> Option<DecodedVideoImage> {
        let meta = self.probe_asset(asset_hash, url, filename, bytes)?;
        let (dec_w, dec_h) =
            compute_decode_dimensions(meta.width, meta.height, target_hint, max_gpu_dim);
        let key = VideoStreamKey {
            asset_hash,
            dec_w,
            dec_h,
        };
        let rgba = decode_frame_bytes(key, &meta, time_secs)?;
        self.sessions.retain(|entry| *entry != key);
        if self.sessions.len() == MAX_ACTIVE_VIDEO_SESSIONS {
            self.sessions.pop_front();
        }
        self.sessions.push_back(key);
        Some(DecodedVideoImage {
            width: dec_w,
            height: dec_h,
            rgba: Arc::new(rgba),
        })
    }

    /// Reads a cached frame for a registered asset identified by `asset_hash`.
    pub fn decode_frame_from_path(
        &mut self,
        asset_hash: u64,
        _path: &Path,
        time_secs: f64,
        target_hint: Option<(u32, u32)>,
        max_gpu_dim: u32,
    ) -> Option<DecodedVideoImage> {
        self.decode_frame(
            asset_hash,
            None,
            "",
            None,
            time_secs,
            target_hint,
            max_gpu_dim,
        )
    }
}
