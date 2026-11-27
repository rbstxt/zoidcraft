//! Project lifecycle (new/open/save/`.pz` export), autosave & recovery
//! snapshots, settings, modals, and status overlays.

use super::*;

pub(crate) const TOAST_LIFETIME: Duration = Duration::from_secs(4);

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const AUTOSAVE_INTERVAL: Duration = Duration::from_secs(60);

#[cfg(not(target_arch = "wasm32"))]
pub(crate) const MAX_RECOVERY_SNAPSHOTS: usize = 10;

/// Key stored in `Sequence::preserved` inside `.zcraft` saves (stripped on `.pz` export) to
/// persist enabled safe built-in plugins across GUI save & reload without polluting vanilla CM3 `.pz`.
pub(crate) const ZOIDCRAFT_ENABLED_PLUGINS_KEY: &str = "zoidcraftEnabledPlugins";

/// Assets larger than 8 MiB with a local file `url` are converted to lightweight external
/// references inside periodic recovery snapshots so autosave never copies or gzips huge videos.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES: usize = 8 * 1024 * 1024;

/// `localStorage` metadata key for browser autosave snapshots.
#[cfg(target_arch = "wasm32")]
pub(crate) const WEB_AUTOSAVE_META_KEY: &str = "zoidcraft.autosave.v1.meta";

/// `localStorage` chunk key prefix (`{WEB_AUTOSAVE_CHUNK_PREFIX}{index}`).
#[cfg(target_arch = "wasm32")]
pub(crate) const WEB_AUTOSAVE_CHUNK_PREFIX: &str = "zoidcraft.autosave.v1.chunk.";

/// Maximum characters stored per `localStorage` chunk key (512 KiB).
#[cfg(target_arch = "wasm32")]
pub(crate) const WEB_AUTOSAVE_CHUNK_CHARS: usize = 512 * 1024;

/// Maximum total base64-encoded bytes permitted for a browser autosave snapshot (~4 MiB,
/// staying comfortably within standard 5 MiB `localStorage` quotas).
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) const WEB_AUTOSAVE_MAX_ENCODED_BYTES: usize = 4 * 1024 * 1024;

/// Threshold above which embedded assets are externalized in web autosave fallback snapshots.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) const WEB_AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES: usize = 256 * 1024;

#[cfg(target_arch = "wasm32")]
pub(crate) const WEB_AUTOSAVE_INTERVAL: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Base64 & Chunked Web Autosave Codec (Zero External Crates)
// ---------------------------------------------------------------------------

#[cfg(any(target_arch = "wasm32", test))]
const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn encode_base64(input: &[u8]) -> String {
    let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
    for chunk in input.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = if chunk.len() > 1 {
            u32::from(chunk[1])
        } else {
            0
        };
        let b2 = if chunk.len() > 2 {
            u32::from(chunk[2])
        } else {
            0
        };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_ALPHABET[((triple >> 18) & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            out.push(BASE64_ALPHABET[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(BASE64_ALPHABET[(triple & 0x3F) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn decode_base64(input: &str) -> Result<Vec<u8>, String> {
    let mut clean = Vec::with_capacity(input.len());
    for b in input.bytes() {
        if !b.is_ascii_whitespace() {
            clean.push(b);
        }
    }
    if !clean.len().is_multiple_of(4) {
        return Err("invalid base64 length".to_owned());
    }
    let decode_val = |b: u8| -> Result<u32, String> {
        match b {
            b'A'..=b'Z' => Ok(u32::from(b - b'A')),
            b'a'..=b'z' => Ok(u32::from(b - b'a') + 26),
            b'0'..=b'9' => Ok(u32::from(b - b'0') + 52),
            b'+' => Ok(62),
            b'/' => Ok(63),
            _ => Err(format!("invalid base64 byte: 0x{b:02x}")),
        }
    };
    let mut out = Vec::with_capacity((clean.len() / 4) * 3);
    for (idx, chunk) in clean.as_chunks::<4>().0.iter().enumerate() {
        let is_last = (idx + 1) * 4 == clean.len();
        let pad = if chunk[2] == b'=' && chunk[3] == b'=' {
            if !is_last {
                return Err("unexpected base64 padding before end".to_owned());
            }
            2
        } else if chunk[3] == b'=' {
            if !is_last {
                return Err("unexpected base64 padding before end".to_owned());
            }
            1
        } else {
            0
        };
        let v0 = decode_val(chunk[0])?;
        let v1 = decode_val(chunk[1])?;
        let v2 = if pad == 2 { 0 } else { decode_val(chunk[2])? };
        let v3 = if pad >= 1 { 0 } else { decode_val(chunk[3])? };
        let triple = (v0 << 18) | (v1 << 12) | (v2 << 6) | v3;
        out.push(((triple >> 16) & 0xFF) as u8);
        if pad < 2 {
            out.push(((triple >> 8) & 0xFF) as u8);
        }
        if pad == 0 {
            out.push((triple & 0xFF) as u8);
        }
    }
    Ok(out)
}

#[cfg(any(target_arch = "wasm32", test))]
fn clone_project_with_asset_byte_cap(src: &Project, max_asset_bytes: usize) -> Project {
    let mut assets = BTreeMap::new();
    for (k, rec) in &src.assets {
        let drop_bytes = rec
            .bytes
            .as_ref()
            .is_some_and(|b| b.len() > max_asset_bytes);
        let copy = if drop_bytes {
            AssetRecord {
                id: rec.id.clone(),
                kind: rec.kind,
                source: rec.source,
                filename: rec.filename.clone(),
                url: rec.url.clone(),
                external: true,
                bytes: None,
                missing: true,
                sha256: rec.sha256.clone(),
                size: rec.size,
                preserved: rec.preserved.clone(),
            }
        } else {
            rec.clone()
        };
        assets.insert(k.clone(), copy);
    }
    Project {
        version: src.version,
        name: src.name.clone(),
        assets,
        media: src.media.clone(),
        sequence: src.sequence.clone(),
        compositions: src.compositions.clone(),
        composition_ids: src.composition_ids.clone(),
        plugin_requirements: src.plugin_requirements.clone(),
        imported_from_pz_version: src.imported_from_pz_version,
        preserved: src.preserved.clone(),
    }
}

/// Encodes `project` into size-guarded, chunked base64 + JSON metadata suitable for browser
/// `window.localStorage` persistence. Automatically externalizes large embedded assets if the
/// full archive would exceed `max_encoded_bytes`.
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn encode_web_autosave_chunks(
    project: &Project,
    timestamp_secs: u64,
    max_encoded_bytes: usize,
    chunk_chars: usize,
) -> Result<(String, Vec<String>, u64), String> {
    let step = chunk_chars.max(1024);
    let mut assets_externalized = false;
    let mut raw_bytes = zc_project::save_zcraft_to_bytes(project)
        .map_err(|e| format!("failed to serialize web autosave snapshot: {e}"))?;
    let mut encoded = encode_base64(&raw_bytes);

    if encoded.len() > max_encoded_bytes {
        let trimmed =
            clone_project_with_asset_byte_cap(project, WEB_AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES);
        raw_bytes = zc_project::save_zcraft_to_bytes(&trimmed)
            .map_err(|e| format!("failed to serialize trimmed web autosave snapshot: {e}"))?;
        encoded = encode_base64(&raw_bytes);
        assets_externalized = true;
    }

    if encoded.len() > max_encoded_bytes {
        let metadata_only = clone_project_with_asset_byte_cap(project, 0);
        raw_bytes = zc_project::save_zcraft_to_bytes(&metadata_only)
            .map_err(|e| format!("failed to serialize lightweight web autosave snapshot: {e}"))?;
        encoded = encode_base64(&raw_bytes);
        assets_externalized = true;
    }

    if encoded.len() > max_encoded_bytes {
        return Err(format!(
            "web autosave snapshot ({} bytes encoded) exceeds browser storage budget ({} bytes)",
            encoded.len(),
            max_encoded_bytes
        ));
    }

    let chunks: Vec<String> = encoded
        .as_bytes()
        .chunks(step)
        .map(|slice| std::str::from_utf8(slice).unwrap_or_default().to_owned())
        .collect();
    let digest = sha256_hex(&raw_bytes);
    let raw_len = raw_bytes.len() as u64;
    let meta = json!({
        "version": 1_u32,
        "project_name": project.name,
        "timestamp_secs": timestamp_secs,
        "raw_bytes": raw_len,
        "chunks": chunks.len(),
        "sha256": digest,
        "assets_externalized": assets_externalized,
    });
    let meta_json =
        serde_json::to_string(&meta).map_err(|e| format!("failed to encode autosave meta: {e}"))?;
    Ok((meta_json, chunks, raw_len))
}

/// Validates and decodes chunked base64 + JSON metadata produced by [`encode_web_autosave_chunks`].
#[cfg(any(target_arch = "wasm32", test))]
pub(crate) fn decode_web_autosave_chunks(
    meta_json: &str,
    chunks: &[String],
) -> Result<(Project, u64, u64), String> {
    let meta: Value = serde_json::from_str(meta_json)
        .map_err(|e| format!("invalid web autosave metadata: {e}"))?;
    let version = meta.get("version").and_then(Value::as_u64).unwrap_or(0);
    if version != 1 {
        return Err(format!("unsupported web autosave version: {version}"));
    }
    let expected_chunks = meta
        .get("chunks")
        .and_then(Value::as_u64)
        .ok_or_else(|| "web autosave metadata missing chunk count".to_owned())?
        as usize;
    if expected_chunks == 0 || chunks.len() != expected_chunks {
        return Err(format!(
            "web autosave chunk count mismatch (expected {expected_chunks}, got {})",
            chunks.len()
        ));
    }
    let expected_sha = meta
        .get("sha256")
        .and_then(Value::as_str)
        .ok_or_else(|| "web autosave metadata missing sha256".to_owned())?;
    let timestamp_secs = meta
        .get("timestamp_secs")
        .and_then(Value::as_u64)
        .unwrap_or(0);

    let total_len: usize = chunks.iter().map(String::len).sum();
    let mut joined = String::with_capacity(total_len);
    for chunk in chunks {
        joined.push_str(chunk);
    }
    let raw_bytes = decode_base64(&joined)?;
    let actual_sha = sha256_hex(&raw_bytes);
    if actual_sha != expected_sha {
        return Err("web autosave snapshot failed SHA-256 integrity check".to_owned());
    }
    let project = zc_project::load_zcraft_from_bytes(&raw_bytes)
        .map_err(|e| format!("failed to load web autosave project: {e}"))?;
    Ok((project, timestamp_secs, raw_bytes.len() as u64))
}

#[cfg(target_arch = "wasm32")]
static WEB_UNSAVED_DIRTY: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
static WEB_BEFOREUNLOAD_REGISTERED: AtomicBool = AtomicBool::new(false);

#[cfg(target_arch = "wasm32")]
struct WebAutosaveRuntime {
    startup_checked: bool,
    last_autosave_at: Instant,
    last_autosave_revision: u64,
}

#[cfg(target_arch = "wasm32")]
thread_local! {
    static WEB_AUTOSAVE_RUNTIME: std::cell::RefCell<WebAutosaveRuntime> =
        std::cell::RefCell::new(WebAutosaveRuntime {
            startup_checked: false,
            last_autosave_at: Instant::now(),
            last_autosave_revision: 0,
        });
}

/// Updates the global unsaved-changes flag read by the browser `beforeunload` listener.
#[cfg(target_arch = "wasm32")]
pub(crate) fn set_web_unsaved_dirty(dirty: bool) {
    WEB_UNSAVED_DIRTY.store(dirty, Ordering::Relaxed);
}

/// Registers a single idempotent `window` `beforeunload` handler that prompts the user before
/// leaving or closing the tab whenever [`set_web_unsaved_dirty`] is `true`.
#[cfg(target_arch = "wasm32")]
pub(crate) fn register_web_beforeunload_guard() -> bool {
    use wasm_bindgen::{JsCast, JsValue, closure::Closure};

    if WEB_BEFOREUNLOAD_REGISTERED.swap(true, Ordering::SeqCst) {
        return true;
    }
    let Some(window) = web_sys::window() else {
        WEB_BEFOREUNLOAD_REGISTERED.store(false, Ordering::SeqCst);
        return false;
    };
    let window_js: &JsValue = window.as_ref();
    let Ok(add_listener_val) =
        js_sys::Reflect::get(window_js, &JsValue::from_str("addEventListener"))
    else {
        WEB_BEFOREUNLOAD_REGISTERED.store(false, Ordering::SeqCst);
        return false;
    };
    let Ok(add_listener) = add_listener_val.dyn_into::<js_sys::Function>() else {
        WEB_BEFOREUNLOAD_REGISTERED.store(false, Ordering::SeqCst);
        return false;
    };

    let handler = Closure::<dyn FnMut(JsValue)>::new(move |event: JsValue| {
        if !WEB_UNSAVED_DIRTY.load(Ordering::Relaxed) {
            return;
        }
        if let Ok(prevent) = js_sys::Reflect::get(&event, &JsValue::from_str("preventDefault"))
            && let Ok(func) = prevent.dyn_into::<js_sys::Function>()
        {
            let _ = func.call0(&event);
        }
        let _ = js_sys::Reflect::set(
            &event,
            &JsValue::from_str("returnValue"),
            &JsValue::from_str(""),
        );
    });

    let attached = add_listener
        .call2(
            window_js,
            &JsValue::from_str("beforeunload"),
            handler.as_ref(),
        )
        .is_ok();
    if attached {
        handler.forget();
        true
    } else {
        WEB_BEFOREUNLOAD_REGISTERED.store(false, Ordering::SeqCst);
        false
    }
}

#[cfg(target_arch = "wasm32")]
fn web_local_storage() -> Option<web_sys::Storage> {
    web_sys::window().and_then(|w| w.local_storage().ok().flatten())
}

#[cfg(target_arch = "wasm32")]
fn read_web_autosave_chunk_count(storage: &web_sys::Storage) -> usize {
    storage
        .get_item(WEB_AUTOSAVE_META_KEY)
        .ok()
        .flatten()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|meta| meta.get("chunks").and_then(Value::as_u64))
        .unwrap_or(0) as usize
}

/// Saves a chunked, quota-guarded recovery snapshot of `session` into browser `localStorage`.
#[cfg(target_arch = "wasm32")]
pub(crate) fn save_web_autosave_snapshot(
    session: &Session,
) -> Result<RecoverySnapshotInfo, String> {
    let storage =
        web_local_storage().ok_or_else(|| "Browser localStorage is unavailable".to_owned())?;
    let project = prepare_project_for_zcraft_save(session);
    let timestamp_secs = (js_sys::Date::now() / 1000.0).max(0.0) as u64;
    let (meta_json, chunks, raw_len) = encode_web_autosave_chunks(
        &project,
        timestamp_secs,
        WEB_AUTOSAVE_MAX_ENCODED_BYTES,
        WEB_AUTOSAVE_CHUNK_CHARS,
    )?;

    let prev_chunks = read_web_autosave_chunk_count(&storage);
    for (idx, chunk) in chunks.iter().enumerate() {
        let key = format!("{WEB_AUTOSAVE_CHUNK_PREFIX}{idx}");
        if storage.set_item(&key, chunk).is_err() {
            // Roll back partial chunks beyond previous snapshot if quota was hit mid-write.
            for cleanup_idx in 0..=idx {
                let cleanup_key = format!("{WEB_AUTOSAVE_CHUNK_PREFIX}{cleanup_idx}");
                let _ = storage.remove_item(&cleanup_key);
            }
            let _ = storage.remove_item(WEB_AUTOSAVE_META_KEY);
            return Err("Browser localStorage quota exceeded while saving autosave".to_owned());
        }
    }
    if storage.set_item(WEB_AUTOSAVE_META_KEY, &meta_json).is_err() {
        for idx in 0..chunks.len() {
            let key = format!("{WEB_AUTOSAVE_CHUNK_PREFIX}{idx}");
            let _ = storage.remove_item(&key);
        }
        return Err(
            "Browser localStorage quota exceeded while writing autosave metadata".to_owned(),
        );
    }
    for stale_idx in chunks.len()..prev_chunks {
        let key = format!("{WEB_AUTOSAVE_CHUNK_PREFIX}{stale_idx}");
        let _ = storage.remove_item(&key);
    }

    Ok(RecoverySnapshotInfo {
        filename: format!("browser-autosave-{timestamp_secs}.zcraft"),
        timestamp_secs,
        size_bytes: raw_len,
        path: PathBuf::from(WEB_AUTOSAVE_META_KEY),
    })
}

/// Reads lightweight metadata for an existing browser `localStorage` recovery snapshot without
/// decoding the full project archive.
#[cfg(target_arch = "wasm32")]
pub(crate) fn peek_web_autosave_info() -> Option<RecoverySnapshotInfo> {
    let storage = web_local_storage()?;
    let meta_json = storage.get_item(WEB_AUTOSAVE_META_KEY).ok()??;
    let meta: Value = serde_json::from_str(&meta_json).ok()?;
    if meta.get("version").and_then(Value::as_u64) != Some(1) {
        return None;
    }
    let chunk_count = meta.get("chunks").and_then(Value::as_u64)? as usize;
    if chunk_count == 0 {
        return None;
    }
    // Quick check that the first chunk exists before advertising recovery.
    storage
        .get_item(&format!("{WEB_AUTOSAVE_CHUNK_PREFIX}0"))
        .ok()??;
    let timestamp_secs = meta
        .get("timestamp_secs")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let size_bytes = meta.get("raw_bytes").and_then(Value::as_u64).unwrap_or(0);
    Some(RecoverySnapshotInfo {
        filename: format!("browser-autosave-{timestamp_secs}.zcraft"),
        timestamp_secs,
        size_bytes,
        path: PathBuf::from(WEB_AUTOSAVE_META_KEY),
    })
}

/// Loads and verifies the saved browser `localStorage` autosave [`Project`] snapshot.
#[cfg(target_arch = "wasm32")]
pub(crate) fn load_web_autosave_snapshot() -> Result<Project, String> {
    let storage =
        web_local_storage().ok_or_else(|| "Browser localStorage is unavailable".to_owned())?;
    let meta_json = storage
        .get_item(WEB_AUTOSAVE_META_KEY)
        .ok()
        .flatten()
        .ok_or_else(|| "No browser autosave snapshot found".to_owned())?;
    let meta: Value = serde_json::from_str(&meta_json)
        .map_err(|e| format!("invalid browser autosave metadata: {e}"))?;
    let chunk_count = meta
        .get("chunks")
        .and_then(Value::as_u64)
        .ok_or_else(|| "browser autosave metadata missing chunk count".to_owned())?
        as usize;
    let mut chunks = Vec::with_capacity(chunk_count);
    for idx in 0..chunk_count {
        let key = format!("{WEB_AUTOSAVE_CHUNK_PREFIX}{idx}");
        let chunk = storage
            .get_item(&key)
            .ok()
            .flatten()
            .ok_or_else(|| format!("missing browser autosave chunk {idx}"))?;
        chunks.push(chunk);
    }
    let (project, _, _) = decode_web_autosave_chunks(&meta_json, &chunks)?;
    Ok(project)
}

/// Removes any stored browser `localStorage` autosave snapshot and all of its chunks.
#[cfg(target_arch = "wasm32")]
pub(crate) fn clear_web_autosave_snapshot() {
    let Some(storage) = web_local_storage() else {
        return;
    };
    let count = read_web_autosave_chunk_count(&storage).max(16);
    for idx in 0..count {
        let key = format!("{WEB_AUTOSAVE_CHUNK_PREFIX}{idx}");
        let _ = storage.remove_item(&key);
    }
    let _ = storage.remove_item(WEB_AUTOSAVE_META_KEY);
}

// ---------------------------------------------------------------------------
// Top-Level ZoidCraft Desktop Application
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PendingUnsavedAction {
    NewProject,
    OpenProject,
    #[cfg(not(target_arch = "wasm32"))]
    OpenDroppedProject(PathBuf),
    #[cfg(target_arch = "wasm32")]
    OpenDroppedWebProject(web::File),
    CloseWindow,
}

/// Computes the bottom-center anchor point inside the viewport for the non-blocking
/// crash-recovery banner so it never obscures the top toolbar, sidebar, top-left viewport
/// HUD, or floating Graph Editor titlebar.
#[cfg(test)]
pub(crate) fn recovery_banner_anchor_pos(viewport_rect: Option<Rect>) -> Option<Pos2> {
    let vp = viewport_rect?;
    if vp.width() <= 60.0 || vp.height() <= 40.0 {
        return None;
    }
    Some(Pos2::new(
        vp.center().x.round(),
        (vp.bottom() - 10.0).round(),
    ))
}

/// Computes `(clamped_banner_width, viewport_inner_bounds)` strictly within
/// `viewport_rect.shrink(8.0)` so viewport banners never spill onto the Media sidebar
/// even at compact `800×540` window sizes.
pub(crate) fn compute_viewport_banner_bounds(
    viewport_rect: Rect,
    max_width: f32,
) -> Option<(f32, Rect)> {
    let bounds = viewport_rect.shrink(8.0);
    if bounds.width() <= 40.0 || bounds.height() <= 30.0 {
        return None;
    }
    let width = bounds.width().min(max_width).floor();
    Some((width, bounds))
}

/// Clamped placement geometry for transient bottom-right status/error toasts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StatusToastBounds {
    pub(crate) bounds: Rect,
    pub(crate) anchor_pos: Pos2,
    pub(crate) max_width: f32,
}

/// Computes strictly clamped bottom-right toast bounds inside `viewport_rect` (left of the
/// 44px Audio Level Meter + 3px divider reserve) with a bounded maximum width so long error
/// messages wrap cleanly and never overlap Track A1 or the right VU meter.
#[cfg(test)]
pub(crate) fn compute_status_toast_bounds(
    viewport_rect: Option<Rect>,
    screen_rect: Rect,
) -> StatusToastBounds {
    compute_status_toast_bounds_with_bottom_inset(viewport_rect, screen_rect, 0.0)
}

/// Computes strictly clamped bottom-right toast bounds inside `viewport_rect` with an optional
/// `extra_bottom_inset` (e.g., to lift the toast strictly above the autosave Recovery banner).
pub(crate) fn compute_status_toast_bounds_with_bottom_inset(
    viewport_rect: Option<Rect>,
    screen_rect: Rect,
    extra_bottom_inset: f32,
) -> StatusToastBounds {
    const EDGE_PAD: f32 = 10.0;
    const VU_METER_RESERVE: f32 = 56.0; // 44px VU meter + 3px divider + 9px margin
    const MAX_TOAST_WIDTH: f32 = 420.0;

    let base = viewport_rect
        .map(|vp| vp.intersect(screen_rect))
        .filter(|vp| vp.width() >= 80.0 && vp.height() >= 40.0)
        .unwrap_or(screen_rect);

    let left = (base.left() + EDGE_PAD).min(base.right());
    let top = (base.top() + EDGE_PAD).min(base.bottom());
    let right_margin = if base.width() > 240.0 {
        VU_METER_RESERVE
    } else {
        EDGE_PAD
    };
    let right = (base.right() - right_margin)
        .max(left + 40.0)
        .min(base.right());
    let bottom_pad = EDGE_PAD + extra_bottom_inset.max(0.0);
    let bottom = (base.bottom() - bottom_pad)
        .max(top + 20.0)
        .min(base.bottom());
    let bounds = Rect::from_min_max(Pos2::new(left, top), Pos2::new(right, bottom));
    let max_width = bounds.width().clamp(40.0, MAX_TOAST_WIDTH).floor();
    StatusToastBounds {
        bounds,
        anchor_pos: Pos2::new(bounds.right().round(), bounds.bottom().round()),
        max_width,
    }
}

/// Formats a concise missing-plugin banner label with explicit total count and `+{N} more`
/// overflow suffix so long lists never clip mid-word (hover tooltip always lists every plugin).
pub(crate) fn format_missing_plugins_banner_summary(
    plugin_names: &[String],
    narrow: bool,
) -> String {
    match plugin_names {
        [] => String::new(),
        [only] => format!("Disabled plugin (1): {only}"),
        [first, second] => {
            if !narrow && first.chars().count() + second.chars().count() <= 24 {
                format!("Disabled plugins (2): {first}, {second}")
            } else {
                format!("Disabled plugins (2): {first}, +1 more")
            }
        }
        [first, second, rest @ ..] => {
            let total = 2 + rest.len();
            if !narrow && first.chars().count() + second.chars().count() <= 22 {
                format!(
                    "Disabled plugins ({total}): {first}, {second}, +{} more",
                    rest.len()
                )
            } else {
                format!("Disabled plugins ({total}): {first}, +{} more", total - 1)
            }
        }
    }
}

/// Formats a Unix timestamp (seconds) as clean human-readable `YYYY-MM-DD HH:MM UTC`
/// using Howard Hinnant's civil-from-days conversion (zero external dependencies).
pub(crate) fn format_recovery_timestamp(timestamp_secs: u64) -> String {
    if timestamp_secs == 0 {
        return "unsaved snapshot".to_owned();
    }
    let secs_per_day = 86_400_u64;
    let days = (timestamp_secs / secs_per_day) as i64;
    let secs_of_day = timestamp_secs % secs_per_day;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;

    // Civil date from days since 1970-01-01.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let day = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let month = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let year = y + i64::from(month <= 2);

    format!("{year:04}-{month:02}-{day:02} {hour:02}:{minute:02} UTC")
}

/// Detects built-in plugins required by the currently open project that are disabled in
/// `session.plugins`, producing zero false positives on a fresh blank project.
pub(crate) fn collect_disabled_required_plugins(session: &Session) -> Vec<String> {
    let mut required = session.active_plugin_dependencies();
    if !session.project.compositions.is_empty() {
        required.insert("precomp-plus".to_owned());
    }
    for track in &session.document().video_tracks {
        for clip in &track.clips {
            if clip.kind.eq_ignore_ascii_case("precomp")
                || clip.kind.to_ascii_lowercase().contains("precomp-plus")
            {
                required.insert("precomp-plus".to_owned());
            }
            for obj in &clip.objects_3d {
                let kind_lc = obj.kind.to_ascii_lowercase();
                if kind_lc == "flare" || kind_lc.contains("optical-flare") {
                    required.insert("optical-flares".to_owned());
                }
            }
        }
    }
    required
        .into_iter()
        .filter(|id| session.plugins.get(id).is_some() && !session.plugins.is_enabled(id))
        .collect()
}

// ---------------------------------------------------------------------------
// Real Disk Autosave & Crash Recovery Helpers (Zero External Crates)
// ---------------------------------------------------------------------------

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn default_recovery_dir() -> PathBuf {
    #[cfg(target_os = "macos")]
    {
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(home)
                .join("Library")
                .join("Application Support")
                .join("ZoidCraft")
                .join("recovery");
        }
    }
    #[cfg(target_os = "windows")]
    {
        if let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty()) {
            return PathBuf::from(appdata).join("ZoidCraft").join("recovery");
        }
    }
    #[cfg(all(not(target_os = "macos"), not(target_os = "windows")))]
    {
        if let Some(xdg) = std::env::var_os("XDG_DATA_HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(xdg).join("zoidcraft").join("recovery");
        }
        if let Some(home) = std::env::var_os("HOME").filter(|v| !v.is_empty()) {
            return PathBuf::from(home)
                .join(".local")
                .join("share")
                .join("zoidcraft")
                .join("recovery");
        }
    }
    std::env::temp_dir().join("zoidcraft-recovery")
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn parse_autosave_timestamp_secs(filename: &str, fallback_secs: u64) -> u64 {
    let Some(stem) = filename
        .strip_prefix("autosave-")
        .and_then(|s| s.strip_suffix(".zcraft"))
    else {
        return fallback_secs;
    };
    let head = stem.split('-').next().unwrap_or(stem);
    head.parse::<u64>().unwrap_or(fallback_secs)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn list_recovery_snapshots_in_dir(dir: &Path) -> Vec<RecoverySnapshotInfo> {
    let Ok(entries) = fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut snapshots = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(filename) = path.file_name().and_then(|n| n.to_str()).map(str::to_owned) else {
            continue;
        };
        if !filename.starts_with("autosave-") || !filename.ends_with(".zcraft") {
            continue;
        }
        let meta = entry.metadata().ok();
        let size_bytes = meta.as_ref().map_or(0, fs::Metadata::len);
        let modified_secs = meta
            .as_ref()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |d| d.as_secs());
        let timestamp_secs = parse_autosave_timestamp_secs(&filename, modified_secs);
        snapshots.push(RecoverySnapshotInfo {
            filename,
            timestamp_secs,
            size_bytes,
            path,
        });
    }
    snapshots.sort_by(|a, b| {
        b.timestamp_secs
            .cmp(&a.timestamp_secs)
            .then_with(|| b.filename.cmp(&a.filename))
    });
    snapshots
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn save_recovery_snapshot_to_dir(
    project: &Project,
    dir: &Path,
    max_keep: usize,
) -> Result<RecoverySnapshotInfo, String> {
    fs::create_dir_all(dir).map_err(|e| format!("failed to create recovery directory: {e}"))?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let millis = now.subsec_millis();
    let mut candidate_name = format!("autosave-{secs}-{millis:03}.zcraft");
    let mut path = dir.join(&candidate_name);
    let mut seq = 1u32;
    while path.exists() {
        candidate_name = format!("autosave-{secs}-{millis:03}-{seq}.zcraft");
        path = dir.join(&candidate_name);
        seq = seq.saturating_add(1);
    }

    save_zcraft(project, &path).map_err(|e| format!("failed to write recovery snapshot: {e}"))?;
    let size_bytes = fs::metadata(&path).map_or(0, |m| m.len());

    // Prune older snapshots beyond `max_keep`, keeping the newest entries.
    let all = list_recovery_snapshots_in_dir(dir);
    for old in all.into_iter().skip(max_keep.max(1)) {
        let _ = fs::remove_file(&old.path);
    }

    Ok(RecoverySnapshotInfo {
        filename: candidate_name,
        timestamp_secs: secs,
        size_bytes,
        path,
    })
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn paths_refer_to_same_file(a: &Path, b: &Path) -> bool {
    if a == b {
        return true;
    }
    if let (Ok(ca), Ok(cb)) = (fs::canonicalize(a), fs::canonicalize(b))
        && ca == cb
    {
        return true;
    }
    // Also check canonical parent + exact filename in case one path does not exist yet.
    if let (Some(ap), Some(af), Some(bp), Some(bf)) =
        (a.parent(), a.file_name(), b.parent(), b.file_name())
        && af == bf
        && let (Ok(cap), Ok(cbp)) = (fs::canonicalize(ap), fs::canonicalize(bp))
        && cap == cbp
    {
        return true;
    }
    false
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn validate_save_without_assets_target(
    current_file: Option<&Path>,
    target_path: &Path,
) -> Result<(), String> {
    if let Some(current) = current_file
        && paths_refer_to_same_file(current, target_path)
    {
        return Err(
            "Choose a separate output path when saving without assets so embedded media in the current project is preserved."
                .to_owned(),
        );
    }
    Ok(())
}

/// Prepares a cloned [`Project`] for native `.zcraft` saving, syncing active plugin
/// requirements and recording all currently enabled safe built-in plugins inside
/// `sequence.preserved["zoidcraftEnabledPlugins"]` so reopening preserves the exact look.
pub(crate) fn prepare_project_for_zcraft_save(session: &Session) -> Project {
    let mut project = session.project.clone();
    for dep in session.active_plugin_dependencies() {
        project.plugin_requirements.insert(dep);
    }
    let mut enabled: Vec<String> = session
        .plugins
        .enabled_ids()
        .iter()
        .filter(|id| id.as_str() != "core")
        .cloned()
        .collect();
    enabled.sort();
    enabled.dedup();
    if enabled.is_empty() {
        project
            .sequence
            .preserved
            .remove(ZOIDCRAFT_ENABLED_PLUGINS_KEY);
    } else {
        project
            .sequence
            .preserved
            .insert(ZOIDCRAFT_ENABLED_PLUGINS_KEY.to_owned(), json!(enabled));
    }
    project
}

/// Builds a lightweight recovery [`Project`] snapshot for background periodic autosave
/// without copying or gzipping embedded assets larger than
/// [`AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES`] when they already have a local `url`.
/// Leaves `session.project` completely untouched.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn prepare_project_for_autosave(session: &Session) -> Project {
    let src = &session.project;
    let mut assets = BTreeMap::new();
    for (k, rec) in &src.assets {
        let should_externalize = rec.url.is_some()
            && rec
                .bytes
                .as_ref()
                .is_some_and(|b| b.len() > AUTOSAVE_EXTERNALIZE_THRESHOLD_BYTES);
        let copy = if should_externalize {
            AssetRecord {
                id: rec.id.clone(),
                kind: rec.kind,
                source: rec.source,
                filename: rec.filename.clone(),
                url: rec.url.clone(),
                external: true,
                bytes: None,
                missing: rec.missing,
                sha256: rec.sha256.clone(),
                size: rec.size,
                preserved: rec.preserved.clone(),
            }
        } else {
            rec.clone()
        };
        assets.insert(k.clone(), copy);
    }

    let mut plugin_requirements = src.plugin_requirements.clone();
    for dep in session.active_plugin_dependencies() {
        plugin_requirements.insert(dep);
    }
    let mut sequence = src.sequence.clone();
    let mut enabled: Vec<String> = session
        .plugins
        .enabled_ids()
        .iter()
        .filter(|id| id.as_str() != "core")
        .cloned()
        .collect();
    enabled.sort();
    enabled.dedup();
    if enabled.is_empty() {
        sequence.preserved.remove(ZOIDCRAFT_ENABLED_PLUGINS_KEY);
    } else {
        sequence
            .preserved
            .insert(ZOIDCRAFT_ENABLED_PLUGINS_KEY.to_owned(), json!(enabled));
    }

    Project {
        version: src.version,
        name: src.name.clone(),
        assets,
        media: src.media.clone(),
        sequence,
        compositions: src.compositions.clone(),
        composition_ids: src.composition_ids.clone(),
        plugin_requirements,
        imported_from_pz_version: src.imported_from_pz_version,
        preserved: src.preserved.clone(),
    }
}

/// Extracts persisted enabled built-in plugin IDs saved into `sequence.preserved` by
/// [`prepare_project_for_zcraft_save`].
pub(crate) fn extract_saved_enabled_plugins(project: &Project) -> Vec<String> {
    let Some(arr) = project
        .sequence
        .preserved
        .get(ZOIDCRAFT_ENABLED_PLUGINS_KEY)
        .and_then(Value::as_array)
    else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

pub(crate) fn prepare_project_for_pz_export(session: &Session) -> Project {
    let mut project = session.project.clone();
    for dep in session.active_plugin_dependencies() {
        project.plugin_requirements.insert(dep);
    }
    // Never pollute vanilla CM3 `.pz` archives with ZoidCraft-only persisted plugin state.
    project
        .sequence
        .preserved
        .remove(ZOIDCRAFT_ENABLED_PLUGINS_KEY);
    project
}

pub(crate) fn sanitize_filename(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim();
    if trimmed.is_empty() {
        "Project".to_owned()
    } else {
        trimmed.to_owned()
    }
}

impl ZoidCraftApp {
    pub(crate) fn show_toast(&mut self, message: impl Into<String>) {
        self.status_toast = Some((message.into(), Instant::now()));
    }

    pub(crate) fn has_unsaved_changes(&self) -> bool {
        self.session.is_dirty() || self.session.history.is_in_transaction()
    }

    pub(crate) fn needs_confirmation_before_replace_or_close(&self) -> bool {
        self.has_unsaved_changes() || self.active_export.is_some()
    }

    pub(crate) fn update_window_title(&mut self, ctx: &egui::Context) {
        let dirty = if self.has_unsaved_changes() { "*" } else { "" };
        let mode_tag = if self.session.plugins.is_vanilla_cm3_mode() {
            " (Core Mode)"
        } else {
            ""
        };
        let name = &self.session.document().project_name;
        let title = format!("ZoidCraft{mode_tag} - {name}{dirty}");
        if title != self.last_title {
            self.last_title = title.clone();
            ctx.send_viewport_cmd(egui::ViewportCommand::Title(title));
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.maybe_init_web_persistence();
            set_web_unsaved_dirty(self.needs_confirmation_before_replace_or_close());
            self.maybe_periodic_web_autosave(ctx);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn maybe_init_web_persistence(&mut self) {
        let should_check = WEB_AUTOSAVE_RUNTIME.with(|rt| {
            let mut rt = rt.borrow_mut();
            if rt.startup_checked {
                false
            } else {
                rt.startup_checked = true;
                rt.last_autosave_at = Instant::now();
                rt.last_autosave_revision = self.session.revision();
                true
            }
        });
        if should_check {
            let _ = register_web_beforeunload_guard();
            if self.startup_recovery_banner.is_none()
                && !self.has_unsaved_changes()
                && let Some(info) = peek_web_autosave_info()
            {
                self.startup_recovery_banner = Some(info);
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn maybe_periodic_web_autosave(&mut self, ctx: &egui::Context) {
        let rev = self.session.revision();
        let needs_snapshot = self.session.is_dirty()
            && WEB_AUTOSAVE_RUNTIME.with(|rt| rt.borrow().last_autosave_revision != rev);
        if !needs_snapshot {
            return;
        }
        let elapsed = WEB_AUTOSAVE_RUNTIME.with(|rt| rt.borrow().last_autosave_at.elapsed());
        if elapsed >= WEB_AUTOSAVE_INTERVAL {
            WEB_AUTOSAVE_RUNTIME.with(|rt| {
                rt.borrow_mut().last_autosave_at = Instant::now();
            });
            if self.session.history.is_in_transaction() {
                return;
            }
            match save_web_autosave_snapshot(&self.session) {
                Ok(_) => {
                    WEB_AUTOSAVE_RUNTIME.with(|rt| {
                        rt.borrow_mut().last_autosave_revision = rev;
                    });
                }
                Err(err) => {
                    tracing::warn!("web autosave snapshot failed: {err}");
                }
            }
        } else {
            ctx.request_repaint_after(Duration::from_secs(5));
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn handle_restore_web_autosave(&mut self) {
        match load_web_autosave_snapshot() {
            Ok(project) => {
                let before_restore = self.session.snapshot();
                self.load_project_into_session(project, None);
                self.session
                    .history
                    .push("Restore Browser Autosave", before_restore);
                self.startup_recovery_banner = None;
                let rev = self.session.revision();
                WEB_AUTOSAVE_RUNTIME.with(|rt| {
                    let mut rt = rt.borrow_mut();
                    rt.last_autosave_at = Instant::now();
                    rt.last_autosave_revision = rev;
                });
                self.show_toast("Restored browser autosave snapshot");
            }
            Err(err) => {
                self.show_toast(format!("Failed to restore browser autosave: {err}"));
            }
        }
    }

    pub(crate) fn reset_ephemeral_ui_state(&mut self) {
        self.pending_eyedropper = None;
        self.eyedropper_swallow_pointer = false;
        self.last_viewport_render_error = None;

        // Reset ephemeral panel states (plugin_notice, shader_editor, expr_editor,
        // tree drag/rename states, pickers, floating windows) while preserving user
        // layout splits, preferences, export settings, and probed FFmpeg status.
        let mut fresh_panels = PanelsState::default();
        std::mem::swap(
            &mut fresh_panels.objects_split,
            &mut self.workspace.panels.objects_split,
        );
        std::mem::swap(
            &mut fresh_panels.effects_split,
            &mut self.workspace.panels.effects_split,
        );
        std::mem::swap(
            &mut fresh_panels.prop_columns,
            &mut self.workspace.panels.prop_columns,
        );
        std::mem::swap(
            &mut fresh_panels.export_settings,
            &mut self.workspace.panels.export_settings,
        );
        std::mem::swap(
            &mut fresh_panels.export_status,
            &mut self.workspace.panels.export_status,
        );
        std::mem::swap(
            &mut fresh_panels.export_queue,
            &mut self.workspace.panels.export_queue,
        );
        fresh_panels.export_queue_running = self.workspace.panels.export_queue_running;
        std::mem::swap(
            &mut fresh_panels.recovery_snapshots,
            &mut self.workspace.panels.recovery_snapshots,
        );
        std::mem::swap(
            &mut fresh_panels.app_settings,
            &mut self.workspace.panels.app_settings,
        );
        self.workspace.panels = fresh_panels;
        self.workspace.panels.plugin_notice = None;
        self.workspace.panels.floating_windows.clear_focus();
        self.workspace.audio_meter = Default::default();
        self.workspace.viewport_draw_tool = None;

        self.timeline.drag_state = TimelineDragState::None;
        self.timeline.snap_guide_x = None;
        self.timeline.box_selected_clips.clear();
        self.viewport.drag_state = ViewportDragState::None;
        self.viewport.active_draw_tool = None;
        self.viewport.active_camera_id = None;
        self.viewport.reset_2d_view();

        self.graph_editor.focused_property = None;
        self.graph_editor.selected_keys.clear();
        self.graph_editor.handles.clear();
        self.graph_editor.drag_state = GraphDragState::None;
        self.graph_editor.snap_guide_frame = None;
        self.graph_editor.solo_track = None;
        self.graph_editor.hidden_tracks.clear();
        self.graph_editor.show_all_easings_popup = false;
    }

    pub(crate) fn handle_new_project(&mut self) {
        if self.needs_confirmation_before_replace_or_close() {
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::NewProject);
            return;
        }
        self.perform_new_project();
    }

    pub(crate) fn perform_new_project(&mut self) {
        self.detach_active_export();
        self.pending_media_imports.clear();
        #[cfg(target_arch = "wasm32")]
        self.web.clear_pending();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pending_autosave = None;
        }
        if self.session.history.is_in_transaction() {
            let _ = self.session.cancel_transaction();
        }
        let _ = self.session.execute(Command::NewProject);
        apply_default_project_fps(
            &mut self.session,
            self.workspace.panels.app_settings.default_fps,
        );
        self.session.selection.clear_all();
        auto_select_default_scene_clip(&mut self.session);
        self.session.playback.editing_camera = false;
        self.current_file = None;
        self.pending_unsaved_dialog = None;
        #[cfg(target_arch = "wasm32")]
        {
            set_web_unsaved_dirty(false);
            clear_web_autosave_snapshot();
        }
        self.invalidate_project_hash_caches();
        self.cached_missing_plugins_revision = None;
        self.cached_missing_plugins.clear();
        self.dismissed_missing_plugins.clear();
        self.reset_live_audio_output();
        self.audio_clock.set_frame(0.0);
        self.last_audio_epoch = self.session.playback.epoch_id;
        self.last_audio_frame = 0.0;
        self.decoded_audio.clear();
        self.asset_waveform_peaks.clear();
        self.timeline.audio_peaks.clear();
        self.reset_ephemeral_ui_state();
        self.viewport.camera.reset();
        self.force_single_render = true;
        sync_export_settings_from_session(
            &self.session,
            &mut self.workspace.panels.export_settings,
        );
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
        self.show_toast("Created new default project");
    }

    pub(crate) fn handle_open_project(&mut self) {
        if self.needs_confirmation_before_replace_or_close() {
            self.pending_unsaved_dialog = Some(PendingUnsavedAction::OpenProject);
            return;
        }
        self.perform_open_project();
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn perform_open_project(&mut self) {
        self.pause_playback_for_modal_dialog();
        let chosen = rfd::FileDialog::new()
            .add_filter("ZoidCraft & .pz Projects", &["zcraft", "pz"])
            .pick_file();
        self.last_tick = Instant::now();
        let Some(path) = chosen else {
            return;
        };
        self.open_project_from_path(&path);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn open_project_from_path(&mut self, path: &Path) {
        let is_pz = path
            .extension()
            .and_then(|e| e.to_str())
            .is_some_and(|ext| ext.eq_ignore_ascii_case("pz"));

        if is_pz {
            self.finish_open_pz(path, import_pz(path));
        } else {
            self.finish_open_zcraft(path, load_zcraft(path));
        }
    }

    pub(crate) fn finish_open_pz(
        &mut self,
        path: &Path,
        result: Result<zc_project::PzImportResult, zc_project::ProjectError>,
    ) {
        let plugins = self.session.plugins.clone();
        match result {
            Ok(imported) => {
                let warn_count = imported.warnings.len();
                let missing_plugins =
                    plugins.missing_required_plugins(&imported.project.plugin_requirements);
                self.load_project_into_session(imported.project, None);
                if !missing_plugins.is_empty() {
                    let missing_labels: Vec<&str> = missing_plugins
                        .iter()
                        .map(|id| plugins.get(id).map_or(id.as_str(), |p| p.name.as_str()))
                        .collect();
                    self.workspace.panels.plugin_notice = Some(format!(
                        "Imported project references disabled plugins: {}",
                        missing_labels.join(", ")
                    ));
                }
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project.pz");
                if warn_count > 0 {
                    self.show_toast(format!(
                        "Imported {name} ({warn_count} compatibility warning(s))"
                    ));
                } else {
                    self.show_toast(format!("Imported {name}"));
                }
            }
            Err(err) => self.show_toast(format!("Failed to import .pz: {err}")),
        }
    }

    pub(crate) fn finish_open_zcraft(
        &mut self,
        path: &Path,
        result: Result<Project, zc_project::ProjectError>,
    ) {
        match result {
            Ok(project) => {
                self.load_project_into_session(project, Some(path.to_path_buf()));
                let missing_plugins = self
                    .session
                    .plugins
                    .missing_required_plugins(&self.session.project.plugin_requirements);
                if !missing_plugins.is_empty() {
                    let missing_labels: Vec<&str> = missing_plugins
                        .iter()
                        .map(|id| {
                            self.session
                                .plugins
                                .get(id)
                                .map_or(id.as_str(), |p| p.name.as_str())
                        })
                        .collect();
                    self.workspace.panels.plugin_notice = Some(format!(
                        "Loaded project requires disabled plugins: {}",
                        missing_labels.join(", ")
                    ));
                }
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project.zcraft");
                self.show_toast(format!("Opened {name}"));
            }
            Err(err) => self.show_toast(format!("Failed to open .zcraft: {err}")),
        }
    }

    pub(crate) fn load_project_into_session(&mut self, project: Project, path: Option<PathBuf>) {
        self.detach_active_export();
        self.pending_media_imports.clear();
        #[cfg(target_arch = "wasm32")]
        self.web.clear_pending();
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.pending_autosave = None;
        }
        let mut plugins = self.session.plugins.clone();
        // Automatically re-enable safe built-in plugins recorded when this `.zcraft` was saved
        // so reopening a saved project renders identically without a false warning banner.
        for plugin_id in extract_saved_enabled_plugins(&project) {
            let _ = plugins.enable(&plugin_id);
        }
        self.session = Session::from_project(project, plugins);
        let (undo_entries, undo_bytes) = self.workspace.panels.app_settings.undo_limits();
        self.session.history.set_limits(undo_entries, undo_bytes);
        self.session.selection.clear_all();
        self.session.mark_saved();
        self.current_file = path;
        self.pending_unsaved_dialog = None;
        self.invalidate_project_hash_caches();
        self.cached_missing_plugins_revision = None;
        self.cached_missing_plugins.clear();
        self.dismissed_missing_plugins.clear();
        self.reset_live_audio_output();
        self.audio_clock
            .set_frame(self.session.playback.current_frame);
        self.last_audio_epoch = self.session.playback.epoch_id;
        self.last_audio_frame = self.session.playback.current_frame;
        self.rebuild_audio_caches();
        #[cfg(target_arch = "wasm32")]
        self.register_web_project_videos();
        self.reset_ephemeral_ui_state();
        self.viewport.camera.reset();
        self.force_single_render = true;
        sync_export_settings_from_session(
            &self.session,
            &mut self.workspace.panels.export_settings,
        );
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
    }

    pub(crate) fn handle_save_project(&mut self, include_assets: bool) -> bool {
        // Commit any active inspector/viewport drag transaction before saving so the saved
        // snapshot and `mark_saved()` revision agree cleanly.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }

        let target_path = if include_assets && let Some(existing) = self.current_file.clone() {
            existing
        } else {
            self.pause_playback_for_modal_dialog();
            let base = sanitize_filename(&self.session.document().project_name);
            let default_name = if include_assets {
                format!("{base}.zcraft")
            } else {
                format!("{base}-no-assets.zcraft")
            };
            #[cfg(target_arch = "wasm32")]
            let chosen = Some(PathBuf::from(default_name));
            #[cfg(not(target_arch = "wasm32"))]
            let chosen = rfd::FileDialog::new()
                .add_filter("ZoidCraft Project", &["zcraft"])
                .set_file_name(default_name)
                .save_file();
            self.last_tick = Instant::now();
            let Some(chosen) = chosen else {
                return false;
            };
            chosen
        };

        self.save_project_to_path(&target_path, include_assets)
    }

    pub(crate) fn save_project_to_path(
        &mut self,
        target_path: &Path,
        include_assets: bool,
    ) -> bool {
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }

        #[cfg(not(target_arch = "wasm32"))]
        if !include_assets
            && let Err(msg) =
                validate_save_without_assets_target(self.current_file.as_deref(), target_path)
        {
            self.show_toast(msg);
            return false;
        }

        let mut project_to_save = prepare_project_for_zcraft_save(&self.session);
        let mut omitted_embedded_count = 0usize;
        if !include_assets {
            for rec in project_to_save.assets.values_mut() {
                if rec.bytes.take().is_some() {
                    omitted_embedded_count += 1;
                    rec.external = true;
                    rec.missing = true;
                } else {
                    rec.external = true;
                }
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        let result = save_zcraft(&project_to_save, target_path).map_err(|err| err.to_string());
        #[cfg(target_arch = "wasm32")]
        let result = web::download_project(
            &project_to_save,
            target_path,
            zc_project::save_zcraft_to_bytes,
        );
        match result {
            Ok(()) => {
                if include_assets {
                    self.current_file = Some(target_path.to_path_buf());
                    self.session.mark_saved();
                    #[cfg(target_arch = "wasm32")]
                    {
                        set_web_unsaved_dirty(false);
                        clear_web_autosave_snapshot();
                    }
                }
                let name = target_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project.zcraft");
                let msg = if include_assets {
                    format!("Saved {name}")
                } else if omitted_embedded_count > 0 {
                    format!(
                        "Saved {name} without assets ({omitted_embedded_count} embedded asset(s) omitted; current project intact)"
                    )
                } else {
                    format!("Saved {name} (without assets)")
                };
                self.show_toast(msg);
                true
            }
            Err(err) => {
                self.show_toast(format!("Failed to save project: {err}"));
                false
            }
        }
    }

    pub(crate) fn handle_export_pz(&mut self) -> bool {
        // Commit any active inspector/viewport drag transaction before exporting so the
        // exported archive includes the latest edits cleanly.
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }

        self.pause_playback_for_modal_dialog();
        let base = sanitize_filename(&self.session.document().project_name);
        let default_name = format!("{base}.pz");
        #[cfg(target_arch = "wasm32")]
        let chosen = Some(PathBuf::from(default_name));
        #[cfg(not(target_arch = "wasm32"))]
        let chosen = rfd::FileDialog::new()
            .add_filter("Project Archive (.pz)", &["pz"])
            .set_file_name(default_name)
            .save_file();
        self.last_tick = Instant::now();
        let Some(target_path) = chosen else {
            return false;
        };

        let project_to_export = prepare_project_for_pz_export(&self.session);
        #[cfg(not(target_arch = "wasm32"))]
        let result = export_pz(&project_to_export, &target_path).map_err(|err| err.to_string());
        #[cfg(target_arch = "wasm32")]
        let result = web::download_project(
            &project_to_export,
            &target_path,
            zc_project::export_pz_to_bytes,
        );
        match result {
            Ok(()) => {
                let name = target_path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("project.pz");
                self.show_toast(format!("Exported .pz project: {name}"));
                true
            }
            Err(err) => {
                self.show_toast(format!("Failed to export .pz project: {err}"));
                false
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn refresh_recovery_snapshots(&mut self) {
        self.workspace.panels.recovery_snapshots =
            list_recovery_snapshots_in_dir(&self.recovery_dir);
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn poll_pending_autosave(&mut self, ctx: &egui::Context) {
        let Some(rx) = self.pending_autosave.as_ref() else {
            return;
        };
        match rx.try_recv() {
            Ok(Ok(rev)) => {
                self.pending_autosave = None;
                self.last_autosave_revision = rev;
                self.refresh_recovery_snapshots();
            }
            Ok(Err(err)) => {
                self.pending_autosave = None;
                tracing::warn!("autosave recovery snapshot failed: {err}");
            }
            Err(TryRecvError::Disconnected) => {
                self.pending_autosave = None;
            }
            Err(TryRecvError::Empty) => {
                ctx.request_repaint_after(Duration::from_millis(200));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn maybe_periodic_autosave(&mut self, ctx: &egui::Context) {
        self.poll_pending_autosave(ctx);
        if self.pending_autosave.is_some() {
            return;
        }
        let rev = self.session.revision();
        let needs_snapshot = self.session.is_dirty() && rev != self.last_autosave_revision;
        if !needs_snapshot {
            return;
        }
        if self.last_autosave_at.elapsed() >= AUTOSAVE_INTERVAL {
            self.last_autosave_at = Instant::now();
            if self.session.history.is_in_transaction() {
                return;
            }
            let recovery_project = prepare_project_for_autosave(&self.session);
            let recovery_dir = self.recovery_dir.clone();
            let (tx, rx) = mpsc::channel();
            let spawn_res =
                thread::Builder::new()
                    .name("zc-autosave".to_owned())
                    .spawn(move || {
                        let res = save_recovery_snapshot_to_dir(
                            &recovery_project,
                            &recovery_dir,
                            MAX_RECOVERY_SNAPSHOTS,
                        )
                        .map(|_| rev);
                        let _ = tx.send(res);
                    });
            match spawn_res {
                Ok(_) => {
                    self.pending_autosave = Some(rx);
                    ctx.request_repaint_after(Duration::from_millis(200));
                }
                Err(err) => {
                    tracing::warn!("failed to spawn zc-autosave thread: {err}");
                }
            }
        } else {
            ctx.request_repaint_after(Duration::from_secs(5));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn handle_save_recovery_snapshot_now(&mut self) {
        if self.session.history.is_in_transaction() {
            let _ = self.session.commit_transaction(true);
        }
        let project_to_save = prepare_project_for_zcraft_save(&self.session);
        match save_recovery_snapshot_to_dir(
            &project_to_save,
            &self.recovery_dir,
            MAX_RECOVERY_SNAPSHOTS,
        ) {
            Ok(info) => {
                self.last_autosave_at = Instant::now();
                self.last_autosave_revision = self.session.revision();
                self.refresh_recovery_snapshots();
                self.show_toast(format!("Saved recovery snapshot: {}", info.filename));
            }
            Err(err) => {
                self.show_toast(format!("Failed to save recovery snapshot: {err}"));
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn handle_restore_recovery_snapshot(&mut self, path: &Path) {
        match load_zcraft(path) {
            Ok(project) => {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("recovery snapshot")
                    .to_owned();
                let before_restore = self.session.snapshot();
                self.load_project_into_session(project, None);
                // Mark the restored snapshot dirty (`*`) so unsaved-changes protection triggers
                // until the user explicitly saves, while allowing Cmd+Z to undo the restore.
                self.session
                    .history
                    .push("Restore Recovery Snapshot", before_restore);
                self.startup_recovery_banner = None;
                self.last_autosave_at = Instant::now();
                self.last_autosave_revision = self.session.revision();
                self.show_toast(format!("Restored recovery snapshot: {name}"));
            }
            Err(err) => {
                self.show_toast(format!("Failed to restore recovery snapshot: {err}"));
            }
        }
    }

    pub(crate) fn visible_missing_required_plugins(&mut self) -> Vec<String> {
        let rev = self.session.revision();
        if self.cached_missing_plugins_revision != Some(rev) {
            self.cached_missing_plugins = collect_disabled_required_plugins(&self.session);
            self.cached_missing_plugins_revision = Some(rev);
        }
        self.cached_missing_plugins
            .iter()
            .filter(|id| !self.dismissed_missing_plugins.contains(*id))
            .cloned()
            .collect()
    }

    pub(crate) fn enable_required_plugins(&mut self, plugin_ids: &[String]) {
        let mut enabled_names = Vec::new();
        for plugin_id in plugin_ids {
            if self
                .session
                .execute(Command::EnablePlugin {
                    plugin_id: plugin_id.clone(),
                })
                .is_ok()
            {
                let label = self
                    .session
                    .plugins
                    .get(plugin_id)
                    .map_or_else(|| plugin_id.clone(), |p| p.name.clone());
                enabled_names.push(label);
                self.dismissed_missing_plugins.remove(plugin_id);
            }
        }
        self.invalidate_project_hash_caches();
        self.cached_missing_plugins_revision = None;
        self.workspace.panels.plugin_notice = None;
        self.force_single_render = true;
        if let Some(r) = self.renderer.as_mut() {
            r.invalidate();
        }
        if !enabled_names.is_empty() {
            self.show_toast(format!(
                "Enabled required {}: {}",
                if enabled_names.len() == 1 {
                    "plugin"
                } else {
                    "plugins"
                },
                enabled_names.join(", ")
            ));
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn handle_delete_recovery_snapshot(&mut self, path: &Path) {
        match fs::remove_file(path) {
            Ok(()) => {
                if self
                    .startup_recovery_banner
                    .as_ref()
                    .is_some_and(|b| b.path == path)
                {
                    self.startup_recovery_banner = None;
                }
                self.refresh_recovery_snapshots();
                self.show_toast("Deleted recovery snapshot");
            }
            Err(err) => {
                self.show_toast(format!("Failed to delete recovery snapshot: {err}"));
            }
        }
    }

    pub(crate) fn show_unsaved_changes_modal(&mut self, ctx: &egui::Context) {
        let Some(pending) = self.pending_unsaved_dialog.clone() else {
            return;
        };

        let has_unsaved = self.has_unsaved_changes();
        let has_export = self.active_export.is_some();

        let action_label = match &pending {
            PendingUnsavedAction::NewProject => "creating a new project",
            PendingUnsavedAction::OpenProject => "opening another project",
            #[cfg(not(target_arch = "wasm32"))]
            PendingUnsavedAction::OpenDroppedProject(_) => "opening another project",
            #[cfg(target_arch = "wasm32")]
            PendingUnsavedAction::OpenDroppedWebProject(_) => "opening another project",
            PendingUnsavedAction::CloseWindow => "closing ZoidCraft",
        };

        let (modal_title, prompt_text) = match (has_export, has_unsaved) {
            (true, true) => (
                "Export In Progress & Unsaved Changes",
                format!(
                    "An export is currently running and '{}' has unsaved changes. Cancel export and save before {action_label}?",
                    self.session.document().project_name
                ),
            ),
            (true, false) => (
                "Export In Progress",
                format!("An export is currently running. Cancel the export before {action_label}?"),
            ),
            (false, _) => (
                "Unsaved Changes",
                format!(
                    "Save changes to '{}' before {action_label}?",
                    self.session.document().project_name
                ),
            ),
        };

        let mut choice: Option<u8> = None;
        Area::new(Id::unique("zc_unsaved_backdrop"))
            .order(Order::Foreground)
            .fixed_pos(Pos2::ZERO)
            .show(ctx, |ui| {
                let screen = ctx.input(|i| i.viewport_rect());
                let _ = ui.allocate_rect(screen, egui::Sense::click());
                ui.painter().rect_filled(
                    screen,
                    CornerRadius::ZERO,
                    Color32::from_black_alpha(150),
                );
            });

        egui::Window::new(modal_title)
            .id(Id::unique("zc_unsaved_changes_modal"))
            .order(Order::Tooltip)
            .collapsible(false)
            .resizable(false)
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .frame(
                Frame::NONE
                    .fill(BG_PANEL)
                    .stroke(Stroke::new(1.0, self.theme.accent()))
                    .inner_margin(egui::Margin::same(14)),
            )
            .show(ctx, |ui| {
                ui.label(
                    egui::RichText::new(prompt_text)
                        .font(self.theme.font(FONT_SMALL))
                        .color(TEXT_BRIGHT),
                );
                ui.add_space(12.0);
                let primary_fill = Color32::from_rgb(0x2d, 0x67, 0xc4);
                let primary_stroke = Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4));
                let destructive_fill = Color32::from_rgb(0x6e, 0x25, 0x25);
                let destructive_stroke = Stroke::new(1.0, WARNING_BAR);
                let destructive_text = Color32::from_rgb(0xf6, 0xd8, 0xd8);
                let neutral_stroke = Stroke::new(1.0, BORDER_PANEL);
                let btn_min_size = Vec2::new(78.0, 26.0);

                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 8.0;
                    if has_unsaved {
                        let save_lbl = if has_export {
                            "Cancel Export & Save"
                        } else {
                            "Save"
                        };
                        let discard_lbl = if has_export {
                            "Cancel Export & Don't Save"
                        } else {
                            "Don't Save"
                        };
                        let save_btn = egui::Button::new(
                            egui::RichText::new(save_lbl)
                                .font(self.theme.font(FONT_SMALL))
                                .color(TEXT_BRIGHT),
                        )
                        .fill(primary_fill)
                        .stroke(primary_stroke)
                        .corner_radius(CornerRadius::same(2))
                        .min_size(btn_min_size);
                        if ui.add(save_btn).clicked() {
                            choice = Some(1);
                        }

                        let discard_btn = egui::Button::new(
                            egui::RichText::new(discard_lbl)
                                .font(self.theme.font(FONT_SMALL))
                                .color(destructive_text),
                        )
                        .fill(destructive_fill)
                        .stroke(destructive_stroke)
                        .corner_radius(CornerRadius::same(2))
                        .min_size(btn_min_size);
                        if ui.add(discard_btn).clicked() {
                            choice = Some(2);
                        }
                    } else {
                        let proceed_lbl = if matches!(pending, PendingUnsavedAction::CloseWindow) {
                            "Cancel Export & Close"
                        } else {
                            "Cancel Export & Proceed"
                        };
                        let proceed_btn = egui::Button::new(
                            egui::RichText::new(proceed_lbl)
                                .font(self.theme.font(FONT_SMALL))
                                .color(destructive_text),
                        )
                        .fill(destructive_fill)
                        .stroke(destructive_stroke)
                        .corner_radius(CornerRadius::same(2))
                        .min_size(btn_min_size);
                        if ui.add(proceed_btn).clicked() {
                            choice = Some(2);
                        }
                    }
                    let cancel_lbl = if has_export { "Keep Running" } else { "Cancel" };
                    let cancel_btn = egui::Button::new(
                        egui::RichText::new(cancel_lbl)
                            .font(self.theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    )
                    .fill(BG_BUTTON)
                    .stroke(neutral_stroke)
                    .corner_radius(CornerRadius::same(2))
                    .min_size(btn_min_size);
                    if ui.add(cancel_btn).clicked() || ctx.input(|i| i.key_pressed(Key::Escape)) {
                        choice = Some(0);
                    }
                });
            });

        match choice {
            Some(0) => {
                self.pending_unsaved_dialog = None;
            }
            Some(1) => {
                if self.session.history.is_in_transaction() {
                    let _ = self.session.commit_transaction(true);
                }
                if self.handle_save_project(true) {
                    self.pending_unsaved_dialog = None;
                    match pending {
                        PendingUnsavedAction::NewProject => self.perform_new_project(),
                        PendingUnsavedAction::OpenProject => self.perform_open_project(),
                        #[cfg(not(target_arch = "wasm32"))]
                        PendingUnsavedAction::OpenDroppedProject(path) => {
                            self.open_project_from_path(&path);
                        }
                        #[cfg(target_arch = "wasm32")]
                        PendingUnsavedAction::OpenDroppedWebProject(file) => {
                            self.open_web_project(file)
                        }
                        PendingUnsavedAction::CloseWindow => {
                            self.detach_active_export();
                            self.allow_close_now = true;
                            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                }
            }
            Some(2) => {
                if self.session.history.is_in_transaction() {
                    let _ = self.session.cancel_transaction();
                }
                self.pending_unsaved_dialog = None;
                match pending {
                    PendingUnsavedAction::NewProject => self.perform_new_project(),
                    PendingUnsavedAction::OpenProject => self.perform_open_project(),
                    #[cfg(not(target_arch = "wasm32"))]
                    PendingUnsavedAction::OpenDroppedProject(path) => {
                        self.open_project_from_path(&path);
                    }
                    #[cfg(target_arch = "wasm32")]
                    PendingUnsavedAction::OpenDroppedWebProject(file) => {
                        self.open_web_project(file)
                    }
                    PendingUnsavedAction::CloseWindow => {
                        self.detach_active_export();
                        self.allow_close_now = true;
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn show_status_overlays(&mut self, ctx: &egui::Context) {
        // OS file drag-and-drop hover indicator.
        let hovered_count = ctx.input(|i| i.raw.hovered_files.len());
        if hovered_count > 0 {
            Area::new(Id::unique("zc_drop_overlay"))
                .order(Order::Foreground)
                .interactable(false)
                .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
                .show(ctx, |ui| {
                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(2.0, self.theme.accent()))
                        .inner_margin(egui::Margin::symmetric(18, 12))
                        .show(ui, |ui| {
                            let label = if hovered_count == 1 {
                                "Drop .zcraft, .pz, or media file to open or import".to_owned()
                            } else {
                                format!(
                                    "Drop {hovered_count} files (.zcraft, .pz, or media) to open or import"
                                )
                            };
                            ui.label(
                                egui::RichText::new(label)
                                    .font(self.theme.font(FONT_SMALL))
                                    .color(TEXT_BRIGHT),
                            );
                        });
                });
        }

        // Non-blocking missing-plugin warning banner anchored at top-center of the viewport
        // (strictly clamped within `viewport_rect.shrink(8.0)`) with 1-click plugin enablement.
        if self.pending_eyedropper.is_none() {
            let missing_plugins = self.visible_missing_required_plugins();
            if !missing_plugins.is_empty() {
                let plugin_names: Vec<String> = missing_plugins
                    .iter()
                    .map(|id| {
                        self.session
                            .plugins
                            .get(id)
                            .map_or_else(|| id.clone(), |p| p.name.clone())
                    })
                    .collect();
                let full_list = plugin_names.join(", ");
                let mut enable_clicked = false;
                let mut dismiss_clicked = false;

                let (banner_w, clamp_bounds) = self
                    .last_viewport_rect
                    .and_then(|vp| compute_viewport_banner_bounds(vp, 560.0))
                    .map_or((480.0, None), |(w, b)| (w, Some(b)));
                let inner_w = (banner_w - 18.0).max(120.0);

                let mut area =
                    Area::new(Id::unique("zc_missing_plugins_banner")).order(Order::Middle);
                if let Some(bounds) = clamp_bounds {
                    let left = (bounds.center().x - banner_w * 0.5)
                        .clamp(
                            bounds.left(),
                            (bounds.right() - banner_w).max(bounds.left()),
                        )
                        .round();
                    area = area
                        .constrain_to(bounds)
                        .fixed_pos(Pos2::new(left, bounds.top().round()));
                } else {
                    area = area.anchor(Align2::CENTER_TOP, Vec2::new(0.0, 10.0));
                }

                let warn_border = Color32::from_rgb(0xd4, 0x9b, 0x38);
                let enable_fill = Color32::from_rgb(0x2d, 0x67, 0xc4);
                let enable_stroke = Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4));
                let narrow = banner_w < 380.0;
                let text_font = self
                    .theme
                    .font(if narrow { FONT_COMPACT } else { FONT_SMALL });
                let enable_label = if narrow {
                    "Enable Plugins"
                } else {
                    "Enable Required Plugins"
                };

                area.show(ctx, |ui| {
                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(1.0, warn_border))
                        .inner_margin(egui::Margin::symmetric(8, 5))
                        .show(ui, |ui| {
                            ui.set_min_width(inner_w);
                            ui.set_max_width(inner_w);
                            ui.allocate_ui_with_layout(
                                Vec2::new(inner_w, 24.0),
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.spacing_mut().item_spacing.x = 6.0;
                                    let dismiss_btn = egui::Button::new(
                                        egui::RichText::new("×")
                                            .font(self.theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .fill(BG_BUTTON)
                                    .stroke(Stroke::new(1.0, BORDER_PANEL))
                                    .corner_radius(CornerRadius::same(2))
                                    .min_size(Vec2::new(22.0, 22.0));
                                    if ui
                                        .add(dismiss_btn)
                                        .on_hover_text("Dismiss for this session")
                                        .clicked()
                                    {
                                        dismiss_clicked = true;
                                    }

                                    let enable_btn = egui::Button::new(
                                        egui::RichText::new(enable_label)
                                            .font(text_font.clone())
                                            .color(TEXT_BRIGHT),
                                    )
                                    .fill(enable_fill)
                                    .stroke(enable_stroke)
                                    .corner_radius(CornerRadius::same(2))
                                    .min_size(Vec2::new(0.0, 22.0));
                                    if ui.add(enable_btn).clicked() {
                                        enable_clicked = true;
                                    }

                                    let avail_lbl_w = ui.available_width().max(40.0);
                                    let msg = format_missing_plugins_banner_summary(
                                        &plugin_names,
                                        narrow,
                                    );
                                    let tooltip = format!(
                                        "This project uses features from {} disabled {}:\n{}",
                                        plugin_names.len(),
                                        if plugin_names.len() == 1 {
                                            "plugin"
                                        } else {
                                            "plugins"
                                        },
                                        full_list
                                    );
                                    ui.add_sized(
                                        [avail_lbl_w, 20.0],
                                        egui::Label::new(
                                            egui::RichText::new(msg)
                                                .font(text_font)
                                                .color(Color32::from_rgb(0xf2, 0xd1, 0x8b)),
                                        )
                                        .truncate(),
                                    )
                                    .on_hover_text(tooltip);
                                },
                            );
                        });
                });

                if enable_clicked {
                    self.enable_required_plugins(&missing_plugins);
                } else if dismiss_clicked {
                    for id in missing_plugins {
                        self.dismissed_missing_plugins.insert(id);
                    }
                }
            }
        }

        // Non-blocking startup crash-recovery banner positioned cleanly at the bottom-center
        // of the viewport and strictly clamped inside `viewport_rect.shrink(8.0)` so even at
        // 800×540 it never spills left onto the Media sidebar.
        let mut recovery_banner_extra_bottom_inset = 0.0_f32;
        if self.pending_eyedropper.is_none()
            && let Some(snap) = self.startup_recovery_banner.clone()
        {
            let mut restore_clicked = false;
            let mut dismiss_clicked = false;
            let formatted_ts = format_recovery_timestamp(snap.timestamp_secs);
            let (banner_w, clamp_bounds) = self
                .last_viewport_rect
                .and_then(|vp| compute_viewport_banner_bounds(vp, 460.0))
                .map_or((420.0, None), |(w, b)| (w, Some(b)));
            let inner_w = (banner_w - 18.0).max(120.0);
            let narrow = banner_w < 360.0;
            let text_font = self
                .theme
                .font(if narrow { FONT_COMPACT } else { FONT_SMALL });

            let mut area = Area::new(Id::unique("zc_recovery_banner")).order(Order::Middle);
            if let Some(bounds) = clamp_bounds {
                let left = (bounds.center().x - banner_w * 0.5)
                    .clamp(
                        bounds.left(),
                        (bounds.right() - banner_w).max(bounds.left()),
                    )
                    .round();
                area = area
                    .constrain_to(bounds)
                    .pivot(Align2::LEFT_BOTTOM)
                    .fixed_pos(Pos2::new(left, bounds.bottom().round()));
            } else {
                area = area.anchor(Align2::CENTER_BOTTOM, Vec2::new(0.0, -14.0));
            }
            let banner_resp = area.show(ctx, |ui| {
                Frame::NONE
                    .fill(BG_TITLE)
                    .stroke(Stroke::new(1.0, self.theme.accent()))
                    .inner_margin(egui::Margin::symmetric(8, 5))
                    .show(ui, |ui| {
                        ui.set_min_width(inner_w);
                        ui.set_max_width(inner_w);
                        ui.allocate_ui_with_layout(
                            Vec2::new(inner_w, 24.0),
                            egui::Layout::right_to_left(egui::Align::Center),
                            |ui| {
                                ui.spacing_mut().item_spacing.x = 6.0;
                                let dismiss_btn = egui::Button::new(
                                    egui::RichText::new("Dismiss")
                                        .font(text_font.clone())
                                        .color(TEXT_PRIMARY),
                                )
                                .fill(BG_BUTTON)
                                .stroke(Stroke::new(1.0, BORDER_PANEL))
                                .corner_radius(CornerRadius::same(2))
                                .min_size(Vec2::new(0.0, 22.0));
                                if ui.add(dismiss_btn).clicked() {
                                    dismiss_clicked = true;
                                }

                                let restore_btn = egui::Button::new(
                                    egui::RichText::new("Restore")
                                        .font(text_font.clone())
                                        .color(TEXT_BRIGHT),
                                )
                                .fill(Color32::from_rgb(0x2d, 0x67, 0xc4))
                                .stroke(Stroke::new(1.0, Color32::from_rgb(0x52, 0x8b, 0xe4)))
                                .corner_radius(CornerRadius::same(2))
                                .min_size(Vec2::new(0.0, 22.0));
                                if ui.add(restore_btn).clicked() {
                                    restore_clicked = true;
                                }

                                let label_w = ui.available_width().max(40.0);
                                let banner_label = format!("Recovery: {formatted_ts}");
                                ui.add_sized(
                                    [label_w, 20.0],
                                    egui::Label::new(
                                        egui::RichText::new(banner_label)
                                            .font(text_font)
                                            .color(TEXT_BRIGHT),
                                    )
                                    .truncate(),
                                )
                                .on_hover_text(format!("Autosave snapshot: {}", snap.filename));
                            },
                        );
                    });
            });
            recovery_banner_extra_bottom_inset =
                (banner_resp.response.rect.height() + 10.0).max(46.0);
            if restore_clicked {
                #[cfg(not(target_arch = "wasm32"))]
                self.handle_restore_recovery_snapshot(&snap.path);
                #[cfg(target_arch = "wasm32")]
                self.handle_restore_web_autosave();
            } else if dismiss_clicked {
                self.startup_recovery_banner = None;
            }
        }

        // Eyedropper active banner.
        if self.pending_eyedropper.is_some() {
            Area::new(Id::unique("zc_eyedropper_banner"))
                .order(Order::Foreground)
                .anchor(Align2::CENTER_TOP, Vec2::new(0.0, 10.0))
                .show(ctx, |ui| {
                    Frame::NONE
                        .fill(BG_TITLE)
                        .stroke(Stroke::new(1.0, self.theme.accent()))
                        .inner_margin(egui::Margin::symmetric(10, 5))
                        .show(ui, |ui| {
                            ui.label(
                                egui::RichText::new(
                                    "Eyedropper active: click inside viewport to pick color (Esc to cancel)",
                                )
                                .font(self.theme.font(FONT_SMALL))
                                .color(TEXT_BRIGHT),
                            );
                        });
                });
        }

        // Transient status/error toast clamped strictly inside viewport_rect (left of the VU meter
        // and offset strictly above the autosave Recovery banner whenever visible).
        if let Some((msg, created)) = &self.status_toast {
            if created.elapsed() < TOAST_LIFETIME {
                let text = msg.clone();
                let is_error = text.starts_with("Shader error:")
                    || text.starts_with("Viewport render error:")
                    || text.starts_with("Failed");
                let screen_rect = ctx.input(|i| i.viewport_rect());
                let placement = compute_status_toast_bounds_with_bottom_inset(
                    self.last_viewport_rect,
                    screen_rect,
                    recovery_banner_extra_bottom_inset,
                );
                let inner_max_w = (placement.max_width - 20.0).max(40.0);
                Area::new(Id::unique("zc_status_toast"))
                    .order(Order::Foreground)
                    .interactable(false)
                    .pivot(Align2::RIGHT_BOTTOM)
                    .fixed_pos(placement.anchor_pos)
                    .constrain_to(placement.bounds)
                    .show(ctx, |ui| {
                        Frame::NONE
                            .fill(BG_PANEL)
                            .stroke(Stroke::new(
                                1.0,
                                if is_error { WARNING_BAR } else { BORDER_PANEL },
                            ))
                            .corner_radius(CornerRadius::ZERO)
                            .inner_margin(egui::Margin::symmetric(10, 6))
                            .show(ui, |ui| {
                                ui.set_max_width(inner_max_w);
                                ui.add(
                                    egui::Label::new(
                                        egui::RichText::new(text)
                                            .font(self.theme.font(FONT_SMALL))
                                            .color(TEXT_PRIMARY),
                                    )
                                    .wrap(),
                                );
                            });
                    });
                ctx.request_repaint_after(Duration::from_millis(250));
            } else {
                self.status_toast = None;
            }
        }
    }
}

#[cfg(test)]
mod web_persistence_tests {
    use super::*;

    #[test]
    fn base64_roundtrip_handles_all_padding_lengths() {
        for sample in [
            b"".as_slice(),
            b"a",
            b"ab",
            b"abc",
            b"abcd",
            b"ZoidCraft web persistence 0123456789+/=",
            &[0x00, 0x01, 0x02, 0xfe, 0xff, 0x80, 0x7f],
        ] {
            let enc = encode_base64(sample);
            let dec = decode_base64(&enc).expect("decode valid base64");
            assert_eq!(dec, sample);
        }
        assert!(decode_base64("abc").is_err(), "bad length must fail");
        assert!(decode_base64("!!!!").is_err(), "invalid char must fail");
    }

    #[test]
    fn web_autosave_chunks_roundtrip_and_reject_corruption() {
        let mut project = Project::new_cm3_default();
        project.name = "WebRecoveryDemo".to_owned();
        let aid = project.insert_binary_asset(
            AssetKind::Image,
            "badge.png",
            vec![0x89, b'P', b'N', b'G', 1, 2, 3, 4],
        );

        let (meta_json, mut chunks, raw_len) = encode_web_autosave_chunks(
            &project,
            1_730_000_000,
            WEB_AUTOSAVE_MAX_ENCODED_BYTES,
            1024,
        )
        .expect("encode web autosave chunks");
        assert!(raw_len > 0);
        assert!(!chunks.is_empty());

        let (restored, ts, decoded_len) =
            decode_web_autosave_chunks(&meta_json, &chunks).expect("decode web autosave chunks");
        assert_eq!(ts, 1_730_000_000);
        assert_eq!(decoded_len, raw_len);
        assert_eq!(restored.name, "WebRecoveryDemo");
        assert_eq!(
            restored.assets.get(&aid).and_then(|a| a.bytes.as_deref()),
            Some([0x89, b'P', b'N', b'G', 1, 2, 3, 4].as_slice())
        );

        // Corrupt a byte in the first chunk and ensure SHA-256 verification fails.
        if let Some(first) = chunks.first_mut() {
            let flipped = if first.starts_with('A') { 'B' } else { 'A' };
            first.replace_range(0..1, &flipped.to_string());
        }
        assert!(
            decode_web_autosave_chunks(&meta_json, &chunks).is_err(),
            "corrupted chunk must fail integrity check"
        );
    }

    #[test]
    fn web_autosave_externalizes_oversized_assets_to_stay_within_quota() {
        let mut project = Project::new_cm3_default();
        project.name = "QuotaFallback".to_owned();
        // Create pseudo-random incompressible payload that exceeds a tight 16 KiB test quota.
        let mut big = Vec::with_capacity(24 * 1024);
        let mut state: u32 = 0x1234_5678;
        for _ in 0..(24 * 1024) {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            big.push((state >> 24) as u8);
        }
        let aid = project.insert_binary_asset(AssetKind::Av, "clip.mp4", big);

        let (meta_json, chunks, _) =
            encode_web_autosave_chunks(&project, 1_730_000_123, 16 * 1024, 2048)
                .expect("fallback metadata-only snapshot must fit within quota");
        let meta: Value = serde_json::from_str(&meta_json).expect("meta json");
        assert_eq!(
            meta.get("assets_externalized").and_then(Value::as_bool),
            Some(true)
        );
        let (restored, _, _) =
            decode_web_autosave_chunks(&meta_json, &chunks).expect("decode fallback snapshot");
        assert_eq!(restored.name, "QuotaFallback");
        let rec = restored
            .assets
            .get(&aid)
            .expect("asset record preserved in fallback");
        assert!(rec.external);
        assert!(rec.bytes.is_none());
    }
}
