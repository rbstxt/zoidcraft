use std::collections::HashMap;
use std::io::Write;
use std::sync::Arc;

use zc_plugin::PluginRegistry;
use zc_project::Project;

/// Default finished-frame RGBA8 cache capacity (512 MiB per Spec 04 §5.1).
pub const DEFAULT_FRAME_CACHE_MAX_BYTES: usize = 512 * 1024 * 1024;

/// Hard upper bound on cached frame entries (per Spec 04 §5.1).
pub const MAX_FRAME_CACHE_ENTRIES: usize = 8192;

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// Assets up to 64 KiB are hashed in full on every check even when `sha256` is populated,
/// guaranteeing direct in-memory byte mutations always invalidate cached frames.
const FULL_ASSET_HASH_THRESHOLD_BYTES: usize = 64 * 1024;
/// Edge/midpoint sample span (1 KiB) for large assets (> 64 KiB) that already carry a valid SHA-256.
const LARGE_ASSET_SAMPLE_BYTES: usize = 1024;

#[inline]
fn fnv_mix(hash: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *hash ^= u64::from(b);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

#[inline]
fn fnv_mix_u64(hash: &mut u64, val: u64) {
    fnv_mix(hash, &val.to_le_bytes());
}

#[inline]
fn fnv_mix_str(hash: &mut u64, s: &str) {
    fnv_mix_u64(hash, s.len() as u64);
    fnv_mix(hash, s.as_bytes());
}

struct FnvWriter<'a>(&'a mut u64);

impl Write for FnvWriter<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        fnv_mix(self.0, buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Computes a deterministic 64-bit fingerprint of all enabled plugins and their per-plugin
/// configuration values in [`PluginRegistry`].
#[must_use]
pub fn plugin_registry_hash(plugins: &PluginRegistry) -> u64 {
    let mut h = FNV_OFFSET_BASIS;
    let enabled = plugins.enabled_ids();
    fnv_mix_u64(&mut h, enabled.len() as u64);
    for id in enabled {
        fnv_mix_str(&mut h, id);
        if let Some(cfg) = plugins.plugin_config(id) {
            fnv_mix(&mut h, &[1]);
            let _ = serde_json::to_writer(FnvWriter(&mut h), cfg);
        } else {
            fnv_mix(&mut h, &[0]);
        }
    }
    h
}

/// Computes a deterministic, zero-allocation 64-bit render fingerprint covering every
/// rendered field of `project` and `plugins`:
/// - `project.version`, `imported_from_pz_version`, `plugin_requirements`, and `preserved`
/// - `project.sequence` (resolution, fps, length, background color, motion blur, controls,
///   video tracks including `is_3d`/`enabled`/`solo`, clip `time_curve`, layers, masks and
///   `preserved` fields, 3D objects including `text_scale_mode`/`model_properties`/`euler_order`/
///   materials/effectors/constraints, `shadow_mode` compat mode, fonts, custom shaders,
///   `Composite` children, `EffectGroup` children, expressions, and Bezier handles)
/// - `project.compositions` (nested Precomp+ sequences)
/// - `project.assets` metadata AND binary `bytes` (full content hash when `sha256` is empty,
///   when `size` mismatches `bytes.len()`, or when `bytes.len() <= 64 KiB`; head/mid/tail
///   content sampling for larger assets with a valid SHA-256)
/// - Enabled plugin IDs and per-plugin configurations in `plugins`
#[must_use]
pub fn project_render_hash(project: &Project, plugins: &PluginRegistry) -> u64 {
    let mut h = FNV_OFFSET_BASIS;

    fnv_mix(&mut h, b"zc_proj_v1");
    fnv_mix_u64(&mut h, u64::from(project.version));
    match project.imported_from_pz_version {
        Some(v) => {
            fnv_mix(&mut h, &[1]);
            fnv_mix_u64(&mut h, u64::from(v));
        }
        None => fnv_mix(&mut h, &[0]),
    }

    fnv_mix_u64(&mut h, project.plugin_requirements.len() as u64);
    for req in &project.plugin_requirements {
        fnv_mix_str(&mut h, req);
    }

    fnv_mix(&mut h, b"|preserved|");
    let _ = serde_json::to_writer(FnvWriter(&mut h), &project.preserved);

    fnv_mix(&mut h, b"|sequence|");
    let _ = serde_json::to_writer(FnvWriter(&mut h), &project.sequence);

    fnv_mix(&mut h, b"|compositions|");
    let _ = serde_json::to_writer(FnvWriter(&mut h), &project.compositions);

    fnv_mix(&mut h, b"|assets|");
    fnv_mix_u64(&mut h, project.assets.len() as u64);
    for (asset_key, rec) in &project.assets {
        fnv_mix_str(&mut h, asset_key);
        fnv_mix_str(&mut h, &rec.id);
        fnv_mix_u64(&mut h, u64::from(rec.kind.cm3_id()));
        fnv_mix_u64(&mut h, u64::from(rec.source.cm3_id()));
        fnv_mix_str(&mut h, &rec.filename);
        match &rec.url {
            Some(url) => {
                fnv_mix(&mut h, &[1]);
                fnv_mix_str(&mut h, url);
            }
            None => fnv_mix(&mut h, &[0]),
        }
        fnv_mix(&mut h, &[u8::from(rec.external), u8::from(rec.missing)]);
        fnv_mix_str(&mut h, &rec.sha256);
        fnv_mix_u64(&mut h, rec.size);
        let _ = serde_json::to_writer(FnvWriter(&mut h), &rec.preserved);

        match rec.bytes.as_deref() {
            None => fnv_mix(&mut h, &[0]),
            Some(bytes) => {
                fnv_mix(&mut h, &[1]);
                fnv_mix_u64(&mut h, bytes.len() as u64);
                let hash_full = rec.sha256.is_empty()
                    || rec.size != bytes.len() as u64
                    || bytes.len() <= FULL_ASSET_HASH_THRESHOLD_BYTES;
                if hash_full {
                    fnv_mix(&mut h, bytes);
                } else {
                    let len = bytes.len();
                    let span = LARGE_ASSET_SAMPLE_BYTES.min(len);
                    fnv_mix(&mut h, &bytes[..span]);
                    let mid = (len / 2).saturating_sub(span / 2);
                    fnv_mix(&mut h, &bytes[mid..mid + span]);
                    fnv_mix(&mut h, &bytes[len - span..]);
                }
            }
        }
    }

    fnv_mix(&mut h, b"|plugins|");
    fnv_mix_u64(&mut h, plugin_registry_hash(plugins));

    h
}

/// Unique key identifying a rendered finished frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FrameCacheKey {
    pub frame_index: i64,
    pub width: u32,
    pub height: u32,
    pub revision: u64,
}

impl FrameCacheKey {
    pub const fn new(frame_index: i64, width: u32, height: u32, revision: u64) -> Self {
        Self {
            frame_index,
            width,
            height,
            revision,
        }
    }

    /// Quantizes a fractional frame number into millisecond-precision `frame_index` (`* 1000`).
    #[must_use]
    pub fn from_frame(frame: f64, width: u32, height: u32, revision: u64) -> Self {
        Self::new((frame * 1000.0).round() as i64, width, height, revision)
    }

    /// Builds a cache key whose `revision` fingerprints the full rendered state of
    /// `project` and `plugins`.
    #[must_use]
    pub fn for_project(
        frame: f64,
        width: u32,
        height: u32,
        project: &Project,
        plugins: &PluginRegistry,
    ) -> Self {
        Self::from_frame(frame, width, height, project_render_hash(project, plugins))
    }

    /// Folds the active plugin fingerprint into `self.revision` so callers supplying an
    /// external revision never collide across plugin enable/disable or config changes.
    #[must_use]
    pub fn with_plugins(self, plugins: &PluginRegistry) -> Self {
        self.with_salt(plugin_registry_hash(plugins))
    }

    /// Folds an additional 64-bit discriminator (e.g., preview scale, camera override,
    /// selection highlight) into `self.revision`.
    #[must_use]
    pub const fn with_salt(mut self, salt: u64) -> Self {
        let mut h = FNV_OFFSET_BASIS;
        let rev_bytes = self.revision.to_le_bytes();
        let salt_bytes = salt.to_le_bytes();
        let mut i = 0;
        while i < 8 {
            h ^= rev_bytes[i] as u64;
            h = h.wrapping_mul(FNV_PRIME);
            i += 1;
        }
        let mut j = 0;
        while j < 8 {
            h ^= salt_bytes[j] as u64;
            h = h.wrapping_mul(FNV_PRIME);
            j += 1;
        }
        self.revision = h;
        self
    }
}

/// Immutable finished RGBA8 frame stored in [`FrameCache`].
#[derive(Debug, Clone)]
pub struct CachedFrame {
    pub width: u32,
    pub height: u32,
    pub pixels: Arc<[u8]>,
}

#[derive(Debug)]
struct LruNode {
    key: FrameCacheKey,
    frame: CachedFrame,
    bytes: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

/// Bounded O(1) LRU finished-frame RGBA8 cache with both a byte cap and a hard
/// entry limit (`8192`).
#[derive(Debug)]
pub struct FrameCache {
    nodes: Vec<Option<LruNode>>,
    free_slots: Vec<usize>,
    index: HashMap<FrameCacheKey, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    current_bytes: usize,
    max_bytes: usize,
    max_entries: usize,
}

impl Default for FrameCache {
    fn default() -> Self {
        Self::new(DEFAULT_FRAME_CACHE_MAX_BYTES)
    }
}

impl FrameCache {
    /// Default capacity constant (`512 MiB`).
    pub const DEFAULT_MAX_BYTES: usize = DEFAULT_FRAME_CACHE_MAX_BYTES;
    /// Hard maximum number of cached entries (`8192`).
    pub const MAX_ENTRIES: usize = MAX_FRAME_CACHE_ENTRIES;

    /// Creates a new bounded LRU frame cache with the specified byte budget.
    pub fn new(max_bytes: usize) -> Self {
        Self {
            nodes: Vec::new(),
            free_slots: Vec::new(),
            index: HashMap::new(),
            head: None,
            tail: None,
            current_bytes: 0,
            max_bytes,
            max_entries: Self::MAX_ENTRIES,
        }
    }

    /// Looks up a cached frame and promotes it to most-recently-used on hit.
    pub fn get(&mut self, key: &FrameCacheKey) -> Option<CachedFrame> {
        let slot = *self.index.get(key)?;
        self.move_to_front(slot);
        self.nodes[slot].as_ref().map(|n| n.frame.clone())
    }

    /// Non-mutating cache lookup that does not alter LRU ordering.
    pub fn peek(&self, key: &FrameCacheKey) -> Option<&CachedFrame> {
        let slot = *self.index.get(key)?;
        self.nodes[slot].as_ref().map(|n| &n.frame)
    }

    /// Returns `true` if `key` is currently stored in the cache.
    pub fn contains(&self, key: &FrameCacheKey) -> bool {
        self.index.contains_key(key)
    }

    /// Inserts or replaces a finished RGBA8 frame, evicting least-recently-used
    /// entries until both `max_bytes` and `MAX_ENTRIES` are satisfied.
    ///
    /// Returns `true` if the frame was stored, or `false` if the frame exceeds
    /// the cache's total byte budget.
    pub fn insert(&mut self, key: FrameCacheKey, width: u32, height: u32, pixels: Vec<u8>) -> bool {
        self.insert_arc(key, width, height, Arc::from(pixels))
    }

    /// Zero-copy variant of [`Self::insert`] accepting a pre-wrapped `Arc<[u8]>` so callers
    /// that return the same `Arc<[u8]>` do not clone the pixel buffer.
    pub fn insert_arc(
        &mut self,
        key: FrameCacheKey,
        width: u32,
        height: u32,
        pixels: Arc<[u8]>,
    ) -> bool {
        let bytes = pixels.len();
        if self.max_bytes == 0 || bytes > self.max_bytes || self.max_entries == 0 {
            self.remove(&key);
            return false;
        }

        let frame = CachedFrame {
            width,
            height,
            pixels,
        };

        if let Some(&slot) = self.index.get(&key) {
            if let Some(node) = self.nodes[slot].as_mut() {
                self.current_bytes = self
                    .current_bytes
                    .saturating_sub(node.bytes)
                    .saturating_add(bytes);
                node.frame = frame;
                node.bytes = bytes;
            }
            self.move_to_front(slot);
        } else {
            let slot = self.alloc_slot(LruNode {
                key,
                frame,
                bytes,
                prev: None,
                next: None,
            });
            self.index.insert(key, slot);
            self.current_bytes = self.current_bytes.saturating_add(bytes);
            self.attach_front(slot);
        }

        self.evict_excess();
        true
    }

    /// Removes a specific frame from the cache if present.
    pub fn remove(&mut self, key: &FrameCacheKey) -> bool {
        let Some(slot) = self.index.remove(key) else {
            return false;
        };
        self.detach(slot);
        if let Some(node) = self.nodes[slot].take() {
            self.current_bytes = self.current_bytes.saturating_sub(node.bytes);
        }
        self.free_slots.push(slot);
        true
    }

    /// Evicts every cached frame immediately.
    pub fn invalidate_all(&mut self) {
        self.nodes.clear();
        self.free_slots.clear();
        self.index.clear();
        self.head = None;
        self.tail = None;
        self.current_bytes = 0;
    }

    /// Evicts all cached frames matching `revision` and returns the number of entries removed.
    pub fn invalidate_revision(&mut self, revision: u64) -> usize {
        let stale: Vec<FrameCacheKey> = self
            .index
            .keys()
            .copied()
            .filter(|k| k.revision == revision)
            .collect();
        let count = stale.len();
        for key in stale {
            self.remove(&key);
        }
        count
    }

    /// Evicts all cached frames whose `revision` differs from `keep_revision` and returns
    /// the number of stale entries removed.
    pub fn invalidate_stale_revisions(&mut self, keep_revision: u64) -> usize {
        let stale: Vec<FrameCacheKey> = self
            .index
            .keys()
            .copied()
            .filter(|k| k.revision != keep_revision)
            .collect();
        let count = stale.len();
        for key in stale {
            self.remove(&key);
        }
        count
    }

    /// Evicts all cached frames whose `frame_index` falls within `[start_frame_index, end_frame_index]`
    /// (inclusive) and returns the number of entries removed.
    pub fn invalidate_frame_range(
        &mut self,
        start_frame_index: i64,
        end_frame_index: i64,
    ) -> usize {
        let (lo, hi) = if start_frame_index <= end_frame_index {
            (start_frame_index, end_frame_index)
        } else {
            (end_frame_index, start_frame_index)
        };
        let matching: Vec<FrameCacheKey> = self
            .index
            .keys()
            .copied()
            .filter(|k| k.frame_index >= lo && k.frame_index <= hi)
            .collect();
        let count = matching.len();
        for key in matching {
            self.remove(&key);
        }
        count
    }

    /// Shrinks the maximum byte budget to 90% under VRAM/memory pressure and
    /// evicts least-recently-used entries to fit.
    pub fn shrink_for_pressure(&mut self) {
        let shrunk = (self.max_bytes / 10).saturating_mul(9);
        self.set_max_bytes(shrunk);
    }

    /// Updates the maximum byte budget and immediately evicts excess entries.
    pub fn set_max_bytes(&mut self, max_bytes: usize) {
        self.max_bytes = max_bytes;
        self.evict_excess();
    }

    /// Current total RGBA8 pixel bytes stored in the cache.
    pub fn current_bytes(&self) -> usize {
        self.current_bytes
    }

    /// Configured maximum byte budget.
    pub fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Current number of cached frames.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns `true` if no frames are currently cached.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn alloc_slot(&mut self, node: LruNode) -> usize {
        if let Some(slot) = self.free_slots.pop() {
            self.nodes[slot] = Some(node);
            slot
        } else {
            let slot = self.nodes.len();
            self.nodes.push(Some(node));
            slot
        }
    }

    fn attach_front(&mut self, slot: usize) {
        let old_head = self.head;
        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = old_head;
        }
        if let Some(h) = old_head {
            if let Some(head_node) = self.nodes[h].as_mut() {
                head_node.prev = Some(slot);
            }
        } else {
            self.tail = Some(slot);
        }
        self.head = Some(slot);
    }

    fn detach(&mut self, slot: usize) {
        let (prev, next) = match self.nodes[slot].as_ref() {
            Some(node) => (node.prev, node.next),
            None => return,
        };

        if let Some(p) = prev {
            if let Some(prev_node) = self.nodes[p].as_mut() {
                prev_node.next = next;
            }
        } else {
            self.head = next;
        }

        if let Some(n) = next {
            if let Some(next_node) = self.nodes[n].as_mut() {
                next_node.prev = prev;
            }
        } else {
            self.tail = prev;
        }

        if let Some(node) = self.nodes[slot].as_mut() {
            node.prev = None;
            node.next = None;
        }
    }

    fn move_to_front(&mut self, slot: usize) {
        if self.head == Some(slot) {
            return;
        }
        self.detach(slot);
        self.attach_front(slot);
    }

    fn evict_excess(&mut self) {
        if self.max_bytes == 0 || self.max_entries == 0 {
            self.invalidate_all();
            return;
        }
        while self.current_bytes > self.max_bytes || self.index.len() > self.max_entries {
            let Some(lru_slot) = self.tail else {
                break;
            };
            self.detach(lru_slot);
            if let Some(node) = self.nodes[lru_slot].take() {
                self.index.remove(&node.key);
                self.current_bytes = self.current_bytes.saturating_sub(node.bytes);
            }
            self.free_slots.push(lru_slot);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use zc_project::{
        AssetKind, AssetRecord, AssetSource, CompatMode, CustomProperty, EffectInstance,
        EffectKind, Layer, LayerKind, Object3D, Object3DKind, PropertyKeyframe, PropertyTrack,
        PropertyValue, Sequence, Uuid,
    };

    #[test]
    fn project_render_hash_invalidates_on_all_rendered_mutations() {
        let mut project = Project::new_cm3_default();
        let mut plugins = PluginRegistry::new();

        let base = project_render_hash(&project, &plugins);
        assert_eq!(
            base,
            project_render_hash(&project, &plugins),
            "hash must be deterministic"
        );

        // 1. Resolution change
        project.sequence.resolution = [1280, 720];
        let h_res = project_render_hash(&project, &plugins);
        assert_ne!(h_res, base, "resolution change must invalidate cache hash");

        // 2. Track 3D flag change
        project.sequence.video_tracks[0].is_3d = true;
        let h_3d = project_render_hash(&project, &plugins);
        assert_ne!(h_3d, h_res, "track.is_3d toggle must invalidate cache hash");

        // 3. Clip time_curve expression & keyframe change
        project.sequence.video_tracks[0].clips[0]
            .time_curve
            .expression = Some("time * 2.0".to_owned());
        let h_time = project_render_hash(&project, &plugins);
        assert_ne!(
            h_time, h_3d,
            "time_curve expression change must invalidate cache hash"
        );

        // 4. Layer preserved mask / shadows / backgroundColor mutation
        project.sequence.video_tracks[0].clips[0]
            .layer
            .preserved
            .insert("mask".to_owned(), json!({"enabled": true, "invert": false}));
        let h_mask = project_render_hash(&project, &plugins);
        assert_ne!(
            h_mask, h_time,
            "layer.preserved mask change must invalidate cache hash"
        );

        // 5. Scene3D shadow_mode (compat mode) change
        if let LayerKind::Scene3D { shadow_mode, .. } =
            &mut project.sequence.video_tracks[0].clips[0].layer.kind
        {
            *shadow_mode = CompatMode::Legacy;
        }
        let h_compat = project_render_hash(&project, &plugins);
        assert_ne!(
            h_compat, h_mask,
            "Scene3D compat shadow_mode change must invalidate cache hash"
        );

        // 6. Custom shader + uniform + Bezier handle mutation
        let mut custom_track = PropertyTrack::number(1.0);
        custom_track.animated = true;
        custom_track.keyframes = vec![
            PropertyKeyframe::linear(0.0, PropertyValue::Number(0.0)),
            PropertyKeyframe::linear(30.0, PropertyValue::Number(1.0)),
        ];
        project.sequence.video_tracks[0].clips[0]
            .layer
            .effects
            .push(EffectInstance {
                id: Uuid::new_v4(),
                name: "Custom".to_owned(),
                kind: EffectKind::CustomShader {
                    frag_shader: "void main() { gl_FragColor = vec4(1.0); }".to_owned(),
                    custom_properties: vec![CustomProperty {
                        name: "Strength".to_owned(),
                        prop_type: Some(0),
                        track: custom_track,
                        metadata: Default::default(),
                    }],
                },
                enabled: PropertyTrack::option(1),
                params: Default::default(),
                preserved: Default::default(),
            });
        let h_shader = project_render_hash(&project, &plugins);
        assert_ne!(
            h_shader, h_compat,
            "adding custom shader effect must invalidate cache hash"
        );

        if let EffectKind::CustomShader { frag_shader, .. } =
            &mut project.sequence.video_tracks[0].clips[0].layer.effects[0].kind
        {
            *frag_shader = "void main() { gl_FragColor = vec4(0.5); }".to_owned();
        }
        let h_frag = project_render_hash(&project, &plugins);
        assert_ne!(
            h_frag, h_shader,
            "custom shader source edit must invalidate cache hash"
        );

        if let EffectKind::CustomShader {
            custom_properties, ..
        } = &mut project.sequence.video_tracks[0].clips[0].layer.effects[0].kind
        {
            custom_properties[0].track.keyframes[1].control_points[0] = [-15.0, 0.25];
        }
        let h_bez = project_render_hash(&project, &plugins);
        assert_ne!(
            h_bez, h_frag,
            "Bezier control point edit must invalidate cache hash"
        );

        // 7. Font change on a Text2D layer
        let text_layer = Layer::new(
            "Title",
            LayerKind::Text2D {
                text: "Hello".to_owned(),
                font: Some("font_a".to_owned()),
                size: 48,
                shapes: Vec::new(),
            },
            [1920, 1080],
        );
        project.sequence.video_tracks[0].clips[0].layer = Layer::new(
            "Comp",
            LayerKind::Composite {
                layers: vec![text_layer],
            },
            [1920, 1080],
        );
        let h_font_a = project_render_hash(&project, &plugins);
        if let LayerKind::Composite { layers } =
            &mut project.sequence.video_tracks[0].clips[0].layer.kind
            && let LayerKind::Text2D { font, .. } = &mut layers[0].kind
        {
            *font = Some("font_b".to_owned());
        }
        let h_font_b = project_render_hash(&project, &plugins);
        assert_ne!(
            h_font_b, h_font_a,
            "font change inside Composite Text2D must invalidate cache hash"
        );

        // 8. Equal-length asset byte replacement with empty sha256
        project.assets.insert(
            "raw_tex".to_owned(),
            AssetRecord {
                id: "raw_tex".to_owned(),
                kind: AssetKind::Image,
                source: AssetSource::File,
                filename: "tex.png".to_owned(),
                url: None,
                external: false,
                bytes: Some(vec![1, 2, 3, 4]),
                missing: false,
                sha256: String::new(),
                size: 4,
                preserved: Default::default(),
            },
        );
        let h_asset_1 = project_render_hash(&project, &plugins);
        project.assets.get_mut("raw_tex").unwrap().bytes = Some(vec![1, 2, 3, 5]);
        let h_asset_2 = project_render_hash(&project, &plugins);
        assert_ne!(
            h_asset_2, h_asset_1,
            "equal-length asset byte change with empty sha256 must invalidate cache hash"
        );

        // 9. Nested composition (Precomp+) mutation
        let comp_id = Uuid::new_v4();
        let comp_seq = Sequence {
            background_color: [0.2, 0.3, 0.4, 1.0],
            ..Sequence::default()
        };
        project.compositions.insert(comp_id, comp_seq);
        let h_comp_1 = project_render_hash(&project, &plugins);
        project
            .compositions
            .get_mut(&comp_id)
            .unwrap()
            .background_color = [0.8, 0.3, 0.4, 1.0];
        let h_comp_2 = project_render_hash(&project, &plugins);
        assert_ne!(
            h_comp_2, h_comp_1,
            "nested composition mutation must invalidate cache hash"
        );

        // 10. Plugin enable/disable AND plugin config mutation
        let was_enabled = plugins.is_enabled("player-plus");
        if was_enabled {
            let _ = plugins.disable("player-plus", &Default::default());
        } else {
            let _ = plugins.enable("player-plus");
        }
        let h_plugin_toggle = project_render_hash(&project, &plugins);
        assert_ne!(
            h_plugin_toggle, h_comp_2,
            "plugin toggle must invalidate cache hash"
        );

        let _ = plugins.enable("player-plus");
        let h_cfg_before = project_render_hash(&project, &plugins);
        let _ = plugins.set_plugin_config("player-plus", json!({"previewScale": 0.5}));
        let h_cfg_after = project_render_hash(&project, &plugins);
        assert_ne!(
            h_cfg_after, h_cfg_before,
            "plugin config change must invalidate cache hash"
        );

        // 11. 3D Text text_scale_mode & Model properties mutation
        let mut text_obj = Object3D::new_perspective_camera();
        text_obj.kind = Object3DKind::Text3D(Box::default());
        let mut scene_proj = Project::new_cm3_default();
        if let LayerKind::Scene3D { objects, .. } =
            &mut scene_proj.sequence.video_tracks[0].clips[0].layer.kind
        {
            objects.push(text_obj);
        }
        let h_text_a = project_render_hash(&scene_proj, &plugins);
        if let LayerKind::Scene3D { objects, .. } =
            &mut scene_proj.sequence.video_tracks[0].clips[0].layer.kind
            && let Object3DKind::Text3D(td) = &mut objects[1].kind
        {
            td.text_scale_mode = CompatMode::Legacy;
        }
        let h_text_b = project_render_hash(&scene_proj, &plugins);
        assert_ne!(
            h_text_b, h_text_a,
            "Text3D text_scale_mode change must invalidate cache hash"
        );
    }

    #[test]
    fn frame_cache_set_max_bytes_preserves_fitting_entries_and_zero_clears_all() {
        let mut cache = FrameCache::new(400);
        let k1 = FrameCacheKey::new(1000, 5, 5, 1);
        let k2 = FrameCacheKey::new(2000, 5, 5, 1);
        let k3 = FrameCacheKey::new(3000, 5, 5, 1);

        assert!(cache.insert(k1, 5, 5, vec![1; 100]));
        assert!(cache.insert(k2, 5, 5, vec![2; 100]));
        assert!(cache.insert(k3, 5, 5, vec![3; 100]));

        // Shrinking budget to 200 preserves the two MRU frames (k2, k3) and evicts only k1.
        cache.set_max_bytes(200);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.current_bytes(), 200);
        assert!(cache.peek(&k1).is_none());
        assert!(cache.peek(&k2).is_some());
        assert!(cache.peek(&k3).is_some());

        // Growing budget does not wipe cached entries.
        cache.set_max_bytes(1024);
        assert_eq!(cache.len(), 2);
        assert!(cache.contains(&k2));
        assert!(cache.contains(&k3));

        // Zero budget evicts everything (even zero-length sentinel payloads).
        let k_zero = FrameCacheKey::new(4000, 0, 0, 1);
        assert!(cache.insert(k_zero, 0, 0, Vec::new()));
        cache.set_max_bytes(0);
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn frame_cache_insert_arc_shares_allocation_and_targeted_invalidation_works() {
        let mut cache = FrameCache::new(1024);
        let pixels: Arc<[u8]> = Arc::from(vec![42_u8; 64]);
        let k_r1_f1 = FrameCacheKey::new(1000, 4, 4, 1);
        let k_r1_f2 = FrameCacheKey::new(2000, 4, 4, 1);
        let k_r2_f1 = FrameCacheKey::new(1000, 4, 4, 2);
        let k_r2_f5 = FrameCacheKey::new(5000, 4, 4, 2);

        assert!(cache.insert_arc(k_r1_f1, 4, 4, Arc::clone(&pixels)));
        let fetched = cache.get(&k_r1_f1).expect("cached frame");
        assert!(
            Arc::ptr_eq(&pixels, &fetched.pixels),
            "insert_arc must store pixels without deep-cloning"
        );

        assert!(cache.insert(k_r1_f2, 4, 4, vec![1; 64]));
        assert!(cache.insert(k_r2_f1, 4, 4, vec![2; 64]));
        assert!(cache.insert(k_r2_f5, 4, 4, vec![3; 64]));

        // Evict stale revisions != 2.
        let removed_stale = cache.invalidate_stale_revisions(2);
        assert_eq!(removed_stale, 2);
        assert!(!cache.contains(&k_r1_f1));
        assert!(!cache.contains(&k_r1_f2));
        assert!(cache.contains(&k_r2_f1));
        assert!(cache.contains(&k_r2_f5));

        // Targeted frame-range invalidation removes only [0, 3000].
        let removed_range = cache.invalidate_frame_range(0, 3000);
        assert_eq!(removed_range, 1);
        assert!(!cache.contains(&k_r2_f1));
        assert!(cache.contains(&k_r2_f5));

        // Explicit revision invalidation clears the remaining revision-2 frame.
        assert_eq!(cache.invalidate_revision(2), 1);
        assert!(cache.is_empty());
        assert_eq!(cache.current_bytes(), 0);
    }

    #[test]
    fn frame_cache_key_with_plugins_and_salt_distinguish_preview_variants() {
        let mut plugins = PluginRegistry::new();
        let base = FrameCacheKey::new(1000, 640, 360, 7);
        let with_default_plugins = base.with_plugins(&plugins);
        let _ = plugins.disable("player-plus", &Default::default());
        let with_toggled_plugins = base.with_plugins(&plugins);
        assert_ne!(
            with_default_plugins, with_toggled_plugins,
            "same numeric revision with different plugin states must produce distinct cache keys"
        );

        let scale_half = with_default_plugins.with_salt(2);
        let scale_quarter = with_default_plugins.with_salt(4);
        assert_ne!(scale_half, scale_quarter);
    }
}
