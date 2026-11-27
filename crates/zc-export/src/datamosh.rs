//! Post-export WebM/Matroska EBML True Datamosh processor (Zoidium Export parity).
//!
//! Parses EBML `Segment -> SeekHead` / `Info` / `Tracks` / `Cluster` (`SimpleBlock` &
//! `BlockGroup` with optional alpha `BlockAdditions`) / `Cues` structures in pure safe Rust.
//! Strips secondary video I-frame (keyframe) blocks while keeping the first keyframe intact so
//! downstream decoders bleed motion vectors cleanly across cuts, and rewrites `Cues` and
//! `SeekHead` byte offsets so players can seek the rewritten container without corruption.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
#[cfg(not(target_arch = "wasm32"))]
use std::fs::File;
#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
use crate::ExportError;
#[cfg(not(target_arch = "wasm32"))]
use crate::exporter::publish_temp_output;
#[cfg(not(target_arch = "wasm32"))]
use crate::temp_audio::TempFileGuard;

/// True Datamosh mode for WebM / MKV exports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DatamoshMode {
    /// Disabled: output file is unmodified.
    #[default]
    Off,
    /// Strip every video keyframe after the initial keyframe across the entire video.
    FullVideo,
    /// Strip video keyframes that fall near clip cut boundaries (`cut_times_ms`).
    AtClipCuts,
}

impl DatamoshMode {
    #[must_use]
    pub const fn from_index(idx: usize) -> Self {
        match idx {
            1 => Self::FullVideo,
            2 => Self::AtClipCuts,
            _ => Self::Off,
        }
    }
}

/// EBML element IDs used by WebM / Matroska.
const ID_EBML: u32 = 0x1A45_DFA3;
const ID_SEGMENT: u32 = 0x1853_8067;
const ID_SEEK_HEAD: u32 = 0x114D_9B74;
const ID_SEEK: u32 = 0x4DBB;
const ID_SEEK_ID: u32 = 0x53AB;
const ID_SEEK_POSITION: u32 = 0x53AC;
const ID_INFO: u32 = 0x1549_A966;
const ID_TIMECODE_SCALE: u32 = 0x002A_D7B1;
const ID_TRACKS: u32 = 0x1654_AE6B;
const ID_TRACK_ENTRY: u32 = 0xAE;
const ID_TRACK_NUMBER: u32 = 0xD7;
const ID_TRACK_TYPE: u32 = 0x83;
const ID_CHAPTERS: u32 = 0x1043_A770;
const ID_CLUSTER: u32 = 0x1F43_B675;
const ID_CLUSTER_TIMECODE: u32 = 0xE7;
const ID_BLOCK_GROUP: u32 = 0xA0;
const ID_BLOCK: u32 = 0xA1;
const ID_SIMPLE_BLOCK: u32 = 0xA3;
const ID_REFERENCE_BLOCK: u32 = 0xFB;
const ID_CUES: u32 = 0x1C53_BB6B;
const ID_CUE_POINT: u32 = 0xBB;
const ID_CUE_TRACK_POSITIONS: u32 = 0xB7;
const ID_CUE_CLUSTER_POSITION: u32 = 0xF1;
const ID_CUE_RELATIVE_POSITION: u32 = 0xF0;
const ID_CUE_BLOCK_NUMBER: u32 = 0x5378;
const ID_ATTACHMENTS: u32 = 0x1941_A469;
const ID_TAGS: u32 = 0x1254_C367;

const fn is_level0_id(id: u32) -> bool {
    matches!(id, ID_EBML | ID_SEGMENT)
}

const fn is_level0_or_level1_id(id: u32) -> bool {
    matches!(
        id,
        ID_EBML
            | ID_SEGMENT
            | ID_SEEK_HEAD
            | ID_INFO
            | ID_TRACKS
            | ID_CHAPTERS
            | ID_CLUSTER
            | ID_CUES
            | ID_ATTACHMENTS
            | ID_TAGS
    )
}

/// Reads an EBML element ID (preserving marker bit) and returns `(id, byte_len)`.
fn read_ebml_id(data: &[u8], pos: usize) -> Option<(u32, usize)> {
    let first = *data.get(pos)?;
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 4 || pos + len > data.len() {
        return None;
    }
    let mut id = 0_u32;
    for &b in &data[pos..pos + len] {
        id = (id << 8) | u32::from(b);
    }
    Some((id, len))
}

/// Reads an EBML VINT data size (stripping marker bit) and returns `(Option<size>, byte_len)`.
/// Returns `None` for the size if it is the all-ones unknown/indeterminate EBML size.
fn read_ebml_size(data: &[u8], pos: usize) -> Option<(Option<usize>, usize)> {
    let first = *data.get(pos)?;
    if first == 0 {
        return None;
    }
    let len = first.leading_zeros() as usize + 1;
    if len > 8 || pos + len > data.len() {
        return None;
    }
    let mask = if len == 8 {
        0_u8
    } else {
        (1_u8 << (8 - len)) - 1
    };
    let mut val = u64::from(first & mask);
    let mut all_ones = (first & mask) == mask;
    for &b in &data[pos + 1..pos + len] {
        val = (val << 8) | u64::from(b);
        if b != 0xFF {
            all_ones = false;
        }
    }
    if all_ones {
        Some((None, len))
    } else {
        usize::try_from(val).ok().map(|v| (Some(v), len))
    }
}

/// Encodes a finite EBML data size using at least `min_width` bytes (`1..=8`).
fn encode_ebml_size(val: u64, min_width: usize) -> Vec<u8> {
    let mut width = min_width.clamp(1, 8);
    while width < 8 {
        let max_val = (1_u64 << (7 * width)) - 2;
        if val <= max_val {
            break;
        }
        width += 1;
    }
    let mut out = vec![0_u8; width];
    let mut v = val;
    for byte in out.iter_mut().rev() {
        *byte = (v & 0xFF) as u8;
        v >>= 8;
    }
    out[0] |= 1_u8 << (8 - width);
    out
}

fn encode_uint_be(val: u64) -> Vec<u8> {
    let be = val.to_be_bytes();
    let first_nz = be.iter().position(|&b| b != 0).unwrap_or(be.len() - 1);
    be[first_nz..].to_vec()
}

fn encode_ebml_elem(id_bytes: &[u8], payload: &[u8]) -> Vec<u8> {
    let size_bytes = encode_ebml_size(payload.len() as u64, 1);
    let mut out = Vec::with_capacity(id_bytes.len() + size_bytes.len() + payload.len());
    out.extend_from_slice(id_bytes);
    out.extend_from_slice(&size_bytes);
    out.extend_from_slice(payload);
    out
}

fn read_uint_be(slice: &[u8]) -> u64 {
    let mut v = 0_u64;
    for &b in slice.iter().take(8) {
        v = (v << 8) | u64::from(b);
    }
    v
}

/// Finds the end offset of an unknown-size `Cluster` by walking child elements until a Level 0 or
/// Level 1 element ID (such as the next `Cluster` or `Cues`) or `limit` is reached.
fn find_unknown_cluster_end(data: &[u8], body_start: usize, limit: usize) -> usize {
    let mut p = body_start;
    while p < limit {
        let Some((id, id_len)) = read_ebml_id(data, p) else {
            return limit;
        };
        if is_level0_or_level1_id(id) {
            return p;
        }
        let Some((size_opt, sz_len)) = read_ebml_size(data, p + id_len) else {
            return limit;
        };
        let child_body = p + id_len + sz_len;
        let next_p = match size_opt {
            Some(sz) => child_body.saturating_add(sz).min(limit),
            None => limit,
        };
        if next_p <= p {
            return limit;
        }
        p = next_p;
    }
    p
}

/// Finds the end offset of an unknown-size `Segment` by walking Level 1 elements until the next
/// Level 0 element (`EBML` or `Segment`) or `limit` is reached.
fn find_unknown_segment_end(data: &[u8], body_start: usize, limit: usize) -> usize {
    let mut p = body_start;
    while p < limit {
        let Some((id, id_len)) = read_ebml_id(data, p) else {
            return limit;
        };
        if is_level0_id(id) {
            return p;
        }
        let Some((size_opt, sz_len)) = read_ebml_size(data, p + id_len) else {
            return limit;
        };
        let child_body = p + id_len + sz_len;
        let next_p = match size_opt {
            Some(sz) => child_body.saturating_add(sz).min(limit),
            None if id == ID_CLUSTER => find_unknown_cluster_end(data, child_body, limit),
            None => limit,
        };
        if next_p <= p {
            return limit;
        }
        p = next_p;
    }
    p
}

fn parse_info_timecode_scale(payload: &[u8]) -> u64 {
    let mut p = 0;
    while p < payload.len() {
        let Some((id, id_len)) = read_ebml_id(payload, p) else {
            break;
        };
        let Some((Some(sz), sz_len)) = read_ebml_size(payload, p + id_len) else {
            break;
        };
        let body_start = p + id_len + sz_len;
        let body_end = body_start.saturating_add(sz).min(payload.len());
        if id == ID_TIMECODE_SCALE {
            let scale = read_uint_be(&payload[body_start..body_end]);
            if scale > 0 {
                return scale;
            }
        }
        p = body_end;
    }
    1_000_000
}

fn parse_video_track_numbers(tracks_payload: &[u8]) -> Vec<u64> {
    let mut video_tracks = Vec::new();
    let mut p = 0;
    while p < tracks_payload.len() {
        let Some((id, id_len)) = read_ebml_id(tracks_payload, p) else {
            break;
        };
        let Some((Some(sz), sz_len)) = read_ebml_size(tracks_payload, p + id_len) else {
            break;
        };
        let body_start = p + id_len + sz_len;
        let body_end = body_start.saturating_add(sz).min(tracks_payload.len());
        if id == ID_TRACK_ENTRY {
            let mut ep = body_start;
            let mut track_num = None;
            let mut track_type = None;
            while ep < body_end {
                let Some((eid, eid_len)) = read_ebml_id(tracks_payload, ep) else {
                    break;
                };
                let Some((Some(esz), esz_len)) = read_ebml_size(tracks_payload, ep + eid_len)
                else {
                    break;
                };
                let eb_start = ep + eid_len + esz_len;
                let eb_end = eb_start.saturating_add(esz).min(body_end);
                if eid == ID_TRACK_NUMBER {
                    track_num = Some(read_uint_be(&tracks_payload[eb_start..eb_end]));
                } else if eid == ID_TRACK_TYPE {
                    track_type = Some(read_uint_be(&tracks_payload[eb_start..eb_end]));
                }
                ep = eb_end;
            }
            if track_type == Some(1)
                && let Some(num) = track_num
            {
                video_tracks.push(num);
            }
        }
        p = body_end;
    }
    video_tracks
}

/// Inspects a `BlockGroup` (`0xA0`) payload (used by FFmpeg for VP8/VP9 frames with alpha
/// `BlockAdditions`) and returns `Some(relative_timecode)` if the block belongs to a video track
/// and is a keyframe (indicated by the absence of `ReferenceBlock` `0xFB` per RFC 9559).
fn parse_block_group_keyframe_rel_tc(payload: &[u8], video_tracks: &[u64]) -> Option<i16> {
    let mut p = 0;
    let mut video_rel_tc: Option<i16> = None;
    let mut has_reference_block = false;
    while p < payload.len() {
        let Some((id, id_len)) = read_ebml_id(payload, p) else {
            break;
        };
        let Some((Some(sz), sz_len)) = read_ebml_size(payload, p + id_len) else {
            break;
        };
        let body_start = p + id_len + sz_len;
        let body_end = body_start.saturating_add(sz).min(payload.len());
        if id == ID_BLOCK {
            let block = &payload[body_start..body_end];
            if let Some((Some(track_num), tn_len)) = read_ebml_size(block, 0)
                && block.len() >= tn_len + 3
                && video_tracks.contains(&(track_num as u64))
            {
                let rel_tc = i16::from_be_bytes([block[tn_len], block[tn_len + 1]]);
                video_rel_tc = Some(rel_tc);
            }
        } else if id == ID_REFERENCE_BLOCK {
            has_reference_block = true;
        }
        p = body_end;
    }
    if has_reference_block {
        None
    } else {
        video_rel_tc
    }
}

#[derive(Debug, Clone)]
struct SeekEntry {
    seek_id: Vec<u8>,
    old_pos: u64,
}

fn parse_seek_head_entries(payload: &[u8]) -> Vec<SeekEntry> {
    let mut entries = Vec::new();
    let mut p = 0;
    while p < payload.len() {
        let Some((id, id_len)) = read_ebml_id(payload, p) else {
            break;
        };
        let Some((Some(sz), sz_len)) = read_ebml_size(payload, p + id_len) else {
            break;
        };
        let body_start = p + id_len + sz_len;
        let body_end = body_start.saturating_add(sz).min(payload.len());
        if id == ID_SEEK {
            let mut sp = body_start;
            let mut seek_id: Option<Vec<u8>> = None;
            let mut seek_pos: Option<u64> = None;
            while sp < body_end {
                let Some((sid, sid_len)) = read_ebml_id(payload, sp) else {
                    break;
                };
                let Some((Some(ssz), ssz_len)) = read_ebml_size(payload, sp + sid_len) else {
                    break;
                };
                let sb_start = sp + sid_len + ssz_len;
                let sb_end = sb_start.saturating_add(ssz).min(body_end);
                if sid == ID_SEEK_ID {
                    seek_id = Some(payload[sb_start..sb_end].to_vec());
                } else if sid == ID_SEEK_POSITION {
                    seek_pos = Some(read_uint_be(&payload[sb_start..sb_end]));
                }
                sp = sb_end;
            }
            if let (Some(seek_id), Some(old_pos)) = (seek_id, seek_pos) {
                entries.push(SeekEntry { seek_id, old_pos });
            }
        }
        p = body_end;
    }
    entries
}

fn rewrite_seek_head_element(entries: &[SeekEntry], offset_map: &BTreeMap<u64, u64>) -> Vec<u8> {
    let mut body = Vec::new();
    for entry in entries {
        if let Some(&new_pos) = offset_map.get(&entry.old_pos) {
            let mut seek_payload = Vec::new();
            seek_payload.extend_from_slice(&encode_ebml_elem(&[0x53, 0xAB], &entry.seek_id));
            seek_payload
                .extend_from_slice(&encode_ebml_elem(&[0x53, 0xAC], &encode_uint_be(new_pos)));
            body.extend_from_slice(&encode_ebml_elem(&[0x4D, 0xBB], &seek_payload));
        }
    }
    if body.is_empty() {
        Vec::new()
    } else {
        encode_ebml_elem(&[0x11, 0x4D, 0x9B, 0x74], &body)
    }
}

/// Rewrites `Cues` (`0x1C53BB6B`) so every `CueClusterPosition` (`0xF1`) points to its Cluster's
/// updated Segment-relative byte offset. Drops stale `CueRelativePosition` / `CueBlockNumber`
/// inside modified Clusters and drops `CueTrackPositions` pointing to nonexistent Clusters.
fn rewrite_cues_element(raw_payload: &[u8], cluster_map: &BTreeMap<u64, (u64, bool)>) -> Vec<u8> {
    let mut cues_body = Vec::with_capacity(raw_payload.len());
    let mut p = 0;
    while p < raw_payload.len() {
        let Some((id, id_len)) = read_ebml_id(raw_payload, p) else {
            break;
        };
        let Some((Some(sz), sz_len)) = read_ebml_size(raw_payload, p + id_len) else {
            break;
        };
        let body_start = p + id_len + sz_len;
        let body_end = body_start.saturating_add(sz).min(raw_payload.len());

        if id == ID_CUE_POINT {
            let mut point_body = Vec::with_capacity(body_end.saturating_sub(body_start));
            let mut kept_track_positions = 0_usize;
            let mut cp = body_start;
            while cp < body_end {
                let Some((cid, cid_len)) = read_ebml_id(raw_payload, cp) else {
                    break;
                };
                let Some((Some(csz), csz_len)) = read_ebml_size(raw_payload, cp + cid_len) else {
                    break;
                };
                let cb_start = cp + cid_len + csz_len;
                let cb_end = cb_start.saturating_add(csz).min(body_end);
                if cid == ID_CUE_TRACK_POSITIONS {
                    if let Some(rebuilt_tp) =
                        rewrite_cue_track_positions(&raw_payload[cb_start..cb_end], cluster_map)
                    {
                        point_body.extend_from_slice(&rebuilt_tp);
                        kept_track_positions += 1;
                    }
                } else {
                    point_body.extend_from_slice(&raw_payload[cp..cb_end]);
                }
                cp = cb_end;
            }
            if kept_track_positions > 0 {
                cues_body.extend_from_slice(&encode_ebml_elem(&[0xBB], &point_body));
            }
        } else {
            cues_body.extend_from_slice(&raw_payload[p..body_end]);
        }
        p = body_end;
    }

    if cues_body.is_empty() {
        Vec::new()
    } else {
        encode_ebml_elem(&[0x1C, 0x53, 0xBB, 0x6B], &cues_body)
    }
}

fn rewrite_cue_track_positions(
    tp_payload: &[u8],
    cluster_map: &BTreeMap<u64, (u64, bool)>,
) -> Option<Vec<u8>> {
    // First locate CueClusterPosition to validate and determine whether the target Cluster changed.
    let mut old_cluster_pos: Option<u64> = None;
    let mut scan = 0;
    while scan < tp_payload.len() {
        let (id, id_len) = read_ebml_id(tp_payload, scan)?;
        let (Some(sz), sz_len) = read_ebml_size(tp_payload, scan + id_len)? else {
            return None;
        };
        let b_start = scan + id_len + sz_len;
        let b_end = b_start.saturating_add(sz).min(tp_payload.len());
        if id == ID_CUE_CLUSTER_POSITION {
            old_cluster_pos = Some(read_uint_be(&tp_payload[b_start..b_end]));
            break;
        }
        scan = b_end;
    }
    let (new_cluster_pos, cluster_modified) = *cluster_map.get(&old_cluster_pos?)?;

    let mut rebuilt = Vec::with_capacity(tp_payload.len() + 8);
    let mut p = 0;
    while p < tp_payload.len() {
        let Some((id, id_len)) = read_ebml_id(tp_payload, p) else {
            break;
        };
        let Some((Some(sz), sz_len)) = read_ebml_size(tp_payload, p + id_len) else {
            break;
        };
        let b_start = p + id_len + sz_len;
        let b_end = b_start.saturating_add(sz).min(tp_payload.len());
        if id == ID_CUE_CLUSTER_POSITION {
            rebuilt.extend_from_slice(&encode_ebml_elem(&[0xF1], &encode_uint_be(new_cluster_pos)));
        } else if cluster_modified && (id == ID_CUE_RELATIVE_POSITION || id == ID_CUE_BLOCK_NUMBER)
        {
            // Omit block-local offset/number when blocks inside the Cluster were stripped.
        } else {
            rebuilt.extend_from_slice(&tp_payload[p..b_end]);
        }
        p = b_end;
    }
    Some(encode_ebml_elem(&[0xB7], &rebuilt))
}

fn should_strip_keyframe(
    mode: DatamoshMode,
    abs_time_ms: u64,
    cut_times_ms: &[u64],
    seen_first_keyframe: &mut bool,
) -> bool {
    if !*seen_first_keyframe {
        *seen_first_keyframe = true;
        return false;
    }
    match mode {
        DatamoshMode::Off => false,
        DatamoshMode::FullVideo => true,
        DatamoshMode::AtClipCuts => {
            if cut_times_ms.is_empty() {
                false
            } else {
                cut_times_ms
                    .iter()
                    .any(|&cut| abs_time_ms.abs_diff(cut) <= 150)
            }
        }
    }
}

enum SegItem<'a> {
    SeekHead {
        old_offset: u64,
        entries: Vec<SeekEntry>,
    },
    Cues {
        old_offset: u64,
        raw_payload: &'a [u8],
    },
    Cluster {
        old_offset: u64,
        rendered: Vec<u8>,
        modified: bool,
    },
    Other {
        old_offset: u64,
        raw: &'a [u8],
    },
}

impl SegItem<'_> {
    const fn old_offset(&self) -> u64 {
        match self {
            Self::SeekHead { old_offset, .. }
            | Self::Cues { old_offset, .. }
            | Self::Cluster { old_offset, .. }
            | Self::Other { old_offset, .. } => *old_offset,
        }
    }

    fn static_bytes(&self) -> Option<&[u8]> {
        match self {
            Self::Cluster { rendered, .. } => Some(rendered.as_slice()),
            Self::Other { raw, .. } => Some(raw),
            Self::SeekHead { .. } | Self::Cues { .. } => None,
        }
    }
}

/// Applies EBML `SimpleBlock` / `BlockGroup` keyframe stripping to an in-memory WebM/Matroska
/// byte stream, rewriting `Cues` and `SeekHead` byte offsets to keep seeking intact.
///
/// Returns `(rewritten_bytes, stripped_keyframe_count)`.
#[must_use]
pub fn apply_webm_datamosh(
    bytes: &[u8],
    mode: DatamoshMode,
    cut_times_ms: &[u64],
) -> (Vec<u8>, usize) {
    if mode == DatamoshMode::Off
        || bytes.is_empty()
        || (mode == DatamoshMode::AtClipCuts && cut_times_ms.is_empty())
    {
        return (bytes.to_vec(), 0);
    }

    let mut out = Vec::with_capacity(bytes.len());
    let mut pos = 0_usize;
    let mut total_stripped = 0_usize;

    while pos < bytes.len() {
        let Some((id, id_len)) = read_ebml_id(bytes, pos) else {
            out.extend_from_slice(&bytes[pos..]);
            break;
        };
        let Some((size_opt, sz_len)) = read_ebml_size(bytes, pos + id_len) else {
            out.extend_from_slice(&bytes[pos..]);
            break;
        };
        let header_end = pos + id_len + sz_len;
        let elem_end = match size_opt {
            Some(sz) => header_end.saturating_add(sz).min(bytes.len()),
            None if id == ID_SEGMENT => find_unknown_segment_end(bytes, header_end, bytes.len()),
            None => bytes.len(),
        };

        if id != ID_SEGMENT {
            out.extend_from_slice(&bytes[pos..elem_end]);
            pos = elem_end;
            continue;
        }

        // First scan inside Segment to discover TimecodeScale and video track numbers.
        let mut timecode_scale_ns: u64 = 1_000_000;
        let mut video_tracks: Vec<u64> = Vec::new();
        {
            let mut sp = header_end;
            while sp < elem_end {
                let Some((cid, cid_len)) = read_ebml_id(bytes, sp) else {
                    break;
                };
                let Some((csz_opt, csz_len)) = read_ebml_size(bytes, sp + cid_len) else {
                    break;
                };
                let cb_start = sp + cid_len + csz_len;
                let cb_end = match csz_opt {
                    Some(sz) => cb_start.saturating_add(sz).min(elem_end),
                    None if cid == ID_CLUSTER => {
                        find_unknown_cluster_end(bytes, cb_start, elem_end)
                    }
                    None => elem_end,
                };
                if cid == ID_INFO {
                    timecode_scale_ns = parse_info_timecode_scale(&bytes[cb_start..cb_end]);
                } else if cid == ID_TRACKS {
                    video_tracks = parse_video_track_numbers(&bytes[cb_start..cb_end]);
                }
                if cb_end <= sp {
                    break;
                }
                sp = cb_end;
            }
        }
        if video_tracks.is_empty() {
            video_tracks.push(1);
        }

        // Pass 1: Collect all Level 1 elements, stripping target keyframes inside Clusters.
        let mut items: Vec<SegItem<'_>> = Vec::new();
        let mut seen_first_keyframe = false;
        let mut seg_stripped = 0_usize;
        let mut sp = header_end;
        while sp < elem_end {
            let old_offset = (sp - header_end) as u64;
            let Some((cid, cid_len)) = read_ebml_id(bytes, sp) else {
                items.push(SegItem::Other {
                    old_offset,
                    raw: &bytes[sp..elem_end],
                });
                break;
            };
            let Some((csz_opt, csz_len)) = read_ebml_size(bytes, sp + cid_len) else {
                items.push(SegItem::Other {
                    old_offset,
                    raw: &bytes[sp..elem_end],
                });
                break;
            };
            let cb_start = sp + cid_len + csz_len;
            let cb_end = match csz_opt {
                Some(sz) => cb_start.saturating_add(sz).min(elem_end),
                None if cid == ID_CLUSTER => find_unknown_cluster_end(bytes, cb_start, elem_end),
                None => elem_end,
            };

            if cid == ID_SEEK_HEAD {
                items.push(SegItem::SeekHead {
                    old_offset,
                    entries: parse_seek_head_entries(&bytes[cb_start..cb_end]),
                });
            } else if cid == ID_CUES {
                items.push(SegItem::Cues {
                    old_offset,
                    raw_payload: &bytes[cb_start..cb_end],
                });
            } else if cid == ID_CLUSTER {
                let mut cluster_body = Vec::with_capacity(cb_end.saturating_sub(cb_start));
                let mut cluster_tc: u64 = 0;
                let mut cluster_stripped = 0_usize;
                let mut cp = cb_start;
                while cp < cb_end {
                    let Some((bid, bid_len)) = read_ebml_id(bytes, cp) else {
                        cluster_body.extend_from_slice(&bytes[cp..cb_end]);
                        break;
                    };
                    let Some((bsz_opt, bsz_len)) = read_ebml_size(bytes, cp + bid_len) else {
                        cluster_body.extend_from_slice(&bytes[cp..cb_end]);
                        break;
                    };
                    let bb_start = cp + bid_len + bsz_len;
                    let bb_end = match bsz_opt {
                        Some(sz) => bb_start.saturating_add(sz).min(cb_end),
                        None => cb_end,
                    };

                    if bid == ID_CLUSTER_TIMECODE {
                        cluster_tc = read_uint_be(&bytes[bb_start..bb_end]);
                        cluster_body.extend_from_slice(&bytes[cp..bb_end]);
                    } else if bid == ID_SIMPLE_BLOCK {
                        let block = &bytes[bb_start..bb_end];
                        let mut strip = false;
                        if let Some((Some(track_num), tn_len)) = read_ebml_size(block, 0)
                            && block.len() >= tn_len + 3
                            && video_tracks.contains(&(track_num as u64))
                        {
                            let rel_tc = i16::from_be_bytes([block[tn_len], block[tn_len + 1]]);
                            let flags = block[tn_len + 2];
                            let is_keyframe = (flags & 0x80) != 0;
                            if is_keyframe {
                                let abs_ticks = cluster_tc.saturating_add_signed(i64::from(rel_tc));
                                let abs_ms = (u128::from(abs_ticks) * u128::from(timecode_scale_ns)
                                    / 1_000_000)
                                    as u64;
                                if should_strip_keyframe(
                                    mode,
                                    abs_ms,
                                    cut_times_ms,
                                    &mut seen_first_keyframe,
                                ) {
                                    strip = true;
                                    cluster_stripped += 1;
                                }
                            }
                        }
                        if !strip {
                            cluster_body.extend_from_slice(&bytes[cp..bb_end]);
                        }
                    } else if bid == ID_BLOCK_GROUP {
                        let mut strip = false;
                        if let Some(rel_tc) = parse_block_group_keyframe_rel_tc(
                            &bytes[bb_start..bb_end],
                            &video_tracks,
                        ) {
                            let abs_ticks = cluster_tc.saturating_add_signed(i64::from(rel_tc));
                            let abs_ms = (u128::from(abs_ticks) * u128::from(timecode_scale_ns)
                                / 1_000_000) as u64;
                            if should_strip_keyframe(
                                mode,
                                abs_ms,
                                cut_times_ms,
                                &mut seen_first_keyframe,
                            ) {
                                strip = true;
                                cluster_stripped += 1;
                            }
                        }
                        if !strip {
                            cluster_body.extend_from_slice(&bytes[cp..bb_end]);
                        }
                    } else {
                        cluster_body.extend_from_slice(&bytes[cp..bb_end]);
                    }
                    if bb_end <= cp {
                        break;
                    }
                    cp = bb_end;
                }

                seg_stripped += cluster_stripped;
                let encoded_size = encode_ebml_size(cluster_body.len() as u64, csz_len);
                let mut rendered =
                    Vec::with_capacity(cid_len + encoded_size.len() + cluster_body.len());
                rendered.extend_from_slice(&bytes[sp..sp + cid_len]);
                rendered.extend_from_slice(&encoded_size);
                rendered.extend_from_slice(&cluster_body);

                items.push(SegItem::Cluster {
                    old_offset,
                    rendered,
                    modified: cluster_stripped > 0,
                });
            } else {
                items.push(SegItem::Other {
                    old_offset,
                    raw: &bytes[sp..cb_end],
                });
            }

            if cb_end <= sp {
                break;
            }
            sp = cb_end;
        }

        total_stripped += seg_stripped;
        if seg_stripped == 0 {
            // Preserve the Segment byte-for-byte when no keyframes were stripped.
            out.extend_from_slice(&bytes[pos..elem_end]);
            pos = elem_end;
            continue;
        }

        // Pass 2: Converge SeekHead & Cues byte offsets via a bounded fixed-point layout pass.
        let seg_body = assemble_segment_body(&items);

        // Emit Segment header with exact finite EBML size.
        out.extend_from_slice(&bytes[pos..pos + id_len]);
        let encoded_seg_size = encode_ebml_size(seg_body.len() as u64, sz_len);
        out.extend_from_slice(&encoded_seg_size);
        out.extend_from_slice(&seg_body);
        pos = elem_end;
    }

    (out, total_stripped)
}

fn assemble_segment_body(items: &[SegItem<'_>]) -> Vec<u8> {
    let mut dynamic: Vec<Vec<u8>> = vec![Vec::new(); items.len()];
    let mut active: Vec<bool> = vec![true; items.len()];
    let mut converged = false;

    for _ in 0..10 {
        let mut offsets: Vec<u64> = Vec::with_capacity(items.len());
        let mut cursor = 0_u64;
        for (idx, item) in items.iter().enumerate() {
            offsets.push(cursor);
            let len = item
                .static_bytes()
                .map_or_else(|| dynamic[idx].len(), <[u8]>::len);
            cursor = cursor.saturating_add(len as u64);
        }

        let mut offset_map = BTreeMap::new();
        let mut cluster_map = BTreeMap::new();
        for (idx, item) in items.iter().enumerate() {
            if active[idx] {
                offset_map.insert(item.old_offset(), offsets[idx]);
            }
            if let SegItem::Cluster {
                old_offset,
                modified,
                ..
            } = item
            {
                cluster_map.insert(*old_offset, (offsets[idx], *modified));
            }
        }

        let mut next_dynamic = dynamic.clone();
        let mut next_active = active.clone();
        for (idx, item) in items.iter().enumerate() {
            match item {
                SegItem::Cues { raw_payload, .. } => {
                    let rebuilt = rewrite_cues_element(raw_payload, &cluster_map);
                    next_active[idx] = !rebuilt.is_empty();
                    next_dynamic[idx] = rebuilt;
                }
                SegItem::SeekHead { entries, .. } => {
                    let rebuilt = rewrite_seek_head_element(entries, &offset_map);
                    next_active[idx] = !rebuilt.is_empty();
                    next_dynamic[idx] = rebuilt;
                }
                SegItem::Cluster { .. } | SegItem::Other { .. } => {}
            }
        }

        if next_dynamic == dynamic && next_active == active {
            converged = true;
            break;
        }
        dynamic = next_dynamic;
        active = next_active;
    }

    if !converged {
        // Safety fallback: omit SeekHead and converge Cues alone so no stale offsets remain.
        for (idx, item) in items.iter().enumerate() {
            if matches!(item, SegItem::SeekHead { .. }) {
                dynamic[idx].clear();
                active[idx] = false;
            }
        }
        for _ in 0..8 {
            let mut offsets: Vec<u64> = Vec::with_capacity(items.len());
            let mut cursor = 0_u64;
            for (idx, item) in items.iter().enumerate() {
                offsets.push(cursor);
                let len = item
                    .static_bytes()
                    .map_or_else(|| dynamic[idx].len(), <[u8]>::len);
                cursor = cursor.saturating_add(len as u64);
            }
            let mut cluster_map = BTreeMap::new();
            for (idx, item) in items.iter().enumerate() {
                if let SegItem::Cluster {
                    old_offset,
                    modified,
                    ..
                } = item
                {
                    cluster_map.insert(*old_offset, (offsets[idx], *modified));
                }
            }
            let mut changed = false;
            for (idx, item) in items.iter().enumerate() {
                if let SegItem::Cues { raw_payload, .. } = item {
                    let rebuilt = rewrite_cues_element(raw_payload, &cluster_map);
                    if rebuilt != dynamic[idx] {
                        dynamic[idx] = rebuilt;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    let total_len: usize = items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            item.static_bytes()
                .map_or_else(|| dynamic[idx].len(), <[u8]>::len)
        })
        .sum();
    let mut seg_body = Vec::with_capacity(total_len);
    for (idx, item) in items.iter().enumerate() {
        if let Some(bytes) = item.static_bytes() {
            seg_body.extend_from_slice(bytes);
        } else {
            seg_body.extend_from_slice(&dynamic[idx]);
        }
    }
    seg_body
}

#[cfg(not(target_arch = "wasm32"))]
fn datamosh_temp_path(path: &Path) -> PathBuf {
    let mut tmp = path.as_os_str().to_os_string();
    tmp.push(".datamosh.tmp");
    PathBuf::from(tmp)
}

/// Post-processes a WebM/MKV file on disk atomically with True Datamosh keyframe stripping.
///
/// Writes the rewritten stream to `<path>.datamosh.tmp` and atomically renames over `path`,
/// removing the temporary file if any error occurs.
#[cfg(not(target_arch = "wasm32"))]
pub fn post_process_webm_datamosh_file(
    path: &Path,
    mode: DatamoshMode,
    cut_times_ms: &[u64],
) -> Result<usize, ExportError> {
    if mode == DatamoshMode::Off || (mode == DatamoshMode::AtClipCuts && cut_times_ms.is_empty()) {
        return Ok(0);
    }
    let raw = std::fs::read(path)?;
    let (rewritten, stripped) = apply_webm_datamosh(&raw, mode, cut_times_ms);
    if stripped > 0 {
        let mut tmp_guard = TempFileGuard::new(datamosh_temp_path(path));
        {
            let mut file = File::create(tmp_guard.path())?;
            file.write_all(&rewritten)?;
            file.flush()?;
        }
        publish_temp_output(tmp_guard.path(), path)?;
        tmp_guard.disarm();
    }
    Ok(stripped)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ebml_elem(id: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        // 2-byte VINT size supports up to 16382 bytes.
        let sz = payload.len() as u16;
        out.push(0x40 | ((sz >> 8) as u8 & 0x3F));
        out.push((sz & 0xFF) as u8);
        out.extend_from_slice(payload);
        out
    }

    fn unknown_size_elem(id: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(id);
        out.extend_from_slice(&[0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        out.extend_from_slice(payload);
        out
    }

    fn simple_block(track_num: u8, rel_tc: i16, keyframe: bool) -> Vec<u8> {
        let mut payload = vec![0x80 | track_num];
        payload.extend_from_slice(&rel_tc.to_be_bytes());
        payload.push(if keyframe { 0x80 } else { 0x00 });
        payload.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
        ebml_elem(&[0xA3], &payload)
    }

    fn block_group_with_alpha(track_num: u8, rel_tc: i16, keyframe: bool) -> Vec<u8> {
        let mut blk_payload = vec![0x80 | track_num];
        blk_payload.extend_from_slice(&rel_tc.to_be_bytes());
        blk_payload.push(0x00); // Block (0xA1) does not carry the SimpleBlock keyframe bit.
        blk_payload.extend_from_slice(&[0x11, 0x22, 0x33]);
        let blk = ebml_elem(&[0xA1], &blk_payload);
        // BlockAdditions (0x75A1) carrying a VP9 alpha plane.
        let block_additions = ebml_elem(&[0x75, 0xA1], &[0xA6, 0x83, 0xAA, 0xBB, 0xCC]);
        let mut group = [blk, block_additions].concat();
        if !keyframe {
            // Non-keyframes inside BlockGroup carry ReferenceBlock (0xFB).
            group.extend_from_slice(&ebml_elem(&[0xFB], &(-33_i16).to_be_bytes()));
        }
        ebml_elem(&[0xA0], &group)
    }

    fn build_info_and_tracks() -> (Vec<u8>, Vec<u8>) {
        // Track 1 = video (TrackType=1), Track 2 = audio (TrackType=2).
        let t1 = ebml_elem(
            &[0xAE],
            &[ebml_elem(&[0xD7], &[1]), ebml_elem(&[0x83], &[1])].concat(),
        );
        let t2 = ebml_elem(
            &[0xAE],
            &[ebml_elem(&[0xD7], &[2]), ebml_elem(&[0x83], &[2])].concat(),
        );
        let tracks = ebml_elem(&[0x16, 0x54, 0xAE, 0x6B], &[t1, t2].concat());
        // Info with TimecodeScale = 1_000_000 ns (1 ms).
        let info = ebml_elem(
            &[0x15, 0x49, 0xA9, 0x66],
            &ebml_elem(&[0x2A, 0xD7, 0xB1], &1_000_000u32.to_be_bytes()),
        );
        (info, tracks)
    }

    fn build_synthetic_webm() -> Vec<u8> {
        let (info, tracks) = build_info_and_tracks();
        // Cluster at 0ms: video KF (0ms), video delta (33ms), audio KF (0ms).
        let c0 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &[0]),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
                simple_block(2, 0, true),
            ]
            .concat(),
        );
        // Cluster at 1000ms: video KF (1000ms), video delta (1033ms).
        let c1 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &1000u16.to_be_bytes()),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
            ]
            .concat(),
        );
        // Cluster at 3000ms: video KF (3000ms).
        let c2 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &3000u16.to_be_bytes()),
                simple_block(1, 0, true),
            ]
            .concat(),
        );
        let mut file = ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &[0x42, 0x86, 0x81, 0x01]);
        file.extend_from_slice(&ebml_elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[info, tracks, c0, c1, c2].concat(),
        ));
        file
    }

    fn build_cue_point(
        cue_time: u64,
        track: u64,
        cluster_pos: u64,
        rel_pos: Option<u64>,
    ) -> Vec<u8> {
        let mut tp = Vec::new();
        tp.extend_from_slice(&encode_ebml_elem(&[0xF7], &encode_uint_be(track)));
        tp.extend_from_slice(&encode_ebml_elem(&[0xF1], &encode_uint_be(cluster_pos)));
        if let Some(rp) = rel_pos {
            tp.extend_from_slice(&encode_ebml_elem(&[0xF0], &encode_uint_be(rp)));
        }
        let mut pt = encode_ebml_elem(&[0xB3], &encode_uint_be(cue_time));
        pt.extend_from_slice(&encode_ebml_elem(&[0xB7], &tp));
        encode_ebml_elem(&[0xBB], &pt)
    }

    fn build_seek_entry(id_bytes: &[u8], pos: u64) -> Vec<u8> {
        let mut s = encode_ebml_elem(&[0x53, 0xAB], id_bytes);
        s.extend_from_slice(&encode_ebml_elem(&[0x53, 0xAC], &encode_uint_be(pos)));
        encode_ebml_elem(&[0x4D, 0xBB], &s)
    }

    fn find_segment_payload(bytes: &[u8]) -> &[u8] {
        let mut p = 0;
        while p < bytes.len() {
            let (id, id_len) = read_ebml_id(bytes, p).expect("ebml id");
            let (sz_opt, sz_len) = read_ebml_size(bytes, p + id_len).expect("ebml size");
            let b_start = p + id_len + sz_len;
            let b_end = match sz_opt {
                Some(sz) => b_start + sz,
                None => bytes.len(),
            };
            if id == ID_SEGMENT {
                return &bytes[b_start..b_end];
            }
            p = b_end;
        }
        panic!("Segment not found");
    }

    #[test]
    fn webm_datamosh_strips_expected_video_keyframes_and_keeps_audio() {
        let webm = build_synthetic_webm();
        let (off_bytes, off_count) = apply_webm_datamosh(&webm, DatamoshMode::Off, &[1000]);
        assert_eq!(off_count, 0);
        assert_eq!(off_bytes, webm);

        // FullVideo keeps first video KF (0ms) & audio KF, strips KFs at 1000ms and 3000ms.
        let (full_bytes, full_count) = apply_webm_datamosh(&webm, DatamoshMode::FullVideo, &[]);
        assert_eq!(full_count, 2);
        assert!(full_bytes.len() < webm.len());

        // AtClipCuts with cut at 1000ms strips only the 1000ms KF, leaving 0ms and 3000ms intact.
        let (cut_bytes, cut_count) = apply_webm_datamosh(&webm, DatamoshMode::AtClipCuts, &[1000]);
        assert_eq!(cut_count, 1);
        assert!(cut_bytes.len() > full_bytes.len() && cut_bytes.len() < webm.len());
    }

    #[test]
    fn at_clip_cuts_with_empty_cut_times_is_clean_noop() {
        let webm = build_synthetic_webm();
        let (out, stripped) = apply_webm_datamosh(&webm, DatamoshMode::AtClipCuts, &[]);
        assert_eq!(stripped, 0);
        assert_eq!(out, webm);
    }

    #[test]
    fn block_group_with_alpha_strips_secondary_keyframes_and_keeps_deltas() {
        let (info, tracks) = build_info_and_tracks();
        let c0 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &[0]),
                block_group_with_alpha(1, 0, true), // first KF -> keep
                block_group_with_alpha(1, 33, false), // delta -> keep
            ]
            .concat(),
        );
        let c1 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &1000u16.to_be_bytes()),
                block_group_with_alpha(1, 0, true), // secondary KF -> strip
                block_group_with_alpha(1, 33, false), // delta -> keep
            ]
            .concat(),
        );
        let mut file = ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &[0x42, 0x86, 0x81, 0x01]);
        file.extend_from_slice(&ebml_elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[info, tracks, c0, c1].concat(),
        ));

        let (out, stripped) = apply_webm_datamosh(&file, DatamoshMode::FullVideo, &[]);
        assert_eq!(stripped, 1);
        assert!(out.len() < file.len());

        // Verify the delta BlockGroup in cluster 1 survived while the keyframe BlockGroup was dropped.
        let seg = find_segment_payload(&out);
        let mut block_groups_seen = 0_usize;
        let mut sp = 0;
        while sp < seg.len() {
            let (cid, cid_len) = read_ebml_id(seg, sp).expect("cid");
            let (Some(csz), csz_len) = read_ebml_size(seg, sp + cid_len).expect("csz") else {
                panic!("rewritten element should have finite size");
            };
            let cb_start = sp + cid_len + csz_len;
            let cb_end = cb_start + csz;
            if cid == ID_CLUSTER {
                let mut cp = cb_start;
                while cp < cb_end {
                    let (bid, bid_len) = read_ebml_id(seg, cp).expect("bid");
                    let (Some(bsz), bsz_len) = read_ebml_size(seg, cp + bid_len).expect("bsz")
                    else {
                        panic!("finite child");
                    };
                    if bid == ID_BLOCK_GROUP {
                        block_groups_seen += 1;
                    }
                    cp += bid_len + bsz_len + bsz;
                }
            }
            sp = cb_end;
        }
        assert_eq!(
            block_groups_seen, 3,
            "first KF + 2 delta BlockGroups should survive"
        );
    }

    #[test]
    fn unknown_size_segment_and_clusters_parse_and_rewrite_cleanly() {
        let (info, tracks) = build_info_and_tracks();
        let c0 = unknown_size_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &[0]),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
            ]
            .concat(),
        );
        let c1 = unknown_size_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &1000u16.to_be_bytes()),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
            ]
            .concat(),
        );
        let c2 = unknown_size_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &2000u16.to_be_bytes()),
                simple_block(1, 0, true),
            ]
            .concat(),
        );
        let mut file = ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &[0x42, 0x86, 0x81, 0x01]);
        file.extend_from_slice(&unknown_size_elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[info, tracks, c0, c1, c2].concat(),
        ));

        let (out, stripped) = apply_webm_datamosh(&file, DatamoshMode::FullVideo, &[]);
        assert_eq!(
            stripped, 2,
            "both secondary keyframes across unknown-size Clusters must be stripped"
        );

        // Ensure all 3 Clusters are still individually parseable with finite sizes.
        let seg = find_segment_payload(&out);
        let mut cluster_timecodes = Vec::new();
        let mut sp = 0;
        while sp < seg.len() {
            let (cid, cid_len) = read_ebml_id(seg, sp).expect("cid");
            let (Some(csz), csz_len) = read_ebml_size(seg, sp + cid_len).expect("csz") else {
                panic!("rewritten element must have finite size");
            };
            let cb_start = sp + cid_len + csz_len;
            let cb_end = cb_start + csz;
            if cid == ID_CLUSTER {
                let (bid, bid_len) = read_ebml_id(seg, cb_start).expect("tc id");
                let (Some(bsz), bsz_len) = read_ebml_size(seg, cb_start + bid_len).expect("tc sz")
                else {
                    panic!("finite tc");
                };
                assert_eq!(bid, ID_CLUSTER_TIMECODE);
                let tc_start = cb_start + bid_len + bsz_len;
                cluster_timecodes.push(read_uint_be(&seg[tc_start..tc_start + bsz]));
            }
            sp = cb_end;
        }
        assert_eq!(cluster_timecodes, vec![0, 1000, 2000]);
    }

    #[test]
    fn seek_head_and_cues_offsets_are_rewritten_accurately() {
        let (info, tracks) = build_info_and_tracks();
        let c0 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &[0]),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
            ]
            .concat(),
        );
        let c1 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &1000u16.to_be_bytes()),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
            ]
            .concat(),
        );
        let c2 = ebml_elem(
            &[0x1F, 0x43, 0xB6, 0x75],
            &[
                ebml_elem(&[0xE7], &2000u16.to_be_bytes()),
                simple_block(1, 0, true),
                simple_block(1, 33, false),
            ]
            .concat(),
        );

        // Compute exact Segment-relative offsets with a fixed-size SeekHead preamble.
        let make_sh = |info_p: u64, tracks_p: u64, cues_p: u64| {
            let body = [
                build_seek_entry(&[0x15, 0x49, 0xA9, 0x66], info_p),
                build_seek_entry(&[0x16, 0x54, 0xAE, 0x6B], tracks_p),
                build_seek_entry(&[0x1C, 0x53, 0xBB, 0x6B], cues_p),
                // Stale/bogus Seek entry pointing to a nonexistent offset should be stripped.
                build_seek_entry(&[0x12, 0x54, 0xC3, 0x67], 999_999),
            ]
            .concat();
            ebml_elem(&[0x11, 0x4D, 0x9B, 0x74], &body)
        };
        let mut sh = make_sh(0, 0, 0);
        for _ in 0..4 {
            let info_p = sh.len() as u64;
            let tracks_p = info_p + info.len() as u64;
            let c0_p = tracks_p + tracks.len() as u64;
            let c1_p = c0_p + c0.len() as u64;
            let c2_p = c1_p + c1.len() as u64;
            let cues_p = c2_p + c2.len() as u64;
            sh = make_sh(info_p, tracks_p, cues_p);
        }
        let info_p = sh.len() as u64;
        let tracks_p = info_p + info.len() as u64;
        let c0_p = tracks_p + tracks.len() as u64;
        let c1_p = c0_p + c0.len() as u64;
        let c2_p = c1_p + c1.len() as u64;
        let cues = ebml_elem(
            &[0x1C, 0x53, 0xBB, 0x6B],
            &[
                build_cue_point(0, 1, c0_p, Some(5)),
                build_cue_point(1000, 1, c1_p, Some(6)),
                build_cue_point(2000, 1, c2_p, Some(6)),
            ]
            .concat(),
        );

        let mut file = ebml_elem(&[0x1A, 0x45, 0xDF, 0xA3], &[0x42, 0x86, 0x81, 0x01]);
        file.extend_from_slice(&ebml_elem(
            &[0x18, 0x53, 0x80, 0x67],
            &[sh, info, tracks, c0, c1, c2, cues].concat(),
        ));

        let (out, stripped) = apply_webm_datamosh(&file, DatamoshMode::FullVideo, &[]);
        assert_eq!(stripped, 2);

        let seg = find_segment_payload(&out);

        // Locate rewritten SeekHead and Cues and verify every recorded offset matches the actual
        // element ID and Cluster timecode at that exact byte offset in `seg`.
        let mut seek_entries = Vec::new();
        let mut cue_positions = Vec::new();
        let mut cue_has_rel_pos = Vec::new();

        let mut sp = 0;
        while sp < seg.len() {
            let (cid, cid_len) = read_ebml_id(seg, sp).expect("cid");
            let (Some(csz), csz_len) = read_ebml_size(seg, sp + cid_len).expect("csz") else {
                panic!("finite size");
            };
            let cb_start = sp + cid_len + csz_len;
            let cb_end = cb_start + csz;
            if cid == ID_SEEK_HEAD {
                seek_entries = parse_seek_head_entries(&seg[cb_start..cb_end]);
            } else if cid == ID_CUES {
                let mut cp = cb_start;
                while cp < cb_end {
                    let (pid, pid_len) = read_ebml_id(seg, cp).expect("pid");
                    let (Some(psz), psz_len) = read_ebml_size(seg, cp + pid_len).expect("psz")
                    else {
                        panic!("finite");
                    };
                    let pb_start = cp + pid_len + psz_len;
                    let pb_end = pb_start + psz;
                    if pid == ID_CUE_POINT {
                        let mut cue_time = 0_u64;
                        let mut cluster_pos = None;
                        let mut rel_pos_present = false;
                        let mut q = pb_start;
                        while q < pb_end {
                            let (qid, qid_len) = read_ebml_id(seg, q).expect("qid");
                            let (Some(qsz), qsz_len) =
                                read_ebml_size(seg, q + qid_len).expect("qsz")
                            else {
                                panic!("finite");
                            };
                            let qb_start = q + qid_len + qsz_len;
                            let qb_end = qb_start + qsz;
                            if qid == 0xB3 {
                                cue_time = read_uint_be(&seg[qb_start..qb_end]);
                            } else if qid == ID_CUE_TRACK_POSITIONS {
                                let mut t = qb_start;
                                while t < qb_end {
                                    let (tid, tid_len) = read_ebml_id(seg, t).expect("tid");
                                    let (Some(tsz), tsz_len) =
                                        read_ebml_size(seg, t + tid_len).expect("tsz")
                                    else {
                                        panic!("finite");
                                    };
                                    let tb_start = t + tid_len + tsz_len;
                                    let tb_end = tb_start + tsz;
                                    if tid == ID_CUE_CLUSTER_POSITION {
                                        cluster_pos = Some(read_uint_be(&seg[tb_start..tb_end]));
                                    } else if tid == ID_CUE_RELATIVE_POSITION {
                                        rel_pos_present = true;
                                    }
                                    t = tb_end;
                                }
                            }
                            q = qb_end;
                        }
                        cue_positions.push((cue_time, cluster_pos.expect("cluster_pos")));
                        cue_has_rel_pos.push(rel_pos_present);
                    }
                    cp = pb_end;
                }
            }
            sp = cb_end;
        }

        // Bogus Seek entry was stripped; remaining 3 entries point to exact Info, Tracks, Cues IDs.
        assert_eq!(seek_entries.len(), 3);
        for entry in &seek_entries {
            let off = entry.old_pos as usize;
            assert_eq!(
                &seg[off..off + entry.seek_id.len()],
                entry.seek_id.as_slice(),
                "SeekHead offset {off} did not point to expected ID {:02X?}",
                entry.seek_id
            );
        }

        // Every CueClusterPosition points directly to a Cluster header with matching timecode.
        assert_eq!(cue_positions.len(), 3);
        for (cue_time, pos) in &cue_positions {
            let off = *pos as usize;
            let (id, id_len) = read_ebml_id(seg, off).expect("cluster id at CueClusterPosition");
            assert_eq!(id, ID_CLUSTER);
            let (Some(_), sz_len) = read_ebml_size(seg, off + id_len).expect("cluster sz") else {
                panic!("finite cluster");
            };
            let body_start = off + id_len + sz_len;
            let (tc_id, tc_id_len) = read_ebml_id(seg, body_start).expect("tc id");
            assert_eq!(tc_id, ID_CLUSTER_TIMECODE);
            let (Some(tc_sz), tc_sz_len) =
                read_ebml_size(seg, body_start + tc_id_len).expect("tc sz")
            else {
                panic!("finite tc");
            };
            let tc_val = read_uint_be(
                &seg[body_start + tc_id_len + tc_sz_len
                    ..body_start + tc_id_len + tc_sz_len + tc_sz],
            );
            assert_eq!(tc_val, *cue_time);
        }
        // Cluster 0 was unmodified (keeps CueRelativePosition); Clusters 1 & 2 were stripped
        // (stale CueRelativePosition omitted).
        assert_eq!(cue_has_rel_pos, vec![true, false, false]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn post_process_file_writes_atomically_and_cleans_up_temp_file() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "zc_datamosh_atomic_{}_{nanos}.webm",
            std::process::id()
        ));
        let _guard = TempFileGuard::new(path.clone());
        let webm = build_synthetic_webm();
        std::fs::write(&path, &webm).expect("write initial webm");

        // Empty AtClipCuts leaves file untouched.
        let noop = post_process_webm_datamosh_file(&path, DatamoshMode::AtClipCuts, &[])
            .expect("empty cuts noop");
        assert_eq!(noop, 0);
        assert_eq!(std::fs::read(&path).expect("read unchanged"), webm);

        // FullVideo rewrites via <path>.datamosh.tmp and renames atomically.
        let stripped = post_process_webm_datamosh_file(&path, DatamoshMode::FullVideo, &[])
            .expect("atomic datamosh");
        assert_eq!(stripped, 2);
        let after = std::fs::read(&path).expect("read rewritten");
        assert!(after.len() < webm.len());
        assert!(
            !datamosh_temp_path(&path).exists(),
            "temporary sibling file must not remain after atomic rename"
        );
    }
}
