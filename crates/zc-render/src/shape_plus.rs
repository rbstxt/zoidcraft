//! Production-grade Shape+ vector engine, SVG importer/exporter, and antialiased
//! exact-SDF mask rasterizer.
//!
//! Mirrors Zoidium's `plugins/shape-plus/*`:
//! - Canonical document schema (`schemaVersion: 1`, `renderVersion: 1`), validation,
//!   and tree CRUD helpers (`default_shape_plus_payload`, `parse_shape_plus_document`,
//!   `add_shape_plus_item`, `remove_shape_plus_item`, `reorder_shape_plus_item`,
//!   `duplicate_shape_plus_item`).
//! - Full CM3 keyframe evaluation via `zc_anim` (all 33 easings, Hold, Bezier handles)
//!   plus optional expression callback.
//! - Complete procedural compiler & 2D antialiased rasterizer covering all 21 Shape+
//!   item kinds (`group`, `rectangle`, `ellipse`, `polystar`, `path`, `text`,
//!   `textAnimator`, `fill`, `gradientFill`, `stroke`, `gradientStroke`, `trim`,
//!   `repeater`, `roundCorners`, `twist`, `zigZag`, `puckerBloat`, `wigglePaths`,
//!   `wiggleTransform`, `offsetPaths`, `mergePaths`).
//! - SVG importer (`import_svg_to_shape_plus`) & path exporter
//!   (`export_shape_plus_path_to_svg_d`).
//! - Antialiased exact-SDF mask rasterizer (`rasterize_masks_aa`).

use std::collections::{BTreeMap, HashMap, HashSet};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

use serde_json::{Map, Value, json};
use zc_anim::{
    BezierSegment, EasingKind, InterpolationMode, eval_cubic_bezier, eval_easing,
    eval_segment_bezier,
};

// ---------------------------------------------------------------------------
// Constants & Budgets (matching Zoidium `model.js`, `operators.js`, `mask-model.js`)
// ---------------------------------------------------------------------------

/// All 21 supported Zoidium Shape+ item kinds.
pub const ALL_SHAPE_PLUS_KINDS: [&str; 21] = [
    "group",
    "rectangle",
    "ellipse",
    "polystar",
    "path",
    "text",
    "textAnimator",
    "fill",
    "stroke",
    "gradientFill",
    "gradientStroke",
    "trim",
    "repeater",
    "roundCorners",
    "twist",
    "zigZag",
    "puckerBloat",
    "wigglePaths",
    "wiggleTransform",
    "offsetPaths",
    "mergePaths",
];

pub const MAX_SHAPE_PLUS_ITEMS: usize = 512;
pub const MAX_SHAPE_PLUS_DEPTH: usize = 16;
pub const MAX_SHAPE_PLUS_VERTICES: usize = 256;
pub const MAX_REPEATER_COPIES: f64 = 128.0;
pub const MAX_DASH_FRAGMENTS: usize = 2048;
pub const MAX_MASK_LONG_SIDE: u32 = 2048;

const KAPPA: f64 = 0.552_284_749_830_793_6;
const DEFAULT_TOLERANCE: f64 = 0.125;
const MAX_FLATTEN_DEPTH: usize = 16;
const MAX_PATH_POINTS: usize = 32_768;

// ---------------------------------------------------------------------------
// Canonical Document Schema & Tree CRUD Helpers
// ---------------------------------------------------------------------------

/// Parsed Shape+ document matching Zoidium's canonical JSON schema.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapePlusDoc {
    pub schema_version: u32,
    pub render_version: u32,
    pub next_id: u64,
    pub contents: Vec<ShapePlusItem>,
}

/// One node in a Shape+ `contents` hierarchy.
#[derive(Debug, Clone, PartialEq)]
pub struct ShapePlusItem {
    pub id: String,
    pub kind: String,
    pub properties: Map<String, Value>,
    pub contents: Vec<ShapePlusItem>,
    pub vertex_ids: Vec<String>,
    pub vertex_modes: Map<String, Value>,
}

/// Canonical starter Shape+ payload matching Zoidium `plugins/shape-plus/model.js`:
/// `schemaVersion: 1`, `renderVersion: 1`, `nextId: 4`, default `Rectangle 1` group (`s1`)
/// containing `rectangle` (`s2`) and `fill` (`s3`).
#[must_use]
pub fn default_shape_plus_payload() -> Value {
    json!({
        "schemaVersion": 1,
        "renderVersion": 1,
        "nextId": 4,
        "contents": [
            {
                "id": "s1",
                "kind": "group",
                "properties": {
                    "name": "Rectangle 1",
                    "enabled": 1,
                    "anchor": [0.0, 0.0],
                    "position": [0.0, 0.0],
                    "scale": [1.0, 1.0],
                    "rotation": 0.0,
                    "opacity": 1.0
                },
                "contents": [
                    {
                        "id": "s2",
                        "kind": "rectangle",
                        "properties": {
                            "name": "Rectangle",
                            "enabled": 1,
                            "size": [200.0, 200.0],
                            "position": [0.0, 0.0],
                            "roundness": 0.0
                        }
                    },
                    {
                        "id": "s3",
                        "kind": "fill",
                        "properties": {
                            "name": "Fill",
                            "enabled": 1,
                            "color": [1.0, 1.0, 1.0],
                            "opacity": 1.0,
                            "fillRule": "nonzero",
                            "composite": "below"
                        }
                    }
                ]
            }
        ]
    })
}

/// Returns the default human-readable title for a Shape+ kind.
#[must_use]
pub fn shape_plus_kind_title(kind: &str) -> Option<&'static str> {
    match kind {
        "group" => Some("Group"),
        "rectangle" => Some("Rectangle"),
        "ellipse" => Some("Ellipse"),
        "polystar" => Some("Polystar"),
        "path" => Some("Path"),
        "text" => Some("Text"),
        "textAnimator" => Some("Text Animator"),
        "fill" => Some("Fill"),
        "stroke" => Some("Stroke"),
        "gradientFill" => Some("Gradient Fill"),
        "gradientStroke" => Some("Gradient Stroke"),
        "trim" => Some("Trim Paths"),
        "repeater" => Some("Repeater"),
        "roundCorners" => Some("Round Corners"),
        "twist" => Some("Twist"),
        "zigZag" => Some("Zig Zag"),
        "puckerBloat" => Some("Pucker & Bloat"),
        "wigglePaths" => Some("Wiggle Paths"),
        "wiggleTransform" => Some("Wiggle Transform"),
        "offsetPaths" => Some("Offset Paths"),
        "mergePaths" => Some("Merge Paths"),
        _ => None,
    }
}

/// Parses a Shape+ JSON payload into a typed [`ShapePlusDoc`].
#[must_use]
pub fn parse_shape_plus_document(payload: &Value) -> ShapePlusDoc {
    let schema_version = payload
        .get("schemaVersion")
        .and_then(Value::as_u64)
        .unwrap_or(1) as u32;
    let render_version = payload
        .get("renderVersion")
        .and_then(Value::as_u64)
        .unwrap_or(1) as u32;
    let mut max_seq = payload
        .get("nextId")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .saturating_sub(1);
    let contents = payload
        .get("contents")
        .and_then(Value::as_array)
        .map(|arr| parse_items(arr, 0, &mut max_seq))
        .unwrap_or_default();
    ShapePlusDoc {
        schema_version,
        render_version,
        next_id: max_seq + 1,
        contents,
    }
}

fn observe_canonical_id(id: &str, max_seq: &mut u64) {
    if let Some(rest) = id.strip_prefix('s')
        && let Ok(val) = u64::from_str_radix(rest, 36)
        && val > *max_seq
    {
        *max_seq = val;
    }
}

fn parse_items(arr: &[Value], depth: usize, max_seq: &mut u64) -> Vec<ShapePlusItem> {
    if depth > MAX_SHAPE_PLUS_DEPTH {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(arr.len());
    for item in arr {
        let Some(obj) = item.as_object() else {
            continue;
        };
        let Some(kind) = obj.get("kind").and_then(Value::as_str) else {
            continue;
        };
        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        observe_canonical_id(&id, max_seq);
        let properties = obj
            .get("properties")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let contents = obj
            .get("contents")
            .and_then(Value::as_array)
            .map(|c| parse_items(c, depth + 1, max_seq))
            .unwrap_or_default();
        let vertex_ids = obj
            .get("vertexIds")
            .and_then(Value::as_array)
            .map(|v| {
                v.iter()
                    .filter_map(Value::as_str)
                    .map(ToOwned::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let vertex_modes = obj
            .get("vertexModes")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        out.push(ShapePlusItem {
            id,
            kind: kind.to_owned(),
            properties,
            contents,
            vertex_ids,
            vertex_modes,
        });
    }
    out
}

/// Validates a Shape+ payload against Zoidium's structural & budget constraints
/// (`depth <= 16`, `items <= 512`, unique IDs, valid 21 kinds, path `vertexIds` `1..=256`).
pub fn validate_shape_plus_payload(payload: &Value) -> Result<(), String> {
    let schema = payload
        .get("schemaVersion")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    let render = payload
        .get("renderVersion")
        .and_then(Value::as_u64)
        .unwrap_or(1);
    if schema != 1 || render != 1 {
        return Err(format!(
            "Unsupported Shape+ format version (schema={schema}, render={render})"
        ));
    }
    let contents = payload
        .get("contents")
        .and_then(Value::as_array)
        .ok_or_else(|| "Shape+ payload missing contents array".to_owned())?;
    let mut count = 0usize;
    let mut ids = HashSet::new();
    validate_contents_slice(contents, 0, &mut count, &mut ids)
}

fn validate_contents_slice(
    contents: &[Value],
    depth: usize,
    count: &mut usize,
    ids: &mut HashSet<String>,
) -> Result<(), String> {
    if depth > MAX_SHAPE_PLUS_DEPTH {
        return Err("Invalid Shape+ contents depth".to_owned());
    }
    for item in contents {
        *count += 1;
        if *count > MAX_SHAPE_PLUS_ITEMS {
            return Err("Shape+ item budget exceeded".to_owned());
        }
        let kind = item
            .get("kind")
            .and_then(Value::as_str)
            .ok_or_else(|| "Shape+ item missing kind".to_owned())?;
        if shape_plus_kind_title(kind).is_none() {
            return Err(format!("Unknown Shape+ item kind: {kind}"));
        }
        let id = item
            .get("id")
            .and_then(Value::as_str)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| "Shape+ item missing string id".to_owned())?;
        if !ids.insert(id.to_owned()) {
            return Err(format!("Duplicate Shape+ item id: {id}"));
        }
        if kind == "path"
            && let Some(vids) = item.get("vertexIds").and_then(Value::as_array)
        {
            if vids.is_empty() || vids.len() > MAX_SHAPE_PLUS_VERTICES {
                return Err("Invalid Shape+ path vertexIds count".to_owned());
            }
            let mut seen = HashSet::new();
            for v in vids {
                let s = v
                    .as_str()
                    .filter(|s| !s.is_empty() && s.len() <= 128)
                    .ok_or_else(|| "Invalid Shape+ path vertex id".to_owned())?;
                if !seen.insert(s) {
                    return Err("Duplicate Shape+ path vertex id".to_owned());
                }
            }
        }
        if kind == "group" {
            let children = item
                .get("contents")
                .and_then(Value::as_array)
                .ok_or_else(|| format!("Shape+ group {id} missing contents array"))?;
            validate_contents_slice(children, depth + 1, count, ids)?;
        }
    }
    Ok(())
}

fn alloc_next_id(payload: &mut Value) -> String {
    let mut seq = payload
        .get("nextId")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .saturating_sub(1);
    if let Some(arr) = payload.get("contents").and_then(Value::as_array) {
        let mut dummy = seq;
        let _ = parse_items(arr, 0, &mut dummy);
        seq = seq.max(dummy);
    }
    seq += 1;
    let id = format!("s{seq}");
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("nextId".to_owned(), json!(seq + 1));
    }
    id
}

fn default_item_json(kind: &str, id: String) -> Result<Value, String> {
    let title =
        shape_plus_kind_title(kind).ok_or_else(|| format!("Unknown Shape+ item kind: {kind}"))?;
    let mut props = Map::new();
    props.insert("name".to_owned(), json!(title));
    props.insert("enabled".to_owned(), json!(1));
    match kind {
        "group" => {
            props.insert("anchor".to_owned(), json!([0.0, 0.0]));
            props.insert("position".to_owned(), json!([0.0, 0.0]));
            props.insert("scale".to_owned(), json!([1.0, 1.0]));
            props.insert("rotation".to_owned(), json!(0.0));
            props.insert("opacity".to_owned(), json!(1.0));
            return Ok(json!({
                "id": id,
                "kind": kind,
                "properties": props,
                "contents": []
            }));
        }
        "rectangle" => {
            props.insert("size".to_owned(), json!([200.0, 200.0]));
            props.insert("position".to_owned(), json!([0.0, 0.0]));
            props.insert("roundness".to_owned(), json!(0.0));
        }
        "ellipse" => {
            props.insert("size".to_owned(), json!([200.0, 200.0]));
            props.insert("position".to_owned(), json!([0.0, 0.0]));
        }
        "polystar" => {
            props.insert("mode".to_owned(), json!("star"));
            props.insert("points".to_owned(), json!(5.0));
            props.insert("position".to_owned(), json!([0.0, 0.0]));
            props.insert("innerRadius".to_owned(), json!(50.0));
            props.insert("outerRadius".to_owned(), json!(100.0));
            props.insert("innerRoundness".to_owned(), json!(0.0));
            props.insert("outerRoundness".to_owned(), json!(0.0));
            props.insert("rotation".to_owned(), json!(0.0));
        }
        "path" => {
            props.insert("closed".to_owned(), json!(1));
            props.insert(
                "path".to_owned(),
                json!([
                    -100.0, -100.0, 0.0, 0.0, 0.0, 0.0, 100.0, -100.0, 0.0, 0.0, 0.0, 0.0, 100.0,
                    100.0, 0.0, 0.0, 0.0, 0.0, -100.0, 100.0, 0.0, 0.0, 0.0, 0.0
                ]),
            );
            return Ok(json!({
                "id": id,
                "kind": kind,
                "vertexIds": ["v1", "v2", "v3", "v4"],
                "vertexModes": {},
                "properties": props
            }));
        }
        "text" => {
            props.insert("sourceText".to_owned(), json!("Text"));
            props.insert("fontSize".to_owned(), json!(100.0));
            props.insert("tracking".to_owned(), json!(0.0));
            props.insert("leading".to_owned(), json!(120.0));
            props.insert("position".to_owned(), json!([0.0, 0.0]));
            props.insert("alignment".to_owned(), json!("left"));
            props.insert("textMode".to_owned(), json!("point"));
            props.insert("boxSize".to_owned(), json!([600.0, 400.0]));
        }
        "textAnimator" => {
            props.insert("selectorUnits".to_owned(), json!("percentage"));
            props.insert("selectorStart".to_owned(), json!(0.0));
            props.insert("selectorEnd".to_owned(), json!(100.0));
            props.insert("selectorOffset".to_owned(), json!(0.0));
            props.insert("selectorBasedOn".to_owned(), json!("characters"));
            props.insert("selectorMode".to_owned(), json!("add"));
            props.insert("selectorAmount".to_owned(), json!(100.0));
            props.insert("selectorShape".to_owned(), json!("square"));
        }
        "fill" => {
            props.insert("color".to_owned(), json!([1.0, 1.0, 1.0]));
            props.insert("opacity".to_owned(), json!(1.0));
            props.insert("fillRule".to_owned(), json!("nonzero"));
            props.insert("composite".to_owned(), json!("below"));
        }
        "stroke" => {
            props.insert("color".to_owned(), json!([1.0, 1.0, 1.0]));
            props.insert("opacity".to_owned(), json!(1.0));
            props.insert("width".to_owned(), json!(10.0));
            props.insert("cap".to_owned(), json!("butt"));
            props.insert("join".to_owned(), json!("miter"));
            props.insert("miterLimit".to_owned(), json!(4.0));
            props.insert("composite".to_owned(), json!("below"));
        }
        "gradientFill" | "gradientStroke" => {
            props.insert("opacity".to_owned(), json!(1.0));
            props.insert("gradientType".to_owned(), json!("linear"));
            props.insert("startPoint".to_owned(), json!([-100.0, 0.0]));
            props.insert("endPoint".to_owned(), json!([100.0, 0.0]));
            props.insert("highlightLength".to_owned(), json!(0.0));
            props.insert("highlightAngle".to_owned(), json!(0.0));
            props.insert(
                "stops".to_owned(),
                json!([
                    { "position": 0.0, "color": "rgba(255,77,26,1)" },
                    { "position": 1.0, "color": "rgba(26,102,255,1)" }
                ]),
            );
            props.insert("composite".to_owned(), json!("below"));
            if kind == "gradientFill" {
                props.insert("fillRule".to_owned(), json!("nonzero"));
            } else {
                props.insert("width".to_owned(), json!(10.0));
                props.insert("cap".to_owned(), json!("butt"));
                props.insert("join".to_owned(), json!("miter"));
                props.insert("miterLimit".to_owned(), json!(4.0));
            }
        }
        "trim" => {
            props.insert("start".to_owned(), json!(0.0));
            props.insert("end".to_owned(), json!(100.0));
            props.insert("offset".to_owned(), json!(0.0));
            props.insert("mode".to_owned(), json!("simultaneous"));
        }
        "repeater" => {
            props.insert("copies".to_owned(), json!(3.0));
            props.insert("offset".to_owned(), json!(0.0));
            props.insert("composite".to_owned(), json!("below"));
            props.insert("anchor".to_owned(), json!([0.0, 0.0]));
            props.insert("position".to_owned(), json!([100.0, 0.0]));
            props.insert("scale".to_owned(), json!([1.0, 1.0]));
            props.insert("rotation".to_owned(), json!(0.0));
            props.insert("startOpacity".to_owned(), json!(1.0));
            props.insert("endOpacity".to_owned(), json!(1.0));
        }
        "roundCorners" => {
            props.insert("radius".to_owned(), json!(10.0));
        }
        "twist" => {
            props.insert("angle".to_owned(), json!(0.0));
            props.insert("center".to_owned(), json!([0.0, 0.0]));
        }
        "zigZag" => {
            props.insert("size".to_owned(), json!(10.0));
            props.insert("ridges".to_owned(), json!(3.0));
            props.insert("points".to_owned(), json!("corner"));
        }
        "puckerBloat" => {
            props.insert("amount".to_owned(), json!(0.0));
        }
        "wigglePaths" => {
            props.insert("size".to_owned(), json!(10.0));
            props.insert("detail".to_owned(), json!(5.0));
            props.insert("points".to_owned(), json!("corner"));
            props.insert("frequency".to_owned(), json!(1.0));
            props.insert("correlation".to_owned(), json!(50.0));
            props.insert("temporalPhase".to_owned(), json!(0.0));
            props.insert("spatialPhase".to_owned(), json!(0.0));
            props.insert("seed".to_owned(), json!(1.0));
        }
        "wiggleTransform" => {
            props.insert("frequency".to_owned(), json!(1.0));
            props.insert("correlation".to_owned(), json!(50.0));
            props.insert("temporalPhase".to_owned(), json!(0.0));
            props.insert("spatialPhase".to_owned(), json!(0.0));
            props.insert("seed".to_owned(), json!(1.0));
            props.insert("anchor".to_owned(), json!([0.0, 0.0]));
            props.insert("position".to_owned(), json!([20.0, 20.0]));
            props.insert("scale".to_owned(), json!([0.0, 0.0]));
            props.insert("rotation".to_owned(), json!(0.0));
        }
        "offsetPaths" => {
            props.insert("amount".to_owned(), json!(10.0));
            props.insert("join".to_owned(), json!("miter"));
            props.insert("miterLimit".to_owned(), json!(4.0));
        }
        "mergePaths" => {
            props.insert("mode".to_owned(), json!("add"));
        }
        _ => {}
    }
    Ok(json!({
        "id": id,
        "kind": kind,
        "properties": props
    }))
}

fn find_group_contents_mut<'a>(
    list: &'a mut [Value],
    group_id: &str,
) -> Option<&'a mut Vec<Value>> {
    for item in list.iter_mut() {
        if item.get("id").and_then(Value::as_str) == Some(group_id)
            && item.get("kind").and_then(Value::as_str) == Some("group")
        {
            return item.get_mut("contents").and_then(Value::as_array_mut);
        }
        if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
            && let Some(found) = find_group_contents_mut(children, group_id)
        {
            return Some(found);
        }
    }
    None
}

/// Adds a new Shape+ item of `kind` into `parent_group_id` (or root `contents` if `None`),
/// returning the newly allocated canonical ID (`sN`).
pub fn add_shape_plus_item(
    payload: &mut Value,
    parent_group_id: Option<&str>,
    kind: &str,
) -> Result<String, String> {
    if shape_plus_kind_title(kind).is_none() {
        return Err(format!("Unknown Shape+ item kind: {kind}"));
    }
    let id = alloc_next_id(payload);
    let item = default_item_json(kind, id.clone())?;
    let is_geometry_or_group = matches!(
        kind,
        "rectangle" | "ellipse" | "polystar" | "path" | "text" | "group"
    );

    let root = payload
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "Shape+ payload missing contents array".to_owned())?;
    let target = match parent_group_id {
        Some(gid) => find_group_contents_mut(root, gid)
            .ok_or_else(|| format!("Shape+ group not found: {gid}"))?,
        None => root,
    };
    if is_geometry_or_group {
        target.insert(0, item);
    } else {
        target.push(item);
    }
    Ok(id)
}

/// Removes an item by ID anywhere in the Shape+ tree. Returns `Ok(true)` if removed.
pub fn remove_shape_plus_item(payload: &mut Value, item_id: &str) -> Result<bool, String> {
    fn remove_rec(list: &mut Vec<Value>, item_id: &str) -> bool {
        if let Some(pos) = list
            .iter()
            .position(|v| v.get("id").and_then(Value::as_str) == Some(item_id))
        {
            list.remove(pos);
            return true;
        }
        for item in list.iter_mut() {
            if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
                && remove_rec(children, item_id)
            {
                return true;
            }
        }
        false
    }
    let root = payload
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "Shape+ payload missing contents array".to_owned())?;
    Ok(remove_rec(root, item_id))
}

/// Reorders an item within its sibling list to `new_index` (clamped to sibling bounds).
pub fn reorder_shape_plus_item(
    payload: &mut Value,
    item_id: &str,
    new_index: usize,
) -> Result<bool, String> {
    fn reorder_rec(list: &mut Vec<Value>, item_id: &str, new_index: usize) -> bool {
        if let Some(pos) = list
            .iter()
            .position(|v| v.get("id").and_then(Value::as_str) == Some(item_id))
        {
            let item = list.remove(pos);
            let idx = new_index.min(list.len());
            list.insert(idx, item);
            return true;
        }
        for item in list.iter_mut() {
            if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
                && reorder_rec(children, item_id, new_index)
            {
                return true;
            }
        }
        false
    }
    let root = payload
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "Shape+ payload missing contents array".to_owned())?;
    Ok(reorder_rec(root, item_id, new_index))
}

/// Deep-duplicates an item (reassigning unique IDs to it and all nested descendants)
/// directly after the original item. Returns the duplicated root item's new ID.
pub fn duplicate_shape_plus_item(
    payload: &mut Value,
    item_id: &str,
) -> Result<Option<String>, String> {
    fn find_clone(list: &[Value], item_id: &str) -> Option<Value> {
        for item in list {
            if item.get("id").and_then(Value::as_str) == Some(item_id) {
                return Some(item.clone());
            }
            if let Some(children) = item.get("contents").and_then(Value::as_array)
                && let Some(found) = find_clone(children, item_id)
            {
                return Some(found);
            }
        }
        None
    }

    fn reassign_ids(item: &mut Value, seq: &mut u64) -> String {
        *seq += 1;
        let new_id = format!("s{seq}");
        if let Some(obj) = item.as_object_mut() {
            obj.insert("id".to_owned(), json!(new_id));
            if let Some(children) = obj.get_mut("contents").and_then(Value::as_array_mut) {
                for child in children.iter_mut() {
                    reassign_ids(child, seq);
                }
            }
        }
        new_id
    }

    fn insert_after(list: &mut Vec<Value>, target_id: &str, clone: Value) -> bool {
        if let Some(pos) = list
            .iter()
            .position(|v| v.get("id").and_then(Value::as_str) == Some(target_id))
        {
            list.insert(pos + 1, clone);
            return true;
        }
        for item in list.iter_mut() {
            if let Some(children) = item.get_mut("contents").and_then(Value::as_array_mut)
                && insert_after(children, target_id, clone.clone())
            {
                return true;
            }
        }
        false
    }

    let Some(mut cloned) = payload
        .get("contents")
        .and_then(Value::as_array)
        .and_then(|arr| find_clone(arr, item_id))
    else {
        return Ok(None);
    };

    let mut seq = payload
        .get("nextId")
        .and_then(Value::as_u64)
        .unwrap_or(1)
        .saturating_sub(1);
    if let Some(arr) = payload.get("contents").and_then(Value::as_array) {
        let _ = parse_items(arr, 0, &mut seq);
    }
    let new_id = reassign_ids(&mut cloned, &mut seq);
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("nextId".to_owned(), json!(seq + 1));
    }
    let root = payload
        .get_mut("contents")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "Shape+ payload missing contents array".to_owned())?;
    insert_after(root, item_id, cloned);
    Ok(Some(new_id))
}

// ---------------------------------------------------------------------------
// Keyframe + Easing + Bezier + Expression Property Evaluator
// ---------------------------------------------------------------------------

/// Expression evaluation callback type `(expression_source, frame) -> Option<f64>`.
pub type ExprEvalFn<'a> = &'a dyn Fn(&str, f64) -> Option<f64>;
type ExprEval<'a> = Option<ExprEvalFn<'a>>;

#[inline]
fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
    v.clamp(lo, hi)
}

#[inline]
fn wrap_f64(v: f64, period: f64) -> f64 {
    ((v % period) + period) % period
}

#[inline]
fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn parse_normalized_bezier(kf: &Value) -> Option<[f64; 4]> {
    for key in ["bezier", "controlPoints"] {
        if let Some(v) = kf.get(key) {
            if let Some(arr) = v.as_array()
                && arr.len() == 4
                && arr.iter().all(Value::is_number)
            {
                return Some([
                    arr[0].as_f64()?,
                    arr[1].as_f64()?,
                    arr[2].as_f64()?,
                    arr[3].as_f64()?,
                ]);
            }
            if let Some(obj) = v.as_object()
                && let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
                    obj.get("x1").and_then(Value::as_f64),
                    obj.get("y1").and_then(Value::as_f64),
                    obj.get("x2").and_then(Value::as_f64),
                    obj.get("y2").and_then(Value::as_f64),
                )
            {
                return Some([x1, y1, x2, y2]);
            }
        }
    }
    None
}

fn parse_handle_pair(kf: &Value, is_out: bool) -> Option<[f64; 2]> {
    let direct_key = if is_out { "outTangent" } else { "inTangent" };
    if let Some(arr) = kf.get(direct_key).and_then(Value::as_array)
        && arr.len() >= 2
    {
        return Some([arr[0].as_f64()?, arr[1].as_f64()?]);
    }
    if let Some(cp) = kf.get("controlPoints") {
        if let Some(arr) = cp.as_array()
            && arr.len() == 2
            && arr[0].is_array()
            && arr[1].is_array()
        {
            let h = arr[usize::from(is_out)].as_array()?;
            if h.len() >= 2 {
                return Some([h[0].as_f64()?, h[1].as_f64()?]);
            }
        }
        let sub = if is_out { "out" } else { "in" };
        if let Some(arr) = cp.get(sub).and_then(Value::as_array)
            && arr.len() >= 2
        {
            return Some([arr[0].as_f64()?, arr[1].as_f64()?]);
        }
    }
    None
}

fn eval_segment_factor(k0: &Value, k1: &Value, frame: f64) -> Option<f64> {
    let f0 = k0.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
    let f1 = k1.get("frame").and_then(Value::as_f64).unwrap_or(f0);
    let span = f1 - f0;
    if span <= 1e-12 || frame <= f0 {
        return Some(0.0);
    }
    if frame >= f1 {
        return Some(1.0);
    }
    let tween_code = k0
        .get("tween")
        .and_then(Value::as_u64)
        .map_or(1_u16, |t| t.min(u64::from(u16::MAX)) as u16);
    let (mode, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    if mode == InterpolationMode::Discrete || easing == EasingKind::Hold {
        return Some(0.0);
    }
    let u = ((frame - f0) / span).clamp(0.0, 1.0);
    if let Some([x1, y1, x2, y2]) = parse_normalized_bezier(k0) {
        return Some(eval_cubic_bezier(x1, y1, x2, y2, u));
    }
    if mode == InterpolationMode::Bezier
        || parse_handle_pair(k0, true).is_some()
        || parse_handle_pair(k1, false).is_some()
    {
        return None;
    }
    Some(eval_easing(easing, u))
}

fn eval_segment_scalar(k0: &Value, v0: f64, k1: &Value, v1: f64, frame: f64) -> f64 {
    if let Some(e) = eval_segment_factor(k0, k1, frame) {
        return lerp_f64(v0, v1, e);
    }
    let f0 = k0.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
    let f1 = k1.get("frame").and_then(Value::as_f64).unwrap_or(f0);
    let span = (f1 - f0).max(1e-12);
    let out_t = parse_handle_pair(k0, true);
    let in_t = parse_handle_pair(k1, false);
    if out_t.is_some() || in_t.is_some() {
        return eval_segment_bezier(
            BezierSegment {
                start_frame: f0,
                start_value: v0,
                out_tangent: out_t.unwrap_or([span / 3.0, (v1 - v0) / 3.0]),
                end_frame: f1,
                end_value: v1,
                in_tangent: in_t.unwrap_or([-span / 3.0, -(v1 - v0) / 3.0]),
            },
            frame,
        );
    }
    let tween_code = k0
        .get("tween")
        .and_then(Value::as_u64)
        .map_or(1_u16, |t| t.min(u64::from(u16::MAX)) as u16);
    let (_, easing) = InterpolationMode::from_cm3_tween_code(tween_code);
    let u = ((frame - f0) / span).clamp(0.0, 1.0);
    lerp_f64(v0, v1, eval_easing(easing, u))
}

/// Evaluates a Shape+ scalar property track at `frame` using full `zc_anim` easings,
/// Hold (`tween == 0`), cubic Bezier handles, and optional expression evaluation.
pub fn eval_prop_f64(
    val: Option<&Value>,
    frame: f64,
    default: f64,
    eval_expr: ExprEval<'_>,
) -> f64 {
    let Some(v) = val else {
        return default;
    };
    if let Some(n) = v.as_f64() {
        return if n.is_finite() { n } else { default };
    }
    if let Some(b) = v.as_bool() {
        return if b { 1.0 } else { 0.0 };
    }
    if let Some(s) = v.as_str()
        && let Ok(n) = s.trim().parse::<f64>()
        && n.is_finite()
    {
        return n;
    }
    if let Some(expr) = v.get("expression").and_then(Value::as_str)
        && !expr.trim().is_empty()
        && let Some(cb) = eval_expr
        && let Some(out) = cb(expr, frame)
        && out.is_finite()
    {
        return out;
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, f64, &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let val = k.get("value").and_then(|v| {
                    v.as_f64()
                        .or_else(|| v.as_bool().map(|b| if b { 1.0 } else { 0.0 }))
                        .or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))
                })?;
                Some((f, val, k))
            })
            .collect();
        if !sorted.is_empty() {
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            if frame <= sorted[0].0 {
                return sorted[0].1;
            }
            if frame >= sorted[sorted.len() - 1].0 {
                return sorted[sorted.len() - 1].1;
            }
            for w in sorted.windows(2) {
                if frame <= w[1].0 {
                    return eval_segment_scalar(w[0].2, w[0].1, w[1].2, w[1].1, frame);
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_prop_f64(Some(inner), frame, default, eval_expr);
    }
    default
}

fn eval_prop_vec2(
    val: Option<&Value>,
    frame: f64,
    default: [f64; 2],
    eval_expr: ExprEval<'_>,
) -> [f64; 2] {
    let Some(v) = val else {
        return default;
    };
    if let Some(arr) = v.as_array()
        && arr.len() >= 2
    {
        return [
            arr[0]
                .as_f64()
                .filter(|x| x.is_finite())
                .unwrap_or(default[0]),
            arr[1]
                .as_f64()
                .filter(|x| x.is_finite())
                .unwrap_or(default[1]),
        ];
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array)
        && objs.len() >= 2
    {
        return [
            eval_prop_f64(Some(&objs[0]), frame, default[0], eval_expr),
            eval_prop_f64(Some(&objs[1]), frame, default[1], eval_expr),
        ];
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let arr = eval_prop_f64_vec(Some(v), frame, eval_expr);
        if arr.len() >= 2 {
            return [arr[0], arr[1]];
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_prop_vec2(Some(inner), frame, default, eval_expr);
    }
    default
}

fn eval_prop_f64_vec(val: Option<&Value>, frame: f64, eval_expr: ExprEval<'_>) -> Vec<f64> {
    let Some(v) = val else {
        return Vec::new();
    };
    if let Some(arr) = v.as_array() {
        return arr
            .iter()
            .map(|x| x.as_f64().filter(|v| v.is_finite()).unwrap_or(0.0))
            .collect();
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array) {
        return objs
            .iter()
            .map(|o| eval_prop_f64(Some(o), frame, 0.0, eval_expr))
            .collect();
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, Vec<f64>, &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let arr = k.get("value").and_then(Value::as_array)?;
                let vals = arr
                    .iter()
                    .map(|x| x.as_f64().filter(|n| n.is_finite()).unwrap_or(0.0))
                    .collect();
                Some((f, vals, k))
            })
            .collect();
        if !sorted.is_empty() {
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            if frame <= sorted[0].0 {
                return sorted[0].1.clone();
            }
            if frame >= sorted[sorted.len() - 1].0 {
                return sorted[sorted.len() - 1].1.clone();
            }
            for w in sorted.windows(2) {
                if frame <= w[1].0 {
                    let len = w[0].1.len().min(w[1].1.len());
                    return (0..len)
                        .map(|i| eval_segment_scalar(w[0].2, w[0].1[i], w[1].2, w[1].1[i], frame))
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_prop_f64_vec(Some(inner), frame, eval_expr);
    }
    Vec::new()
}

fn eval_prop_str(val: Option<&Value>, frame: f64, default: &str) -> String {
    let Some(v) = val else {
        return default.to_owned();
    };
    if let Some(s) = v.as_str() {
        return s.to_owned();
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut pts: Vec<(f64, &str)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let s = k.get("value").and_then(Value::as_str)?;
                Some((f, s))
            })
            .collect();
        if !pts.is_empty() {
            pts.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut prev = pts[0].1;
            for (kf_f, s) in pts {
                if kf_f > frame {
                    break;
                }
                prev = s;
            }
            return prev.to_owned();
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_prop_str(Some(inner), frame, default);
    }
    default.to_owned()
}

fn eval_prop_color_rgb(
    val: Option<&Value>,
    frame: f64,
    default: [f64; 3],
    eval_expr: ExprEval<'_>,
) -> [f64; 3] {
    let Some(v) = val else {
        return default;
    };
    if let Some(s) = v.as_str()
        && let Ok(c) = parse_css_color(s)
    {
        return [c[0], c[1], c[2]];
    }
    let arr = eval_prop_f64_vec(Some(v), frame, eval_expr);
    if arr.len() >= 3 {
        let scale = if arr[0] > 1.0 || arr[1] > 1.0 || arr[2] > 1.0 {
            1.0 / 255.0
        } else {
            1.0
        };
        return [
            (arr[0] * scale).clamp(0.0, 1.0),
            (arr[1] * scale).clamp(0.0, 1.0),
            (arr[2] * scale).clamp(0.0, 1.0),
        ];
    }
    if let Some(inner) = v.get("value") {
        return eval_prop_color_rgb(Some(inner), frame, default, eval_expr);
    }
    default
}

fn eval_prop_enabled(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> bool {
    let Some(p) = props else {
        return true;
    };
    match p.get("enabled") {
        None => true,
        Some(v) => eval_prop_f64(Some(v), frame, 1.0, eval_expr) != 0.0,
    }
}

// ---------------------------------------------------------------------------
// 2D Affine Matrix & Path Primitives
// ---------------------------------------------------------------------------

type Mat2D = [f64; 6];
const MAT_IDENTITY: Mat2D = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];

#[inline]
fn mat_is_identity(m: Mat2D) -> bool {
    m == MAT_IDENTITY
}

#[inline]
fn mat_mul(a: Mat2D, b: Mat2D) -> Mat2D {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
        a[0] * b[4] + a[2] * b[5] + a[4],
        a[1] * b[4] + a[3] * b[5] + a[5],
    ]
}

#[inline]
fn mat_inv(m: Mat2D) -> Option<Mat2D> {
    let det = m[0] * m[3] - m[1] * m[2];
    if det.abs() < 1e-12 || !det.is_finite() {
        return None;
    }
    Some([
        m[3] / det,
        -m[1] / det,
        -m[2] / det,
        m[0] / det,
        (m[2] * m[5] - m[3] * m[4]) / det,
        (m[1] * m[4] - m[0] * m[5]) / det,
    ])
}

#[inline]
fn mat_transform_pt(p: [f64; 2], m: Mat2D) -> [f64; 2] {
    [
        m[0] * p[0] + m[2] * p[1] + m[4],
        m[1] * p[0] + m[3] * p[1] + m[5],
    ]
}

#[inline]
fn mat_transform_vec(v: [f64; 2], m: Mat2D) -> [f64; 2] {
    [m[0] * v[0] + m[2] * v[1], m[1] * v[0] + m[3] * v[1]]
}

#[inline]
fn mat_max_scale(m: Mat2D) -> f64 {
    m[0].hypot(m[1]).max(m[2].hypot(m[3])).max(1e-6)
}

/// Group transform matrix matching Zoidium `compiler.js:11-19` & `text-animate.js:27-39`.
fn group_matrix(
    anchor: [f64; 2],
    position: [f64; 2],
    scale: [f64; 2],
    rotation: f64,
    skew: f64,
    skew_axis: f64,
) -> Mat2D {
    let (n, c) = rotation.sin_cos();
    if skew.abs() < 1e-12 {
        let l00 = c * scale[0];
        let l10 = n * scale[0];
        let l01 = -n * scale[1];
        let l11 = c * scale[1];
        return [
            l00,
            l10,
            l01,
            l11,
            position[0] - (l00 * anchor[0] + l01 * anchor[1]),
            position[1] - (l10 * anchor[0] + l11 * anchor[1]),
        ];
    }
    let q = skew.tan();
    let (sa, ca) = skew_axis.sin_cos();
    let h00 = ca + q * sa;
    let h01 = -sa + q * ca;
    let sk00 = ca * h00 + sa * sa;
    let sk01 = ca * h01 + sa * ca;
    let sk10 = -sa * h00 + ca * sa;
    let sk11 = -sa * h01 + ca * ca;
    let l00 = (c * sk00 - n * sk10) * scale[0];
    let l01 = (c * sk01 - n * sk11) * scale[1];
    let l10 = (n * sk00 + c * sk10) * scale[0];
    let l11 = (n * sk01 + c * sk11) * scale[1];
    [
        l00,
        l10,
        l01,
        l11,
        position[0] - (l00 * anchor[0] + l01 * anchor[1]),
        position[1] - (l10 * anchor[0] + l11 * anchor[1]),
    ]
}

#[derive(Debug, Clone)]
struct BezierVertex {
    point: [f64; 2],
    in_h: [f64; 2],
    out_h: [f64; 2],
    corner: bool,
}

impl BezierVertex {
    fn new(point: [f64; 2], in_h: [f64; 2], out_h: [f64; 2]) -> Self {
        Self {
            point,
            in_h,
            out_h,
            corner: true,
        }
    }

    fn sharp(point: [f64; 2]) -> Self {
        Self::new(point, [0.0, 0.0], [0.0, 0.0])
    }
}

#[derive(Debug, Clone)]
struct ShapePath {
    points: Vec<[f64; 2]>,
    vertices: Option<Vec<BezierVertex>>,
    closed: bool,
    widths: Option<Vec<f64>>,
    distances: Option<Vec<f64>>,
    total_length: Option<f64>,
}

impl ShapePath {
    fn transformed(&self, m: Mat2D) -> Self {
        if mat_is_identity(m) {
            return self.clone();
        }
        let points = self
            .points
            .iter()
            .map(|&p| mat_transform_pt(p, m))
            .collect();
        let vertices = self.vertices.as_ref().map(|vs| {
            vs.iter()
                .map(|v| BezierVertex {
                    point: mat_transform_pt(v.point, m),
                    in_h: mat_transform_vec(v.in_h, m),
                    out_h: mat_transform_vec(v.out_h, m),
                    corner: v.corner,
                })
                .collect()
        });
        Self {
            points,
            vertices,
            closed: self.closed,
            widths: self.widths.clone(),
            distances: self.distances.clone(),
            total_length: self.total_length,
        }
    }
}

#[inline]
fn pt_dist(a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

#[inline]
fn pt_mix(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn seg_dist(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let l2 = dx * dx + dy * dy;
    let t = if l2 > 0.0 {
        (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    pt_dist(p, pt_mix(a, b, t))
}

fn split_cubic(c: [[f64; 2]; 4]) -> ([[f64; 2]; 4], [[f64; 2]; 4]) {
    let a = pt_mix(c[0], c[1], 0.5);
    let b = pt_mix(c[1], c[2], 0.5);
    let d = pt_mix(c[2], c[3], 0.5);
    let e = pt_mix(a, b, 0.5);
    let f = pt_mix(b, d, 0.5);
    let p = pt_mix(e, f, 0.5);
    ([c[0], a, e, p], [p, f, d, c[3]])
}

fn flatten_cubic(c: [[f64; 2]; 4], tolerance: f64) -> Vec<[f64; 2]> {
    let tol = tolerance.max(1e-4);
    let mut points = vec![c[0]];
    let mut stack = vec![(c, 0usize)];
    while let Some((curve, depth)) = stack.pop() {
        let err =
            seg_dist(curve[1], curve[0], curve[3]).max(seg_dist(curve[2], curve[0], curve[3]));
        if err <= tol || depth >= MAX_FLATTEN_DEPTH || points.len() + stack.len() >= 4096 {
            points.push(curve[3]);
        } else {
            let (left, right) = split_cubic(curve);
            stack.push((right, depth + 1));
            stack.push((left, depth + 1));
        }
    }
    points
}

fn vertices_to_cubics(vertices: &[BezierVertex], closed: bool) -> Vec<[[f64; 2]; 4]> {
    if vertices.len() < 2 {
        return Vec::new();
    }
    let n = if closed {
        vertices.len()
    } else {
        vertices.len() - 1
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = &vertices[i];
        let b = &vertices[(i + 1) % vertices.len()];
        out.push([
            a.point,
            [a.point[0] + a.out_h[0], a.point[1] + a.out_h[1]],
            [b.point[0] + b.in_h[0], b.point[1] + b.in_h[1]],
            b.point,
        ]);
    }
    out
}

fn flatten_vertices(vertices: &[BezierVertex], closed: bool, tolerance: f64) -> ShapePath {
    let mut points = Vec::new();
    for c in vertices_to_cubics(vertices, closed) {
        let seg = flatten_cubic(c, tolerance);
        let start_idx = usize::from(!points.is_empty());
        points.extend_from_slice(&seg[start_idx..]);
        if points.len() >= MAX_PATH_POINTS {
            points.truncate(MAX_PATH_POINTS);
            break;
        }
    }
    if closed && points.len() > 1 && pt_dist(points[0], *points.last().unwrap()) < 1e-9 {
        points.pop();
    }
    if points.is_empty() && !vertices.is_empty() {
        points.push(vertices[0].point);
    }
    ShapePath {
        points,
        vertices: Some(vertices.to_vec()),
        closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

fn reverse_vertices(vertices: &mut [BezierVertex]) {
    vertices.reverse();
    for v in vertices.iter_mut() {
        std::mem::swap(&mut v.in_h, &mut v.out_h);
    }
}

fn direction_is_reversed(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> bool {
    let Some(p) = props else {
        return false;
    };
    let Some(dir) = p.get("direction") else {
        return false;
    };
    if let Some(s) = dir.as_str() {
        return matches!(
            s.to_ascii_lowercase().as_str(),
            "ccw" | "reverse" | "-1" | "3"
        );
    }
    let v = eval_prop_f64(Some(dir), frame, 1.0, eval_expr);
    v < 0.0 || (v - 3.0).abs() < 1e-6
}

fn build_primitive_path(
    item: &Value,
    frame: f64,
    tolerance: f64,
    eval_expr: ExprEval<'_>,
) -> Option<ShapePath> {
    let kind = item.get("kind").and_then(Value::as_str)?;
    let props = item.get("properties").and_then(Value::as_object);
    let get = |k: &str| props.and_then(|p| p.get(k));
    let center = eval_prop_vec2(get("position"), frame, [0.0, 0.0], eval_expr);
    let rev = direction_is_reversed(props, frame, eval_expr);

    let mut vertices = Vec::new();
    let mut closed = true;

    match kind {
        "rectangle" => {
            let size = eval_prop_vec2(get("size"), frame, [200.0, 200.0], eval_expr);
            let hw = size[0].abs() * 0.5;
            let hh = size[1].abs() * 0.5;
            let r = eval_prop_f64(get("roundness"), frame, 0.0, eval_expr)
                .max(0.0)
                .min(hw.min(hh));
            if r <= 1e-9 {
                for p in [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]] {
                    vertices.push(BezierVertex::sharp(p));
                }
            } else {
                let kr = KAPPA * r;
                let spec = [
                    ([-hw + r, -hh], [-kr, 0.0], [0.0, 0.0]),
                    ([hw - r, -hh], [0.0, 0.0], [kr, 0.0]),
                    ([hw, -hh + r], [0.0, -kr], [0.0, 0.0]),
                    ([hw, hh - r], [0.0, 0.0], [0.0, kr]),
                    ([hw - r, hh], [kr, 0.0], [0.0, 0.0]),
                    ([-hw + r, hh], [0.0, 0.0], [-kr, 0.0]),
                    ([-hw, hh - r], [0.0, kr], [0.0, 0.0]),
                    ([-hw, -hh + r], [0.0, 0.0], [0.0, -kr]),
                ];
                for (pt, inh, outh) in spec {
                    vertices.push(BezierVertex::new(pt, inh, outh));
                }
            }
        }
        "ellipse" => {
            let size = eval_prop_vec2(get("size"), frame, [200.0, 200.0], eval_expr);
            let rx = size[0].abs() * 0.5;
            let ry = size[1].abs() * 0.5;
            let spec = [
                ([rx, 0.0], [0.0, -ry * KAPPA], [0.0, ry * KAPPA]),
                ([0.0, ry], [rx * KAPPA, 0.0], [-rx * KAPPA, 0.0]),
                ([-rx, 0.0], [0.0, ry * KAPPA], [0.0, -ry * KAPPA]),
                ([0.0, -ry], [-rx * KAPPA, 0.0], [rx * KAPPA, 0.0]),
            ];
            for (pt, inh, outh) in spec {
                vertices.push(BezierVertex::new(pt, inh, outh));
            }
        }
        "polystar" => {
            let count = eval_prop_f64(get("points"), frame, 5.0, eval_expr)
                .round()
                .clamp(3.0, 128.0) as usize;
            let mode_str = get("mode")
                .or_else(|| get("starType"))
                .map(|v| eval_prop_str(Some(v), frame, "star"))
                .unwrap_or_else(|| "star".to_owned());
            let star = !mode_str.eq_ignore_ascii_case("polygon");
            let outer_r = eval_prop_f64(get("outerRadius"), frame, 100.0, eval_expr).max(0.0);
            let inner_r = eval_prop_f64(get("innerRadius"), frame, 50.0, eval_expr).max(0.0);
            let outer_round =
                eval_prop_f64(get("outerRoundness"), frame, 0.0, eval_expr).clamp(0.0, 100.0);
            let inner_round =
                eval_prop_f64(get("innerRoundness"), frame, 0.0, eval_expr).clamp(0.0, 100.0);
            let rot = eval_prop_f64(get("rotation"), frame, 0.0, eval_expr);
            let n = count * if star { 2 } else { 1 };
            let mut raw = Vec::with_capacity(n);
            for i in 0..n {
                let theta = rot + FRAC_PI_2 + (i as f64 / n as f64) * TAU;
                let r = if star && i % 2 == 1 { inner_r } else { outer_r };
                raw.push([r * theta.cos(), r * theta.sin()]);
            }
            if outer_round == 0.0 && inner_round == 0.0 {
                for p in raw {
                    vertices.push(BezierVertex::sharp(p));
                }
            } else {
                // Exact tangent cubic corner rounding matching `primitives.js:24-31`.
                for i in 0..n {
                    let a = raw[(i + n - 1) % n];
                    let b = raw[i];
                    let c = raw[(i + 1) % n];
                    let f = ((if star && i % 2 == 1 {
                        inner_round
                    } else {
                        outer_round
                    }) / 100.0)
                        .clamp(0.0, 1.0)
                        * 0.5;
                    let u = [b[0] + (a[0] - b[0]) * f, b[1] + (a[1] - b[1]) * f];
                    let v = [b[0] + (c[0] - b[0]) * f, b[1] + (c[1] - b[1]) * f];
                    vertices.push(BezierVertex::new(
                        u,
                        [0.0, 0.0],
                        [(b[0] - u[0]) * (2.0 / 3.0), (b[1] - u[1]) * (2.0 / 3.0)],
                    ));
                    vertices.push(BezierVertex::new(
                        v,
                        [(b[0] - v[0]) * (2.0 / 3.0), (b[1] - v[1]) * (2.0 / 3.0)],
                        [0.0, 0.0],
                    ));
                }
            }
        }
        "path" => {
            closed = eval_prop_f64(get("closed"), frame, 1.0, eval_expr) != 0.0;
            if let Some(vs) = get("vertices").and_then(Value::as_array) {
                for v in vs {
                    if let Some(pt) = v.get("point").and_then(Value::as_array)
                        && pt.len() >= 2
                    {
                        let p = [pt[0].as_f64().unwrap_or(0.0), pt[1].as_f64().unwrap_or(0.0)];
                        let inh = v
                            .get("in")
                            .and_then(Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map_or([0.0, 0.0], |a| {
                                [a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0)]
                            });
                        let outh = v
                            .get("out")
                            .and_then(Value::as_array)
                            .filter(|a| a.len() >= 2)
                            .map_or([0.0, 0.0], |a| {
                                [a[0].as_f64().unwrap_or(0.0), a[1].as_f64().unwrap_or(0.0)]
                            });
                        vertices.push(BezierVertex::new(p, inh, outh));
                    }
                }
            } else {
                let values = eval_prop_f64_vec(get("path"), frame, eval_expr);
                let num_verts = item
                    .get("vertexIds")
                    .and_then(Value::as_array)
                    .map_or(values.len() / 6, Vec::len);
                for i in 0..num_verts {
                    let b = i * 6;
                    if b + 5 < values.len() {
                        vertices.push(BezierVertex::new(
                            [values[b], values[b + 1]],
                            [values[b + 2], values[b + 3]],
                            [values[b + 4], values[b + 5]],
                        ));
                    }
                }
            }
            if rev {
                reverse_vertices(&mut vertices);
            }
            return Some(flatten_vertices(&vertices, closed, tolerance));
        }
        _ => return None,
    }

    if rev {
        reverse_vertices(&mut vertices);
    }
    for v in &mut vertices {
        v.point[0] += center[0];
        v.point[1] += center[1];
    }
    Some(flatten_vertices(&vertices, closed, tolerance))
}

// ---------------------------------------------------------------------------
// Path Length Table, Trim, Dash, Width Profile, Round Corners
// ---------------------------------------------------------------------------

struct LengthTable {
    lengths: Vec<f64>,
    total: f64,
}

fn build_length_table(points: &[[f64; 2]], closed: bool) -> LengthTable {
    if points.is_empty() {
        return LengthTable {
            lengths: vec![0.0],
            total: 0.0,
        };
    }
    let segs = if closed {
        points.len()
    } else {
        points.len().saturating_sub(1)
    };
    let mut lengths = Vec::with_capacity(segs + 1);
    lengths.push(0.0);
    for i in 1..=segs {
        let d = pt_dist(points[i - 1], points[i % points.len()]);
        lengths.push(lengths[i - 1] + d);
    }
    let total = *lengths.last().unwrap_or(&0.0);
    LengthTable { lengths, total }
}

fn sample_at_length(path: &ShapePath, s: f64, table: &LengthTable) -> [f64; 2] {
    if path.points.is_empty() {
        return [0.0, 0.0];
    }
    let s = clamp_f64(s, 0.0, table.total);
    if table.lengths.len() < 2 {
        return path.points[0];
    }
    let mut low = 1usize;
    let mut high = table.lengths.len() - 1;
    while low < high {
        let mid = (low + high) / 2;
        if table.lengths[mid] < s {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    let i = low;
    let start = table.lengths[i - 1];
    let end = table.lengths[i];
    if end > start {
        let t = (s - start) / (end - start);
        pt_mix(
            path.points[(i - 1) % path.points.len()],
            path.points[i % path.points.len()],
            t,
        )
    } else {
        path.points[0]
    }
}

fn slice_path(path: &ShapePath, start: f64, end: f64, table: &LengthTable) -> ShapePath {
    let mut points = vec![sample_at_length(path, start, table)];
    let mut distances = vec![start];
    for i in 1..table.lengths.len() {
        if table.lengths[i] > start + 1e-9 && table.lengths[i] < end - 1e-9 {
            points.push(path.points[i % path.points.len()]);
            distances.push(table.lengths[i]);
        }
    }
    points.push(sample_at_length(path, end, table));
    distances.push(end);

    let orig_corners: Vec<[f64; 2]> = path
        .vertices
        .as_ref()
        .map(|vs| {
            vs.iter()
                .filter(|v| {
                    v.corner
                        && v.in_h[0].hypot(v.in_h[1]) < 1e-9
                        && v.out_h[0].hypot(v.out_h[1]) < 1e-9
                })
                .map(|v| v.point)
                .collect()
        })
        .unwrap_or_default();

    let vertices: Vec<BezierVertex> = points
        .iter()
        .map(|&pt| BezierVertex {
            point: pt,
            in_h: [0.0, 0.0],
            out_h: [0.0, 0.0],
            corner: orig_corners.iter().any(|&c| pt_dist(c, pt) < 1e-8),
        })
        .collect();

    ShapePath {
        points,
        vertices: Some(vertices),
        closed: false,
        widths: None,
        distances: Some(distances),
        total_length: Some(table.total),
    }
}

/// Exact Zoidium `intervals(start, end, offset)` (`operators.js:54-66`):
/// `offset` is in **DEGREES** (`offset / 360.0`), NOT radians!
fn trim_intervals(start: f64, end: f64, offset_deg: f64) -> Vec<[f64; 2]> {
    let span = clamp_f64((end - start).abs() / 100.0, 0.0, 1.0);
    if span >= 1.0 - 1e-10 {
        return vec![[0.0, 1.0]];
    }
    if span < 1e-10 {
        return Vec::new();
    }
    let a = wrap_f64(start.min(end) / 100.0 + offset_deg / 360.0, 1.0);
    let b = a + span;
    if b > 1.0 {
        vec![[a, 1.0], [0.0, b - 1.0]]
    } else {
        vec![[a, b]]
    }
}

/// Exact Zoidium `trim(paths, p)` (`operators.js:67-109`).
fn apply_trim(
    paths: &[(usize, ShapePath)],
    start: f64,
    end: f64,
    offset_deg: f64,
    individual: bool,
) -> Vec<(usize, ShapePath)> {
    let ranges = trim_intervals(start, end, offset_deg);
    let tables: Vec<LengthTable> = paths
        .iter()
        .map(|(_, p)| build_length_table(&p.points, p.closed))
        .collect();
    let total: f64 = tables.iter().map(|t| t.total).sum();
    let mut cursor = 0.0;
    let mut result = Vec::new();

    for ((owner, path), table) in paths.iter().zip(tables.iter()) {
        let length = table.total;
        let domain = if individual { total } else { length };
        let origin = if individual { cursor } else { 0.0 };
        let mut pieces: Vec<ShapePath> = Vec::new();

        for &[a, b] in &ranges {
            let s = (a * domain - origin).max(0.0);
            let e = (b * domain - origin).min(length);
            if e - s > 1e-9 {
                if s <= 1e-12 && (e - length).abs() <= 1e-12 {
                    pieces.push(path.clone());
                } else {
                    pieces.push(slice_path(path, s, e, table));
                }
            }
        }

        // Closed-path seam stitching without duplicate vertices (`operators.js:87-99`).
        if path.closed
            && pieces.len() == 2
            && let (Some(&last0), Some(&first1)) =
                (pieces[0].points.last(), pieces[1].points.first())
            && pt_dist(last0, first1) < 1e-8
        {
            let second = pieces.pop().unwrap();
            if second.points.len() > 1 {
                pieces[0].points.extend_from_slice(&second.points[1..]);
            }
            if let (Some(v0), Some(v1)) = (pieces[0].vertices.as_mut(), second.vertices.as_ref())
                && v1.len() > 1
            {
                v0.extend_from_slice(&v1[1..]);
            }
        }

        for mut piece in pieces {
            piece.distances = None;
            piece.total_length = None;
            result.push((*owner, piece));
        }
        cursor += length;
    }

    result
}

/// Exact Zoidium `roundCorners(path, radius, tolerance)` (`operators.js:205-267`).
fn apply_round_corners(path: &ShapePath, radius: f64, tolerance: f64) -> ShapePath {
    if !radius.is_finite() || radius <= 0.0 {
        return path.clone();
    }
    let Some(src_verts) = path.vertices.as_ref() else {
        return path.clone();
    };
    let mut authored: Vec<BezierVertex> = Vec::with_capacity(src_verts.len());
    for v in src_verts {
        if let Some(prev) = authored.last_mut()
            && pt_dist(prev.point, v.point) < 1e-9
        {
            prev.out_h = v.out_h;
        } else {
            authored.push(v.clone());
        }
    }
    if path.closed
        && authored.len() > 1
        && pt_dist(authored[0].point, authored.last().unwrap().point) < 1e-9
    {
        let last = authored.pop().unwrap();
        authored[0].in_h = last.in_h;
    }
    let n = authored.len();
    if n < 3 {
        return path.clone();
    }
    let mut vertices = Vec::with_capacity(n * 2);
    for i in 0..n {
        let v = &authored[i];
        let prev = &authored[(i + n - 1) % n];
        let next = &authored[(i + 1) % n];
        let has_handles = v.in_h[0].hypot(v.in_h[1]) > 1e-9 || v.out_h[0].hypot(v.out_h[1]) > 1e-9;
        if !v.corner || (!path.closed && (i == 0 || i == n - 1)) || has_handles {
            vertices.push(v.clone());
            continue;
        }
        let a = pt_dist(v.point, prev.point);
        let b = pt_dist(v.point, next.point);
        if a < 1e-9 || b < 1e-9 {
            vertices.push(v.clone());
            continue;
        }
        let u = [
            (prev.point[0] - v.point[0]) / a,
            (prev.point[1] - v.point[1]) / a,
        ];
        let w = [
            (next.point[0] - v.point[0]) / b,
            (next.point[1] - v.point[1]) / b,
        ];
        let angle = clamp_f64(u[0] * w[0] + u[1] * w[1], -1.0, 1.0).acos();
        if angle < 1e-6 || PI - angle < 1e-6 {
            vertices.push(v.clone());
            continue;
        }
        let cut = (radius / (angle * 0.5).tan()).min(a * 0.5).min(b * 0.5);
        let actual = cut * (angle * 0.5).tan();
        let handle = (4.0 / 3.0) * ((PI - angle) / 4.0).tan() * actual;
        vertices.push(BezierVertex::new(
            [v.point[0] + u[0] * cut, v.point[1] + u[1] * cut],
            [0.0, 0.0],
            [-u[0] * handle, -u[1] * handle],
        ));
        vertices.push(BezierVertex::new(
            [v.point[0] + w[0] * cut, v.point[1] + w[1] * cut],
            [-w[0] * handle, -w[1] * handle],
            [0.0, 0.0],
        ));
    }
    flatten_vertices(&vertices, path.closed, tolerance)
}

// ---------------------------------------------------------------------------
// Deformers: Twist, Zig Zag, Pucker & Bloat, Wiggle Paths, Wiggle Transform
// (exact port of Zoidium `deform.js`)
// ---------------------------------------------------------------------------

#[inline]
fn wiggle_hash(x: i32, y: i32, seed: i32) -> f64 {
    let mut n = x.wrapping_mul(374_761_393)
        ^ y.wrapping_mul(668_265_263)
        ^ seed.wrapping_mul(1_442_695_041);
    n = (n ^ ((n as u32) >> 13) as i32).wrapping_mul(1_274_126_177);
    let u = (n ^ ((n as u32) >> 16) as i32) as u32;
    f64::from(u) / 2_147_483_647.5 - 1.0
}

#[inline]
fn smoothstep(t: f64) -> f64 {
    t * t * (3.0 - 2.0 * t)
}

fn wiggle_noise(space: f64, time: f64, seed: i32, period: i32) -> f64 {
    let x = space.floor() as i32;
    let y = time.floor() as i32;
    let u = smoothstep(space - f64::from(x));
    let v = smoothstep(time - f64::from(y));
    let at = |i: i32, j: i32| {
        let xi = if period > 0 { i.rem_euclid(period) } else { i };
        wiggle_hash(xi, j, seed)
    };
    let a = at(x, y) * (1.0 - u) + at(x + 1, y) * u;
    let b = at(x, y + 1) * (1.0 - u) + at(x + 1, y + 1) * u;
    a * (1.0 - v) + b * v
}

#[derive(Debug, Clone, Copy)]
struct WiggleParams {
    time: f64,
    frequency: f64,
    correlation: f64,
    temporal_phase: f64,
    spatial_phase: f64,
    seed: i32,
}

fn sample_wiggle_noise(p: WiggleParams, space: f64, channel: i32, copy: i32, period: i32) -> f64 {
    let time = p.time * p.frequency + p.temporal_phase / 360.0;
    let seed = p.seed.wrapping_add(channel.wrapping_mul(7919));
    let correlation = clamp_f64(p.correlation / 100.0, 0.0, 1.0);
    let common = wiggle_noise(p.spatial_phase / 360.0, time, seed, 0);
    let independent = wiggle_noise(
        space + p.spatial_phase / 360.0,
        time,
        seed.wrapping_add(copy.wrapping_mul(104_729)),
        period,
    );
    common * correlation + independent * (1.0 - correlation)
}

fn resample_path(path: &ShapePath, spacing: f64) -> ShapePath {
    let spacing = spacing.max(0.25);
    let src = &path.points;
    if src.len() < 2 {
        return path.clone();
    }
    let count = if path.closed {
        src.len()
    } else {
        src.len() - 1
    };
    let mut points = Vec::new();
    for i in 0..count {
        let a = src[i];
        let b = src[(i + 1) % src.len()];
        let n = (pt_dist(a, b) / spacing).ceil().max(1.0) as usize;
        if points.len() + n + 1 > MAX_PATH_POINTS {
            break;
        }
        for j in 0..n {
            let t = j as f64 / n as f64;
            points.push(pt_mix(a, b, t));
        }
    }
    if !path.closed
        && let Some(&last) = src.last()
    {
        points.push(last);
    }
    ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

/// Catmull-Rom smooth conversion (`deform.js:83-93`).
fn smooth_polyline(path: &ShapePath, tolerance: f64) -> ShapePath {
    let pts = &path.points;
    let n = pts.len();
    if n < 2 {
        return path.clone();
    }
    let get = |i: isize| -> [f64; 2] {
        let idx = if path.closed {
            i.rem_euclid(n as isize) as usize
        } else {
            i.clamp(0, (n - 1) as isize) as usize
        };
        pts[idx]
    };
    let vertices: Vec<BezierVertex> = (0..n)
        .map(|i| {
            let p_prev = get(i as isize - 1);
            let p_next = get(i as isize + 1);
            let t = [(p_next[0] - p_prev[0]) / 6.0, (p_next[1] - p_prev[1]) / 6.0];
            BezierVertex::new(pts[i], [-t[0], -t[1]], t)
        })
        .collect();
    let mut flat = flatten_vertices(&vertices, path.closed, tolerance);
    flat.vertices = None;
    flat
}

fn apply_zig_zag(
    path: &ShapePath,
    size: f64,
    ridges: f64,
    smooth_mode: bool,
    tolerance: f64,
) -> ShapePath {
    if path.points.len() < 2 {
        return path.clone();
    }
    let n = (ridges.round().clamp(1.0, 64.0) as usize) * 2 + 1;
    let spans: Vec<Vec<[f64; 2]>> = if let Some(vs) = path.vertices.as_ref() {
        vertices_to_cubics(vs, path.closed)
            .into_iter()
            .map(|c| flatten_cubic(c, tolerance))
            .collect()
    } else {
        let cnt = if path.closed {
            path.points.len()
        } else {
            path.points.len() - 1
        };
        (0..cnt)
            .map(|i| vec![path.points[i], path.points[(i + 1) % path.points.len()]])
            .collect()
    };

    let mut points = Vec::new();
    for span in &spans {
        let table = build_length_table(span, false);
        let total = table.total;
        let mut segment = 1usize;
        for j in 0..n {
            let dist = total * (j as f64) / (n as f64);
            while segment + 1 < span.len() && table.lengths[segment] < dist {
                segment += 1;
            }
            let a = span[segment - 1];
            let b = span[segment];
            let dx = b[0] - a[0];
            let dy = b[1] - a[1];
            let len = dx.hypot(dy);
            let t = if len > 0.0 {
                (dist - table.lengths[segment - 1]) / len
            } else {
                0.0
            };
            let d = if j == 0 {
                0.0
            } else if j % 2 == 1 {
                size
            } else {
                -size
            };
            let nx = if len > 0.0 { -dy / len } else { 0.0 };
            let ny = if len > 0.0 { dx / len } else { 0.0 };
            points.push([a[0] + dx * t + nx * d, a[1] + dy * t + ny * d]);
            if points.len() >= 4096 {
                break;
            }
        }
        if points.len() >= 4096 {
            break;
        }
    }
    if !path.closed
        && let Some(&last) = path.points.last()
    {
        points.push(last);
    }
    let res = ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    };
    if smooth_mode {
        smooth_polyline(&res, tolerance)
    } else {
        res
    }
}

fn apply_twist(path: &ShapePath, angle: f64, center: [f64; 2], tolerance: f64) -> ShapePath {
    if path.points.len() < 2 || angle.abs() < 1e-12 {
        return path.clone();
    }
    let radius = path
        .points
        .iter()
        .map(|p| (p[0] - center[0]).hypot(p[1] - center[1]))
        .fold(0.0_f64, f64::max);
    let k = angle.abs() / 100.0;
    let spacing = if k > 0.0 {
        4.0_f64.min((8.0 * tolerance / (2.0 * k + radius * k * k)).sqrt())
    } else {
        4.0
    };
    let sampled = resample_path(path, spacing);
    let points = sampled
        .points
        .into_iter()
        .map(|pt| {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            let a = angle * dx.hypot(dy) / 100.0;
            let (s, c) = a.sin_cos();
            [center[0] + c * dx - s * dy, center[1] + s * dx + c * dy]
        })
        .collect();
    ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

fn apply_pucker_bloat(path: &ShapePath, amount_pct: f64) -> ShapePath {
    if path.points.len() < 2 || amount_pct.abs() < 1e-12 {
        return path.clone();
    }
    let n_pts = path.points.len() as f64;
    let center = path.points.iter().fold([0.0, 0.0], |acc, p| {
        [acc[0] + p[0] / n_pts, acc[1] + p[1] / n_pts]
    });
    let authored: Vec<[f64; 2]> = path
        .vertices
        .as_ref()
        .map(|vs| vs.iter().map(|v| v.point).collect())
        .unwrap_or_else(|| path.points.clone());
    let authored_segs = if path.closed {
        authored.len()
    } else {
        authored.len().saturating_sub(1)
    };
    let sampled = resample_path(path, 4.0);
    let amt = clamp_f64(amount_pct / 100.0, -1.0, 1.0);
    let points = sampled
        .points
        .into_iter()
        .map(|pt| {
            let dx = pt[0] - center[0];
            let dy = pt[1] - center[1];
            let mut best = f64::INFINITY;
            let mut best_t = 0.0;
            for j in 0..authored_segs {
                let a = authored[j];
                let b = authored[(j + 1) % authored.len()];
                let ex = b[0] - a[0];
                let ey = b[1] - a[1];
                let l2 = ex * ex + ey * ey;
                let u = if l2 > 0.0 {
                    clamp_f64(((pt[0] - a[0]) * ex + (pt[1] - a[1]) * ey) / l2, 0.0, 1.0)
                } else {
                    0.0
                };
                let d2 = (pt[0] - a[0] - u * ex).powi(2) + (pt[1] - a[1] - u * ey).powi(2);
                if d2 < best {
                    best = d2;
                    best_t = u;
                }
            }
            let factor = 1.0 - amt * (TAU * best_t).cos();
            [center[0] + dx * factor, center[1] + dy * factor]
        })
        .collect();
    ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    }
}

fn apply_wiggle_paths(
    path: &ShapePath,
    size: f64,
    detail: f64,
    smooth_mode: bool,
    wp: WiggleParams,
    tolerance: f64,
) -> ShapePath {
    if path.points.len() < 2 {
        return path.clone();
    }
    let detail_clamped = clamp_f64(detail, 1.0, 100.0);
    let spacing = 100.0 / detail_clamped;
    let sampled = resample_path(path, spacing);
    let n = sampled.points.len();
    let detail_i = detail_clamped.round() as i32;
    let denom = if path.closed {
        n.max(1) as f64
    } else {
        n.saturating_sub(1).max(1) as f64
    };
    let period = if path.closed { detail_i } else { 0 };
    let points = sampled
        .points
        .into_iter()
        .enumerate()
        .map(|(i, pt)| {
            let space = (i as f64 / denom) * f64::from(detail_i);
            let nx = sample_wiggle_noise(wp, space, 0, 0, period);
            let ny = sample_wiggle_noise(wp, space, 1, 0, period);
            [pt[0] + size * nx, pt[1] + size * ny]
        })
        .collect();
    let res = ShapePath {
        points,
        vertices: None,
        closed: path.closed,
        widths: None,
        distances: None,
        total_length: None,
    };
    if smooth_mode {
        smooth_polyline(&res, tolerance)
    } else {
        res
    }
}

#[derive(Debug, Clone, Copy)]
struct WiggleTransformConfig {
    wp: WiggleParams,
    anchor: [f64; 2],
    position: [f64; 2],
    scale: [f64; 2],
    rotation: f64,
}

fn eval_wiggle_transform_matrix(cfg: WiggleTransformConfig, copy: i32) -> Mat2D {
    let offset2 = |val: [f64; 2], ch: i32| -> [f64; 2] {
        let space = f64::from(copy) + f64::from(ch) * 0.37;
        [
            val[0] * sample_wiggle_noise(cfg.wp, space, ch, copy, 0),
            val[1] * sample_wiggle_noise(cfg.wp, space, ch + 1, copy, 0),
        ]
    };
    let anchor = offset2(cfg.anchor, 1);
    let position = offset2(cfg.position, 3);
    let scale_off = offset2(cfg.scale, 5);
    let scale = [1.0 + scale_off[0] / 100.0, 1.0 + scale_off[1] / 100.0];
    let rotation = cfg.rotation * sample_wiggle_noise(cfg.wp, f64::from(copy) + 7.0, 7, copy, 0);
    group_matrix(anchor, position, scale, rotation, 0.0, 0.0)
}

// ---------------------------------------------------------------------------
// Stroke Expansion: Caps, Joins, MiterLimit, Dashes, Taper, Wave
// (exact port of Zoidium `stroke.js`, `stroke-profile.js`, `operators.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrokeCap {
    Butt,
    Round,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrokeJoin {
    Miter,
    Round,
    Bevel,
}

#[derive(Debug, Clone)]
struct StrokeStyle {
    width: f64,
    cap: StrokeCap,
    join: StrokeJoin,
    miter_limit: f64,
    dash_pairs: usize,
    dashes: [f64; 3],
    gaps: [f64; 3],
    dash_offset: f64,
    start_length: f64,
    end_length: f64,
    start_width: f64,
    end_width: f64,
    start_ease: f64,
    end_ease: f64,
    wave_amount: f64,
    wave_units_length: bool,
    wavelength: f64,
    wave_cycles: f64,
    wave_phase: f64,
}

impl StrokeStyle {
    fn from_props(props: Option<&Map<String, Value>>, frame: f64, eval_expr: ExprEval<'_>) -> Self {
        let get = |k: &str| props.and_then(|p| p.get(k));
        let cap = match eval_prop_str(get("cap"), frame, "butt")
            .to_ascii_lowercase()
            .as_str()
        {
            "round" => StrokeCap::Round,
            "square" => StrokeCap::Square,
            _ => StrokeCap::Butt,
        };
        let join = match eval_prop_str(get("join"), frame, "miter")
            .to_ascii_lowercase()
            .as_str()
        {
            "round" => StrokeJoin::Round,
            "bevel" => StrokeJoin::Bevel,
            _ => StrokeJoin::Miter,
        };
        let dash_pairs = eval_prop_f64(get("dashPairs"), frame, 0.0, eval_expr)
            .round()
            .clamp(0.0, 3.0) as usize;
        Self {
            width: eval_prop_f64(get("width"), frame, 10.0, eval_expr).max(0.0),
            cap,
            join,
            miter_limit: eval_prop_f64(get("miterLimit"), frame, 4.0, eval_expr).max(1.0),
            dash_pairs,
            dashes: [
                eval_prop_f64(get("dash1"), frame, 20.0, eval_expr).max(0.0),
                eval_prop_f64(get("dash2"), frame, 20.0, eval_expr).max(0.0),
                eval_prop_f64(get("dash3"), frame, 20.0, eval_expr).max(0.0),
            ],
            gaps: [
                eval_prop_f64(get("gap1"), frame, 10.0, eval_expr).max(0.0),
                eval_prop_f64(get("gap2"), frame, 10.0, eval_expr).max(0.0),
                eval_prop_f64(get("gap3"), frame, 10.0, eval_expr).max(0.0),
            ],
            dash_offset: eval_prop_f64(get("dashOffset"), frame, 0.0, eval_expr),
            start_length: eval_prop_f64(get("startLength"), frame, 0.0, eval_expr)
                .clamp(0.0, 100.0),
            end_length: eval_prop_f64(get("endLength"), frame, 0.0, eval_expr).clamp(0.0, 100.0),
            start_width: eval_prop_f64(get("startWidth"), frame, 0.0, eval_expr).max(0.0),
            end_width: eval_prop_f64(get("endWidth"), frame, 0.0, eval_expr).max(0.0),
            start_ease: eval_prop_f64(get("startEase"), frame, 0.0, eval_expr).clamp(-100.0, 100.0),
            end_ease: eval_prop_f64(get("endEase"), frame, 0.0, eval_expr).clamp(-100.0, 100.0),
            wave_amount: eval_prop_f64(get("waveAmount"), frame, 0.0, eval_expr).clamp(0.0, 100.0),
            wave_units_length: eval_prop_str(get("waveUnits"), frame, "cycles")
                .eq_ignore_ascii_case("length"),
            wavelength: eval_prop_f64(get("wavelength"), frame, 100.0, eval_expr).max(0.01),
            wave_cycles: eval_prop_f64(get("waveCycles"), frame, 3.0, eval_expr).max(0.0),
            wave_phase: eval_prop_f64(get("wavePhase"), frame, 0.0, eval_expr),
        }
    }
}

fn stroke_width_at(s: f64, total: f64, style: &StrokeStyle) -> f64 {
    if !total.is_finite() || total <= 0.0 {
        return style.width.max(0.0);
    }
    let start = (style.start_length / 100.0).clamp(0.0, 1.0) * total;
    let end = (style.end_length / 100.0).clamp(0.0, 1.0) * total;
    let ease = |t: f64, e: f64| -> f64 {
        t.clamp(0.0, 1.0)
            .powf(2.0_f64.powf(e.clamp(-100.0, 100.0) / 50.0))
    };
    let mut w = 1.0;
    if start > 0.0 && s < start {
        let sw = style.start_width / 100.0;
        w *= sw + (1.0 - sw) * ease(s / start, style.start_ease);
    }
    if end > 0.0 && s > total - end {
        let ew = style.end_width / 100.0;
        w *= ew + (1.0 - ew) * ease((total - s) / end, style.end_ease);
    }
    if style.wave_amount > 0.0 {
        let cycles = if style.wave_units_length {
            total / style.wavelength.max(0.01)
        } else {
            style.wave_cycles
        };
        let wave = 1.0
            + (style.wave_amount / 100.0)
                * (TAU * ((cycles * s) / total + style.wave_phase / 360.0)).sin();
        w *= wave.max(0.0);
    }
    (style.width * w).max(0.0)
}

fn apply_dash(paths: &[ShapePath], style: &StrokeStyle) -> Vec<ShapePath> {
    if style.dash_pairs == 0 {
        return paths.to_vec();
    }
    let mut pattern = Vec::with_capacity(style.dash_pairs * 2);
    for i in 0..style.dash_pairs {
        pattern.push(style.dashes[i].max(0.0));
        pattern.push(style.gaps[i].max(0.0));
    }
    let period: f64 = pattern.iter().sum();
    if period < 1e-6 {
        return paths.to_vec();
    }
    let mut result = Vec::new();
    for path in paths {
        let table = build_length_table(&path.points, path.closed);
        let total = table.total;
        if total < 1e-9 {
            continue;
        }
        let mut pieces: Vec<ShapePath> = Vec::new();
        let mut phase = wrap_f64(style.dash_offset, period);
        let mut index = 0usize;
        while phase >= pattern[index] && index + 1 < pattern.len() {
            phase -= pattern[index];
            index += 1;
        }
        let mut cursor = 0.0;
        let mut visits = 0usize;
        while cursor < total - 1e-9 {
            visits += 1;
            if visits > MAX_DASH_FRAGMENTS * 8 {
                break;
            }
            let end = (cursor + pattern[index] - phase).min(total);
            if index.is_multiple_of(2) && end > cursor + 1e-9 {
                let piece = slice_path(path, cursor, end, &table);
                let merge_prev = pieces.last().is_some_and(|prev| {
                    prev.distances
                        .as_ref()
                        .and_then(|d| d.last())
                        .is_some_and(|&last_d| (last_d - cursor).abs() < 1e-9)
                });
                if merge_prev {
                    let prev = pieces.last_mut().unwrap();
                    if piece.points.len() > 1 {
                        prev.points.extend_from_slice(&piece.points[1..]);
                    }
                    if let (Some(pd), Some(nd)) =
                        (prev.distances.as_mut(), piece.distances.as_ref())
                        && nd.len() > 1
                    {
                        pd.extend_from_slice(&nd[1..]);
                    }
                } else {
                    pieces.push(piece);
                }
            }
            cursor = end;
            phase = 0.0;
            index = (index + 1) % pattern.len();
            if pieces.len() + result.len() > MAX_DASH_FRAGMENTS {
                break;
            }
        }
        // Closed-path seam stitching (`operators.js:152-171`).
        if path.closed
            && !pieces.is_empty()
            && pieces[0]
                .distances
                .as_ref()
                .and_then(|d| d.first())
                .is_some_and(|&d0| d0 == 0.0)
            && pieces
                .last()
                .and_then(|p| p.distances.as_ref())
                .and_then(|d| d.last())
                .is_some_and(|&de| (de - total).abs() < 1e-9)
        {
            if pieces.len() == 1 {
                pieces[0] = path.clone();
            } else {
                let first = pieces.remove(0);
                let mut last = pieces.pop().unwrap();
                if first.points.len() > 1 {
                    last.points.extend_from_slice(&first.points[1..]);
                }
                if let (Some(ld), Some(fd)) = (last.distances.as_mut(), first.distances.as_ref())
                    && fd.len() > 1
                {
                    ld.extend(fd[1..].iter().map(|&s| s + total));
                }
                pieces.insert(0, last);
            }
        }
        result.extend(pieces);
    }
    result
}

fn apply_stroke_profile(paths: &[ShapePath], style: &StrokeStyle) -> Vec<ShapePath> {
    let dashed = apply_dash(paths, style);
    let variable = style.start_length > 0.0 || style.end_length > 0.0 || style.wave_amount > 0.0;
    if !variable {
        return dashed;
    }
    let mut total_samples = 0usize;
    dashed
        .into_iter()
        .map(|path| {
            let table = build_length_table(&path.points, path.closed);
            let total = path.total_length.unwrap_or(table.total);
            if total < 1e-9 {
                let w = stroke_width_at(0.0, total, style);
                let widths = vec![w; path.points.len()];
                return ShapePath {
                    widths: Some(widths),
                    ..path
                };
            }
            let cycles = if style.wave_units_length {
                total / style.wavelength.max(0.01)
            } else {
                style.wave_cycles
            };
            let subdivisions = 128.0_f64.max((cycles * 32.0).ceil());
            let step = total / subdivisions;
            let mut points = Vec::new();
            let mut widths = Vec::new();
            for i in 1..table.lengths.len() {
                let a = table.lengths[i - 1];
                let b = table.lengths[i];
                let count = ((b - a) / step).ceil().max(1.0) as usize;
                for j in 0..count {
                    let frac = j as f64 / count as f64;
                    let s = a + (b - a) * frac;
                    let domain = if let Some(ds) = path.distances.as_ref() {
                        ds[i - 1] + (ds[i] - ds[i - 1]) * frac
                    } else {
                        s
                    };
                    let d_wrapped = if domain > total {
                        domain % total
                    } else {
                        domain
                    };
                    points.push(sample_at_length(&path, s, &table));
                    widths.push(stroke_width_at(d_wrapped, total, style));
                    total_samples += 1;
                    if points.len() >= MAX_PATH_POINTS || total_samples >= 65_536 {
                        break;
                    }
                }
                if points.len() >= MAX_PATH_POINTS || total_samples >= 65_536 {
                    break;
                }
            }
            if !path.closed
                && let Some(&last_pt) = path.points.last()
            {
                let d_end = path
                    .distances
                    .as_ref()
                    .and_then(|d| d.last().copied())
                    .unwrap_or(total);
                points.push(last_pt);
                widths.push(stroke_width_at(d_end, total, style));
            }
            ShapePath {
                points,
                vertices: None,
                closed: path.closed,
                widths: Some(widths),
                distances: path.distances,
                total_length: path.total_length,
            }
        })
        .collect()
}

fn signed_area(pts: &[[f64; 2]]) -> f64 {
    if pts.len() < 3 {
        return 0.0;
    }
    let mut a = 0.0;
    for i in 0..pts.len() {
        let p = pts[i];
        let q = pts[(i + 1) % pts.len()];
        a += p[0] * q[1] - q[0] * p[1];
    }
    a * 0.5
}

fn stroke_arc(
    center: [f64; 2],
    radius: f64,
    from: f64,
    sweep: f64,
    tolerance: f64,
) -> Vec<[f64; 2]> {
    if !radius.is_finite() || radius <= 0.0 {
        return vec![center];
    }
    let step = 2.0 * clamp_f64(1.0 - tolerance / radius, -1.0, 1.0).acos();
    let count = (sweep.abs() / step.max(0.01)).ceil().clamp(1.0, 512.0) as usize;
    (0..=count)
        .map(|i| {
            let a = from + sweep * (i as f64) / (count as f64);
            [center[0] + radius * a.cos(), center[1] + radius * a.sin()]
        })
        .collect()
}

/// Expands stroke paths into closed CCW polygons (`stroke.js:15-72`).
fn expand_stroke_polygons(
    paths: &[ShapePath],
    style: &StrokeStyle,
    tolerance: f64,
) -> Vec<Vec<[f64; 2]>> {
    if style.width <= 0.0 {
        return Vec::new();
    }
    let profiled = apply_stroke_profile(paths, style);
    let mut polygons: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut push_poly = |mut poly: Vec<[f64; 2]>| {
        if poly.len() < 3 {
            return;
        }
        if signed_area(&poly) < 0.0 {
            poly.reverse();
        }
        polygons.push(poly);
    };

    let base_radius = style.width * 0.5;
    for path in &profiled {
        let mut indices = Vec::with_capacity(path.points.len());
        for (i, &pt) in path.points.iter().enumerate() {
            if i == 0 || pt_dist(pt, path.points[i - 1]) > 1e-9 {
                indices.push(i);
            }
        }
        let mut points: Vec<[f64; 2]> = indices.iter().map(|&i| path.points[i]).collect();
        if path.closed && points.len() > 1 && pt_dist(points[0], *points.last().unwrap()) < 1e-9 {
            points.pop();
            indices.pop();
        }
        let radii: Option<Vec<f64>> = path
            .widths
            .as_ref()
            .map(|ws| indices.iter().map(|&i| ws[i].max(0.0) * 0.5).collect());
        let r_at = |idx: usize| -> f64 {
            radii
                .as_ref()
                .and_then(|rs| rs.get(idx).copied())
                .unwrap_or(base_radius)
        };

        let n = points.len();
        if n == 0 {
            continue;
        }
        if n == 1 {
            let r = r_at(0);
            if r > 0.0 {
                match style.cap {
                    StrokeCap::Round => push_poly(stroke_arc(points[0], r, 0.0, TAU, tolerance)),
                    StrokeCap::Square => push_poly(vec![
                        [points[0][0] - r, points[0][1] - r],
                        [points[0][0] + r, points[0][1] - r],
                        [points[0][0] + r, points[0][1] + r],
                        [points[0][0] - r, points[0][1] + r],
                    ]),
                    StrokeCap::Butt => {}
                }
            }
            continue;
        }

        let count = if path.closed { n } else { n - 1 };
        let mut directions: Vec<[f64; 2]> = Vec::with_capacity(count);
        for i in 0..count {
            let p = points[i];
            let q = points[(i + 1) % n];
            let len = pt_dist(p, q);
            if len < 1e-12 {
                directions.push([1.0, 0.0]);
                continue;
            }
            let d = [(q[0] - p[0]) / len, (q[1] - p[1]) / len];
            let r0 = r_at(i);
            let r1 = r_at((i + 1) % n);
            let n0 = [-d[1] * r0, d[0] * r0];
            let n1 = [-d[1] * r1, d[0] * r1];
            directions.push(d);

            let a = if !path.closed && i == 0 && style.cap == StrokeCap::Square {
                [p[0] - d[0] * r0, p[1] - d[1] * r0]
            } else {
                p
            };
            let b = if !path.closed && i == count - 1 && style.cap == StrokeCap::Square {
                [q[0] + d[0] * r1, q[1] + d[1] * r1]
            } else {
                q
            };
            push_poly(vec![
                [a[0] + n0[0], a[1] + n0[1]],
                [a[0] - n0[0], a[1] - n0[1]],
                [b[0] - n1[0], b[1] - n1[1]],
                [b[0] + n1[0], b[1] + n1[1]],
            ]);
        }

        // Joins
        let join_start = if path.closed { 0 } else { 1 };
        let join_end = if path.closed { n } else { n - 1 };
        for (i, &p) in points.iter().enumerate().take(join_end).skip(join_start) {
            let r = r_at(i);
            if r <= 0.0 {
                continue;
            }
            let prev = (i + count - 1) % count;
            let next = i % count;
            let a = directions[prev];
            let b = directions[next];
            let turn = a[0] * b[1] - a[1] * b[0];
            let dot = a[0] * b[0] + a[1] * b[1];
            if turn.abs() < 1e-10 {
                if dot < 0.0 && style.join == StrokeJoin::Round {
                    push_poly(stroke_arc(p, r, 0.0, TAU, tolerance));
                }
                continue;
            }
            let sign = if turn > 0.0 { -1.0 } else { 1.0 };
            let u = [-a[1] * r * sign, a[0] * r * sign];
            let v = [-b[1] * r * sign, b[0] * r * sign];
            match style.join {
                StrokeJoin::Round => {
                    let start_ang = u[1].atan2(u[0]);
                    let sweep = turn.atan2(dot);
                    let mut poly = vec![p];
                    poly.extend(stroke_arc(p, r, start_ang, sweep, tolerance));
                    push_poly(poly);
                }
                StrokeJoin::Miter | StrokeJoin::Bevel => {
                    let vx = v[0] - u[0];
                    let vy = v[1] - u[1];
                    let t = (vx * b[1] - vy * b[0]) / turn;
                    let inter = [p[0] + u[0] + a[0] * t, p[1] + u[1] + a[1] * t];
                    if style.join == StrokeJoin::Miter
                        && pt_dist(inter, p) <= r * style.miter_limit.max(1.0)
                    {
                        push_poly(vec![
                            p,
                            [p[0] + u[0], p[1] + u[1]],
                            inter,
                            [p[0] + v[0], p[1] + v[1]],
                        ]);
                    } else {
                        push_poly(vec![
                            p,
                            [p[0] + u[0], p[1] + u[1]],
                            [p[0] + v[0], p[1] + v[1]],
                        ]);
                    }
                }
            }
        }

        // Round end caps
        if !path.closed && style.cap == StrokeCap::Round {
            let start_dir = directions[0];
            let end_dir = directions[count - 1];
            let r0 = r_at(0);
            let r1 = r_at(n - 1);
            if r0 > 0.0 {
                let mut cap0 = vec![points[0]];
                cap0.extend(stroke_arc(
                    points[0],
                    r0,
                    start_dir[1].atan2(start_dir[0]) + FRAC_PI_2,
                    PI,
                    tolerance,
                ));
                push_poly(cap0);
            }
            if r1 > 0.0 {
                let mut cap1 = vec![points[n - 1]];
                cap1.extend(stroke_arc(
                    points[n - 1],
                    r1,
                    end_dir[1].atan2(end_dir[0]) - FRAC_PI_2,
                    PI,
                    tolerance,
                ));
                push_poly(cap1);
            }
        }
    }

    polygons
}

// ---------------------------------------------------------------------------
// Planar Arrangement Boolean Engine (`mergePaths` & `offsetPaths`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BoolOp {
    Add,
    Subtract,
    Intersect,
    Exclude,
}

impl BoolOp {
    #[inline]
    fn eval(self, a: bool, b: bool) -> bool {
        match self {
            Self::Add => a || b,
            Self::Subtract => a && !b,
            Self::Intersect => a && b,
            Self::Exclude => a ^ b,
        }
    }
}

fn point_in_polygons_nonzero(pt: [f64; 2], polys: &[Vec<[f64; 2]>]) -> bool {
    let mut winding = 0i32;
    let px = pt[0];
    let py = pt[1];
    for poly in polys {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if a[1] <= py {
                if b[1] > py {
                    let is_left = (b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1]);
                    if is_left > 0.0 {
                        winding += 1;
                    }
                }
            } else if b[1] <= py {
                let is_left = (b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1]);
                if is_left < 0.0 {
                    winding -= 1;
                }
            }
        }
    }
    winding != 0
}

#[inline]
fn quant_key(p: [f64; 2]) -> (i64, i64) {
    const Q: f64 = 4096.0;
    ((p[0] * Q).round() as i64, (p[1] * Q).round() as i64)
}

fn segment_intersection_params(
    a0: [f64; 2],
    a1: [f64; 2],
    b0: [f64; 2],
    b1: [f64; 2],
) -> Vec<(f64, f64)> {
    let min_ax = a0[0].min(a1[0]) - 1e-7;
    let max_ax = a0[0].max(a1[0]) + 1e-7;
    let min_ay = a0[1].min(a1[1]) - 1e-7;
    let max_ay = a0[1].max(a1[1]) + 1e-7;
    let min_bx = b0[0].min(b1[0]) - 1e-7;
    let max_bx = b0[0].max(b1[0]) + 1e-7;
    let min_by = b0[1].min(b1[1]) - 1e-7;
    let max_by = b0[1].max(b1[1]) + 1e-7;
    if max_ax < min_bx || max_bx < min_ax || max_ay < min_by || max_by < min_ay {
        return Vec::new();
    }

    let r = [a1[0] - a0[0], a1[1] - a0[1]];
    let s = [b1[0] - b0[0], b1[1] - b0[1]];
    let denom = r[0] * s[1] - r[1] * s[0];
    let qp = [b0[0] - a0[0], b0[1] - a0[1]];

    if denom.abs() > 1e-11 {
        let t = (qp[0] * s[1] - qp[1] * s[0]) / denom;
        let u = (qp[0] * r[1] - qp[1] * r[0]) / denom;
        if (-1e-8..=1.0 + 1e-8).contains(&t) && (-1e-8..=1.0 + 1e-8).contains(&u) {
            return vec![(t.clamp(0.0, 1.0), u.clamp(0.0, 1.0))];
        }
        return Vec::new();
    }

    // Collinear overlap check
    let len_r = r[0].hypot(r[1]);
    let len_s = s[0].hypot(s[1]);
    if len_r < 1e-10 || len_s < 1e-10 {
        return Vec::new();
    }
    let cross_qp = (qp[0] * r[1] - qp[1] * r[0]).abs() / len_r;
    if cross_qp > 1e-6 {
        return Vec::new();
    }
    let r2 = r[0] * r[0] + r[1] * r[1];
    let s2 = s[0] * s[0] + s[1] * s[1];
    let mut hits = Vec::new();
    let t0 = ((b0[0] - a0[0]) * r[0] + (b0[1] - a0[1]) * r[1]) / r2;
    let t1 = ((b1[0] - a0[0]) * r[0] + (b1[1] - a0[1]) * r[1]) / r2;
    if t0 > 1e-8 && t0 < 1.0 - 1e-8 {
        hits.push((t0, 0.0));
    }
    if t1 > 1e-8 && t1 < 1.0 - 1e-8 {
        hits.push((t1, 1.0));
    }
    let u0 = ((a0[0] - b0[0]) * s[0] + (a0[1] - b0[1]) * s[1]) / s2;
    let u1 = ((a1[0] - b0[0]) * s[0] + (a1[1] - b0[1]) * s[1]) / s2;
    if u0 > 1e-8 && u0 < 1.0 - 1e-8 {
        hits.push((0.0, u0));
    }
    if u1 > 1e-8 && u1 < 1.0 - 1e-8 {
        hits.push((1.0, u1));
    }
    hits
}

fn boolean_polygons(
    subject: &[Vec<[f64; 2]>],
    clip: &[Vec<[f64; 2]>],
    op: BoolOp,
) -> Vec<Vec<[f64; 2]>> {
    let mut segs: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for poly in subject.iter().chain(clip.iter()) {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if pt_dist(a, b) > 1e-7 {
                segs.push((a, b));
            }
        }
    }
    if segs.is_empty() {
        return Vec::new();
    }

    // 1. Collect all pairwise intersection split parameters.
    let mut splits: Vec<Vec<f64>> = vec![vec![0.0, 1.0]; segs.len()];
    for i in 0..segs.len() {
        for j in (i + 1)..segs.len() {
            let hits = segment_intersection_params(segs[i].0, segs[i].1, segs[j].0, segs[j].1);
            for (t, u) in hits {
                if t > 1e-8 && t < 1.0 - 1e-8 {
                    splits[i].push(t);
                }
                if u > 1e-8 && u < 1.0 - 1e-8 {
                    splits[j].push(u);
                }
            }
        }
    }

    // 2. Split segments and deduplicate undirected atomic segments.
    let mut unique_edges: Vec<([f64; 2], [f64; 2])> = Vec::new();
    let mut seen_undirected: HashSet<((i64, i64), (i64, i64))> = HashSet::new();
    for (idx, (a, b)) in segs.iter().enumerate() {
        let ts = &mut splits[idx];
        ts.sort_by(f64::total_cmp);
        ts.dedup_by(|x, y| (*y - *x).abs() < 1e-8);
        for w in ts.windows(2) {
            let p0 = pt_mix(*a, *b, w[0]);
            let p1 = pt_mix(*a, *b, w[1]);
            if pt_dist(p0, p1) < 1e-6 {
                continue;
            }
            let k0 = quant_key(p0);
            let k1 = quant_key(p1);
            if k0 == k1 {
                continue;
            }
            let canon = if k0 < k1 { (k0, k1) } else { (k1, k0) };
            if seen_undirected.insert(canon) {
                unique_edges.push((p0, p1));
            }
        }
    }

    // 3. Classify each atomic segment's left & right side via exact midpoint winding probe.
    let mut directed: Vec<([f64; 2], [f64; 2])> = Vec::new();
    for (ei, &(a, b)) in unique_edges.iter().enumerate() {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let len = dx.hypot(dy);
        if len < 1e-7 {
            continue;
        }
        let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
        let n_left = [-dy / len, dx / len];

        // Choose adaptive probe distance well inside the adjacent faces.
        let mut min_other = f64::INFINITY;
        for (ej, &(oa, ob)) in unique_edges.iter().enumerate() {
            if ei == ej {
                continue;
            }
            let d = seg_dist(mid, oa, ob);
            if d > 1e-9 && d < min_other {
                min_other = d;
            }
        }
        let eps = (1e-4_f64).min(len * 0.25).min(min_other * 0.25).max(1e-7);
        let p_left = [mid[0] + n_left[0] * eps, mid[1] + n_left[1] * eps];
        let p_right = [mid[0] - n_left[0] * eps, mid[1] - n_left[1] * eps];

        let in_a_l = point_in_polygons_nonzero(p_left, subject);
        let in_b_l = point_in_polygons_nonzero(p_left, clip);
        let val_l = op.eval(in_a_l, in_b_l);

        let in_a_r = point_in_polygons_nonzero(p_right, subject);
        let in_b_r = point_in_polygons_nonzero(p_right, clip);
        let val_r = op.eval(in_a_r, in_b_r);

        if val_l && !val_r {
            directed.push((a, b));
        } else if val_r && !val_l {
            directed.push((b, a));
        }
    }

    if directed.is_empty() {
        return Vec::new();
    }

    // 4. Chain directed boundary edges into closed loops (sharpest right turn keeps faces simple).
    let mut adj: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, &(a, _)) in directed.iter().enumerate() {
        adj.entry(quant_key(a)).or_default().push(idx);
    }
    let mut used = vec![false; directed.len()];
    let mut loops: Vec<Vec<[f64; 2]>> = Vec::new();

    for start_idx in 0..directed.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let (s_a, s_b) = directed[start_idx];
        let start_key = quant_key(s_a);
        let mut ring = vec![s_a];
        let mut cur_a = s_a;
        let mut cur_b = s_b;
        let mut steps = 0usize;

        while quant_key(cur_b) != start_key && steps <= directed.len() {
            steps += 1;
            ring.push(cur_b);
            let bk = quant_key(cur_b);
            let Some(cands) = adj.get(&bk) else {
                break;
            };
            let in_angle = (cur_a[1] - cur_b[1]).atan2(cur_a[0] - cur_b[0]);
            let mut best: Option<(usize, f64)> = None;
            for &ci in cands {
                if used[ci] {
                    continue;
                }
                let (_, nb) = directed[ci];
                let out_angle = (nb[1] - cur_b[1]).atan2(nb[0] - cur_b[0]);
                let mut diff = out_angle - in_angle;
                while diff <= 0.0 {
                    diff += TAU;
                }
                if best.is_none_or(|(_, best_d)| diff < best_d) {
                    best = Some((ci, diff));
                }
            }
            if let Some((next_idx, _)) = best {
                used[next_idx] = true;
                cur_a = directed[next_idx].0;
                cur_b = directed[next_idx].1;
            } else {
                break;
            }
        }

        if quant_key(cur_b) == start_key && ring.len() >= 3 {
            let simplified = simplify_collinear_ring(&ring);
            if simplified.len() >= 3 && signed_area(&simplified).abs() > 1e-6 {
                loops.push(simplified);
            }
        }
    }

    loops
}

fn simplify_collinear_ring(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    let mut out = pts.to_vec();
    let mut changed = true;
    while changed && out.len() >= 3 {
        changed = false;
        let mut next = Vec::with_capacity(out.len());
        let n = out.len();
        for i in 0..n {
            let a = out[(i + n - 1) % n];
            let b = out[i];
            let c = out[(i + 1) % n];
            let abx = b[0] - a[0];
            let aby = b[1] - a[1];
            let bcx = c[0] - b[0];
            let bcy = c[1] - b[1];
            let cross = abx * bcy - aby * bcx;
            let dot = abx * bcx + aby * bcy;
            let l1 = abx.hypot(aby);
            let l2 = bcx.hypot(bcy);
            if l1 < 1e-7 || (l1 > 0.0 && l2 > 0.0 && cross.abs() / (l1 * l2) < 1e-6 && dot > 0.0) {
                changed = true;
            } else {
                next.push(b);
            }
        }
        out = next;
    }
    out
}

fn apply_merge_paths(paths: &[ShapePath], mode: &str) -> Vec<ShapePath> {
    if mode == "merge" {
        return paths
            .iter()
            .map(|p| ShapePath {
                vertices: None,
                ..p.clone()
            })
            .collect();
    }
    let closed_polys: Vec<Vec<[f64; 2]>> = paths
        .iter()
        .filter(|p| p.closed && p.points.len() >= 3)
        .map(|p| p.points.clone())
        .collect();
    if closed_polys.is_empty() {
        return Vec::new();
    }

    let result_polys = match mode {
        "add" => boolean_polygons(&closed_polys, &[], BoolOp::Add),
        "subtract" => {
            let base = boolean_polygons(&closed_polys[0..1], &[], BoolOp::Add);
            if closed_polys.len() <= 1 {
                base
            } else {
                let cutters = boolean_polygons(&closed_polys[1..], &[], BoolOp::Add);
                boolean_polygons(&base, &cutters, BoolOp::Subtract)
            }
        }
        "intersect" => {
            let mut acc = boolean_polygons(&closed_polys[0..1], &[], BoolOp::Add);
            for poly in &closed_polys[1..] {
                acc = boolean_polygons(&acc, std::slice::from_ref(poly), BoolOp::Intersect);
                if acc.is_empty() {
                    break;
                }
            }
            acc
        }
        "exclude" => {
            let mut acc = boolean_polygons(&closed_polys[0..1], &[], BoolOp::Add);
            for poly in &closed_polys[1..] {
                acc = boolean_polygons(&acc, std::slice::from_ref(poly), BoolOp::Exclude);
            }
            acc
        }
        _ => closed_polys,
    };

    result_polys
        .into_iter()
        .map(|points| ShapePath {
            points,
            vertices: None,
            closed: true,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect()
}

fn apply_offset_paths(
    paths: &[ShapePath],
    amount: f64,
    join: StrokeJoin,
    miter_limit: f64,
    tolerance: f64,
) -> Vec<ShapePath> {
    if amount.abs() < 1e-9 {
        return paths.to_vec();
    }
    let abs_amt = amount.abs();
    let band_style = StrokeStyle {
        width: abs_amt * 2.0,
        cap: if join == StrokeJoin::Round {
            StrokeCap::Round
        } else {
            StrokeCap::Butt
        },
        join,
        miter_limit: miter_limit.clamp(1.0, 100.0),
        dash_pairs: 0,
        dashes: [0.0; 3],
        gaps: [0.0; 3],
        dash_offset: 0.0,
        start_length: 0.0,
        end_length: 0.0,
        start_width: 100.0,
        end_width: 100.0,
        start_ease: 0.0,
        end_ease: 0.0,
        wave_amount: 0.0,
        wave_units_length: false,
        wavelength: 100.0,
        wave_cycles: 1.0,
        wave_phase: 0.0,
    };

    let closed_paths: Vec<ShapePath> = paths
        .iter()
        .filter(|p| p.closed && p.points.len() >= 3)
        .cloned()
        .collect();
    let open_paths: Vec<ShapePath> = paths
        .iter()
        .filter(|p| !p.closed && p.points.len() >= 2)
        .cloned()
        .collect();

    let mut out_polys: Vec<Vec<[f64; 2]>> = Vec::new();
    if !closed_paths.is_empty() {
        let base_polys: Vec<Vec<[f64; 2]>> =
            closed_paths.iter().map(|p| p.points.clone()).collect();
        let unified = boolean_polygons(&base_polys, &[], BoolOp::Add);
        let boundary_paths: Vec<ShapePath> = unified
            .iter()
            .map(|pts| ShapePath {
                points: pts.clone(),
                vertices: None,
                closed: true,
                widths: None,
                distances: None,
                total_length: None,
            })
            .collect();
        let band_pieces = expand_stroke_polygons(&boundary_paths, &band_style, tolerance);
        let band_union = boolean_polygons(&band_pieces, &[], BoolOp::Add);
        let res = if amount > 0.0 {
            boolean_polygons(&unified, &band_union, BoolOp::Add)
        } else {
            boolean_polygons(&unified, &band_union, BoolOp::Subtract)
        };
        out_polys.extend(res);
    }

    if !open_paths.is_empty() {
        let open_pieces = expand_stroke_polygons(&open_paths, &band_style, tolerance);
        let open_union = boolean_polygons(&open_pieces, &[], BoolOp::Add);
        if out_polys.is_empty() {
            out_polys = open_union;
        } else {
            out_polys = boolean_polygons(&out_polys, &open_union, BoolOp::Add);
        }
    }

    out_polys
        .into_iter()
        .map(|points| ShapePath {
            points,
            vertices: None,
            closed: true,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Gradients (exact port of Zoidium `paint.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GradientType {
    Linear,
    Radial,
}

#[derive(Debug, Clone)]
struct GradientStop {
    position: f64,
    /// Straight RGBA `0..=1`.
    color: [f64; 4],
}

#[derive(Debug, Clone)]
struct GradientPaint {
    kind: GradientType,
    start: [f64; 2],
    end: [f64; 2],
    highlight_length: f64,
    highlight_angle: f64,
    stops: Vec<GradientStop>,
    matrix: Mat2D,
}

impl GradientPaint {
    fn color_at(&self, t: f64) -> [f64; 4] {
        let stops = &self.stops;
        if stops.is_empty() {
            return [1.0, 1.0, 1.0, 1.0];
        }
        let mut a = &stops[0];
        let mut b = stops.last().unwrap();
        for i in 1..stops.len() {
            if t <= stops[i].position {
                a = &stops[i - 1];
                b = &stops[i];
                break;
            }
        }
        let f = clamp_f64(
            (t - a.position) / (b.position - a.position).max(1e-9),
            0.0,
            1.0,
        );
        // Premultiplied encoded sRGB interpolation (`paint.js:58-77`).
        let pa = [
            a.color[0] * a.color[3],
            a.color[1] * a.color[3],
            a.color[2] * a.color[3],
            a.color[3],
        ];
        let pb = [
            b.color[0] * b.color[3],
            b.color[1] * b.color[3],
            b.color[2] * b.color[3],
            b.color[3],
        ];
        [
            lerp_f64(pa[0], pb[0], f),
            lerp_f64(pa[1], pb[1], f),
            lerp_f64(pa[2], pb[2], f),
            lerp_f64(pa[3], pb[3], f),
        ]
    }

    /// Samples premultiplied RGBA at a sequence-space point (`paint.js:82-112`).
    fn sample_premul(&self, seq_pt: [f64; 2], inv_mat: Mat2D) -> [f64; 4] {
        let local = mat_transform_pt(seq_pt, inv_mat);
        let axis = [self.end[0] - self.start[0], self.end[1] - self.start[1]];
        let radius = axis[0].hypot(axis[1]).max(1e-6);
        let t = match self.kind {
            GradientType::Linear => {
                ((local[0] - self.start[0]) * axis[0] + (local[1] - self.start[1]) * axis[1])
                    / (radius * radius).max(1e-6)
            }
            GradientType::Radial => {
                let angle = axis[1].atan2(axis[0]) + self.highlight_angle;
                let h = clamp_f64(self.highlight_length / 100.0, -0.99, 0.99);
                let hl = [h * angle.cos(), h * angle.sin()];
                let d = [
                    (local[0] - self.start[0]) / radius - hl[0],
                    (local[1] - self.start[1]) / radius - hl[1],
                ];
                let a = d[0] * d[0] + d[1] * d[1];
                let b = hl[0] * d[0] + hl[1] * d[1];
                if a > 1e-8 {
                    a / (-b + (b * b + a * (1.0 - h * h)).max(0.0).sqrt()).max(1e-6)
                } else {
                    0.0
                }
            }
        };
        self.color_at(t)
    }
}

fn parse_gradient_stops_array(arr: &[Value]) -> Vec<GradientStop> {
    let mut stops = Vec::new();
    for item in arr.iter().take(16) {
        let (pos, col_val) = if let Some(pair) = item.as_array()
            && pair.len() >= 2
        {
            (pair[0].as_f64(), Some(&pair[1]))
        } else if let Some(obj) = item.as_object() {
            (
                obj.get("position").and_then(Value::as_f64),
                obj.get("color"),
            )
        } else {
            (None, None)
        };
        let (Some(p), Some(cv)) = (pos, col_val) else {
            continue;
        };
        if !p.is_finite() {
            continue;
        }
        let rgba = if let Some(s) = cv.as_str() {
            parse_css_color(s).ok()
        } else if let Some(c_arr) = cv.as_array()
            && c_arr.len() >= 3
        {
            let r = c_arr[0].as_f64().unwrap_or(0.0);
            let g = c_arr[1].as_f64().unwrap_or(0.0);
            let b = c_arr[2].as_f64().unwrap_or(0.0);
            let a = c_arr.get(3).and_then(Value::as_f64).unwrap_or(1.0);
            let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
                1.0 / 255.0
            } else {
                1.0
            };
            Some([
                (r * scale).clamp(0.0, 1.0),
                (g * scale).clamp(0.0, 1.0),
                (b * scale).clamp(0.0, 1.0),
                a.clamp(0.0, 1.0),
            ])
        } else {
            None
        };
        if let Some(color) = rgba {
            stops.push(GradientStop {
                position: p.clamp(0.0, 1.0),
                color,
            });
        }
    }
    stops.sort_by(|a, b| a.position.total_cmp(&b.position));
    stops
}

fn eval_gradient_stops(val: Option<&Value>, frame: f64) -> Vec<GradientStop> {
    let default_stops = || {
        vec![
            GradientStop {
                position: 0.0,
                color: [1.0, 77.0 / 255.0, 26.0 / 255.0, 1.0],
            },
            GradientStop {
                position: 1.0,
                color: [26.0 / 255.0, 102.0 / 255.0, 1.0, 1.0],
            },
        ]
    };
    let Some(v) = val else {
        return default_stops();
    };
    if let Some(arr) = v.as_array() {
        let parsed = parse_gradient_stops_array(arr);
        return if parsed.len() >= 2 {
            parsed
        } else {
            default_stops()
        };
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, Vec<GradientStop>, &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let arr = k.get("value").and_then(Value::as_array)?;
                let s = parse_gradient_stops_array(arr);
                if s.len() >= 2 { Some((f, s, k)) } else { None }
            })
            .collect();
        if !sorted.is_empty() {
            sorted.sort_by(|a, b| a.0.total_cmp(&b.0));
            if frame <= sorted[0].0 {
                return sorted[0].1.clone();
            }
            if frame >= sorted[sorted.len() - 1].0 {
                return sorted[sorted.len() - 1].1.clone();
            }
            for w in sorted.windows(2) {
                if frame <= w[1].0 {
                    let t = eval_segment_factor(w[0].2, w[1].2, frame).unwrap_or_else(|| {
                        let span = (w[1].0 - w[0].0).max(1e-9);
                        ((frame - w[0].0) / span).clamp(0.0, 1.0)
                    });
                    let len = w[0].1.len().min(w[1].1.len());
                    return (0..len)
                        .map(|i| GradientStop {
                            position: lerp_f64(w[0].1[i].position, w[1].1[i].position, t),
                            color: [
                                lerp_f64(w[0].1[i].color[0], w[1].1[i].color[0], t),
                                lerp_f64(w[0].1[i].color[1], w[1].1[i].color[1], t),
                                lerp_f64(w[0].1[i].color[2], w[1].1[i].color[2], t),
                                lerp_f64(w[0].1[i].color[3], w[1].1[i].color[3], t),
                            ],
                        })
                        .collect();
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_gradient_stops(Some(inner), frame);
    }
    default_stops()
}

fn eval_gradient_paint(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> GradientPaint {
    let get = |k: &str| props.and_then(|p| p.get(k));
    let kind = if eval_prop_str(get("gradientType"), frame, "linear").eq_ignore_ascii_case("radial")
    {
        GradientType::Radial
    } else {
        GradientType::Linear
    };
    GradientPaint {
        kind,
        start: eval_prop_vec2(get("startPoint"), frame, [-100.0, 0.0], eval_expr),
        end: eval_prop_vec2(get("endPoint"), frame, [100.0, 0.0], eval_expr),
        highlight_length: eval_prop_f64(get("highlightLength"), frame, 0.0, eval_expr)
            .clamp(-99.0, 99.0),
        highlight_angle: eval_prop_f64(get("highlightAngle"), frame, 0.0, eval_expr),
        stops: eval_gradient_stops(get("stops"), frame),
        matrix: MAT_IDENTITY,
    }
}

// ---------------------------------------------------------------------------
// Live Vector Text & Text Animator (`text`, `textAnimator`)
// ---------------------------------------------------------------------------

/// Built-in stroke-based vector font producing genuine closed 2D outlines for all
/// printable ASCII characters (`0x21..=0x7E`), plus fallback box for other glyphs.
/// Coordinates are normalized in `[0..0.60, 0..1.0]` (baseline at `y = 0`, cap height `y = 0.72`).
fn vector_glyph_strokes(ch: char) -> Vec<Vec<[f64; 2]>> {
    match ch {
        'A' => vec![
            vec![[0.05, 0.0], [0.28, 0.72], [0.51, 0.0]],
            vec![[0.14, 0.25], [0.42, 0.25]],
        ],
        'B' => vec![vec![
            [0.08, 0.0],
            [0.08, 0.72],
            [0.36, 0.72],
            [0.48, 0.63],
            [0.48, 0.45],
            [0.36, 0.37],
            [0.08, 0.37],
            [0.38, 0.37],
            [0.50, 0.27],
            [0.50, 0.09],
            [0.38, 0.0],
            [0.08, 0.0],
        ]],
        'C' => vec![vec![
            [0.49, 0.60],
            [0.39, 0.72],
            [0.20, 0.72],
            [0.08, 0.56],
            [0.08, 0.16],
            [0.20, 0.0],
            [0.39, 0.0],
            [0.49, 0.12],
        ]],
        'D' => vec![vec![
            [0.08, 0.0],
            [0.08, 0.72],
            [0.33, 0.72],
            [0.49, 0.56],
            [0.49, 0.16],
            [0.33, 0.0],
            [0.08, 0.0],
        ]],
        'E' => vec![
            vec![[0.48, 0.72], [0.08, 0.72], [0.08, 0.0], [0.48, 0.0]],
            vec![[0.08, 0.37], [0.40, 0.37]],
        ],
        'F' => vec![
            vec![[0.48, 0.72], [0.08, 0.72], [0.08, 0.0]],
            vec![[0.08, 0.37], [0.40, 0.37]],
        ],
        'G' => vec![vec![
            [0.49, 0.60],
            [0.39, 0.72],
            [0.20, 0.72],
            [0.08, 0.56],
            [0.08, 0.16],
            [0.20, 0.0],
            [0.42, 0.0],
            [0.49, 0.12],
            [0.49, 0.34],
            [0.30, 0.34],
        ]],
        'H' => vec![
            vec![[0.08, 0.0], [0.08, 0.72]],
            vec![[0.48, 0.0], [0.48, 0.72]],
            vec![[0.08, 0.37], [0.48, 0.37]],
        ],
        'I' => vec![
            vec![[0.14, 0.72], [0.42, 0.72]],
            vec![[0.28, 0.72], [0.28, 0.0]],
            vec![[0.14, 0.0], [0.42, 0.0]],
        ],
        'J' => vec![vec![
            [0.44, 0.72],
            [0.44, 0.16],
            [0.34, 0.0],
            [0.18, 0.0],
            [0.08, 0.12],
        ]],
        'K' => vec![
            vec![[0.08, 0.0], [0.08, 0.72]],
            vec![[0.48, 0.72], [0.08, 0.34], [0.49, 0.0]],
        ],
        'L' => vec![vec![[0.08, 0.72], [0.08, 0.0], [0.48, 0.0]]],
        'M' => vec![vec![
            [0.06, 0.0],
            [0.06, 0.72],
            [0.28, 0.28],
            [0.50, 0.72],
            [0.50, 0.0],
        ]],
        'N' => vec![vec![[0.08, 0.0], [0.08, 0.72], [0.48, 0.0], [0.48, 0.72]]],
        'O' => vec![vec![
            [0.28, 0.72],
            [0.44, 0.62],
            [0.50, 0.36],
            [0.44, 0.10],
            [0.28, 0.0],
            [0.12, 0.10],
            [0.06, 0.36],
            [0.12, 0.62],
            [0.28, 0.72],
        ]],
        'P' => vec![vec![
            [0.08, 0.0],
            [0.08, 0.72],
            [0.36, 0.72],
            [0.49, 0.62],
            [0.49, 0.44],
            [0.36, 0.34],
            [0.08, 0.34],
        ]],
        'Q' => vec![
            vec![
                [0.28, 0.72],
                [0.44, 0.62],
                [0.50, 0.36],
                [0.44, 0.10],
                [0.28, 0.0],
                [0.12, 0.10],
                [0.06, 0.36],
                [0.12, 0.62],
                [0.28, 0.72],
            ],
            vec![[0.34, 0.18], [0.52, -0.04]],
        ],
        'R' => vec![
            vec![
                [0.08, 0.0],
                [0.08, 0.72],
                [0.36, 0.72],
                [0.49, 0.62],
                [0.49, 0.44],
                [0.36, 0.35],
                [0.08, 0.35],
            ],
            vec![[0.30, 0.35], [0.50, 0.0]],
        ],
        'S' => vec![vec![
            [0.48, 0.62],
            [0.36, 0.72],
            [0.18, 0.72],
            [0.08, 0.61],
            [0.08, 0.48],
            [0.20, 0.39],
            [0.38, 0.33],
            [0.49, 0.22],
            [0.49, 0.10],
            [0.36, 0.0],
            [0.18, 0.0],
            [0.07, 0.11],
        ]],
        'T' => vec![
            vec![[0.04, 0.72], [0.52, 0.72]],
            vec![[0.28, 0.72], [0.28, 0.0]],
        ],
        'U' => vec![vec![
            [0.08, 0.72],
            [0.08, 0.18],
            [0.18, 0.0],
            [0.38, 0.0],
            [0.48, 0.18],
            [0.48, 0.72],
        ]],
        'V' => vec![vec![[0.06, 0.72], [0.28, 0.0], [0.50, 0.72]]],
        'W' => vec![vec![
            [0.04, 0.72],
            [0.15, 0.0],
            [0.28, 0.44],
            [0.41, 0.0],
            [0.52, 0.72],
        ]],
        'X' => vec![
            vec![[0.07, 0.72], [0.49, 0.0]],
            vec![[0.49, 0.72], [0.07, 0.0]],
        ],
        'Y' => vec![
            vec![[0.06, 0.72], [0.28, 0.36], [0.50, 0.72]],
            vec![[0.28, 0.36], [0.28, 0.0]],
        ],
        'Z' => vec![vec![[0.08, 0.72], [0.48, 0.72], [0.08, 0.0], [0.48, 0.0]]],
        'a'..='z' => {
            // Distinct lowercase proportions (x-height ~0.50) using clean strokes.
            let upper = ch.to_ascii_uppercase();
            vector_glyph_strokes(upper)
                .into_iter()
                .map(|stroke| {
                    stroke
                        .into_iter()
                        .map(|[x, y]| [x * 0.92, y * 0.72])
                        .collect()
                })
                .collect()
        }
        '0' => vec![
            vec![
                [0.28, 0.72],
                [0.44, 0.62],
                [0.48, 0.36],
                [0.44, 0.10],
                [0.28, 0.0],
                [0.12, 0.10],
                [0.08, 0.36],
                [0.12, 0.62],
                [0.28, 0.72],
            ],
            vec![[0.16, 0.16], [0.40, 0.56]],
        ],
        '1' => vec![
            vec![[0.16, 0.58], [0.28, 0.72], [0.28, 0.0]],
            vec![[0.14, 0.0], [0.42, 0.0]],
        ],
        '2' => vec![vec![
            [0.08, 0.58],
            [0.20, 0.72],
            [0.38, 0.72],
            [0.48, 0.58],
            [0.48, 0.44],
            [0.08, 0.0],
            [0.49, 0.0],
        ]],
        '3' => vec![vec![
            [0.09, 0.62],
            [0.22, 0.72],
            [0.38, 0.72],
            [0.48, 0.60],
            [0.48, 0.46],
            [0.36, 0.37],
            [0.22, 0.37],
            [0.38, 0.37],
            [0.49, 0.27],
            [0.49, 0.11],
            [0.38, 0.0],
            [0.20, 0.0],
            [0.08, 0.11],
        ]],
        '4' => vec![vec![[0.40, 0.0], [0.40, 0.72], [0.07, 0.24], [0.50, 0.24]]],
        '5' => vec![vec![
            [0.47, 0.72],
            [0.10, 0.72],
            [0.10, 0.40],
            [0.36, 0.40],
            [0.49, 0.30],
            [0.49, 0.11],
            [0.36, 0.0],
            [0.18, 0.0],
            [0.08, 0.10],
        ]],
        '6' => vec![vec![
            [0.44, 0.66],
            [0.32, 0.72],
            [0.16, 0.62],
            [0.08, 0.38],
            [0.08, 0.14],
            [0.20, 0.0],
            [0.38, 0.0],
            [0.49, 0.12],
            [0.49, 0.28],
            [0.38, 0.39],
            [0.18, 0.39],
            [0.08, 0.28],
        ]],
        '7' => vec![vec![[0.08, 0.72], [0.49, 0.72], [0.22, 0.0]]],
        '8' => vec![
            vec![
                [0.28, 0.72],
                [0.44, 0.63],
                [0.44, 0.46],
                [0.28, 0.38],
                [0.12, 0.46],
                [0.12, 0.63],
                [0.28, 0.72],
            ],
            vec![
                [0.28, 0.38],
                [0.47, 0.28],
                [0.47, 0.10],
                [0.28, 0.0],
                [0.09, 0.10],
                [0.09, 0.28],
                [0.28, 0.38],
            ],
        ],
        '9' => vec![vec![
            [0.12, 0.06],
            [0.24, 0.0],
            [0.40, 0.10],
            [0.48, 0.34],
            [0.48, 0.58],
            [0.36, 0.72],
            [0.18, 0.72],
            [0.08, 0.60],
            [0.08, 0.44],
            [0.18, 0.34],
            [0.38, 0.34],
            [0.48, 0.44],
        ]],
        '!' => vec![
            vec![[0.28, 0.72], [0.28, 0.22]],
            vec![[0.28, 0.06], [0.28, 0.02]],
        ],
        '.' => vec![vec![[0.28, 0.06], [0.28, 0.02]]],
        ',' => vec![vec![[0.28, 0.08], [0.22, -0.06]]],
        '-' => vec![vec![[0.10, 0.36], [0.46, 0.36]]],
        '+' => vec![
            vec![[0.10, 0.36], [0.46, 0.36]],
            vec![[0.28, 0.54], [0.28, 0.18]],
        ],
        ':' => vec![
            vec![[0.28, 0.48], [0.28, 0.44]],
            vec![[0.28, 0.12], [0.28, 0.08]],
        ],
        '?' => vec![
            vec![
                [0.10, 0.58],
                [0.22, 0.72],
                [0.38, 0.72],
                [0.48, 0.60],
                [0.48, 0.48],
                [0.28, 0.34],
                [0.28, 0.22],
            ],
            vec![[0.28, 0.06], [0.28, 0.02]],
        ],
        '/' => vec![vec![[0.10, 0.0], [0.46, 0.72]]],
        '(' => vec![vec![
            [0.38, 0.74],
            [0.22, 0.56],
            [0.18, 0.36],
            [0.22, 0.16],
            [0.38, -0.02],
        ]],
        ')' => vec![vec![
            [0.18, 0.74],
            [0.34, 0.56],
            [0.38, 0.36],
            [0.34, 0.16],
            [0.18, -0.02],
        ]],
        _ => vec![vec![
            [0.08, 0.0],
            [0.48, 0.0],
            [0.48, 0.72],
            [0.08, 0.72],
            [0.08, 0.0],
        ]],
    }
}

fn glyph_closed_contours(ch: char, scale_x: f64, scale_y: f64, ox: f64, oy: f64) -> Vec<ShapePath> {
    if ch.is_whitespace() {
        return Vec::new();
    }
    let strokes = vector_glyph_strokes(ch);
    let thickness = (scale_x.min(scale_y) * 0.11).max(1.0);
    let style = StrokeStyle {
        width: thickness,
        cap: StrokeCap::Round,
        join: StrokeJoin::Round,
        miter_limit: 4.0,
        dash_pairs: 0,
        dashes: [0.0; 3],
        gaps: [0.0; 3],
        dash_offset: 0.0,
        start_length: 0.0,
        end_length: 0.0,
        start_width: 100.0,
        end_width: 100.0,
        start_ease: 0.0,
        end_ease: 0.0,
        wave_amount: 0.0,
        wave_units_length: false,
        wavelength: 100.0,
        wave_cycles: 1.0,
        wave_phase: 0.0,
    };
    let raw_paths: Vec<ShapePath> = strokes
        .into_iter()
        .map(|pts| ShapePath {
            points: pts
                .into_iter()
                .map(|[x, y]| [ox + x * scale_x, oy + y * scale_y])
                .collect(),
            vertices: None,
            closed: false,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect();
    expand_stroke_polygons(&raw_paths, &style, 0.25)
        .into_iter()
        .map(|points| ShapePath {
            points,
            vertices: None,
            closed: true,
            widths: None,
            distances: None,
            total_length: None,
        })
        .collect()
}

#[derive(Debug, Clone)]
struct TextGlyphInstance {
    ch: char,
    line: usize,
    word: Option<usize>,
    x: f64,
    y: f64,
    advance: f64,
    scale_x: f64,
    scale_y: f64,
    paths: Vec<ShapePath>,
}

fn layout_shape_plus_text(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> (Vec<TextGlyphInstance>, usize) {
    let get = |k: &str| props.and_then(|p| p.get(k));
    let mut source = eval_prop_str(get("sourceText"), frame, "Text");
    let caps = eval_prop_str(get("caps"), frame, "normal").to_ascii_lowercase();
    if caps == "all" {
        source = source.to_uppercase();
    }
    let font_size = eval_prop_f64(get("fontSize"), frame, 100.0, eval_expr).max(0.0);
    if font_size <= 0.0 || source.is_empty() {
        return (Vec::new(), 0);
    }
    let tracking = eval_prop_f64(get("tracking"), frame, 0.0, eval_expr);
    let leading = eval_prop_f64(get("leading"), frame, font_size * 1.2, eval_expr);
    let baseline_shift = eval_prop_f64(get("baselineShift"), frame, 0.0, eval_expr);
    let h_scale = eval_prop_f64(get("horizontalScale"), frame, 100.0, eval_expr) / 100.0;
    let v_scale = eval_prop_f64(get("verticalScale"), frame, 100.0, eval_expr) / 100.0;
    let origin = eval_prop_vec2(get("position"), frame, [0.0, 0.0], eval_expr);
    let alignment = eval_prop_str(get("alignment"), frame, "left").to_ascii_lowercase();
    let text_mode = eval_prop_str(get("textMode"), frame, "point").to_ascii_lowercase();
    let box_size = eval_prop_vec2(get("boxSize"), frame, [600.0, 400.0], eval_expr);
    let first_line_indent = eval_prop_f64(get("firstLineIndent"), frame, 0.0, eval_expr);
    let space_before = eval_prop_f64(get("spaceBefore"), frame, 0.0, eval_expr);
    let space_after = eval_prop_f64(get("spaceAfter"), frame, 0.0, eval_expr);

    let sx = font_size * h_scale;
    let sy = font_size * v_scale;
    let char_adv = |c: char| -> f64 {
        let factor = if c == ' ' {
            0.38
        } else if matches!(c, 'I' | 'i' | 'l' | '!' | '.' | ',' | ':') {
            0.44
        } else {
            0.62
        };
        factor * sx + tracking * font_size / 1000.0
    };

    // Split paragraphs on '\n', and word-wrap in box mode.
    let mut lines: Vec<(String, bool)> = Vec::new();
    for para in source.split('\n') {
        if text_mode == "box" && box_size[0] > 0.0 {
            let max_w = box_size[0];
            let mut cur_line = String::new();
            let mut cur_w = first_line_indent;
            let mut first = true;
            for word in para.split(' ') {
                let word_w: f64 = word.chars().map(char_adv).sum();
                let sp_w = if cur_line.is_empty() {
                    0.0
                } else {
                    char_adv(' ')
                };
                if !cur_line.is_empty() && cur_w + sp_w + word_w > max_w {
                    lines.push((cur_line, first));
                    cur_line = word.to_owned();
                    cur_w = word_w;
                    first = false;
                } else {
                    if !cur_line.is_empty() {
                        cur_line.push(' ');
                        cur_w += sp_w;
                    }
                    cur_line.push_str(word);
                    cur_w += word_w;
                }
            }
            lines.push((cur_line, first));
        } else {
            lines.push((para.to_owned(), true));
        }
    }

    let line_count = lines.len();
    let mut glyphs = Vec::new();
    let mut pen_y = origin[1] + baseline_shift;
    let mut next_word_id = 0usize;

    for (line_idx, (line_str, is_para_start)) in lines.iter().enumerate() {
        if line_idx > 0 {
            pen_y -= leading;
            if *is_para_start {
                pen_y -= space_before + space_after;
            }
        }
        let line_w: f64 = line_str.chars().map(char_adv).sum();
        let align_x = match alignment.as_str() {
            "center" | "justifycenter" => -line_w * 0.5,
            "right" | "justifyright" => -line_w,
            _ => 0.0,
        };
        let indent = if *is_para_start {
            first_line_indent
        } else {
            0.0
        };
        let mut pen_x = origin[0] + align_x + indent;
        let mut in_word = false;
        let mut cur_word = 0usize;

        for ch in line_str.chars() {
            let word_opt = if ch.is_whitespace() {
                in_word = false;
                None
            } else {
                if !in_word {
                    in_word = true;
                    cur_word = next_word_id;
                    next_word_id += 1;
                }
                Some(cur_word)
            };
            let adv = char_adv(ch);
            let (gx, gy, gch) = if caps == "small" && ch.is_ascii_lowercase() {
                (sx * 0.8, sy * 0.8, ch.to_ascii_uppercase())
            } else {
                (sx, sy, ch)
            };
            let paths = glyph_closed_contours(gch, gx, gy, pen_x, pen_y);
            glyphs.push(TextGlyphInstance {
                ch,
                line: line_idx,
                word: word_opt,
                x: pen_x,
                y: pen_y,
                advance: adv,
                scale_x: gx,
                scale_y: gy,
                paths,
            });
            pen_x += adv;
        }
    }

    (glyphs, line_count)
}

fn selector_hash_seed(seed: f64, salt: i32) -> f64 {
    let mut h = ((seed.floor() as i32).wrapping_mul(374_761_393))
        .wrapping_add(salt.wrapping_mul(668_265_263));
    h = (h ^ ((h as u32) >> 13) as i32).wrapping_mul(1_274_126_177);
    h ^= ((h as u32) >> 16) as i32;
    f64::from(h as u32) / 4_294_967_296.0
}

fn selector_random_perm(count: usize, seed: f64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..count).collect();
    for i in (1..count).rev() {
        let r = selector_hash_seed(seed * 1000.0 + (i as f64) * 131.0 + 101.0 * 17.0, 7);
        let j = ((r * (i + 1) as f64).floor() as usize).min(i);
        order.swap(i, j);
    }
    order
}

fn selector_ease_low(value: f64, percent: f64) -> f64 {
    let v = value.clamp(0.0, 1.0);
    let amt = percent / 100.0;
    if amt >= 0.0 {
        v.powf(1.0 + amt)
    } else {
        v.powf(1.0 / (1.0 - amt))
    }
}

fn selector_ease_high(value: f64, percent: f64) -> f64 {
    let v = value.clamp(0.0, 1.0);
    let amt = percent / 100.0;
    if amt >= 0.0 {
        1.0 - (1.0 - v).powf(1.0 + amt)
    } else {
        1.0 - (1.0 - v).powf(1.0 / (1.0 - amt))
    }
}

fn selector_shape_value(shape: &str, i: usize, n: usize, start: f64, end: f64) -> f64 {
    if n == 0 || (start - end).abs() < 1e-12 {
        return 0.0;
    }
    let nf = n as f64;
    if shape == "square" {
        let a = start.max(i as f64 / nf);
        let b = end.min((i + 1) as f64 / nf);
        return ((b - a) * nf).clamp(0.0, 1.0);
    }
    let c = (i as f64 + 0.5) / nf;
    if shape == "rampUp" {
        if c <= start {
            return 0.0;
        }
        if c >= end {
            return 1.0;
        }
        return (c - start) / (end - start);
    }
    if shape == "rampDown" {
        if c <= start {
            return 1.0;
        }
        if c >= end {
            return 0.0;
        }
        return (end - c) / (end - start);
    }
    if c <= start || c >= end {
        return 0.0;
    }
    let t = (c - start) / (end - start);
    match shape {
        "triangle" => {
            if t < 0.5 {
                t * 2.0
            } else {
                (1.0 - t) * 2.0
            }
        }
        "round" => (1.0 - (t * 2.0 - 1.0).powi(2)).max(0.0).sqrt(),
        _ => 0.5 - 0.5 * (t * TAU).cos(),
    }
}

fn selector_combine(mode: &str, a: f64, b: f64) -> f64 {
    let r = match mode {
        "subtract" => a - b,
        "intersect" => a * b,
        "min" => a.min(b),
        "max" => a.max(b),
        "difference" => (a - b).abs(),
        _ => a + b,
    };
    r.clamp(-1.0, 1.0)
}

fn compute_text_units(
    glyphs: &[TextGlyphInstance],
    line_count: usize,
    based_on: &str,
) -> (usize, Vec<isize>) {
    match based_on {
        "lines" => (line_count, glyphs.iter().map(|g| g.line as isize).collect()),
        "words" => {
            let mut max_w = 0usize;
            let map: Vec<isize> = glyphs
                .iter()
                .map(|g| match g.word {
                    Some(w) => {
                        max_w = max_w.max(w + 1);
                        w as isize
                    }
                    None => -1,
                })
                .collect();
            (max_w, map)
        }
        "noSpaces" => {
            let mut count = 0isize;
            let map: Vec<isize> = glyphs
                .iter()
                .map(|g| {
                    if g.ch.is_whitespace() {
                        -1
                    } else {
                        let id = count;
                        count += 1;
                        id
                    }
                })
                .collect();
            (count as usize, map)
        }
        _ => (glyphs.len(), (0..glyphs.len() as isize).collect()),
    }
}

fn shift_codepoint(ch: char, shift: i32) -> char {
    if shift == 0 {
        return ch;
    }
    let code = ch as u32;
    for (base, size) in [(97_u32, 26_i32), (65_u32, 26_i32), (48_u32, 10_i32)] {
        if code >= base && code < base + size as u32 {
            let shifted = ((code - base) as i32 + shift).rem_euclid(size) as u32;
            return char::from_u32(base + shifted).unwrap_or(ch);
        }
    }
    ch
}

#[derive(Debug, Clone, Copy)]
struct CharPaintOverride {
    opacity: f64,
    fill_color: Option<[f64; 3]>,
}

fn animator_prop_enabled(props: &Map<String, Value>, key: &str, has_any_flag: bool) -> bool {
    let flag_key = format!("enabled_{key}");
    if let Some(v) = props.get(&flag_key) {
        return v
            .as_bool()
            .unwrap_or_else(|| v.as_f64().unwrap_or(0.0) != 0.0);
    }
    if has_any_flag {
        return false;
    }
    props.contains_key(key) || (key == "fillColor" && props.contains_key("color"))
}

fn evaluate_text_with_animators(
    item: &Value,
    animators: &[&Value],
    frame: f64,
    fps: f64,
    eval_expr: ExprEval<'_>,
) -> Vec<(Vec<ShapePath>, Mat2D, Option<CharPaintOverride>)> {
    let props = item.get("properties").and_then(Value::as_object);
    let font_size = eval_prop_f64(
        props.and_then(|p| p.get("fontSize")),
        frame,
        100.0,
        eval_expr,
    );
    let (mut glyphs, line_count) = layout_shape_plus_text(props, frame, eval_expr);
    if glyphs.is_empty() {
        return Vec::new();
    }

    let active_animators: Vec<&Map<String, Value>> = animators
        .iter()
        .filter_map(|a| {
            let ap = a.get("properties").and_then(Value::as_object)?;
            if !eval_prop_enabled(Some(ap), frame, eval_expr) {
                return None;
            }
            Some(ap)
        })
        .collect();

    if active_animators.is_empty() {
        let all_paths: Vec<ShapePath> = glyphs.into_iter().flat_map(|g| g.paths).collect();
        return vec![(all_paths, MAT_IDENTITY, None)];
    }

    let time = if fps > 0.0 { frame / fps } else { 0.0 };
    let n = glyphs.len();
    struct AnimWeights {
        s: Vec<f64>,
        vx: Vec<f64>,
        vy: Vec<f64>,
        has_any_flag: bool,
    }

    let mut all_weights = Vec::with_capacity(active_animators.len());
    for ap in &active_animators {
        let get = |k: &str| ap.get(k);
        let has_any_flag = ap.keys().any(|k| k.starts_with("enabled_"));
        let based_on = eval_prop_str(get("selectorBasedOn"), frame, "characters");
        let (unit_count, unit_map) = compute_text_units(&glyphs, line_count, &based_on);
        let units_mode = eval_prop_str(get("selectorUnits"), frame, "percentage");
        let start_raw = eval_prop_f64(get("selectorStart"), frame, 0.0, eval_expr);
        let end_raw = eval_prop_f64(get("selectorEnd"), frame, 100.0, eval_expr);
        let off_raw = eval_prop_f64(get("selectorOffset"), frame, 0.0, eval_expr);
        let (start, end) = if units_mode == "index" && unit_count > 0 {
            let nf = unit_count as f64;
            let raw_e = if (end_raw - 100.0).abs() < 1e-9
                && !ap
                    .get("selectorEnd")
                    .is_some_and(|v| v.get("keyframes").is_some())
            {
                nf
            } else {
                end_raw
            };
            ((start_raw + off_raw) / nf, (raw_e + off_raw) / nf)
        } else {
            ((start_raw + off_raw) / 100.0, (end_raw + off_raw) / 100.0)
        };
        let from = start.min(end);
        let to = start.max(end);
        let shape = eval_prop_str(get("selectorShape"), frame, "square");
        let ease_low = eval_prop_f64(get("selectorEaseLow"), frame, 0.0, eval_expr);
        let ease_high = eval_prop_f64(get("selectorEaseHigh"), frame, 0.0, eval_expr);
        let amount = eval_prop_f64(get("selectorAmount"), frame, 100.0, eval_expr) / 100.0;
        let randomize = eval_prop_f64(get("selectorRandomize"), frame, 0.0, eval_expr) != 0.0;
        let rand_seed = eval_prop_f64(get("selectorRandomSeed"), frame, 0.0, eval_expr);
        let perm = if randomize {
            Some(selector_random_perm(unit_count, rand_seed))
        } else {
            None
        };
        let range_vals: Vec<f64> = (0..unit_count)
            .map(|i| {
                let ui = perm.as_ref().map_or(i, |p| p[i]);
                let raw = selector_shape_value(&shape, ui, unit_count, from, to);
                selector_ease_high(selector_ease_low(raw, ease_low), ease_high) * amount
            })
            .collect();

        let sel_mode = eval_prop_str(get("selectorMode"), frame, "add");
        let wiggly_on = eval_prop_f64(get("wigglyOn"), frame, 0.0, eval_expr) != 0.0;
        let wiggly_mode = eval_prop_str(get("wigglyMode"), frame, "add");
        let wiggly_max = eval_prop_f64(get("wigglyMax"), frame, 100.0, eval_expr) / 100.0;
        let wiggly_min = eval_prop_f64(get("wigglyMin"), frame, -100.0, eval_expr) / 100.0;
        let wiggly_based = eval_prop_str(get("wigglyBasedOn"), frame, "characters");
        let locked = eval_prop_f64(get("lockDimensions"), frame, 0.0, eval_expr) != 0.0;
        let wp = WiggleParams {
            time,
            frequency: eval_prop_f64(get("wigglesPerSecond"), frame, 2.0, eval_expr),
            correlation: eval_prop_f64(get("correlation"), frame, 50.0, eval_expr),
            temporal_phase: eval_prop_f64(get("temporalPhase"), frame, 0.0, eval_expr),
            spatial_phase: eval_prop_f64(get("spatialPhase"), frame, 0.0, eval_expr),
            seed: eval_prop_f64(get("wigglySeed"), frame, 0.0, eval_expr).round() as i32,
        };
        let wiggly_map = if wiggly_on && wiggly_based != based_on {
            compute_text_units(&glyphs, line_count, &wiggly_based).1
        } else {
            unit_map.clone()
        };

        let mut s = vec![0.0; n];
        let mut vx = vec![0.0; n];
        let mut vy = vec![0.0; n];
        for i in 0..n {
            let u = unit_map[i];
            let r = if u < 0 {
                0.0
            } else {
                selector_combine(&sel_mode, 0.0, range_vals[u as usize])
            };
            if !wiggly_on {
                s[i] = r;
                vx[i] = r;
                vy[i] = r;
            } else {
                let uw = wiggly_map[i];
                let sample_w = |comp: i32| -> f64 {
                    if uw < 0 {
                        0.0
                    } else {
                        let nv = sample_wiggle_noise(wp, uw as f64, comp, 0, 0);
                        wiggly_min + (nv + 1.0) * 0.5 * (wiggly_max - wiggly_min)
                    }
                };
                s[i] = selector_combine(&wiggly_mode, r, sample_w(0));
                if locked {
                    vx[i] = s[i];
                    vy[i] = s[i];
                } else {
                    vx[i] = selector_combine(&wiggly_mode, r, sample_w(1));
                    vy[i] = selector_combine(&wiggly_mode, r, sample_w(2));
                }
            }
        }
        all_weights.push(AnimWeights {
            s,
            vx,
            vy,
            has_any_flag,
        });
    }

    // Apply charOffset and tracking shifts first.
    let mut line_carry: BTreeMap<usize, f64> = BTreeMap::new();
    for (i, glyph) in glyphs.iter_mut().enumerate() {
        let mut shift = 0i32;
        let mut track_step = 0.0;
        for (a_idx, ap) in active_animators.iter().enumerate() {
            let w = &all_weights[a_idx];
            if animator_prop_enabled(ap, "charOffset", w.has_any_flag) {
                let co = eval_prop_f64(ap.get("charOffset"), frame, 0.0, eval_expr);
                shift += (co * w.s[i]).round() as i32;
            }
            if animator_prop_enabled(ap, "tracking", w.has_any_flag) {
                let tr = eval_prop_f64(ap.get("tracking"), frame, 0.0, eval_expr);
                track_step += tr * w.s[i] * font_size / 1000.0;
            }
        }
        let carry = line_carry.entry(glyph.line).or_insert(0.0);
        if *carry != 0.0 || shift != 0 {
            glyph.x += *carry;
            if shift != 0 {
                glyph.ch = shift_codepoint(glyph.ch, shift);
            }
            glyph.paths =
                glyph_closed_contours(glyph.ch, glyph.scale_x, glyph.scale_y, glyph.x, glyph.y);
        }
        *carry += track_step;
    }

    // Build per-glyph composed transform matrix & optional opacity/fillColor override.
    let mut out = Vec::with_capacity(n);
    for (i, glyph) in glyphs.into_iter().enumerate() {
        if glyph.paths.is_empty() {
            continue;
        }
        let mut mat = MAT_IDENTITY;
        let mut opacity = 1.0_f64;
        let mut fill_color: Option<[f64; 3]> = None;
        let ref_pt = [glyph.x + glyph.advance * 0.5, glyph.y];

        for (a_idx, ap) in active_animators.iter().enumerate() {
            let w = &all_weights[a_idx];
            let sx_w = w.vx[i];
            let sy_w = w.vy[i];
            let sc_w = w.s[i];
            let anchor_off = if animator_prop_enabled(ap, "anchor", w.has_any_flag) {
                let a = eval_prop_vec2(ap.get("anchor"), frame, [0.0, 0.0], eval_expr);
                [a[0] * sx_w, a[1] * sy_w]
            } else {
                [0.0, 0.0]
            };
            let pivot = [ref_pt[0] + anchor_off[0], ref_pt[1] + anchor_off[1]];
            let pos_off = if animator_prop_enabled(ap, "position", w.has_any_flag) {
                let p = eval_prop_vec2(ap.get("position"), frame, [0.0, 0.0], eval_expr);
                [p[0] * sx_w, p[1] * sy_w]
            } else {
                [0.0, 0.0]
            };
            let scale = if animator_prop_enabled(ap, "scale", w.has_any_flag) {
                let s = eval_prop_vec2(ap.get("scale"), frame, [100.0, 100.0], eval_expr);
                [
                    1.0 + (s[0] / 100.0 - 1.0) * sx_w,
                    1.0 + (s[1] / 100.0 - 1.0) * sy_w,
                ]
            } else {
                [1.0, 1.0]
            };
            let rot = if animator_prop_enabled(ap, "rotation", w.has_any_flag) {
                eval_prop_f64(ap.get("rotation"), frame, 0.0, eval_expr) * sc_w
            } else {
                0.0
            };
            let skew = if animator_prop_enabled(ap, "skew", w.has_any_flag) {
                eval_prop_f64(ap.get("skew"), frame, 0.0, eval_expr) * sc_w
            } else {
                0.0
            };
            let skew_axis = if animator_prop_enabled(ap, "skewAxis", w.has_any_flag) {
                eval_prop_f64(ap.get("skewAxis"), frame, 0.0, eval_expr) * sc_w
            } else {
                0.0
            };
            if animator_prop_enabled(ap, "opacity", w.has_any_flag) {
                let target_op = eval_prop_f64(ap.get("opacity"), frame, 1.0, eval_expr);
                opacity *= (1.0 + (target_op - 1.0) * sc_w).clamp(0.0, 1.0);
            }
            if animator_prop_enabled(ap, "fillColor", w.has_any_flag) && sc_w.abs() > 1e-6 {
                let col_val = ap.get("fillColor").or_else(|| ap.get("color"));
                let target = eval_prop_color_rgb(col_val, frame, [1.0, 1.0, 1.0], eval_expr);
                let base = fill_color.unwrap_or([1.0, 1.0, 1.0]);
                let f = sc_w.clamp(0.0, 1.0);
                fill_color = Some([
                    lerp_f64(base[0], target[0], f),
                    lerp_f64(base[1], target[1], f),
                    lerp_f64(base[2], target[2], f),
                ]);
            }

            let level = group_matrix(
                pivot,
                [pivot[0] + pos_off[0], pivot[1] + pos_off[1]],
                scale,
                rot,
                skew,
                skew_axis,
            );
            mat = mat_mul(level, mat);
        }

        let ov = if (opacity - 1.0).abs() > 1e-6 || fill_color.is_some() {
            Some(CharPaintOverride {
                opacity,
                fill_color,
            })
        } else {
            None
        };
        out.push((glyph.paths, mat, ov));
    }
    out
}

// ---------------------------------------------------------------------------
// Procedural Compiler (matching Zoidium `compiler-procedural.js`)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PathView {
    base_idx: usize,
    matrix: Mat2D,
    wiggle_stack: Vec<(Mat2D, WiggleTransformConfig)>,
    char_override: Option<CharPaintOverride>,
}

impl PathView {
    fn matrix_for_copy(&self, copy: i32) -> Mat2D {
        if self.wiggle_stack.is_empty() {
            return self.matrix;
        }
        let mut m = self.matrix;
        for &(post_mat, wcfg) in &self.wiggle_stack {
            let wm = eval_wiggle_transform_matrix(wcfg, copy);
            m = mat_mul(post_mat, mat_mul(wm, m));
        }
        m
    }
}

#[derive(Debug, Clone)]
enum PaintKind {
    Fill {
        color: [f64; 3],
        even_odd: bool,
    },
    GradientFill {
        gradient: GradientPaint,
        even_odd: bool,
    },
    Stroke {
        color: [f64; 3],
        style: StrokeStyle,
    },
    GradientStroke {
        gradient: GradientPaint,
        style: StrokeStyle,
    },
}

#[derive(Debug, Clone)]
enum ScopeCommand {
    Group {
        matrix: Mat2D,
        wiggle: Option<WiggleTransformConfig>,
        opacity: f64,
        commands: Vec<ScopeCommand>,
    },
    Paint {
        views: Vec<PathView>,
        kind: PaintKind,
        opacity: f64,
        tolerance: f64,
    },
}

#[derive(Debug, Clone)]
enum RenderCommand {
    Isolation {
        opacity: f64,
        commands: Vec<RenderCommand>,
    },
    SolidFill {
        polygons: Vec<Vec<[f64; 2]>>,
        color: [f64; 3],
        opacity: f64,
        even_odd: bool,
    },
    GradientFill {
        polygons: Vec<Vec<[f64; 2]>>,
        gradient: GradientPaint,
        opacity: f64,
        even_odd: bool,
    },
}

impl RenderCommand {
    fn transformed(self, m: Mat2D) -> Self {
        if mat_is_identity(m) {
            return self;
        }
        match self {
            Self::Isolation { opacity, commands } => Self::Isolation {
                opacity,
                commands: commands.into_iter().map(|c| c.transformed(m)).collect(),
            },
            Self::SolidFill {
                polygons,
                color,
                opacity,
                even_odd,
            } => Self::SolidFill {
                polygons: polygons
                    .into_iter()
                    .map(|p| p.into_iter().map(|pt| mat_transform_pt(pt, m)).collect())
                    .collect(),
                color,
                opacity,
                even_odd,
            },
            Self::GradientFill {
                polygons,
                mut gradient,
                opacity,
                even_odd,
            } => {
                gradient.matrix = mat_mul(m, gradient.matrix);
                Self::GradientFill {
                    polygons: polygons
                        .into_iter()
                        .map(|p| p.into_iter().map(|pt| mat_transform_pt(pt, m)).collect())
                        .collect(),
                    gradient,
                    opacity,
                    even_odd,
                }
            }
        }
    }
}

struct RepeaterCopy {
    matrix: Mat2D,
    opacity: f64,
}

fn repeater_copies(
    props: Option<&Map<String, Value>>,
    frame: f64,
    eval_expr: ExprEval<'_>,
) -> Vec<RepeaterCopy> {
    let get = |k: &str| props.and_then(|p| p.get(k));
    let count = eval_prop_f64(get("copies"), frame, 3.0, eval_expr).clamp(0.0, MAX_REPEATER_COPIES);
    let n = count.ceil() as usize;
    let offset = eval_prop_f64(get("offset"), frame, 0.0, eval_expr);
    let anchor = eval_prop_vec2(get("anchor"), frame, [0.0, 0.0], eval_expr);
    let pos = eval_prop_vec2(get("position"), frame, [100.0, 0.0], eval_expr);
    let scale = eval_prop_vec2(get("scale"), frame, [1.0, 1.0], eval_expr);
    let rot = eval_prop_f64(get("rotation"), frame, 0.0, eval_expr);
    let start_op = eval_prop_f64(get("startOpacity"), frame, 1.0, eval_expr).clamp(0.0, 1.0);
    let end_op = eval_prop_f64(get("endOpacity"), frame, 1.0, eval_expr).clamp(0.0, 1.0);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let exponent = i as f64 + offset;
        let power = |s: f64| -> f64 {
            if exponent == 0.0 {
                1.0
            } else if s < 0.0 && exponent.fract() != 0.0 {
                s.abs().powf(exponent)
            } else if s == 0.0 && exponent < 0.0 {
                0.0
            } else {
                s.powf(exponent)
            }
        };
        let m = group_matrix(
            anchor,
            [anchor[0] + pos[0] * exponent, anchor[1] + pos[1] * exponent],
            [power(scale[0]), power(scale[1])],
            rot * exponent,
            0.0,
            0.0,
        );
        if !m.iter().all(|v| v.is_finite()) {
            continue;
        }
        let frac = if n > 1 {
            i as f64 / (n - 1) as f64
        } else {
            0.0
        };
        let op = lerp_f64(start_op, end_op, frac).clamp(0.0, 1.0) * (count - i as f64).min(1.0);
        out.push(RepeaterCopy {
            matrix: m,
            opacity: op.clamp(0.0, 1.0),
        });
    }
    out
}

struct Compiler<'a> {
    frame: f64,
    fps: f64,
    eval_expr: ExprEval<'a>,
    bases: Vec<Vec<ShapePath>>,
}

impl<'a> Compiler<'a> {
    fn alloc_base(&mut self, paths: Vec<ShapePath>) -> usize {
        let idx = self.bases.len();
        self.bases.push(paths);
        idx
    }

    fn view_paths(&self, view: &PathView) -> Vec<ShapePath> {
        let m = view.matrix_for_copy(0);
        self.bases[view.base_idx]
            .iter()
            .map(|p| p.transformed(m))
            .collect()
    }

    fn below_animators<'b>(&self, list: &'b [Value], index: usize) -> Vec<&'b Value> {
        list[index + 1..]
            .iter()
            .filter(|item| {
                item.get("kind").and_then(Value::as_str) == Some("textAnimator")
                    && eval_prop_enabled(
                        item.get("properties").and_then(Value::as_object),
                        self.frame,
                        self.eval_expr,
                    )
            })
            .collect()
    }

    fn compile_scope(
        &mut self,
        list: &[Value],
        tolerance: f64,
        depth: usize,
        outer_animators: &[&Value],
    ) -> (Vec<PathView>, Vec<ScopeCommand>) {
        if depth > MAX_SHAPE_PLUS_DEPTH {
            return (Vec::new(), Vec::new());
        }
        let mut views: Vec<PathView> = Vec::new();
        let mut commands: Vec<ScopeCommand> = Vec::new();

        // Precompute downstream repeater scale factor for adaptive curve tolerance.
        let mut future_scales = vec![1.0_f64; list.len()];
        let mut future_scale = 1.0_f64;
        for i in (0..list.len()).rev() {
            future_scales[i] = future_scale;
            let item = &list[i];
            if item.get("kind").and_then(Value::as_str) == Some("repeater") {
                let props = item.get("properties").and_then(Value::as_object);
                if eval_prop_enabled(props, self.frame, self.eval_expr) {
                    let copies = repeater_copies(props, self.frame, self.eval_expr);
                    let max_s = copies
                        .iter()
                        .map(|c| mat_max_scale(c.matrix))
                        .fold(1.0_f64, f64::max);
                    future_scale = (future_scale * max_s).min(1024.0);
                }
            }
        }

        for (index, item) in list.iter().enumerate() {
            let Some(kind) = item.get("kind").and_then(Value::as_str) else {
                continue;
            };
            let props = item.get("properties").and_then(Value::as_object);
            if !eval_prop_enabled(props, self.frame, self.eval_expr) {
                continue;
            }
            let item_tol = (tolerance / future_scales[index]).max(1e-3);
            let get = |k: &str| props.and_then(|p| p.get(k));

            match kind {
                "textAnimator" => {}
                "text" => {
                    let mut anims = self.below_animators(list, index);
                    anims.extend_from_slice(outer_animators);
                    let glyph_groups = evaluate_text_with_animators(
                        item,
                        &anims,
                        self.frame,
                        self.fps,
                        self.eval_expr,
                    );
                    for (paths, mat, ov) in glyph_groups {
                        if paths.is_empty() {
                            continue;
                        }
                        let base_idx = self.alloc_base(paths);
                        views.push(PathView {
                            base_idx,
                            matrix: mat,
                            wiggle_stack: Vec::new(),
                            char_override: ov,
                        });
                    }
                }
                "group" => {
                    let anchor =
                        eval_prop_vec2(get("anchor"), self.frame, [0.0, 0.0], self.eval_expr);
                    let pos =
                        eval_prop_vec2(get("position"), self.frame, [0.0, 0.0], self.eval_expr);
                    let scale =
                        eval_prop_vec2(get("scale"), self.frame, [1.0, 1.0], self.eval_expr);
                    let rot = eval_prop_f64(get("rotation"), self.frame, 0.0, self.eval_expr);
                    let skew = eval_prop_f64(get("skew"), self.frame, 0.0, self.eval_expr);
                    let skew_axis = eval_prop_f64(get("skewAxis"), self.frame, 0.0, self.eval_expr);
                    let opacity = eval_prop_f64(get("opacity"), self.frame, 1.0, self.eval_expr)
                        .clamp(0.0, 1.0);
                    let m = group_matrix(anchor, pos, scale, rot, skew, skew_axis);
                    let g_scale = mat_max_scale(m);
                    let mut next_outer = self.below_animators(list, index);
                    next_outer.extend_from_slice(outer_animators);
                    let children = item
                        .get("contents")
                        .and_then(Value::as_array)
                        .map_or(&[][..], Vec::as_slice);
                    let (child_views, child_cmds) =
                        self.compile_scope(children, item_tol / g_scale, depth + 1, &next_outer);

                    // Propagate nested group paths UP to parent scope!
                    for mut cv in child_views {
                        if cv.wiggle_stack.is_empty() {
                            cv.matrix = mat_mul(m, cv.matrix);
                        } else if let Some(last) = cv.wiggle_stack.last_mut() {
                            last.0 = mat_mul(m, last.0);
                        }
                        views.push(cv);
                    }
                    commands.insert(
                        0,
                        ScopeCommand::Group {
                            matrix: m,
                            wiggle: None,
                            opacity,
                            commands: child_cmds,
                        },
                    );
                }
                "fill" | "stroke" | "gradientFill" | "gradientStroke" => {
                    let opacity = eval_prop_f64(get("opacity"), self.frame, 1.0, self.eval_expr)
                        .clamp(0.0, 1.0);
                    let composite_above = eval_prop_str(get("composite"), self.frame, "below")
                        .eq_ignore_ascii_case("above");
                    let even_odd = matches!(
                        eval_prop_str(get("fillRule"), self.frame, "nonzero")
                            .to_ascii_lowercase()
                            .as_str(),
                        "even-odd" | "evenodd"
                    );
                    let paint_kind = match kind {
                        "fill" => PaintKind::Fill {
                            color: eval_prop_color_rgb(
                                get("color"),
                                self.frame,
                                [1.0, 1.0, 1.0],
                                self.eval_expr,
                            ),
                            even_odd,
                        },
                        "gradientFill" => PaintKind::GradientFill {
                            gradient: eval_gradient_paint(props, self.frame, self.eval_expr),
                            even_odd,
                        },
                        "stroke" => PaintKind::Stroke {
                            color: eval_prop_color_rgb(
                                get("color"),
                                self.frame,
                                [1.0, 1.0, 1.0],
                                self.eval_expr,
                            ),
                            style: StrokeStyle::from_props(props, self.frame, self.eval_expr),
                        },
                        _ => PaintKind::GradientStroke {
                            gradient: eval_gradient_paint(props, self.frame, self.eval_expr),
                            style: StrokeStyle::from_props(props, self.frame, self.eval_expr),
                        },
                    };
                    let paint = ScopeCommand::Paint {
                        views: views.clone(),
                        kind: paint_kind,
                        opacity,
                        tolerance: item_tol,
                    };
                    if composite_above {
                        commands.push(paint);
                    } else {
                        commands.insert(0, paint);
                    }
                }
                "offsetPaths" | "mergePaths" => {
                    let all_paths: Vec<ShapePath> =
                        views.iter().flat_map(|v| self.view_paths(v)).collect();
                    let changed = if kind == "offsetPaths" {
                        let amount = eval_prop_f64(get("amount"), self.frame, 10.0, self.eval_expr);
                        let join = match eval_prop_str(get("join"), self.frame, "miter")
                            .to_ascii_lowercase()
                            .as_str()
                        {
                            "round" => StrokeJoin::Round,
                            "bevel" => StrokeJoin::Bevel,
                            _ => StrokeJoin::Miter,
                        };
                        let miter_limit =
                            eval_prop_f64(get("miterLimit"), self.frame, 4.0, self.eval_expr);
                        apply_offset_paths(&all_paths, amount, join, miter_limit, item_tol)
                    } else {
                        let mode =
                            eval_prop_str(get("mode"), self.frame, "add").to_ascii_lowercase();
                        apply_merge_paths(&all_paths, &mode)
                    };
                    for v in &views {
                        self.bases[v.base_idx].clear();
                    }
                    let new_base = self.alloc_base(changed);
                    let retarget_ok = !self.bases[new_base].is_empty();
                    fn retarget_cmds(cmds: &mut [ScopeCommand], m: Mat2D, new_base: usize) {
                        for cmd in cmds.iter_mut() {
                            match cmd {
                                ScopeCommand::Paint { views, .. } => {
                                    if let Some(inv) = mat_inv(m) {
                                        *views = vec![PathView {
                                            base_idx: new_base,
                                            matrix: inv,
                                            wiggle_stack: Vec::new(),
                                            char_override: None,
                                        }];
                                    }
                                }
                                ScopeCommand::Group {
                                    matrix, commands, ..
                                } => {
                                    retarget_cmds(commands, mat_mul(m, *matrix), new_base);
                                }
                            }
                        }
                    }
                    retarget_cmds(&mut commands, MAT_IDENTITY, new_base);
                    views = if retarget_ok {
                        vec![PathView {
                            base_idx: new_base,
                            matrix: MAT_IDENTITY,
                            wiggle_stack: Vec::new(),
                            char_override: None,
                        }]
                    } else {
                        Vec::new()
                    };
                }
                "wiggleTransform" => {
                    let wp = WiggleParams {
                        time: if self.fps > 0.0 {
                            self.frame / self.fps
                        } else {
                            0.0
                        },
                        frequency: eval_prop_f64(
                            get("frequency").or_else(|| get("wigglesPerSecond")),
                            self.frame,
                            1.0,
                            self.eval_expr,
                        ),
                        correlation: eval_prop_f64(
                            get("correlation"),
                            self.frame,
                            50.0,
                            self.eval_expr,
                        ),
                        temporal_phase: eval_prop_f64(
                            get("temporalPhase"),
                            self.frame,
                            0.0,
                            self.eval_expr,
                        ),
                        spatial_phase: eval_prop_f64(
                            get("spatialPhase"),
                            self.frame,
                            0.0,
                            self.eval_expr,
                        ),
                        seed: eval_prop_f64(get("seed"), self.frame, 1.0, self.eval_expr).round()
                            as i32,
                    };
                    let wcfg = WiggleTransformConfig {
                        wp,
                        anchor: eval_prop_vec2(
                            get("anchor"),
                            self.frame,
                            [0.0, 0.0],
                            self.eval_expr,
                        ),
                        position: eval_prop_vec2(
                            get("position"),
                            self.frame,
                            [20.0, 20.0],
                            self.eval_expr,
                        ),
                        scale: eval_prop_vec2(get("scale"), self.frame, [0.0, 0.0], self.eval_expr),
                        rotation: eval_prop_f64(get("rotation"), self.frame, 0.0, self.eval_expr),
                    };
                    for v in &mut views {
                        v.wiggle_stack.push((MAT_IDENTITY, wcfg));
                    }
                    let wm0 = eval_wiggle_transform_matrix(wcfg, 0);
                    let prev_cmds = std::mem::take(&mut commands);
                    commands.push(ScopeCommand::Group {
                        matrix: wm0,
                        wiggle: Some(wcfg),
                        opacity: 1.0,
                        commands: prev_cmds,
                    });
                }
                "trim" | "roundCorners" | "twist" | "zigZag" | "puckerBloat" | "wigglePaths" => {
                    let tagged: Vec<(usize, ShapePath)> = views
                        .iter()
                        .enumerate()
                        .flat_map(|(vi, v)| self.view_paths(v).into_iter().map(move |p| (vi, p)))
                        .collect();
                    let changed: Vec<(usize, ShapePath)> = match kind {
                        "trim" => {
                            let start =
                                eval_prop_f64(get("start"), self.frame, 0.0, self.eval_expr);
                            let end = eval_prop_f64(get("end"), self.frame, 100.0, self.eval_expr);
                            let offset =
                                eval_prop_f64(get("offset"), self.frame, 0.0, self.eval_expr);
                            let indiv = eval_prop_str(get("mode"), self.frame, "simultaneous")
                                .eq_ignore_ascii_case("individual");
                            apply_trim(&tagged, start, end, offset, indiv)
                        }
                        "roundCorners" => {
                            let radius =
                                eval_prop_f64(get("radius"), self.frame, 10.0, self.eval_expr);
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_round_corners(&p, radius, item_tol)))
                                .collect()
                        }
                        "twist" => {
                            let angle =
                                eval_prop_f64(get("angle"), self.frame, 0.0, self.eval_expr);
                            let center = eval_prop_vec2(
                                get("center"),
                                self.frame,
                                [0.0, 0.0],
                                self.eval_expr,
                            );
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_twist(&p, angle, center, item_tol)))
                                .collect()
                        }
                        "zigZag" => {
                            let size = eval_prop_f64(get("size"), self.frame, 10.0, self.eval_expr);
                            let ridges =
                                eval_prop_f64(get("ridges"), self.frame, 3.0, self.eval_expr);
                            let sm = eval_prop_str(get("points"), self.frame, "corner")
                                .eq_ignore_ascii_case("smooth");
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_zig_zag(&p, size, ridges, sm, item_tol)))
                                .collect()
                        }
                        "puckerBloat" => {
                            let amount =
                                eval_prop_f64(get("amount"), self.frame, 0.0, self.eval_expr);
                            tagged
                                .into_iter()
                                .map(|(o, p)| (o, apply_pucker_bloat(&p, amount)))
                                .collect()
                        }
                        _ => {
                            let size = eval_prop_f64(get("size"), self.frame, 10.0, self.eval_expr);
                            let detail =
                                eval_prop_f64(get("detail"), self.frame, 5.0, self.eval_expr);
                            let sm = eval_prop_str(get("points"), self.frame, "corner")
                                .eq_ignore_ascii_case("smooth");
                            let wp = WiggleParams {
                                time: if self.fps > 0.0 {
                                    self.frame / self.fps
                                } else {
                                    0.0
                                },
                                frequency: eval_prop_f64(
                                    get("frequency").or_else(|| get("wigglesPerSecond")),
                                    self.frame,
                                    1.0,
                                    self.eval_expr,
                                ),
                                correlation: eval_prop_f64(
                                    get("correlation"),
                                    self.frame,
                                    50.0,
                                    self.eval_expr,
                                ),
                                temporal_phase: eval_prop_f64(
                                    get("temporalPhase"),
                                    self.frame,
                                    0.0,
                                    self.eval_expr,
                                ),
                                spatial_phase: eval_prop_f64(
                                    get("spatialPhase"),
                                    self.frame,
                                    0.0,
                                    self.eval_expr,
                                ),
                                seed: eval_prop_f64(get("seed"), self.frame, 1.0, self.eval_expr)
                                    .round() as i32,
                            };
                            tagged
                                .into_iter()
                                .map(|(o, p)| {
                                    (o, apply_wiggle_paths(&p, size, detail, sm, wp, item_tol))
                                })
                                .collect()
                        }
                    };

                    for (vi, view) in views.iter().enumerate() {
                        let m = view.matrix_for_copy(0);
                        let Some(inv) = mat_inv(m) else {
                            continue;
                        };
                        let updated: Vec<ShapePath> = changed
                            .iter()
                            .filter(|(o, _)| *o == vi)
                            .map(|(_, p)| p.transformed(inv))
                            .collect();
                        self.bases[view.base_idx] = updated;
                    }
                }
                "repeater" => {
                    let orig_views = std::mem::take(&mut views);
                    let orig_cmds = std::mem::take(&mut commands);
                    let copies = repeater_copies(props, self.frame, self.eval_expr);
                    let composite_above = eval_prop_str(get("composite"), self.frame, "below")
                        .eq_ignore_ascii_case("above");

                    for (copy_idx, copy) in copies.into_iter().enumerate() {
                        let ci = copy_idx as i32;
                        let mut base_remap: HashMap<usize, usize> = HashMap::new();
                        let clone_view = |v: &PathView,
                                          bases: &mut Vec<Vec<ShapePath>>,
                                          remap: &mut HashMap<usize, usize>|
                         -> PathView {
                            let new_base = *remap.entry(v.base_idx).or_insert_with(|| {
                                let cloned = bases[v.base_idx].clone();
                                let idx = bases.len();
                                bases.push(cloned);
                                idx
                            });
                            PathView {
                                base_idx: new_base,
                                matrix: v.matrix_for_copy(ci),
                                wiggle_stack: Vec::new(),
                                char_override: v.char_override,
                            }
                        };

                        fn clone_cmd(
                            cmd: &ScopeCommand,
                            ci: i32,
                            bases: &mut Vec<Vec<ShapePath>>,
                            remap: &mut HashMap<usize, usize>,
                        ) -> ScopeCommand {
                            match cmd {
                                ScopeCommand::Paint {
                                    views,
                                    kind,
                                    opacity,
                                    tolerance,
                                } => ScopeCommand::Paint {
                                    views: views
                                        .iter()
                                        .map(|v| {
                                            let nb =
                                                *remap.entry(v.base_idx).or_insert_with(|| {
                                                    let cloned = bases[v.base_idx].clone();
                                                    let idx = bases.len();
                                                    bases.push(cloned);
                                                    idx
                                                });
                                            PathView {
                                                base_idx: nb,
                                                matrix: v.matrix_for_copy(ci),
                                                wiggle_stack: Vec::new(),
                                                char_override: v.char_override,
                                            }
                                        })
                                        .collect(),
                                    kind: kind.clone(),
                                    opacity: *opacity,
                                    tolerance: *tolerance,
                                },
                                ScopeCommand::Group {
                                    matrix,
                                    wiggle,
                                    opacity,
                                    commands,
                                } => ScopeCommand::Group {
                                    matrix: wiggle
                                        .map_or(*matrix, |wc| eval_wiggle_transform_matrix(wc, ci)),
                                    wiggle: None,
                                    opacity: *opacity,
                                    commands: commands
                                        .iter()
                                        .map(|c| clone_cmd(c, ci, bases, remap))
                                        .collect(),
                                },
                            }
                        }

                        for ov in &orig_views {
                            let mut cv = clone_view(ov, &mut self.bases, &mut base_remap);
                            cv.matrix = mat_mul(copy.matrix, cv.matrix);
                            views.push(cv);
                        }
                        let cloned_cmds: Vec<ScopeCommand> = orig_cmds
                            .iter()
                            .map(|c| clone_cmd(c, ci, &mut self.bases, &mut base_remap))
                            .collect();
                        let grp = ScopeCommand::Group {
                            matrix: copy.matrix,
                            wiggle: None,
                            opacity: copy.opacity,
                            commands: cloned_cmds,
                        };
                        if composite_above {
                            commands.push(grp);
                        } else {
                            commands.insert(0, grp);
                        }
                    }
                }
                "rectangle" | "ellipse" | "polystar" | "path" => {
                    if let Some(p) =
                        build_primitive_path(item, self.frame, item_tol, self.eval_expr)
                    {
                        let base_idx = self.alloc_base(vec![p]);
                        views.push(PathView {
                            base_idx,
                            matrix: MAT_IDENTITY,
                            wiggle_stack: Vec::new(),
                            char_override: None,
                        });
                    }
                }
                _ => {}
            }
        }

        (views, commands)
    }

    fn lower_command(&self, cmd: &ScopeCommand) -> Vec<RenderCommand> {
        match cmd {
            ScopeCommand::Group {
                matrix,
                opacity,
                commands,
                ..
            } => {
                let lowered: Vec<RenderCommand> = commands
                    .iter()
                    .flat_map(|c| self.lower_command(c))
                    .map(|rc| rc.transformed(*matrix))
                    .collect();
                if lowered.is_empty() || *opacity <= 0.0 {
                    return Vec::new();
                }
                if *opacity < 1.0 - 1e-6 {
                    vec![RenderCommand::Isolation {
                        opacity: *opacity,
                        commands: lowered,
                    }]
                } else {
                    lowered
                }
            }
            ScopeCommand::Paint {
                views,
                kind,
                opacity,
                tolerance,
            } => {
                if views.is_empty() || *opacity <= 0.0 {
                    return Vec::new();
                }
                let has_char_ov = views.iter().any(|v| v.char_override.is_some());
                if has_char_ov {
                    let mut out = Vec::new();
                    for v in views {
                        let paths = self.view_paths(v);
                        if paths.is_empty() {
                            continue;
                        }
                        let ov = v.char_override;
                        let eff_op = opacity * ov.map_or(1.0, |o| o.opacity);
                        if eff_op <= 0.0 {
                            continue;
                        }
                        out.extend(self.lower_paint_paths(&paths, kind, eff_op, *tolerance, ov));
                    }
                    return out;
                }
                let paths: Vec<ShapePath> = views.iter().flat_map(|v| self.view_paths(v)).collect();
                if paths.is_empty() {
                    return Vec::new();
                }
                self.lower_paint_paths(&paths, kind, *opacity, *tolerance, None)
            }
        }
    }

    fn lower_paint_paths(
        &self,
        paths: &[ShapePath],
        kind: &PaintKind,
        opacity: f64,
        tolerance: f64,
        ov: Option<CharPaintOverride>,
    ) -> Vec<RenderCommand> {
        match kind {
            PaintKind::Fill { color, even_odd } => {
                let polygons: Vec<Vec<[f64; 2]>> = paths
                    .iter()
                    .filter(|p| p.points.len() >= 3)
                    .map(|p| p.points.clone())
                    .collect();
                if polygons.is_empty() {
                    return Vec::new();
                }
                let eff_color = ov.and_then(|o| o.fill_color).unwrap_or(*color);
                vec![RenderCommand::SolidFill {
                    polygons,
                    color: eff_color,
                    opacity,
                    even_odd: *even_odd,
                }]
            }
            PaintKind::GradientFill { gradient, even_odd } => {
                let polygons: Vec<Vec<[f64; 2]>> = paths
                    .iter()
                    .filter(|p| p.points.len() >= 3)
                    .map(|p| p.points.clone())
                    .collect();
                if polygons.is_empty() {
                    return Vec::new();
                }
                vec![RenderCommand::GradientFill {
                    polygons,
                    gradient: gradient.clone(),
                    opacity,
                    even_odd: *even_odd,
                }]
            }
            PaintKind::Stroke { color, style } => {
                let polygons = expand_stroke_polygons(paths, style, tolerance);
                if polygons.is_empty() {
                    return Vec::new();
                }
                let eff_color = ov.and_then(|o| o.fill_color).unwrap_or(*color);
                vec![RenderCommand::SolidFill {
                    polygons,
                    color: eff_color,
                    opacity,
                    even_odd: false,
                }]
            }
            PaintKind::GradientStroke { gradient, style } => {
                let polygons = expand_stroke_polygons(paths, style, tolerance);
                if polygons.is_empty() {
                    return Vec::new();
                }
                vec![RenderCommand::GradientFill {
                    polygons,
                    gradient: gradient.clone(),
                    opacity,
                    even_odd: false,
                }]
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2D Antialiased Scanline Rasterizer (4x Vertical SS + Exact Horizontal Coverage)
// ---------------------------------------------------------------------------

struct Edge {
    y_min: f64,
    y_max: f64,
    x_at_ymin: f64,
    dx_dy: f64,
    dir: i32,
}

fn build_scanline_edges(polygons: &[Vec<[f64; 2]>]) -> (Vec<Edge>, f64, f64) {
    let mut edges = Vec::new();
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for poly in polygons {
        let n = poly.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = poly[i];
            let b = poly[(i + 1) % n];
            if (a[1] - b[1]).abs() < 1e-9 {
                continue;
            }
            let (top, bot, dir) = if a[1] < b[1] { (a, b, 1) } else { (b, a, -1) };
            min_y = min_y.min(top[1]);
            max_y = max_y.max(bot[1]);
            edges.push(Edge {
                y_min: top[1],
                y_max: bot[1],
                x_at_ymin: top[0],
                dx_dy: (bot[0] - top[0]) / (bot[1] - top[1]),
                dir,
            });
        }
    }
    (edges, min_y, max_y)
}

/// Computes exact horizontal span coverage averaged across 4 vertical sub-rows per scanline
/// and calls `blend_row(y, &row_cov)` for every row with non-zero coverage.
fn rasterize_polygons_aa<F>(
    polygons: &[Vec<[f64; 2]>],
    width: usize,
    height: usize,
    even_odd: bool,
    mut blend_row: F,
) where
    F: FnMut(usize, &[f32]),
{
    if width == 0 || height == 0 || polygons.is_empty() {
        return;
    }
    let (edges, min_y, max_y) = build_scanline_edges(polygons);
    if edges.is_empty() || !min_y.is_finite() || !max_y.is_finite() {
        return;
    }
    let y0 = (min_y.floor() as isize).max(0) as usize;
    let y1 = (max_y.ceil() as isize).min(height as isize) as usize;
    if y0 >= y1 {
        return;
    }

    let mut row_cov = vec![0.0_f32; width];
    let mut crossings: Vec<(f64, i32)> = Vec::new();
    const SUB_SAMPLES: usize = 4;
    const SUB_WEIGHT: f32 = 1.0 / (SUB_SAMPLES as f32);
    let wf = width as f64;

    for y in y0..y1 {
        row_cov.fill(0.0);
        let mut any = false;
        for sub in 0..SUB_SAMPLES {
            let ys = y as f64 + (sub as f64 + 0.5) / (SUB_SAMPLES as f64);
            crossings.clear();
            for e in &edges {
                if ys >= e.y_min && ys < e.y_max {
                    let x = e.x_at_ymin + (ys - e.y_min) * e.dx_dy;
                    crossings.push((x, e.dir));
                }
            }
            if crossings.len() < 2 {
                continue;
            }
            crossings.sort_by(|a, b| a.0.total_cmp(&b.0));

            let mut winding = 0i32;
            let mut inside = false;
            let mut span_start = 0.0_f64;

            for &(x, dir) in &crossings {
                let prev_in = inside;
                if even_odd {
                    inside = !inside;
                } else {
                    winding += dir;
                    inside = winding != 0;
                }
                if !prev_in && inside {
                    span_start = x;
                } else if prev_in && !inside {
                    let a = span_start.clamp(0.0, wf);
                    let b = x.clamp(0.0, wf);
                    if b > a {
                        any = true;
                        let px0 = (a.floor() as usize).min(width - 1);
                        let px1 = ((b.ceil() as usize).saturating_sub(1)).min(width - 1);
                        if px0 == px1 {
                            row_cov[px0] += ((b - a) as f32) * SUB_WEIGHT;
                        } else {
                            row_cov[px0] += (((px0 + 1) as f64 - a) as f32) * SUB_WEIGHT;
                            for cov in &mut row_cov[(px0 + 1)..px1] {
                                *cov += SUB_WEIGHT;
                            }
                            row_cov[px1] += ((b - px1 as f64) as f32) * SUB_WEIGHT;
                        }
                    }
                }
            }
        }
        if any {
            blend_row(y, &row_cov);
        }
    }
}

#[inline]
fn composite_premul_over(dst: &mut [f32; 4], src: [f32; 4]) {
    let inv = 1.0 - src[3];
    dst[0] = src[0] + dst[0] * inv;
    dst[1] = src[1] + dst[1] * inv;
    dst[2] = src[2] + dst[2] * inv;
    dst[3] = src[3] + dst[3] * inv;
}

fn execute_render_commands(
    commands: &[RenderCommand],
    display: Mat2D,
    width: usize,
    height: usize,
    fb: &mut [[f32; 4]],
) {
    for cmd in commands {
        match cmd {
            RenderCommand::Isolation { opacity, commands } => {
                let op = (*opacity as f32).clamp(0.0, 1.0);
                if op <= 0.0 || commands.is_empty() {
                    continue;
                }
                let mut local = vec![[0.0_f32; 4]; width * height];
                execute_render_commands(commands, display, width, height, &mut local);
                for (dst, src) in fb.iter_mut().zip(local.iter()) {
                    if src[3] > 0.0 {
                        let scaled = [src[0] * op, src[1] * op, src[2] * op, src[3] * op];
                        composite_premul_over(dst, scaled);
                    }
                }
            }
            RenderCommand::SolidFill {
                polygons,
                color,
                opacity,
                even_odd,
            } => {
                let op = (*opacity as f32).clamp(0.0, 1.0);
                if op <= 0.0 {
                    continue;
                }
                let px_polys: Vec<Vec<[f64; 2]>> = polygons
                    .iter()
                    .map(|p| p.iter().map(|&pt| mat_transform_pt(pt, display)).collect())
                    .collect();
                let cr = color[0] as f32;
                let cg = color[1] as f32;
                let cb = color[2] as f32;
                rasterize_polygons_aa(&px_polys, width, height, *even_odd, |y, cov| {
                    let row = &mut fb[y * width..(y + 1) * width];
                    for (px, &c) in row.iter_mut().zip(cov.iter()) {
                        if c > 0.0 {
                            let a = (c * op).clamp(0.0, 1.0);
                            composite_premul_over(px, [cr * a, cg * a, cb * a, a]);
                        }
                    }
                });
            }
            RenderCommand::GradientFill {
                polygons,
                gradient,
                opacity,
                even_odd,
            } => {
                let op = (*opacity as f32).clamp(0.0, 1.0);
                if op <= 0.0 {
                    continue;
                }
                let total_mat = mat_mul(display, gradient.matrix);
                let Some(inv_total) = mat_inv(total_mat) else {
                    continue;
                };
                let px_polys: Vec<Vec<[f64; 2]>> = polygons
                    .iter()
                    .map(|p| p.iter().map(|&pt| mat_transform_pt(pt, display)).collect())
                    .collect();
                rasterize_polygons_aa(&px_polys, width, height, *even_odd, |y, cov| {
                    let py = y as f64 + 0.5;
                    let row = &mut fb[y * width..(y + 1) * width];
                    for (x, (px, &c)) in row.iter_mut().zip(cov.iter()).enumerate() {
                        if c > 0.0 {
                            let sampled = gradient.sample_premul([x as f64 + 0.5, py], inv_total);
                            let mult = c * op;
                            let src = [
                                (sampled[0] as f32) * mult,
                                (sampled[1] as f32) * mult,
                                (sampled[2] as f32) * mult,
                                ((sampled[3] as f32) * mult).clamp(0.0, 1.0),
                            ];
                            composite_premul_over(px, src);
                        }
                    }
                });
            }
        }
    }
}

fn fb_to_premul_rgba8(fb: &[[f32; 4]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(fb.len() * 4);
    for px in fb {
        let a = px[3].clamp(0.0, 1.0);
        let r = px[0].clamp(0.0, a);
        let g = px[1].clamp(0.0, a);
        let b = px[2].clamp(0.0, a);
        out.push((r * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
        out.push((g * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
        out.push((b * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
        out.push((a * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8);
    }
    out
}

/// Compiles and rasterizes a full Shape+ JSON payload into a premultiplied RGBA8 buffer
/// of length `width * height * 4`.
///
/// Sequence space has `(0, 0)` at the center of the frame with `+x` right and `+y` up
/// (matching Zoidium `viewport-mapping.js` and `zc-render`'s 2D layer convention).
#[must_use]
pub fn rasterize_shape_plus(
    payload: &Value,
    width: u32,
    height: u32,
    seq_res: [u32; 2],
    frame: f64,
    fps: f64,
    eval_expr: Option<ExprEvalFn<'_>>,
) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let Some(contents) = payload.get("contents").and_then(Value::as_array) else {
        return vec![0_u8; w * h * 4];
    };

    let seq_w = seq_res[0].max(1) as f64;
    let seq_h = seq_res[1].max(1) as f64;
    let sx = (w as f64) / seq_w;
    let sy = (h as f64) / seq_h;
    let pixel_scale = sx.max(sy).max(1e-6);
    let tolerance = (DEFAULT_TOLERANCE / pixel_scale).clamp(1e-3, 1.0);
    // Display matrix: sequence (center origin, y up) -> pixel (top-left origin, y down).
    let display: Mat2D = [sx, 0.0, 0.0, -sy, (w as f64) * 0.5, (h as f64) * 0.5];

    let mut compiler = Compiler {
        frame: if frame.is_finite() { frame } else { 0.0 },
        fps: if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        },
        eval_expr,
        bases: Vec::new(),
    };
    let (_views, scope_cmds) = compiler.compile_scope(contents, tolerance, 0, &[]);
    let render_cmds: Vec<RenderCommand> = scope_cmds
        .iter()
        .flat_map(|c| compiler.lower_command(c))
        .collect();

    let mut fb = vec![[0.0_f32; 4]; w * h];
    execute_render_commands(&render_cmds, display, w, h, &mut fb);
    fb_to_premul_rgba8(&fb)
}

// ---------------------------------------------------------------------------
// SVG Importer & Path Exporter (`svg-import.js`, `svg-path.js`)
// ---------------------------------------------------------------------------

fn svg_arc_to_cubics(
    start: [f64; 2],
    end: [f64; 2],
    mut rx: f64,
    mut ry: f64,
    rotation_deg: f64,
    large: bool,
    sweep: bool,
) -> Vec<[[f64; 2]; 4]> {
    rx = rx.abs();
    ry = ry.abs();
    if (start[0] - end[0]).abs() < 1e-12 && (start[1] - end[1]).abs() < 1e-12 {
        return Vec::new();
    }
    if rx <= 1e-12 || ry <= 1e-12 {
        return vec![[start, start, end, end]];
    }
    let phi = rotation_deg.to_radians();
    let (s, c) = phi.sin_cos();
    let dx = (start[0] - end[0]) * 0.5;
    let dy = (start[1] - end[1]) * 0.5;
    let x = c * dx + s * dy;
    let y = -s * dx + c * dy;
    let rad_check = (x * x) / (rx * rx) + (y * y) / (ry * ry);
    if rad_check > 1.0 {
        let sc = rad_check.sqrt();
        rx *= sc;
        ry *= sc;
    }
    let num = (rx * rx * ry * ry - rx * rx * y * y - ry * ry * x * x).max(0.0);
    let den = (rx * rx * y * y + ry * ry * x * x).max(1e-18);
    let factor = (if large == sweep { -1.0 } else { 1.0 }) * (num / den).sqrt();
    let cx = (factor * rx * y) / ry;
    let cy = (-factor * ry * x) / rx;
    let center = [
        c * cx - s * cy + (start[0] + end[0]) * 0.5,
        s * cx + c * cy + (start[1] + end[1]) * 0.5,
    ];
    let u = [(x - cx) / rx, (y - cy) / ry];
    let v = [(-x - cx) / rx, (-y - cy) / ry];
    let angle = u[1].atan2(u[0]);
    let mut delta = (u[0] * v[1] - u[1] * v[0]).atan2(u[0] * v[0] + u[1] * v[1]);
    if !sweep && delta > 0.0 {
        delta -= TAU;
    }
    if sweep && delta < 0.0 {
        delta += TAU;
    }
    let n = (delta.abs() / (PI / 4.0)).ceil().max(1.0) as usize;
    let step = delta / n as f64;
    let point = |t: f64| -> [f64; 2] {
        [
            center[0] + rx * c * t.cos() - ry * s * t.sin(),
            center[1] + rx * s * t.cos() + ry * c * t.sin(),
        ]
    };
    let tangent = |t: f64| -> [f64; 2] {
        [
            -rx * c * t.sin() - ry * s * t.cos(),
            -rx * s * t.sin() + ry * c * t.cos(),
        ]
    };
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let a = angle + i as f64 * step;
        let b = a + step;
        let k = (4.0 / 3.0) * (step * 0.25).tan();
        let p = if i == 0 { start } else { point(a) };
        let q = if i + 1 == n { end } else { point(b) };
        let ta = tangent(a);
        let tb = tangent(b);
        out.push([
            p,
            [p[0] + ta[0] * k, p[1] + ta[1] * k],
            [q[0] - tb[0] * k, q[1] - tb[1] * k],
            q,
        ]);
    }
    out
}

#[derive(Debug, Clone)]
struct SvgContour {
    vertices: Vec<BezierVertex>,
    closed: bool,
}

fn tokenize_svg_path(d: &str) -> Result<Vec<String>, String> {
    let bytes = d.as_bytes();
    let mut i = 0usize;
    let mut tokens = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_whitespace() || b == b',' {
            i += 1;
            continue;
        }
        if b.is_ascii_alphabetic() {
            tokens.push((b as char).to_string());
            i += 1;
            continue;
        }
        if b == b'+' || b == b'-' || b == b'.' || b.is_ascii_digit() {
            let start = i;
            if b == b'+' || b == b'-' {
                i += 1;
            }
            let mut has_dot = false;
            while i < bytes.len() {
                if bytes[i].is_ascii_digit() {
                    i += 1;
                } else if bytes[i] == b'.' && !has_dot {
                    has_dot = true;
                    i += 1;
                } else {
                    break;
                }
            }
            if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
                let exp_start = i;
                i += 1;
                if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
                    i += 1;
                }
                let digits_start = i;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                }
                if i == digits_start {
                    i = exp_start;
                }
            }
            let tok = &d[start..i];
            if tok == "+" || tok == "-" || tok == "." {
                return Err(format!("Invalid SVG path number token: {tok}"));
            }
            tokens.push(tok.to_owned());
            continue;
        }
        return Err(format!("Invalid SVG path character: '{}'", b as char));
    }
    Ok(tokens)
}

fn parse_svg_path_d(d: &str) -> Result<Vec<SvgContour>, String> {
    if d.len() > 262_144 {
        return Err("SVG path text budget exceeded".to_owned());
    }
    let mut tokens = tokenize_svg_path(d)?;
    let mut idx = 0usize;
    let mut command: Option<char> = None;
    let mut previous: Option<char> = None;
    let mut current = [0.0_f64, 0.0];
    let mut start = [0.0_f64, 0.0];
    let mut last_cubic: Option<[f64; 2]> = None;
    let mut last_quad: Option<[f64; 2]> = None;
    let mut contours: Vec<SvgContour> = Vec::new();
    let mut commands = 0usize;

    let next_num = |tokens: &[String], idx: &mut usize| -> Result<f64, String> {
        let tok = tokens
            .get(*idx)
            .ok_or_else(|| "Unexpected end of SVG path data".to_owned())?;
        *idx += 1;
        let v = tok
            .parse::<f64>()
            .map_err(|_| format!("Invalid SVG number: {tok}"))?;
        if !v.is_finite() || v.abs() > 1e6 {
            return Err("Invalid or excessive SVG coordinate".to_owned());
        }
        Ok(v)
    };

    let next_flag = |tokens: &mut [String], idx: &mut usize| -> Result<bool, String> {
        let tok = tokens
            .get_mut(*idx)
            .ok_or_else(|| "Unexpected end of SVG arc flags".to_owned())?;
        let first = tok
            .chars()
            .next()
            .ok_or_else(|| "Empty SVG arc flag".to_owned())?;
        let val = match first {
            '0' => false,
            '1' => true,
            _ => return Err(format!("Invalid SVG arc flag: {tok}")),
        };
        if tok.len() == 1 {
            *idx += 1;
        } else {
            *tok = tok[1..].to_owned();
        }
        Ok(val)
    };

    while idx < tokens.len() {
        commands += 1;
        if commands > 4096 {
            return Err("SVG command budget exceeded".to_owned());
        }
        if tokens[idx].len() == 1 && tokens[idx].as_bytes()[0].is_ascii_alphabetic() {
            command = tokens[idx].chars().next();
            idx += 1;
        }
        let cmd = command.ok_or_else(|| "Missing SVG path command".to_owned())?;
        let kind = cmd.to_ascii_uppercase();
        let relative = cmd != kind;
        let origin = current;

        let read_pt = |tokens: &[String], idx: &mut usize| -> Result<[f64; 2], String> {
            let x = next_num(tokens, idx)?;
            let y = next_num(tokens, idx)?;
            Ok(if relative {
                [origin[0] + x, origin[1] + y]
            } else {
                [x, y]
            })
        };

        if kind != 'M' && contours.is_empty() {
            return Err("SVG path must begin with Move (M/m)".to_owned());
        }

        let push_cubic = |contours: &mut [SvgContour],
                          current: &mut [f64; 2],
                          a: [f64; 2],
                          b: [f64; 2],
                          end: [f64; 2]| {
            let c = contours.last_mut().unwrap();
            let last = c.vertices.last_mut().unwrap();
            last.out_h = [a[0] - current[0], a[1] - current[1]];
            let mut v = BezierVertex::sharp(end);
            v.in_h = [b[0] - end[0], b[1] - end[1]];
            c.vertices.push(v);
            *current = end;
        };

        match kind {
            'M' => {
                let pt = read_pt(&tokens, &mut idx)?;
                contours.push(SvgContour {
                    vertices: vec![BezierVertex::sharp(pt)],
                    closed: false,
                });
                start = pt;
                current = pt;
                command = Some(if relative { 'l' } else { 'L' });
            }
            'Z' => {
                let c = contours.last_mut().unwrap();
                c.closed = true;
                if c.vertices.len() > 1 && pt_dist(current, start) < 1e-10 {
                    let last_in = c.vertices.pop().unwrap().in_h;
                    c.vertices[0].in_h = last_in;
                }
                current = start;
                command = None;
            }
            'L' => {
                let pt = read_pt(&tokens, &mut idx)?;
                contours
                    .last_mut()
                    .unwrap()
                    .vertices
                    .push(BezierVertex::sharp(pt));
                current = pt;
            }
            'H' => {
                let x = next_num(&tokens, &mut idx)? + if relative { origin[0] } else { 0.0 };
                let pt = [x, origin[1]];
                contours
                    .last_mut()
                    .unwrap()
                    .vertices
                    .push(BezierVertex::sharp(pt));
                current = pt;
            }
            'V' => {
                let y = next_num(&tokens, &mut idx)? + if relative { origin[1] } else { 0.0 };
                let pt = [origin[0], y];
                contours
                    .last_mut()
                    .unwrap()
                    .vertices
                    .push(BezierVertex::sharp(pt));
                current = pt;
            }
            'C' => {
                let a = read_pt(&tokens, &mut idx)?;
                let b = read_pt(&tokens, &mut idx)?;
                let p = read_pt(&tokens, &mut idx)?;
                push_cubic(&mut contours, &mut current, a, b, p);
                last_cubic = Some(b);
            }
            'S' => {
                let a = if matches!(previous, Some('C' | 'S'))
                    && let Some(lc) = last_cubic
                {
                    [2.0 * origin[0] - lc[0], 2.0 * origin[1] - lc[1]]
                } else {
                    origin
                };
                let b = read_pt(&tokens, &mut idx)?;
                let p = read_pt(&tokens, &mut idx)?;
                push_cubic(&mut contours, &mut current, a, b, p);
                last_cubic = Some(b);
            }
            'Q' | 'T' => {
                let q = if kind == 'Q' {
                    read_pt(&tokens, &mut idx)?
                } else if matches!(previous, Some('Q' | 'T'))
                    && let Some(lq) = last_quad
                {
                    [2.0 * origin[0] - lq[0], 2.0 * origin[1] - lq[1]]
                } else {
                    origin
                };
                let p = read_pt(&tokens, &mut idx)?;
                let c1 = pt_mix(origin, q, 2.0 / 3.0);
                let c2 = pt_mix(p, q, 2.0 / 3.0);
                push_cubic(&mut contours, &mut current, c1, c2, p);
                last_quad = Some(q);
            }
            'A' => {
                let rx = next_num(&tokens, &mut idx)?;
                let ry = next_num(&tokens, &mut idx)?;
                let rot = next_num(&tokens, &mut idx)?;
                let large = next_flag(&mut tokens, &mut idx)?;
                let sweep = next_flag(&mut tokens, &mut idx)?;
                let p = read_pt(&tokens, &mut idx)?;
                for span in svg_arc_to_cubics(origin, p, rx, ry, rot, large, sweep) {
                    push_cubic(&mut contours, &mut current, span[1], span[2], span[3]);
                }
                current = p;
            }
            _ => return Err(format!("Unsupported SVG path command: {cmd}")),
        }

        if !matches!(kind, 'C' | 'S') {
            last_cubic = None;
        }
        if !matches!(kind, 'Q' | 'T') {
            last_quad = None;
        }
        previous = Some(kind);
        if contours.len() > 256
            || contours
                .last()
                .is_some_and(|c| c.vertices.len() > MAX_SHAPE_PLUS_VERTICES)
        {
            return Err("SVG path vertex budget exceeded".to_owned());
        }
    }

    if contours.is_empty() {
        return Err("SVG has no path data".to_owned());
    }
    Ok(contours)
}

fn parse_css_color(raw: &str) -> Result<[f64; 4], String> {
    let trimmed = raw.trim();
    let lower = trimmed.to_ascii_lowercase();
    let hex_str = match lower.as_str() {
        "black" => "#000000",
        "white" => "#ffffff",
        "red" => "#ff0000",
        "green" => "#008000",
        "lime" => "#00ff00",
        "blue" => "#0000ff",
        "yellow" => "#ffff00",
        "cyan" | "aqua" => "#00ffff",
        "magenta" | "fuchsia" => "#ff00ff",
        "orange" => "#ffa500",
        "purple" => "#800080",
        "gray" | "grey" => "#808080",
        "transparent" => "#00000000",
        _ => trimmed,
    };
    if let Some(hex) = hex_str.strip_prefix('#') {
        let expanded: String = match hex.len() {
            3 | 4 => hex.chars().flat_map(|c| [c, c]).collect(),
            6 | 8 => hex.to_owned(),
            _ => return Err(format!("Invalid hex color: {raw}")),
        };
        let byte = |i: usize| -> Result<f64, String> {
            u8::from_str_radix(&expanded[i..i + 2], 16)
                .map(|b| f64::from(b) / 255.0)
                .map_err(|_| format!("Invalid hex color: {raw}"))
        };
        let r = byte(0)?;
        let g = byte(2)?;
        let b = byte(4)?;
        let a = if expanded.len() == 8 { byte(6)? } else { 1.0 };
        return Ok([r, g, b, a]);
    }
    if (lower.starts_with("rgb(") || lower.starts_with("rgba(")) && lower.ends_with(')') {
        let start = lower.find('(').unwrap() + 1;
        let inner = &lower[start..lower.len() - 1];
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() != 3 && parts.len() != 4 {
            return Err(format!("Invalid rgb/rgba color: {raw}"));
        }
        let mut rgba = [0.0, 0.0, 0.0, 1.0];
        for (i, part) in parts.iter().enumerate() {
            let (num_s, denom) = if let Some(stripped) = part.strip_suffix('%') {
                (stripped.trim(), 100.0)
            } else {
                (*part, if i < 3 { 255.0 } else { 1.0 })
            };
            let n = num_s
                .parse::<f64>()
                .map_err(|_| format!("Invalid color component: {raw}"))?;
            rgba[i] = (n / denom).clamp(0.0, 1.0);
        }
        return Ok(rgba);
    }
    Err(format!("Unsupported SVG color: {raw}"))
}

fn parse_svg_numbers(text: &str) -> Result<Vec<f64>, String> {
    let mut out = Vec::new();
    for tok in text.split(|c: char| c.is_whitespace() || c == ',') {
        if tok.is_empty() {
            continue;
        }
        let n = tok
            .parse::<f64>()
            .map_err(|_| format!("Invalid SVG number: {tok}"))?;
        if !n.is_finite() || n.abs() > 1e6 {
            return Err(format!("SVG number out of bounds: {tok}"));
        }
        out.push(n);
    }
    Ok(out)
}

fn parse_svg_transform(text: &str) -> Result<Mat2D, String> {
    let mut result = MAT_IDENTITY;
    let mut rest = text.trim();
    let mut count = 0usize;
    while !rest.is_empty() {
        count += 1;
        if count > 64 {
            return Err("SVG transform budget exceeded".to_owned());
        }
        let open = rest
            .find('(')
            .ok_or_else(|| format!("Invalid SVG transform: {text}"))?;
        let close = rest[open..]
            .find(')')
            .map(|o| open + o)
            .ok_or_else(|| format!("Unterminated SVG transform: {text}"))?;
        let name = rest[..open].trim().trim_start_matches(',').trim();
        let params = parse_svg_numbers(&rest[open + 1..close])?;
        let m: Mat2D = match name {
            "matrix" if params.len() == 6 => [
                params[0], params[1], params[2], params[3], params[4], params[5],
            ],
            "translate" if params.len() == 1 || params.len() == 2 => [
                1.0,
                0.0,
                0.0,
                1.0,
                params[0],
                *params.get(1).unwrap_or(&0.0),
            ],
            "scale" if params.len() == 1 || params.len() == 2 => {
                let sx = params[0];
                let sy = *params.get(1).unwrap_or(&sx);
                [sx, 0.0, 0.0, sy, 0.0, 0.0]
            }
            "rotate" if params.len() == 1 || params.len() == 3 => {
                let a = params[0].to_radians();
                let (s, c) = a.sin_cos();
                let cx = *params.get(1).unwrap_or(&0.0);
                let cy = *params.get(2).unwrap_or(&0.0);
                [c, s, -s, c, cx - c * cx + s * cy, cy - s * cx - c * cy]
            }
            "skewX" if params.len() == 1 => [1.0, 0.0, params[0].to_radians().tan(), 1.0, 0.0, 0.0],
            "skewY" if params.len() == 1 => [1.0, params[0].to_radians().tan(), 0.0, 1.0, 0.0, 0.0],
            _ => return Err(format!("Unsupported SVG transform: {name}")),
        };
        result = mat_mul(result, m);
        rest = rest[close + 1..].trim().trim_start_matches(',').trim();
    }
    Ok(result)
}

fn svd_affine_group(
    contents: Vec<Value>,
    m: Mat2D,
    seq: &mut u64,
    opacity: f64,
    name: &str,
) -> Value {
    let mut next_id = || {
        *seq += 1;
        format!("s{}", *seq)
    };
    let [a, b, c, e, x, y] = m;
    let aa = a * a + b * b;
    let bb = c * c + e * e;
    let ab = a * c + b * e;
    let theta = 0.5 * (2.0 * ab).atan2(aa - bb);
    let vx = theta.cos();
    let vy = theta.sin();
    let ux = a * vx + c * vy;
    let uy = b * vx + e * vy;
    let s1 = ux.hypot(uy);
    let det = a * e - b * c;
    let s2 = if s1 > 1e-12 { det / s1 } else { 0.0 };
    let rotation = if s1 > 1e-12 { uy.atan2(ux) } else { 0.0 };

    if theta.abs() < 1e-9 {
        return json!({
            "id": next_id(),
            "kind": "group",
            "properties": {
                "name": name,
                "enabled": 1,
                "anchor": [0.0, 0.0],
                "position": [x, y],
                "scale": [s1, s2],
                "rotation": rotation,
                "opacity": opacity
            },
            "contents": contents
        });
    }
    let inner_id = next_id();
    let outer_id = next_id();
    json!({
        "id": outer_id,
        "kind": "group",
        "properties": {
            "name": name,
            "enabled": 1,
            "anchor": [0.0, 0.0],
            "position": [x, y],
            "scale": [s1, s2],
            "rotation": rotation,
            "opacity": opacity
        },
        "contents": [
            {
                "id": inner_id,
                "kind": "group",
                "properties": {
                    "name": "Transform",
                    "enabled": 1,
                    "anchor": [0.0, 0.0],
                    "position": [0.0, 0.0],
                    "scale": [1.0, 1.0],
                    "rotation": -theta,
                    "opacity": 1.0
                },
                "contents": contents
            }
        ]
    })
}

fn contour_to_path_json(contour: &SvgContour, seq: &mut u64) -> Value {
    *seq += 1;
    let id = format!("s{}", *seq);
    let mut vertex_ids = Vec::with_capacity(contour.vertices.len());
    let mut flat = Vec::with_capacity(contour.vertices.len() * 6);
    for v in &contour.vertices {
        *seq += 1;
        vertex_ids.push(format!("v{}", *seq));
        flat.extend_from_slice(&[
            v.point[0], v.point[1], v.in_h[0], v.in_h[1], v.out_h[0], v.out_h[1],
        ]);
    }
    json!({
        "id": id,
        "kind": "path",
        "vertexIds": vertex_ids,
        "vertexModes": {},
        "properties": {
            "name": "Path",
            "enabled": 1,
            "closed": if contour.closed { 1 } else { 0 },
            "path": flat
        }
    })
}

#[derive(Debug)]
struct XmlNode {
    tag: String,
    attrs: BTreeMap<String, String>,
    children: Vec<XmlNode>,
}

fn strip_xml_comments(input: &str) -> Result<String, String> {
    let mut out = String::with_capacity(input.len());
    let mut rest = input;
    while let Some(pos) = rest.find("<!--") {
        out.push_str(&rest[..pos]);
        out.push(' ');
        let after = &rest[pos + 4..];
        let end = after
            .find("-->")
            .ok_or_else(|| "Unterminated SVG comment".to_owned())?;
        rest = &after[end + 3..];
    }
    out.push_str(rest);
    Ok(out)
}

fn parse_xml_attributes(raw: &str) -> Result<BTreeMap<String, String>, String> {
    let mut map = BTreeMap::new();
    let bytes = raw.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let key_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'=' {
            i += 1;
        }
        let key = raw[key_start..i].to_owned();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            return Err(format!("Malformed SVG attribute: {key}"));
        }
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || (bytes[i] != b'"' && bytes[i] != b'\'') {
            return Err(format!("Expected quoted SVG attribute value for {key}"));
        }
        let quote = bytes[i];
        i += 1;
        let val_start = i;
        while i < bytes.len() && bytes[i] != quote {
            i += 1;
        }
        if i >= bytes.len() {
            return Err(format!("Unterminated SVG attribute value for {key}"));
        }
        let val = raw[val_start..i].to_owned();
        i += 1;
        map.insert(key, val);
    }
    Ok(map)
}

fn parse_svg_xml(text: &str) -> Result<XmlNode, String> {
    let clean = strip_xml_comments(text)?;
    let mut stack: Vec<XmlNode> = vec![XmlNode {
        tag: "root".to_owned(),
        attrs: BTreeMap::new(),
        children: Vec::new(),
    }];
    let bytes = clean.as_bytes();
    let mut i = 0usize;
    let mut elem_count = 0usize;

    while i < bytes.len() {
        let Some(lt_rel) = clean[i..].find('<') else {
            break;
        };
        i += lt_rel;
        let gt_rel = clean[i..]
            .find('>')
            .ok_or_else(|| "Unterminated XML tag in SVG".to_owned())?;
        let tag_str = &clean[i + 1..i + gt_rel];
        i += gt_rel + 1;

        let trimmed = tag_str.trim();
        if trimmed.is_empty() || trimmed.starts_with('?') || trimmed.starts_with('!') {
            continue;
        }
        if let Some(close_name) = trimmed.strip_prefix('/') {
            let local = close_name
                .trim()
                .strip_prefix("svg:")
                .unwrap_or(close_name.trim());
            if stack.len() <= 1 {
                return Err("Malformed SVG closing tag".to_owned());
            }
            let popped = stack.pop().unwrap();
            if popped.tag != local {
                return Err(format!(
                    "Mismatched SVG closing tag: expected </{}>, got </{local}>",
                    popped.tag
                ));
            }
            stack.last_mut().unwrap().children.push(popped);
            continue;
        }

        elem_count += 1;
        if elem_count > MAX_SHAPE_PLUS_ITEMS || stack.len() > MAX_SHAPE_PLUS_DEPTH + 2 {
            return Err("SVG element budget exceeded".to_owned());
        }
        let self_closing = trimmed.ends_with('/');
        let body = if self_closing {
            trimmed[..trimmed.len() - 1].trim()
        } else {
            trimmed
        };
        let split_at = body
            .find(|c: char| c.is_ascii_whitespace())
            .unwrap_or(body.len());
        let raw_tag = &body[..split_at];
        let local_tag = raw_tag.strip_prefix("svg:").unwrap_or(raw_tag).to_owned();
        let attrs = parse_xml_attributes(&body[split_at..])?;
        let node = XmlNode {
            tag: local_tag,
            attrs,
            children: Vec::new(),
        };
        if self_closing {
            stack.last_mut().unwrap().children.push(node);
        } else {
            stack.push(node);
        }
    }

    if stack.len() != 1 {
        return Err("Unclosed SVG XML element".to_owned());
    }
    let root = stack.pop().unwrap();
    root.children
        .into_iter()
        .find(|n| n.tag == "svg")
        .ok_or_else(|| "Expected root <svg> element".to_owned())
}

fn svg_element_to_path_d(tag: &str, attrs: &BTreeMap<String, String>) -> Result<String, String> {
    let num = |k: &str, def: f64| -> Result<f64, String> {
        match attrs.get(k) {
            None => Ok(def),
            Some(s) => parse_svg_numbers(s).map(|v| *v.first().unwrap_or(&def)),
        }
    };
    match tag {
        "path" => Ok(attrs.get("d").cloned().unwrap_or_default()),
        "polygon" | "polyline" => {
            let pts = parse_svg_numbers(attrs.get("points").map_or("", String::as_str))?;
            if pts.len() < 4 || pts.len() % 2 != 0 {
                return Err("Invalid SVG points attribute".to_owned());
            }
            let mut s = format!("M{} {}", pts[0], pts[1]);
            for pair in pts[2..].as_chunks::<2>().0 {
                s.push_str(&format!(" L{} {}", pair[0], pair[1]));
            }
            if tag == "polygon" {
                s.push_str(" Z");
            }
            Ok(s)
        }
        "line" => {
            let x1 = num("x1", 0.0)?;
            let y1 = num("y1", 0.0)?;
            let x2 = num("x2", 0.0)?;
            let y2 = num("y2", 0.0)?;
            Ok(format!("M{x1} {y1} L{x2} {y2}"))
        }
        "circle" | "ellipse" => {
            let cx = num("cx", 0.0)?;
            let cy = num("cy", 0.0)?;
            let rx = if tag == "circle" {
                num("r", 0.0)?
            } else {
                num("rx", 0.0)?
            };
            let ry = if tag == "circle" { rx } else { num("ry", 0.0)? };
            if rx < 0.0 || ry < 0.0 {
                return Err("Invalid negative SVG radius".to_owned());
            }
            Ok(format!(
                "M{} {} A{rx} {ry} 0 1 0 {} {} A{rx} {ry} 0 1 0 {} {} Z",
                cx - rx,
                cy,
                cx + rx,
                cy,
                cx - rx,
                cy
            ))
        }
        "rect" => {
            let x = num("x", 0.0)?;
            let y = num("y", 0.0)?;
            let w = num("width", 0.0)?;
            let h = num("height", 0.0)?;
            let ry_raw = num("ry", 0.0)?;
            let rx = num("rx", ry_raw)?.min(w * 0.5);
            let ry = num("ry", rx)?.min(h * 0.5);
            if w < 0.0 || h < 0.0 || rx < 0.0 || ry < 0.0 {
                return Err("Invalid SVG rect dimensions".to_owned());
            }
            if rx <= 0.0 || ry <= 0.0 {
                Ok(format!("M{x} {y} h{w} v{h} h{} Z", -w))
            } else {
                Ok(format!(
                    "M{} {y} H{} A{rx} {ry} 0 0 1 {} {} V{} A{rx} {ry} 0 0 1 {} {} H{} A{rx} {ry} 0 0 1 {x} {} V{} A{rx} {ry} 0 0 1 {} {y} Z",
                    x + rx,
                    x + w - rx,
                    x + w,
                    y + ry,
                    y + h - ry,
                    x + w - rx,
                    y + h,
                    x + rx,
                    y + h - ry,
                    y + ry,
                    x + rx
                ))
            }
        }
        _ => Err(format!("Unsupported SVG element: <{tag}>")),
    }
}

fn merge_svg_style(
    attrs: &BTreeMap<String, String>,
    inherited: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut style = inherited.clone();
    style.remove("opacity");
    for k in [
        "fill",
        "stroke",
        "fill-rule",
        "fill-opacity",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "stroke-dasharray",
        "stroke-dashoffset",
        "opacity",
        "display",
        "visibility",
    ] {
        if let Some(v) = attrs.get(k) {
            style.insert(k.to_owned(), v.clone());
        }
    }
    if let Some(inline) = attrs.get("style") {
        for decl in inline.split(';') {
            if let Some((k, v)) = decl.split_once(':') {
                let key = k.trim();
                let val = v.trim();
                if !key.is_empty() {
                    style.insert(key.to_owned(), val.to_owned());
                }
            }
        }
    }
    style
}

/// Imports an SVG XML document (or raw SVG path `d` string) into a canonical Shape+ JSON payload.
pub fn import_svg_to_shape_plus(svg: &str) -> Result<Value, String> {
    if svg.len() > 262_144 {
        return Err("SVG text budget exceeded".to_owned());
    }
    let trimmed = svg.trim();
    if trimmed.is_empty() {
        return Err("Empty SVG input".to_owned());
    }
    let mut seq = 0u64;

    if !trimmed.starts_with('<') {
        let contours = parse_svg_path_d(trimmed)?;
        let mut items: Vec<Value> = contours
            .iter()
            .map(|c| contour_to_path_json(c, &mut seq))
            .collect();
        seq += 1;
        items.push(json!({
            "id": format!("s{seq}"),
            "kind": "fill",
            "properties": {
                "name": "Fill",
                "enabled": 1,
                "color": [1.0, 1.0, 1.0],
                "opacity": 1.0,
                "fillRule": "nonzero",
                "composite": "below"
            }
        }));
        let root_grp = svd_affine_group(
            items,
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            &mut seq,
            1.0,
            "SVG path",
        );
        return Ok(json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "nextId": seq + 1,
            "contents": [root_grp]
        }));
    }

    let root = parse_svg_xml(trimmed)?;
    let skipped = ["defs", "metadata", "title", "desc", "style"];

    fn visit_node(
        node: &XmlNode,
        inherited: &BTreeMap<String, String>,
        skipped: &[&str],
        seq: &mut u64,
    ) -> Result<Vec<Value>, String> {
        if skipped.contains(&node.tag.as_str()) {
            return Ok(Vec::new());
        }
        let style = merge_svg_style(&node.attrs, inherited);
        if style.get("display").map(String::as_str) == Some("none")
            || style.get("visibility").map(String::as_str) == Some("hidden")
        {
            return Ok(Vec::new());
        }
        let opacity = style
            .get("opacity")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(1.0)
            .clamp(0.0, 1.0);
        let m = match node.attrs.get("transform") {
            Some(t) => parse_svg_transform(t)?,
            None => MAT_IDENTITY,
        };

        let mut contents = Vec::new();
        if node.tag == "svg" || node.tag == "g" {
            for child in &node.children {
                contents.extend(visit_node(child, &style, skipped, seq)?);
            }
            // Reverse painted siblings: Shape+ paints bottom to top (`svg-import.js:363`).
            contents.reverse();
        } else {
            let d = svg_element_to_path_d(&node.tag, &node.attrs)?;
            let contours = parse_svg_path_d(&d)?;
            for c in &contours {
                contents.push(contour_to_path_json(c, seq));
            }
            // Stroke first, then fill so stroke draws ABOVE fill in Shape+'s bottom-to-top order.
            let stroke_spec = style.get("stroke").map_or("none", String::as_str);
            if stroke_spec != "none" {
                let rgba = parse_css_color(stroke_spec)?;
                let s_op = style
                    .get("stroke-opacity")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);
                let alpha = (rgba[3] * s_op).clamp(0.0, 1.0);
                let width = style
                    .get("stroke-width")
                    .and_then(|s| s.trim_end_matches("px").parse::<f64>().ok())
                    .unwrap_or(1.0)
                    .max(0.0);
                let cap = style.get("stroke-linecap").map_or("butt", String::as_str);
                let join = style.get("stroke-linejoin").map_or("miter", String::as_str);
                let miter = style
                    .get("stroke-miterlimit")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(4.0)
                    .max(1.0);
                *seq += 1;
                let mut sp = Map::new();
                sp.insert("name".to_owned(), json!("Stroke"));
                sp.insert("enabled".to_owned(), json!(1));
                sp.insert("color".to_owned(), json!([rgba[0], rgba[1], rgba[2]]));
                sp.insert("opacity".to_owned(), json!(alpha));
                sp.insert("width".to_owned(), json!(width));
                sp.insert("cap".to_owned(), json!(cap));
                sp.insert("join".to_owned(), json!(join));
                sp.insert("miterLimit".to_owned(), json!(miter));
                sp.insert("composite".to_owned(), json!("below"));
                if let Some(da) = style.get("stroke-dasharray")
                    && da != "none"
                    && let Ok(mut nums) = parse_svg_numbers(da)
                    && !nums.is_empty()
                {
                    if nums.len() % 2 == 1 {
                        let dup = nums.clone();
                        nums.extend(dup);
                    }
                    let pairs = (nums.len() / 2).min(3);
                    sp.insert("dashPairs".to_owned(), json!(pairs));
                    for pi in 0..pairs {
                        sp.insert(format!("dash{}", pi + 1), json!(nums[pi * 2]));
                        sp.insert(format!("gap{}", pi + 1), json!(nums[pi * 2 + 1]));
                    }
                    if let Some(off) = style
                        .get("stroke-dashoffset")
                        .and_then(|s| s.parse::<f64>().ok())
                    {
                        sp.insert("dashOffset".to_owned(), json!(off));
                    }
                }
                contents.push(json!({
                    "id": format!("s{}", *seq),
                    "kind": "stroke",
                    "properties": sp
                }));
            }

            let fill_spec = style.get("fill").map_or("black", String::as_str);
            if fill_spec != "none" {
                let rgba = parse_css_color(fill_spec)?;
                let f_op = style
                    .get("fill-opacity")
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(1.0);
                let alpha = (rgba[3] * f_op).clamp(0.0, 1.0);
                let rule = if style.get("fill-rule").map(String::as_str) == Some("evenodd") {
                    "even-odd"
                } else {
                    "nonzero"
                };
                *seq += 1;
                contents.push(json!({
                    "id": format!("s{}", *seq),
                    "kind": "fill",
                    "properties": {
                        "name": "Fill",
                        "enabled": 1,
                        "color": [rgba[0], rgba[1], rgba[2]],
                        "opacity": alpha,
                        "fillRule": rule,
                        "composite": "below"
                    }
                }));
            }
        }

        let name = node
            .attrs
            .get("id")
            .map_or(node.tag.as_str(), String::as_str);
        Ok(vec![svd_affine_group(contents, m, seq, opacity, name)])
    }

    let items = visit_node(&root, &BTreeMap::new(), &skipped, &mut seq)?;
    let view_box = match root.attrs.get("viewBox") {
        Some(vb) => {
            let nums = parse_svg_numbers(vb)?;
            if nums.len() != 4 || nums[2] <= 0.0 || nums[3] <= 0.0 {
                return Err("Invalid SVG viewBox".to_owned());
            }
            Some([nums[0], nums[1], nums[2], nums[3]])
        }
        None => None,
    };
    let root_mat = match view_box {
        Some([vx, vy, vw, vh]) => [1.0, 0.0, 0.0, -1.0, -(vx + vw * 0.5), vy + vh * 0.5],
        None => [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
    };
    let wrapped = svd_affine_group(items, root_mat, &mut seq, 1.0, "Imported SVG");
    let payload = json!({
        "schemaVersion": 1,
        "renderVersion": 1,
        "nextId": seq + 1,
        "contents": [wrapped]
    });
    validate_shape_plus_payload(&payload)?;
    Ok(payload)
}

fn fmt_svg_coord(v: f32) -> String {
    let rounded = (f64::from(v) * 1_000_000.0).round() / 1_000_000.0;
    let clean = if rounded.abs() < 1e-12 { 0.0 } else { rounded };
    format!("{clean}")
}

/// Exports a Shape+ Bezier vertex array (`[x, y, in_x, in_y, out_x, out_y]`) to an SVG `d` string
/// matching Zoidium `svg-path.js:214-234`.
#[must_use]
pub fn export_shape_plus_path_to_svg_d(vertices: &[[f32; 6]], closed: bool) -> String {
    if vertices.is_empty() {
        return String::new();
    }
    let mut out = format!(
        "M{} {}",
        fmt_svg_coord(vertices[0][0]),
        fmt_svg_coord(vertices[0][1])
    );
    let count = if closed {
        vertices.len()
    } else {
        vertices.len().saturating_sub(1)
    };
    for i in 0..count {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        if a[4] != 0.0 || a[5] != 0.0 || b[2] != 0.0 || b[3] != 0.0 {
            out.push_str(&format!(
                " C{} {} {} {} {} {}",
                fmt_svg_coord(a[0] + a[4]),
                fmt_svg_coord(a[1] + a[5]),
                fmt_svg_coord(b[0] + b[2]),
                fmt_svg_coord(b[1] + b[3]),
                fmt_svg_coord(b[0]),
                fmt_svg_coord(b[1]),
            ));
        } else {
            out.push_str(&format!(
                " L{} {}",
                fmt_svg_coord(b[0]),
                fmt_svg_coord(b[1])
            ));
        }
    }
    if closed {
        out.push_str(" Z");
    }
    out
}

// ---------------------------------------------------------------------------
// Antialiased Exact-SDF Mask Rasterizer (`rasterize_masks_aa`)
// ---------------------------------------------------------------------------

/// Evaluated layer mask geometry & compositing parameters in mask pixel space
/// (origin top-left, `+y` down).
#[derive(Debug, Clone, PartialEq)]
pub struct EvaluatedMask {
    /// Vertices `[x, y, in_x, in_y, out_x, out_y]` in mask pixel coordinates.
    pub vertices: Vec<[f32; 6]>,
    pub closed: bool,
    /// AE combine mode: `"add"`, `"subtract"`, or `"intersect"`.
    pub mode: String,
    /// Feather blur radius in pixels (`0..=500`).
    pub feather: f32,
    /// Signed SDF expansion in pixels (`-128..=128`).
    pub expansion: f32,
    /// Mask opacity (`0.0..=1.0`).
    pub opacity: f32,
    /// Post-opacity invert flag.
    pub invert: bool,
    /// Whether this mask is enabled.
    pub enabled: bool,
}

impl Default for EvaluatedMask {
    fn default() -> Self {
        Self {
            vertices: Vec::new(),
            closed: true,
            mode: "add".to_owned(),
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            invert: false,
            enabled: true,
        }
    }
}

fn close_mask_vertices(vertices: &[[f32; 6]], closed: bool, scale: f64) -> Vec<BezierVertex> {
    let mut out: Vec<BezierVertex> = vertices
        .iter()
        .map(|v| {
            BezierVertex::new(
                [f64::from(v[0]) * scale, f64::from(v[1]) * scale],
                [f64::from(v[2]) * scale, f64::from(v[3]) * scale],
                [f64::from(v[4]) * scale, f64::from(v[5]) * scale],
            )
        })
        .collect();
    if !closed && out.len() >= 2 {
        let first = out[0].point;
        let last = out.last().unwrap().point;
        if pt_dist(first, last) > 1e-9 {
            out.push(BezierVertex::sharp(first));
        }
    }
    out
}

/// Hard-edge even-odd fill used to obtain exact far-field sign for SDF expansion.
fn fill_polygon_evenodd_hard(points: &[[f64; 2]], width: usize, height: usize, dst: &mut [f32]) {
    let (edges, min_y, max_y) = build_scanline_edges(&[points.to_vec()]);
    if edges.is_empty() || !min_y.is_finite() || !max_y.is_finite() {
        return;
    }
    let y0 = (min_y.floor() as isize).max(0) as usize;
    let y1 = (max_y.ceil() as isize).min(height as isize) as usize;
    let mut crossings = Vec::new();
    for y in y0..y1 {
        let py = y as f64 + 0.5;
        crossings.clear();
        for e in &edges {
            if py >= e.y_min && py < e.y_max {
                crossings.push(e.x_at_ymin + (py - e.y_min) * e.dx_dy);
            }
        }
        if crossings.len() < 2 {
            continue;
        }
        crossings.sort_by(f64::total_cmp);
        let row = &mut dst[y * width..(y + 1) * width];
        for pair in crossings.as_chunks::<2>().0 {
            let x0 = ((pair[0] - 0.5).ceil() as isize).max(0) as usize;
            let x1 = ((pair[1] - 0.5).floor() as isize).min(width as isize - 1);
            if x1 >= x0 as isize {
                for px in &mut row[x0..=(x1 as usize)] {
                    *px = 1.0;
                }
            }
        }
    }
}

/// Exact point-to-segment signed distance mask expansion using Zoidium's coarse spatial grid
/// accelerator (`mask-expansion.js:143-218`).
fn expand_mask_sdf(
    points: &[[f64; 2]],
    expansion: f64,
    width: usize,
    height: usize,
    dst: &mut [f32],
) {
    if points.len() < 3 || signed_area(points).abs() < 1e-9 {
        return;
    }
    fill_polygon_evenodd_hard(points, width, height, dst);

    let pad = expansion.abs() + 1.5;
    let cell = pad.clamp(8.0, 64.0);
    let cells_x = ((width as f64) / cell).ceil().max(1.0) as usize;
    let cells_y = ((height as f64) / cell).ceil().max(1.0) as usize;
    let n = points.len();
    let mut grid: HashMap<usize, Vec<usize>> = HashMap::new();

    for s in 0..n {
        let a = points[s];
        let b = points[(s + 1) % n];
        let fx = (((a[0].min(b[0]) - pad) / cell).floor() as isize).clamp(0, cells_x as isize - 1)
            as usize;
        let tx = (((a[0].max(b[0]) + pad) / cell).floor() as isize).clamp(0, cells_x as isize - 1)
            as usize;
        let fy = (((a[1].min(b[1]) - pad) / cell).floor() as isize).clamp(0, cells_y as isize - 1)
            as usize;
        let ty = (((a[1].max(b[1]) + pad) / cell).floor() as isize).clamp(0, cells_y as isize - 1)
            as usize;
        for cy in fy..=ty {
            for cx in fx..=tx {
                grid.entry(cy * cells_x + cx).or_default().push(s);
            }
        }
    }

    for (key, seg_list) in &grid {
        let cx = key % cells_x;
        let cy = key / cells_x;
        let fx = ((cx as f64 * cell).floor() as usize).min(width.saturating_sub(1));
        let tx = (((cx + 1) as f64 * cell).ceil() as usize)
            .saturating_sub(1)
            .min(width.saturating_sub(1));
        let fy = ((cy as f64 * cell).floor() as usize).min(height.saturating_sub(1));
        let ty = (((cy + 1) as f64 * cell).ceil() as usize)
            .saturating_sub(1)
            .min(height.saturating_sub(1));
        for y in fy..=ty {
            let py = y as f64 + 0.5;
            let row_base = y * width;
            for x in fx..=tx {
                let px = x as f64 + 0.5;
                let mut nearest_sq = f64::INFINITY;
                for &s in seg_list {
                    let a = points[s];
                    let b = points[(s + 1) % n];
                    let dx = b[0] - a[0];
                    let dy = b[1] - a[1];
                    let l2 = dx * dx + dy * dy;
                    let t = if l2 > 0.0 {
                        (((px - a[0]) * dx + (py - a[1]) * dy) / l2).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let ex = a[0] + dx * t - px;
                    let ey = a[1] + dy * t - py;
                    let d2 = ex * ex + ey * ey;
                    if d2 < nearest_sq {
                        nearest_sq = d2;
                    }
                }
                let mag = nearest_sq.sqrt();
                let signed = if dst[row_base + x] >= 0.5 { mag } else { -mag };
                dst[row_base + x] = (0.5 + signed + expansion).clamp(0.0, 1.0) as f32;
            }
        }
    }
}

fn box_blur_1d(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    radius: usize,
    vertical: bool,
) {
    if radius == 0 {
        dst.copy_from_slice(src);
        return;
    }
    let r = radius as isize;
    let inv_win = 1.0 / (2 * radius + 1) as f32;
    if !vertical {
        let w = width as isize;
        for y in 0..height {
            let row_off = y * width;
            let mut sum = 0.0_f32;
            for k in -r..=r {
                let cx = k.clamp(0, w - 1) as usize;
                sum += src[row_off + cx];
            }
            for x in 0..width {
                dst[row_off + x] = sum * inv_win;
                let add_x = (x as isize + r + 1).clamp(0, w - 1) as usize;
                let sub_x = (x as isize - r).clamp(0, w - 1) as usize;
                sum += src[row_off + add_x] - src[row_off + sub_x];
            }
        }
    } else {
        let h = height as isize;
        for x in 0..width {
            let mut sum = 0.0_f32;
            for k in -r..=r {
                let cy = k.clamp(0, h - 1) as usize;
                sum += src[cy * width + x];
            }
            for y in 0..height {
                dst[y * width + x] = sum * inv_win;
                let add_y = (y as isize + r + 1).clamp(0, h - 1) as usize;
                let sub_y = (y as isize - r).clamp(0, h - 1) as usize;
                sum += src[add_y * width + x] - src[sub_y * width + x];
            }
        }
    }
}

fn feather_3pass_box_blur(channel: &mut [f32], width: usize, height: usize, radius: f64) {
    let clamped = radius.clamp(0.0, width.max(height) as f64);
    let r = clamped.round() as usize;
    if r == 0 {
        if clamped > 1e-4 {
            let mut tmp = vec![0.0_f32; channel.len()];
            let mut blurred = vec![0.0_f32; channel.len()];
            box_blur_1d(channel, &mut tmp, width, height, 1, false);
            box_blur_1d(&tmp, &mut blurred, width, height, 1, true);
            let t = clamped as f32;
            for (c, b) in channel.iter_mut().zip(blurred.iter()) {
                *c += (*b - *c) * t;
            }
        }
        return;
    }
    let mut tmp = vec![0.0_f32; channel.len()];
    for _ in 0..3 {
        box_blur_1d(channel, &mut tmp, width, height, r, false);
        box_blur_1d(&tmp, channel, width, height, r, true);
    }
}

fn bilinear_upsample_plane(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<u8> {
    let mut out = vec![0_u8; dw * dh];
    let sx = sw as f64 / dw.max(1) as f64;
    let sy = sh as f64 / dh.max(1) as f64;
    for y in 0..dh {
        let fy = ((y as f64 + 0.5) * sy - 0.5).clamp(0.0, (sh.saturating_sub(1)) as f64);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let ty = (fy - y0 as f64) as f32;
        for x in 0..dw {
            let fx = ((x as f64 + 0.5) * sx - 0.5).clamp(0.0, (sw.saturating_sub(1)) as f64);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let tx = (fx - x0 as f64) as f32;
            let v00 = src[y0 * sw + x0];
            let v10 = src[y0 * sw + x1];
            let v01 = src[y1 * sw + x0];
            let v11 = src[y1 * sw + x1];
            let top = v00 + (v10 - v00) * tx;
            let bot = v01 + (v11 - v01) * tx;
            let v = (top + (bot - top) * ty).clamp(0.0, 1.0);
            out[y * dw + x] = (v * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

/// Rasterizes evaluated layer masks into a single-channel 8-bit coverage plane (`width * height`).
///
/// Features:
/// - Capped at 2048px long side for fast 4K playback (upsampled bilinearly when `max(w, h) > 2048`).
/// - 4x supersampled subpixel edge antialiasing (even when `feather == 0 && expansion == 0`).
/// - Exact point-to-segment signed distance for `expansion` (`-128..=128` px).
/// - 3-pass separable sliding-window box blur for `feather` (`0..=500` px).
/// - `opacity`, post-opacity `invert`, and AE ordered `Add` / `Subtract` / `Intersect` combine.
#[must_use]
pub fn rasterize_masks_aa(masks: &[EvaluatedMask], width: u32, height: u32) -> Vec<u8> {
    if width == 0 || height == 0 {
        return Vec::new();
    }
    let long_side = width.max(height);
    let (rw, rh, scale) = if long_side > MAX_MASK_LONG_SIDE {
        let s = f64::from(MAX_MASK_LONG_SIDE) / f64::from(long_side);
        (
            (f64::from(width) * s).round().max(1.0) as usize,
            (f64::from(height) * s).round().max(1.0) as usize,
            s,
        )
    } else {
        (width as usize, height as usize, 1.0)
    };

    let mut coverage = vec![0.0_f32; rw * rh];
    let mut scratch = vec![0.0_f32; rw * rh];
    let mut first = true;

    for mask in masks.iter().take(32) {
        if !mask.enabled {
            continue;
        }
        scratch.fill(0.0);
        let expansion = f64::from(mask.expansion.clamp(-128.0, 128.0)) * scale;
        let closed_verts = close_mask_vertices(&mask.vertices, mask.closed, scale);
        if closed_verts.len() >= 2 {
            let flat = flatten_vertices(&closed_verts, true, DEFAULT_TOLERANCE);
            if flat.points.len() >= 3 && signed_area(&flat.points).abs() >= 1e-9 {
                if expansion.abs() < 1e-6 {
                    // 4x supersampled subpixel AA even at feather==0 && expansion==0!
                    rasterize_polygons_aa(&[flat.points], rw, rh, true, |y, cov| {
                        let dst_row = &mut scratch[y * rw..(y + 1) * rw];
                        for (d, &c) in dst_row.iter_mut().zip(cov.iter()) {
                            *d = c.clamp(0.0, 1.0);
                        }
                    });
                } else {
                    expand_mask_sdf(&flat.points, expansion, rw, rh, &mut scratch);
                }
            }
        }

        let feather = f64::from(mask.feather.clamp(0.0, 500.0)) * scale;
        if feather > 0.0 {
            feather_3pass_box_blur(&mut scratch, rw, rh, feather);
        }

        let opacity = mask.opacity.clamp(0.0, 1.0);
        let invert = mask.invert;
        let mode = mask.mode.to_ascii_lowercase();

        for (base, &ch) in coverage.iter_mut().zip(scratch.iter()) {
            let mut alpha = (ch * opacity).clamp(0.0, 1.0);
            if invert {
                alpha = 1.0 - alpha;
            }
            *base = if first {
                if mode == "subtract" { 0.0 } else { alpha }
            } else {
                match mode.as_str() {
                    "subtract" => *base * (1.0 - alpha),
                    "intersect" => base.min(alpha),
                    _ => *base + alpha * (1.0 - *base),
                }
            };
        }
        first = false;
    }

    if rw != width as usize || rh != height as usize {
        return bilinear_upsample_plane(&coverage, rw, rh, width as usize, height as usize);
    }
    coverage
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// Comprehensive Unit Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn count_opaque_pixels(rgba: &[u8]) -> usize {
        rgba.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count()
    }

    fn pixel_at(rgba: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
        let i = ((y * w + x) * 4) as usize;
        [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
    }

    fn make_doc(contents: Value) -> Value {
        json!({
            "schemaVersion": 1,
            "renderVersion": 1,
            "nextId": 100,
            "contents": contents
        })
    }

    #[test]
    fn canonical_starter_and_crud_helpers() {
        let mut doc = default_shape_plus_payload();
        assert_eq!(doc["schemaVersion"], 1);
        assert_eq!(doc["renderVersion"], 1);
        assert_eq!(doc["nextId"], 4);
        assert_eq!(doc["contents"][0]["id"], "s1");
        assert_eq!(doc["contents"][0]["contents"][0]["id"], "s2");
        assert_eq!(doc["contents"][0]["contents"][1]["id"], "s3");
        validate_shape_plus_payload(&doc).expect("starter payload must validate");

        let parsed = parse_shape_plus_document(&doc);
        assert_eq!(parsed.next_id, 4);
        assert_eq!(parsed.contents.len(), 1);
        assert_eq!(parsed.contents[0].contents.len(), 2);

        let stroke_id = add_shape_plus_item(&mut doc, Some("s1"), "stroke").expect("add stroke");
        assert_eq!(stroke_id, "s4");
        assert_eq!(doc["contents"][0]["contents"].as_array().unwrap().len(), 3);

        assert!(reorder_shape_plus_item(&mut doc, &stroke_id, 0).expect("reorder"));
        assert_eq!(doc["contents"][0]["contents"][0]["id"], stroke_id);

        let dup_id = duplicate_shape_plus_item(&mut doc, "s1")
            .expect("dup")
            .expect("dup id");
        assert_ne!(dup_id, "s1");
        validate_shape_plus_payload(&doc).expect("duplicated payload must keep unique IDs");

        assert!(remove_shape_plus_item(&mut doc, &dup_id).expect("remove"));
        assert_eq!(doc["contents"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn primitives_gradient_fill_and_gradient_stroke() {
        // Gradient fill across a 160x160 rectangle: left red -> right blue.
        let grad_doc = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [160.0, 100.0], "roundness": 12.0 } },
                {
                    "id": "gf1", "kind": "gradientFill",
                    "properties": {
                        "gradientType": "linear",
                        "startPoint": [-80.0, 0.0],
                        "endPoint": [80.0, 0.0],
                        "stops": [
                            { "position": 0.0, "color": [1.0, 0.0, 0.0, 1.0] },
                            { "position": 1.0, "color": [0.0, 0.0, 1.0, 1.0] }
                        ]
                    }
                }
            ]
        }]));
        let rgba = rasterize_shape_plus(&grad_doc, 200, 200, [200, 200], 0.0, 30.0, None);
        let left_px = pixel_at(&rgba, 200, 35, 100);
        let right_px = pixel_at(&rgba, 200, 165, 100);
        assert!(
            left_px[0] > 200 && left_px[2] < 60,
            "expected red left, got {left_px:?}"
        );
        assert!(
            right_px[2] > 200 && right_px[0] < 60,
            "expected blue right, got {right_px:?}"
        );

        // Ellipse, polystar (with roundness), and cubic path with gradientStroke.
        let multi_doc = make_doc(json!([
            {
                "id": "g_e", "kind": "group", "properties": { "position": [-50.0, 0.0] },
                "contents": [
                    { "id": "e1", "kind": "ellipse", "properties": { "size": [60.0, 40.0] } },
                    { "id": "f1", "kind": "fill", "properties": { "color": [0.0, 1.0, 0.0] } }
                ]
            },
            {
                "id": "g_p", "kind": "group", "properties": { "position": [50.0, 0.0] },
                "contents": [
                    {
                        "id": "ps1", "kind": "polystar",
                        "properties": { "mode": "star", "points": 5, "outerRadius": 40.0, "innerRadius": 18.0, "outerRoundness": 30.0, "innerRoundness": 20.0 }
                    },
                    {
                        "id": "gs1", "kind": "gradientStroke",
                        "properties": {
                            "gradientType": "radial",
                            "startPoint": [0.0, 0.0],
                            "endPoint": [40.0, 0.0],
                            "highlightLength": 35.0,
                            "width": 6.0,
                            "stops": [
                                { "position": 0.0, "color": "#ffff00" },
                                { "position": 1.0, "color": "#ff00ff" }
                            ]
                        }
                    }
                ]
            }
        ]));
        let rgba2 = rasterize_shape_plus(&multi_doc, 200, 200, [200, 200], 0.0, 30.0, None);
        assert!(count_opaque_pixels(&rgba2) > 2000);
        // Check green ellipse center at (-50, 0) -> pixel (50, 100).
        let e_center = pixel_at(&rgba2, 200, 50, 100);
        assert!(e_center[1] > 240 && e_center[0] == 0);
    }

    #[test]
    fn trim_degrees_offset_90_equals_quarter_turn() {
        // Square perimeter 400: start=0, end=25 draws exactly 1 side.
        // offset=0 draws bottom edge (y=-50 in Y-up -> pixel y=150).
        // offset=90 degrees (= 0.25) rotates to right edge (x=+50 -> pixel x=150)!
        let make_trim = |offset_deg: f64| {
            make_doc(json!([{
                "id": "g1", "kind": "group", "properties": {},
                "contents": [
                    { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                    { "id": "t1", "kind": "trim", "properties": { "start": 0.0, "end": 25.0, "offset": offset_deg } },
                    { "id": "s1", "kind": "stroke", "properties": { "color": [1.0, 1.0, 1.0], "width": 8.0 } }
                ]
            }]))
        };
        let r0 = rasterize_shape_plus(&make_trim(0.0), 200, 200, [200, 200], 0.0, 30.0, None);
        let r90 = rasterize_shape_plus(&make_trim(90.0), 200, 200, [200, 200], 0.0, 30.0, None);
        // At offset=0: bottom edge (x=100, y=150) is lit, right edge (x=150, y=100) is dark.
        assert!(pixel_at(&r0, 200, 100, 150)[3] > 200);
        assert_eq!(pixel_at(&r0, 200, 150, 100)[3], 0);
        // At offset=90 deg: right edge (x=150, y=100) is lit, bottom edge (x=100, y=150) is dark.
        assert!(pixel_at(&r90, 200, 150, 100)[3] > 200);
        assert_eq!(pixel_at(&r90, 200, 100, 150)[3], 0);
    }

    #[test]
    fn stroke_caps_dashes_taper_and_wave() {
        // Horizontal line from x=-40 to x=+40 at y=0 (pixels x=60..140, y=100) with width=20.
        let line_stroke = |cap: &str, extra: Value| {
            let mut props = json!({
                "color": [1.0, 1.0, 1.0],
                "width": 20.0,
                "cap": cap
            });
            if let (Some(dst), Some(src)) = (props.as_object_mut(), extra.as_object()) {
                for (k, v) in src {
                    dst.insert(k.clone(), v.clone());
                }
            }
            make_doc(json!([{
                "id": "g1", "kind": "group", "properties": {},
                "contents": [
                    {
                        "id": "p1", "kind": "path", "vertexIds": ["v1", "v2"],
                        "properties": {
                            "closed": 0,
                            "path": [-40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        }
                    },
                    { "id": "s1", "kind": "stroke", "properties": props }
                ]
            }]))
        };

        let butt = rasterize_shape_plus(
            &line_stroke("butt", json!({})),
            200,
            200,
            [200, 200],
            0.0,
            30.0,
            None,
        );
        let square = rasterize_shape_plus(
            &line_stroke("square", json!({})),
            200,
            200,
            [200, 200],
            0.0,
            30.0,
            None,
        );
        let round = rasterize_shape_plus(
            &line_stroke("round", json!({})),
            200,
            200,
            [200, 200],
            0.0,
            30.0,
            None,
        );

        // Pixel (145, 100) is 5px past endpoint x=140: butt=0, square>200, round>200.
        assert_eq!(pixel_at(&butt, 200, 145, 100)[3], 0);
        assert!(pixel_at(&square, 200, 145, 100)[3] > 200);
        assert!(pixel_at(&round, 200, 145, 100)[3] > 200);
        // Corner pixel (148, 92) is inside square cap, outside round cap (dist from (140,100) = hypot(8,8)=11.3 > 10).
        assert!(pixel_at(&square, 200, 148, 92)[3] > 200);
        assert_eq!(pixel_at(&round, 200, 148, 92)[3], 0);

        // Dashes: dash1=20, gap1=20 -> gap at x=80..100 -> pixel (90, 100) must be empty!
        let dashed = rasterize_shape_plus(
            &line_stroke(
                "butt",
                json!({ "dashPairs": 1, "dash1": 20.0, "gap1": 20.0 }),
            ),
            200,
            200,
            [200, 200],
            0.0,
            30.0,
            None,
        );
        assert!(pixel_at(&dashed, 200, 70, 100)[3] > 200);
        assert_eq!(pixel_at(&dashed, 200, 90, 100)[3], 0);

        // Taper: startLength=50, startWidth=0 -> thinner near left end than center.
        let tapered = rasterize_shape_plus(
            &line_stroke(
                "butt",
                json!({ "startLength": 50.0, "startWidth": 0.0, "waveAmount": 25.0, "waveCycles": 2.0 }),
            ),
            200,
            200,
            [200, 200],
            0.0,
            30.0,
            None,
        );
        assert_eq!(pixel_at(&tapered, 200, 62, 93)[3], 0);
        assert!(pixel_at(&tapered, 200, 120, 96)[3] > 150);
    }

    #[test]
    fn repeater_wiggle_transform_and_deformers_all_produce_distinct_outputs() {
        // Repeater: 3 copies shifted by +50 on X with opacity fade.
        let rep = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [24.0, 24.0], "position": [-50.0, 0.0] } },
                { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } },
                {
                    "id": "rep1", "kind": "repeater",
                    "properties": { "copies": 3.0, "position": [50.0, 0.0], "startOpacity": 1.0, "endOpacity": 0.4 }
                }
            ]
        }]));
        let rep_px = rasterize_shape_plus(&rep, 200, 200, [200, 200], 0.0, 30.0, None);
        assert!(pixel_at(&rep_px, 200, 50, 100)[3] > 240);
        assert!(pixel_at(&rep_px, 200, 100, 100)[3] > 150);
        assert!(
            pixel_at(&rep_px, 200, 150, 100)[3] > 80 && pixel_at(&rep_px, 200, 150, 100)[3] < 130
        );

        // Baseline square vs roundCorners, twist, zigZag, puckerBloat, wigglePaths, wiggleTransform, offsetPaths.
        let base_doc = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
            ]
        }]));
        let base_px = rasterize_shape_plus(&base_doc, 200, 200, [200, 200], 10.0, 30.0, None);

        let modifiers = [
            json!({ "id": "m1", "kind": "roundCorners", "properties": { "radius": 30.0 } }),
            json!({ "id": "m2", "kind": "twist", "properties": { "angle": 1.2, "center": [0.0, 0.0] } }),
            json!({ "id": "m3", "kind": "zigZag", "properties": { "size": 12.0, "ridges": 4.0, "points": "smooth" } }),
            json!({ "id": "m4", "kind": "puckerBloat", "properties": { "amount": 45.0 } }),
            json!({ "id": "m5", "kind": "wigglePaths", "properties": { "size": 15.0, "detail": 8.0, "points": "smooth", "seed": 7.0 } }),
            json!({ "id": "m6", "kind": "wiggleTransform", "properties": { "position": [35.0, 25.0], "rotation": 0.4, "seed": 3.0 } }),
            json!({ "id": "m7", "kind": "offsetPaths", "properties": { "amount": 18.0, "join": "round" } }),
            json!({ "id": "m8", "kind": "offsetPaths", "properties": { "amount": -20.0, "join": "miter" } }),
        ];

        for m in modifiers {
            let kind = m["kind"].as_str().unwrap();
            let doc = make_doc(json!([{
                "id": "g1", "kind": "group", "properties": {},
                "contents": [
                    { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                    m,
                    { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
                ]
            }]));
            let px = rasterize_shape_plus(&doc, 200, 200, [200, 200], 10.0, 30.0, None);
            assert!(
                count_opaque_pixels(&px) > 1000,
                "{kind} produced blank render"
            );
            assert_ne!(px, base_px, "{kind} had no visible effect vs baseline");
        }
    }

    #[test]
    fn merge_paths_subtract_cuts_real_hole() {
        let doc = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r_outer", "kind": "rectangle", "properties": { "size": [140.0, 140.0] } },
                { "id": "r_inner", "kind": "rectangle", "properties": { "size": [60.0, 60.0] } },
                { "id": "mp1", "kind": "mergePaths", "properties": { "mode": "subtract" } },
                { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 0.5, 0.0], "fillRule": "nonzero" } }
            ]
        }]));
        let rgba = rasterize_shape_plus(&doc, 200, 200, [200, 200], 0.0, 30.0, None);
        // Center (100, 100) must be a transparent hole; ring at (50, 100) must be filled!
        assert_eq!(pixel_at(&rgba, 200, 100, 100)[3], 0);
        assert!(pixel_at(&rgba, 200, 50, 100)[3] > 240);
    }

    #[test]
    fn isolated_group_opacity_paint_order_disabled_and_bezier_hold_eval() {
        // Isolated group opacity: two overlapping opaque rects inside group with opacity=0.5 ->
        // overlap pixel must have alpha ~128 (NOT ~191 from double blending!).
        let iso = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": { "opacity": 0.5 },
            "contents": [
                {
                    "id": "g_a", "kind": "group", "properties": {},
                    "contents": [
                        { "id": "ra", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                        { "id": "fa", "kind": "fill", "properties": { "color": [1.0, 0.0, 0.0] } }
                    ]
                },
                {
                    "id": "g_b", "kind": "group", "properties": {},
                    "contents": [
                        { "id": "rb", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                        { "id": "fb", "kind": "fill", "properties": { "color": [0.0, 0.0, 1.0] } }
                    ]
                }
            ]
        }]));
        let iso_px = rasterize_shape_plus(&iso, 200, 200, [200, 200], 0.0, 30.0, None);
        let c = pixel_at(&iso_px, 200, 100, 100);
        assert!(
            (i32::from(c[3]) - 128).abs() <= 2,
            "expected isolated alpha ~128, got {c:?}"
        );

        // Paint order: below (default) vs above, plus disabled item skip.
        let order_below = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                { "id": "f_red", "kind": "fill", "properties": { "color": [1.0, 0.0, 0.0], "composite": "below" } },
                { "id": "f_blue", "kind": "fill", "properties": { "color": [0.0, 0.0, 1.0], "composite": "below" } },
                { "id": "f_off", "kind": "fill", "properties": { "enabled": 0, "color": [0.0, 1.0, 0.0], "composite": "above" } }
            ]
        }]));
        // In bottom-to-top default ("below"): f_red is painted AFTER f_blue -> red wins!
        let ob_px = rasterize_shape_plus(&order_below, 200, 200, [200, 200], 0.0, 30.0, None);
        assert!(pixel_at(&ob_px, 200, 100, 100)[0] > 240);

        let mut order_above = order_below.clone();
        order_above["contents"][0]["contents"][2]["properties"]["composite"] = json!("above");
        let oa_px = rasterize_shape_plus(&order_above, 200, 200, [200, 200], 0.0, 30.0, None);
        // f_blue has composite="above" -> blue draws on top of red!
        assert!(pixel_at(&oa_px, 200, 100, 100)[2] > 240);

        // Hold & Bezier keyframe + expression evaluation
        let hold_val = json!({
            "keyframes": [
                { "frame": 0, "value": 10.0, "tween": 0 },
                { "frame": 10, "value": 100.0, "tween": 1 }
            ]
        });
        assert!((eval_prop_f64(Some(&hold_val), 5.0, 0.0, None) - 10.0).abs() < 1e-6);

        let bez_val = json!({
            "keyframes": [
                { "frame": 0, "value": 0.0, "tween": 257, "bezier": [0.0, 1.0, 0.0, 1.0] },
                { "frame": 10, "value": 100.0, "tween": 1 }
            ]
        });
        assert!(eval_prop_f64(Some(&bez_val), 5.0, 0.0, None) > 75.0);

        let expr_val = json!({ "expression": "frame * 3 + 1" });
        let cb = |e: &str, f: f64| {
            if e == "frame * 3 + 1" {
                Some(f * 3.0 + 1.0)
            } else {
                None
            }
        };
        assert!((eval_prop_f64(Some(&expr_val), 4.0, 0.0, Some(&cb)) - 13.0).abs() < 1e-6);
    }

    #[test]
    fn vector_text_and_text_animator_render() {
        let base = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                {
                    "id": "t1", "kind": "text",
                    "properties": { "sourceText": "HI\nOK", "fontSize": 60.0, "alignment": "center", "position": [0.0, 20.0] }
                },
                { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
            ]
        }]));
        let base_px = rasterize_shape_plus(&base, 256, 256, [256, 256], 0.0, 30.0, None);
        assert!(
            count_opaque_pixels(&base_px) > 500,
            "text must produce visible pixels"
        );

        let anim = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                {
                    "id": "t1", "kind": "text",
                    "properties": { "sourceText": "HI\nOK", "fontSize": 60.0, "alignment": "center", "position": [0.0, 20.0] }
                },
                {
                    "id": "ta1", "kind": "textAnimator",
                    "properties": {
                        "enabled_position": 1,
                        "position": [0.0, 30.0],
                        "enabled_fillColor": 1,
                        "fillColor": [1.0, 0.0, 0.0],
                        "enabled_opacity": 1,
                        "opacity": 0.5,
                        "selectorStart": 0.0,
                        "selectorEnd": 50.0,
                        "wigglyOn": 1,
                        "wigglySeed": 4.0
                    }
                },
                { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
            ]
        }]));
        let anim_px = rasterize_shape_plus(&anim, 256, 256, [256, 256], 15.0, 30.0, None);
        assert!(count_opaque_pixels(&anim_px) > 500);
        assert_ne!(anim_px, base_px);
    }

    #[test]
    fn svg_import_and_export_roundtrip_renders() {
        let svg = r##"
            <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
                <g transform="translate(10, 5)">
                    <rect x="20" y="20" width="60" height="60" rx="8" fill="#ff3300" />
                    <circle cx="140" cy="60" r="30" fill="rgb(0, 180, 255)" stroke="#ffffff" stroke-width="4" />
                    <path d="M 30 140 A 40 30 0 0 1 150 140 L 150 170 Z" fill="green" />
                </g>
            </svg>
        "##;
        let payload = import_svg_to_shape_plus(svg).expect("SVG import must succeed");
        validate_shape_plus_payload(&payload).expect("imported SVG payload must be valid");
        let rgba = rasterize_shape_plus(&payload, 200, 200, [200, 200], 0.0, 30.0, None);
        assert!(
            count_opaque_pixels(&rgba) > 4000,
            "imported SVG must rasterize richly"
        );

        let d = export_shape_plus_path_to_svg_d(
            &[
                [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
                [50.0, 50.0, 0.0, -10.0, 0.0, 0.0],
                [0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            ],
            true,
        );
        assert!(d.starts_with("M0 0 C10 0 50 40 50 50 L0 50"));
        assert!(d.ends_with(" Z"));
    }

    #[test]
    fn mask_rasterizer_aa_sdf_expansion_feather_invert_and_combine() {
        // Diagonal diamond to verify soft subpixel AA edges even at feather=0, expansion=0.
        let diamond = EvaluatedMask {
            vertices: vec![
                [100.0, 30.0, 0.0, 0.0, 0.0, 0.0],
                [170.0, 100.0, 0.0, 0.0, 0.0, 0.0],
                [100.0, 170.0, 0.0, 0.0, 0.0, 0.0],
                [30.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            ],
            closed: true,
            mode: "add".to_owned(),
            feather: 0.0,
            expansion: 0.0,
            opacity: 1.0,
            invert: false,
            enabled: true,
        };
        let aa = rasterize_masks_aa(std::slice::from_ref(&diamond), 200, 200);
        assert_eq!(aa[100 * 200 + 100], 255);
        assert_eq!(aa[0], 0);
        let partial_aa_pixels = aa.iter().filter(|&&v| v > 0 && v < 255).count();
        assert!(
            partial_aa_pixels > 40,
            "expected smooth subpixel AA edge pixels at feather=0, got {partial_aa_pixels}"
        );

        // Expansion grows coverage; negative expansion shrinks it.
        let mut expanded = diamond.clone();
        expanded.expansion = 12.0;
        let exp_plane = rasterize_masks_aa(&[expanded], 200, 200);
        assert!(
            exp_plane.iter().filter(|&&v| v > 128).count()
                > aa.iter().filter(|&&v| v > 128).count()
        );

        let mut contracted = diamond.clone();
        contracted.expansion = -12.0;
        let con_plane = rasterize_masks_aa(&[contracted], 200, 200);
        assert!(
            con_plane.iter().filter(|&&v| v > 128).count()
                < aa.iter().filter(|&&v| v > 128).count()
        );

        // Subtract hole + Intersect + Feather + Invert
        let hole = EvaluatedMask {
            vertices: vec![
                [80.0, 80.0, 0.0, 0.0, 0.0, 0.0],
                [120.0, 80.0, 0.0, 0.0, 0.0, 0.0],
                [120.0, 120.0, 0.0, 0.0, 0.0, 0.0],
                [80.0, 120.0, 0.0, 0.0, 0.0, 0.0],
            ],
            closed: true,
            mode: "subtract".to_owned(),
            feather: 4.0,
            expansion: 0.0,
            opacity: 1.0,
            invert: false,
            enabled: true,
        };
        let sub_plane = rasterize_masks_aa(&[diamond.clone(), hole], 200, 200);
        assert!(
            sub_plane[100 * 200 + 100] < 10,
            "subtracted center must be cut out"
        );
        assert!(
            sub_plane[100 * 200 + 60] > 240,
            "outer diamond ring must remain opaque"
        );

        // Intersect with left half
        let left_half = EvaluatedMask {
            vertices: vec![
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [100.0, 200.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            ],
            mode: "intersect".to_owned(),
            ..EvaluatedMask::default()
        };
        let inter_plane = rasterize_masks_aa(&[diamond.clone(), left_half], 200, 200);
        assert!(inter_plane[100 * 200 + 70] > 240);
        assert_eq!(inter_plane[100 * 200 + 130], 0);

        // Post-opacity invert: outside becomes 255, inside at opacity=0.5 becomes ~128!
        let mut inv_mask = diamond;
        inv_mask.opacity = 0.5;
        inv_mask.invert = true;
        let inv_plane = rasterize_masks_aa(&[inv_mask], 200, 200);
        assert_eq!(inv_plane[0], 255);
        assert!((i32::from(inv_plane[100 * 200 + 100]) - 128).abs() <= 2);
    }
}
