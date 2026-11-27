//! Shape+ document schema, tree CRUD, keyframe evaluation, and mask coverage model.

use super::*;

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

pub(crate) fn observe_canonical_id(id: &str, max_seq: &mut u64) {
    if let Some(rest) = id.strip_prefix('s')
        && let Ok(val) = u64::from_str_radix(rest, 36)
        && val > *max_seq
    {
        *max_seq = val;
    }
}

pub(crate) fn parse_items(arr: &[Value], depth: usize, max_seq: &mut u64) -> Vec<ShapePlusItem> {
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

pub(crate) fn validate_contents_slice(
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

pub(crate) fn alloc_next_id(payload: &mut Value) -> String {
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

pub(crate) fn default_item_json(kind: &str, id: String) -> Result<Value, String> {
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
            props.insert("position".to_owned(), json!([0.0, 0.0]));
            props.insert("scale".to_owned(), json!([100.0, 100.0]));
            props.insert("rotation".to_owned(), json!(0.0));
            props.insert("opacity".to_owned(), json!(100.0));
            props.insert("tracking".to_owned(), json!(0.0));
            props.insert("fillColor".to_owned(), json!([1.0, 1.0, 1.0]));
            props.insert("strokeColor".to_owned(), json!([1.0, 1.0, 1.0]));
            props.insert("strokeWidth".to_owned(), json!(0.0));
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
            props.insert("dashLength".to_owned(), json!(0.0));
            props.insert("dashGap".to_owned(), json!(0.0));
            props.insert("dashOffset".to_owned(), json!(0.0));
            props.insert("taperUnits".to_owned(), json!("percentage"));
            props.insert("taperStartLength".to_owned(), json!(0.0));
            props.insert("taperEndLength".to_owned(), json!(0.0));
            props.insert("taperStartWidth".to_owned(), json!(0.0));
            props.insert("taperEndWidth".to_owned(), json!(0.0));
            props.insert("taperStartEase".to_owned(), json!(0.0));
            props.insert("taperEndEase".to_owned(), json!(0.0));
            props.insert("waveAmount".to_owned(), json!(0.0));
            props.insert("waveUnits".to_owned(), json!("pixels"));
            props.insert("waveLength".to_owned(), json!(20.0));
            props.insert("wavePhase".to_owned(), json!(0.0));
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

pub(crate) fn find_group_contents_mut<'a>(
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
pub(crate) type ExprEval<'a> = Option<ExprEvalFn<'a>>;

#[inline]
pub(crate) fn clamp_f64(v: f64, lo: f64, hi: f64) -> f64 {
    v.clamp(lo, hi)
}

#[inline]
pub(crate) fn wrap_f64(v: f64, period: f64) -> f64 {
    ((v % period) + period) % period
}

#[inline]
pub(crate) fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Parses a named easing string (e.g. `"linear"`, `"hold"`, `"easeInOutCubic"`, `"quadIn"`)
/// into `(InterpolationMode, EasingKind)`.
#[must_use]
pub fn parse_easing_name(name: &str) -> Option<(InterpolationMode, EasingKind)> {
    let norm: String = name
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect();
    match norm.as_str() {
        "hold" | "none" | "step" | "discrete" => {
            Some((InterpolationMode::Discrete, EasingKind::Hold))
        }
        "linear" => Some((InterpolationMode::Interpolated, EasingKind::Linear)),
        "bezier" | "custombezier" => Some((InterpolationMode::Bezier, EasingKind::Linear)),
        "quadin" | "easeinquad" | "quadraticin" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuadIn))
        }
        "quadout" | "easeoutquad" | "quadraticout" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuadOut))
        }
        "quadinout" | "easeinoutquad" | "quadraticinout" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuadInOut))
        }
        "cubicin" | "easeincubic" | "easein" => {
            Some((InterpolationMode::Interpolated, EasingKind::CubicIn))
        }
        "cubicout" | "easeoutcubic" | "easeout" => {
            Some((InterpolationMode::Interpolated, EasingKind::CubicOut))
        }
        "cubicinout" | "easeinoutcubic" | "easeinout" | "ease" => {
            Some((InterpolationMode::Interpolated, EasingKind::CubicInOut))
        }
        "cubicoutin" | "easeoutincubic" => {
            Some((InterpolationMode::Interpolated, EasingKind::CubicOutIn))
        }
        "quartin" | "easeinquart" | "quarticin" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuartIn))
        }
        "quartout" | "easeoutquart" | "quarticout" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuartOut))
        }
        "quartinout" | "easeinoutquart" | "quarticinout" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuartInOut))
        }
        "quintin" | "easeinquint" | "quinticin" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuintIn))
        }
        "quintout" | "easeoutquint" | "quinticout" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuintOut))
        }
        "quintinout" | "easeinoutquint" | "quinticinout" => {
            Some((InterpolationMode::Interpolated, EasingKind::QuintInOut))
        }
        "sinein" | "easeinsine" | "sinusoidalin" => {
            Some((InterpolationMode::Interpolated, EasingKind::SineIn))
        }
        "sineout" | "easeoutsine" | "sinusoidalout" => {
            Some((InterpolationMode::Interpolated, EasingKind::SineOut))
        }
        "sineinout" | "easeinoutsine" | "sinusoidalinout" => {
            Some((InterpolationMode::Interpolated, EasingKind::SineInOut))
        }
        "expoin" | "easeinexpo" | "exponentialin" => {
            Some((InterpolationMode::Interpolated, EasingKind::ExpoIn))
        }
        "expoout" | "easeoutexpo" | "exponentialout" => {
            Some((InterpolationMode::Interpolated, EasingKind::ExpoOut))
        }
        "expoinout" | "easeinoutexpo" | "exponentialinout" => {
            Some((InterpolationMode::Interpolated, EasingKind::ExpoInOut))
        }
        "circin" | "easeincirc" | "circularin" => {
            Some((InterpolationMode::Interpolated, EasingKind::CircIn))
        }
        "circout" | "easeoutcirc" | "circularout" => {
            Some((InterpolationMode::Interpolated, EasingKind::CircOut))
        }
        "circinout" | "easeinoutcirc" | "circularinout" => {
            Some((InterpolationMode::Interpolated, EasingKind::CircInOut))
        }
        "elasticin" | "easeinelastic" => {
            Some((InterpolationMode::Interpolated, EasingKind::ElasticIn))
        }
        "elasticout" | "easeoutelastic" => {
            Some((InterpolationMode::Interpolated, EasingKind::ElasticOut))
        }
        "elasticinout" | "easeinoutelastic" => {
            Some((InterpolationMode::Interpolated, EasingKind::ElasticInOut))
        }
        "backin" | "easeinback" => Some((InterpolationMode::Interpolated, EasingKind::BackIn)),
        "backout" | "easeoutback" => Some((InterpolationMode::Interpolated, EasingKind::BackOut)),
        "backinout" | "easeinoutback" => {
            Some((InterpolationMode::Interpolated, EasingKind::BackInOut))
        }
        "bouncein" | "easeinbounce" => {
            Some((InterpolationMode::Interpolated, EasingKind::BounceIn))
        }
        "bounceout" | "easeoutbounce" => {
            Some((InterpolationMode::Interpolated, EasingKind::BounceOut))
        }
        "bounceinout" | "easeinoutbounce" => {
            Some((InterpolationMode::Interpolated, EasingKind::BounceInOut))
        }
        _ => None,
    }
}

pub(crate) fn parse_keyframe_mode_easing(kf: &Value) -> (InterpolationMode, EasingKind) {
    for key in ["tween", "easing"] {
        if let Some(v) = kf.get(key) {
            if let Some(code) = v.as_u64() {
                return InterpolationMode::from_cm3_tween_code(code.min(u64::from(u16::MAX)) as u16);
            }
            if let Some(name) = v.as_str()
                && let Some(parsed) = parse_easing_name(name)
            {
                return parsed;
            }
        }
    }
    (InterpolationMode::Interpolated, EasingKind::Linear)
}

pub(crate) fn parse_normalized_bezier(kf: &Value) -> Option<[f64; 4]> {
    let parse_obj = |obj: &Map<String, Value>| -> Option<[f64; 4]> {
        if let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
            obj.get("x1").and_then(Value::as_f64),
            obj.get("y1").and_then(Value::as_f64),
            obj.get("x2").and_then(Value::as_f64),
            obj.get("y2").and_then(Value::as_f64),
        ) {
            return Some([x1, y1, x2, y2]);
        }
        if let Some(inner) = obj.get("CustomBezier").and_then(Value::as_object) {
            return Some([
                inner.get("x1").and_then(Value::as_f64)?,
                inner.get("y1").and_then(Value::as_f64)?,
                inner.get("x2").and_then(Value::as_f64)?,
                inner.get("y2").and_then(Value::as_f64)?,
            ]);
        }
        None
    };

    for key in ["bezier", "controlPoints", "easing"] {
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
                && let Some(b) = parse_obj(obj)
            {
                return Some(b);
            }
        }
    }
    None
}

pub(crate) fn parse_handle_pair(kf: &Value, is_out: bool) -> Option<[f64; 2]> {
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

pub(crate) fn eval_segment_factor(k0: &Value, k1: &Value, frame: f64) -> Option<f64> {
    let f0 = k0.get("frame").and_then(Value::as_f64).unwrap_or(0.0);
    let f1 = k1.get("frame").and_then(Value::as_f64).unwrap_or(f0);
    let span = f1 - f0;
    if span <= 1e-12 || frame <= f0 {
        return Some(0.0);
    }
    if frame >= f1 {
        return Some(1.0);
    }
    let (mode, easing) = parse_keyframe_mode_easing(k0);
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

pub(crate) fn eval_segment_scalar(k0: &Value, v0: f64, k1: &Value, v1: f64, frame: f64) -> f64 {
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
    let (_, easing) = parse_keyframe_mode_easing(k0);
    let u = ((frame - f0) / span).clamp(0.0, 1.0);
    lerp_f64(v0, v1, eval_easing(easing, u))
}

#[inline]
pub(crate) fn prop_explicitly_static(v: &Value) -> bool {
    v.get("animated").and_then(Value::as_bool) == Some(false) && v.get("value").is_some()
}

/// Returns `true` if a Shape+ property JSON value is animated via keyframes, an active
/// expression, or an animated child channel in `objects`.
#[must_use]
pub fn shape_plus_prop_is_animated(val: &Value) -> bool {
    let Some(obj) = val.as_object() else {
        return false;
    };
    if obj
        .get("expression")
        .and_then(Value::as_str)
        .is_some_and(|s| !s.trim().is_empty())
    {
        return true;
    }
    if obj.get("animated").and_then(Value::as_bool) == Some(true) {
        return true;
    }
    if obj.get("animated").and_then(Value::as_bool) != Some(false)
        && obj
            .get("keyframes")
            .and_then(Value::as_array)
            .is_some_and(|kfs| kfs.len() > 1)
    {
        return true;
    }
    if let Some(objs) = obj.get("objects").and_then(Value::as_array)
        && objs.iter().any(shape_plus_prop_is_animated)
    {
        return true;
    }
    false
}

/// Evaluates a Shape+ scalar property track at `frame` using full `zc_anim` easings,
/// Hold (`tween == 0` / `easing: "hold"`), cubic Bezier handles, and optional expression evaluation.
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
    if prop_explicitly_static(v)
        && let Some(inner) = v.get("value")
    {
        return eval_prop_f64(Some(inner), frame, default, eval_expr);
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

/// Evaluates a 2D vector Shape+ property track (array, `{ objects: [x, y] }`, or keyframed track).
pub fn eval_prop_vec2(
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
    if prop_explicitly_static(v)
        && let Some(inner) = v.get("value")
    {
        return eval_prop_vec2(Some(inner), frame, default, eval_expr);
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

/// Evaluates a variable-length numeric vector Shape+ property track (array, `{ objects }`, or keyframes).
pub fn eval_prop_f64_vec(val: Option<&Value>, frame: f64, eval_expr: ExprEval<'_>) -> Vec<f64> {
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
    if prop_explicitly_static(v)
        && let Some(inner) = v.get("value")
    {
        return eval_prop_f64_vec(Some(inner), frame, eval_expr);
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

/// Evaluates a string Shape+ property track (static string or stepped/held keyframes).
pub fn eval_prop_str(val: Option<&Value>, frame: f64, default: &str) -> String {
    let Some(v) = val else {
        return default.to_owned();
    };
    if let Some(s) = v.as_str() {
        return s.to_owned();
    }
    if prop_explicitly_static(v)
        && let Some(inner) = v.get("value")
    {
        return eval_prop_str(Some(inner), frame, default);
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

fn parse_rgb_value(v: &Value) -> Option<[f64; 3]> {
    if let Some(s) = v.as_str()
        && let Ok(c) = parse_css_color(s)
    {
        return Some([c[0], c[1], c[2]]);
    }
    if let Some(arr) = v.as_array()
        && arr.len() >= 3
    {
        let r = arr[0].as_f64()?;
        let g = arr[1].as_f64()?;
        let b = arr[2].as_f64()?;
        if !r.is_finite() || !g.is_finite() || !b.is_finite() {
            return None;
        }
        let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
            1.0 / 255.0
        } else {
            1.0
        };
        return Some([
            (r * scale).clamp(0.0, 1.0),
            (g * scale).clamp(0.0, 1.0),
            (b * scale).clamp(0.0, 1.0),
        ]);
    }
    None
}

/// Evaluates an RGB color Shape+ property track (normalized `[0..1]`, `[0..255]`, CSS color string,
/// `{ objects: [r, g, b] }`, or keyframed color values).
pub fn eval_prop_color_rgb(
    val: Option<&Value>,
    frame: f64,
    default: [f64; 3],
    eval_expr: ExprEval<'_>,
) -> [f64; 3] {
    let Some(v) = val else {
        return default;
    };
    if let Some(rgb) = parse_rgb_value(v) {
        return rgb;
    }
    if let Some(objs) = v.get("objects").and_then(Value::as_array)
        && objs.len() >= 3
    {
        let r = eval_prop_f64(Some(&objs[0]), frame, default[0], eval_expr);
        let g = eval_prop_f64(Some(&objs[1]), frame, default[1], eval_expr);
        let b = eval_prop_f64(Some(&objs[2]), frame, default[2], eval_expr);
        let scale = if r > 1.0 || g > 1.0 || b > 1.0 {
            1.0 / 255.0
        } else {
            1.0
        };
        return [
            (r * scale).clamp(0.0, 1.0),
            (g * scale).clamp(0.0, 1.0),
            (b * scale).clamp(0.0, 1.0),
        ];
    }
    if prop_explicitly_static(v)
        && let Some(inner) = v.get("value")
    {
        return eval_prop_color_rgb(Some(inner), frame, default, eval_expr);
    }
    if let Some(kfs) = v.get("keyframes").and_then(Value::as_array)
        && !kfs.is_empty()
    {
        let mut sorted: Vec<(f64, [f64; 3], &Value)> = kfs
            .iter()
            .filter_map(|k| {
                let f = k.get("frame").and_then(Value::as_f64)?;
                let rgb = k.get("value").and_then(parse_rgb_value)?;
                Some((f, rgb, k))
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
                    return [
                        eval_segment_scalar(w[0].2, w[0].1[0], w[1].2, w[1].1[0], frame)
                            .clamp(0.0, 1.0),
                        eval_segment_scalar(w[0].2, w[0].1[1], w[1].2, w[1].1[1], frame)
                            .clamp(0.0, 1.0),
                        eval_segment_scalar(w[0].2, w[0].1[2], w[1].2, w[1].1[2], frame)
                            .clamp(0.0, 1.0),
                    ];
                }
            }
        }
    }
    if let Some(inner) = v.get("value") {
        return eval_prop_color_rgb(Some(inner), frame, default, eval_expr);
    }
    default
}

/// Evaluates an `enabled` boolean flag on a Shape+ item's property map (defaults to `true`).
pub fn eval_prop_enabled(
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

/// Evaluates a Shape+ gradient `stops` property track at `frame`, returning `(position, [r, g, b, a])`
/// pairs sorted by stop position.
#[must_use]
pub fn eval_prop_gradient_stops(val: Option<&Value>, frame: f64) -> Vec<(f64, [f64; 4])> {
    eval_gradient_stops(val, frame)
        .into_iter()
        .map(|s| (s.position, s.color))
        .collect()
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

pub(crate) fn close_mask_vertices(
    vertices: &[[f32; 6]],
    closed: bool,
    scale: f64,
) -> Vec<BezierVertex> {
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

/// Returns `true` when `points` spans a non-degenerate 2D bounding box, allowing
/// self-intersecting polygons (such as symmetric bowties whose signed area sums to zero)
/// to rasterize cleanly.
pub(crate) fn has_non_degenerate_bbox(points: &[[f64; 2]]) -> bool {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for &[x, y] in points {
        if x.is_finite() && y.is_finite() {
            min_x = min_x.min(x);
            max_x = max_x.max(x);
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }
    (max_x - min_x) > 1e-6 && (max_y - min_y) > 1e-6
}

/// Hard-edge even-odd fill used to obtain exact far-field sign for SDF expansion.
pub(crate) fn fill_polygon_evenodd_hard(
    points: &[[f64; 2]],
    width: usize,
    height: usize,
    dst: &mut [f32],
) {
    if width == 0 || height == 0 {
        return;
    }
    let (edges, min_y, max_y) = build_scanline_edges(&[points.to_vec()]);
    if edges.is_empty() || !min_y.is_finite() || !max_y.is_finite() {
        return;
    }
    let y0 = (min_y.floor() as isize).clamp(0, height as isize) as usize;
    let y1 = (max_y.ceil() as isize).clamp(0, height as isize) as usize;
    if y0 >= y1 {
        return;
    }
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
            let x0 = ((pair[0] - 0.5).ceil() as isize).clamp(0, width as isize) as usize;
            let x1_excl = (((pair[1] - 0.5).floor() as isize).saturating_add(1))
                .clamp(0, width as isize) as usize;
            if x0 < x1_excl {
                for px in &mut row[x0..x1_excl] {
                    *px = 1.0;
                }
            }
        }
    }
}

/// Exact point-to-segment signed distance mask expansion using Zoidium's coarse spatial grid
/// accelerator (`mask-expansion.js:143-218`).
pub(crate) fn expand_mask_sdf(
    points: &[[f64; 2]],
    expansion: f64,
    width: usize,
    height: usize,
    dst: &mut [f32],
) {
    if width == 0 || height == 0 || points.len() < 3 || !has_non_degenerate_bbox(points) {
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
        let seg_min_x = a[0].min(b[0]) - pad;
        let seg_max_x = a[0].max(b[0]) + pad;
        let seg_min_y = a[1].min(b[1]) - pad;
        let seg_max_y = a[1].max(b[1]) + pad;
        if seg_max_x < 0.0
            || seg_min_x >= width as f64
            || seg_max_y < 0.0
            || seg_min_y >= height as f64
        {
            continue;
        }
        let fx = ((seg_min_x / cell).floor() as isize).clamp(0, cells_x as isize - 1) as usize;
        let tx = ((seg_max_x / cell).floor() as isize).clamp(0, cells_x as isize - 1) as usize;
        let fy = ((seg_min_y / cell).floor() as isize).clamp(0, cells_y as isize - 1) as usize;
        let ty = ((seg_max_y / cell).floor() as isize).clamp(0, cells_y as isize - 1) as usize;
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

pub(crate) fn box_blur_1d(
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

pub(crate) fn feather_3pass_box_blur(
    channel: &mut [f32],
    width: usize,
    height: usize,
    radius: f64,
) {
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

pub(crate) fn bilinear_upsample_plane(
    src: &[f32],
    sw: usize,
    sh: usize,
    dw: usize,
    dh: usize,
) -> Vec<u8> {
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
        if !mask.enabled || mask.vertices.len() < 3 {
            continue;
        }
        scratch.fill(0.0);
        let expansion = f64::from(mask.expansion.clamp(-128.0, 128.0)) * scale;
        let closed_verts = close_mask_vertices(&mask.vertices, mask.closed, scale);
        if closed_verts.len() >= 2 {
            let flat = flatten_vertices(&closed_verts, true, DEFAULT_TOLERANCE);
            if flat.points.len() >= 3 && has_non_degenerate_bbox(&flat.points) {
                if expansion.abs() < 1e-6 {
                    // 4x supersampled subpixel AA even at feather==0 && expansion==0!
                    rasterize_polygons_aa(&[flat.points], rw, rh, true, |y, x0, cov| {
                        let row_start = y * rw + x0;
                        let dst_row = &mut scratch[row_start..row_start + cov.len()];
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

    if first {
        return vec![255_u8; width as usize * height as usize];
    }

    if rw != width as usize || rh != height as usize {
        return bilinear_upsample_plane(&coverage, rw, rh, width as usize, height as usize);
    }
    coverage
        .into_iter()
        .map(|v| (v.clamp(0.0, 1.0) * 255.0 + 0.5).floor().clamp(0.0, 255.0) as u8)
        .collect()
}
