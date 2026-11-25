use super::*;

/// Owned effector configuration parsed from a Zoidium `expressProperties` /
/// `customProperties` / `objects` JSON block.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressEffectorConfig {
    pub kind: EffectorKind,
    pub properties: BTreeMap<String, EtValue>,
}

impl ExpressEffectorConfig {
    pub fn as_layer(&self) -> EffectorLayer<'_> {
        EffectorLayer {
            kind: self.kind,
            props: &self.properties,
        }
    }
}

impl EtPropReader for ExpressEffectorConfig {
    fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
        self.properties.read(key, frame)
    }
}

/// Complete container-level ExpressText+ configuration parsed from JSON.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpressTextConfig {
    pub letters_mode: LettersMode,
    pub rotation_order: String,
    pub effectors: Vec<ExpressEffectorConfig>,
}

impl Default for ExpressTextConfig {
    fn default() -> Self {
        Self {
            letters_mode: LettersMode::PerText,
            rotation_order: "XYZ".to_string(),
            effectors: Vec::new(),
        }
    }
}

fn json_to_et_value(raw: &serde_json::Value) -> Option<EtValue> {
    match raw {
        serde_json::Value::Number(n) => n.as_f64().filter(|v| v.is_finite()).map(EtValue::Number),
        serde_json::Value::Bool(b) => Some(EtValue::Number(if *b { 1.0 } else { 0.0 })),
        serde_json::Value::String(s) => Some(EtValue::Text(s.clone())),
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                let x = arr[0].as_f64()?;
                let y = arr[1].as_f64()?;
                let z = arr[2].as_f64()?;
                if x.is_finite() && y.is_finite() && z.is_finite() {
                    return Some(EtValue::Vec3([x, y, z]));
                }
            }
            None
        }
        serde_json::Value::Object(map) => {
            // CM3 serialized property wrapper: `{ "value": ... }` or `{ "keyframes": [{ "value": ... }] }`.
            if let Some(inner) = map.get("value")
                && let Some(v) = json_to_et_value(inner)
            {
                return Some(v);
            }
            if let Some(serde_json::Value::Array(kfs)) = map.get("keyframes")
                && let Some(first) = kfs.first()
                && let Some(inner) = first.get("value").or_else(|| first.get("v"))
                && let Some(v) = json_to_et_value(inner)
            {
                return Some(v);
            }
            // Vector3 serialized as `{ "objects": [ { "value": x }, { "value": y }, { "value": z } ] }`.
            if let Some(objs) = map.get("objects") {
                let items = json_raw_array(objs);
                if items.len() >= 3 {
                    let x = json_to_et_value(items[0]).map(|v| v.as_number(0.0));
                    let y = json_to_et_value(items[1]).map(|v| v.as_number(0.0));
                    let z = json_to_et_value(items[2]).map(|v| v.as_number(0.0));
                    if let (Some(x), Some(y), Some(z)) = (x, y, z) {
                        return Some(EtValue::Vec3([x, y, z]));
                    }
                }
            }
            None
        }
        serde_json::Value::Null => None,
    }
}

/// Normalizes either a JSON array `[a, b]` or CM3 numeric-keyed list `{"0": a, "1": b, "length": 2}`.
fn json_raw_array(val: &serde_json::Value) -> Vec<&serde_json::Value> {
    match val {
        serde_json::Value::Array(arr) => arr.iter().collect(),
        serde_json::Value::Object(map) => {
            let len = map
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .map(|n| n.min(256) as usize);
            let mut out = Vec::new();
            if let Some(count) = len {
                for i in 0..count {
                    if let Some(v) = map.get(&i.to_string()) {
                        out.push(v);
                    }
                }
            } else {
                let mut indexed: Vec<(usize, &serde_json::Value)> = map
                    .iter()
                    .filter_map(|(k, v)| k.parse::<usize>().ok().map(|idx| (idx, v)))
                    .collect();
                indexed.sort_by_key(|(idx, _)| *idx);
                out.extend(indexed.into_iter().map(|(_, v)| v));
            }
            out
        }
        _ => Vec::new(),
    }
}

fn parse_flat_express_props_map(
    map: &serde_json::Map<String, serde_json::Value>,
) -> Vec<ExpressEffectorConfig> {
    let mut grouped: BTreeMap<usize, BTreeMap<String, EtValue>> = BTreeMap::new();
    for (key, raw_val) in map {
        if let Some(kind) = EffectorKind::from_property_key(key)
            && let Some(et_val) = json_to_et_value(raw_val)
        {
            let order_idx = EffectorKind::ALL
                .iter()
                .position(|&k| k == kind)
                .unwrap_or(0);
            grouped
                .entry(order_idx)
                .or_default()
                .insert(key.clone(), et_val);
        }
    }
    grouped
        .into_iter()
        .map(|(idx, properties)| ExpressEffectorConfig {
            kind: EffectorKind::ALL[idx],
            properties,
        })
        .collect()
}

fn parse_custom_section(item: &serde_json::Value) -> Option<ExpressEffectorConfig> {
    let obj = item.as_object()?;
    let mut props = BTreeMap::new();
    let mut detected_kind: Option<EffectorKind> = obj
        .get("kind")
        .or_else(|| obj.get("name"))
        .and_then(serde_json::Value::as_str)
        .and_then(EffectorKind::from_name)
        .or_else(|| {
            obj.get("type")
                .and_then(|t| {
                    t.as_str()
                        .or_else(|| t.get("name").and_then(serde_json::Value::as_str))
                })
                .and_then(EffectorKind::from_name)
        });

    let def_rows = obj
        .get("type")
        .and_then(|t| t.get("objects"))
        .map(json_raw_array)
        .unwrap_or_default();
    let live_rows = obj.get("objects").map(json_raw_array).unwrap_or_default();

    for (idx, row) in live_rows.iter().enumerate() {
        let key = row
            .get("effectorKey")
            .and_then(serde_json::Value::as_str)
            .or_else(|| {
                row.get("definition")
                    .and_then(|d| d.get("effectorKey"))
                    .and_then(serde_json::Value::as_str)
            })
            .or_else(|| {
                def_rows
                    .get(idx)
                    .and_then(|d| d.get("effectorKey"))
                    .and_then(serde_json::Value::as_str)
            });
        if let Some(k) = key {
            if detected_kind.is_none() {
                detected_kind = EffectorKind::from_property_key(k);
            }
            if let Some(v) = json_to_et_value(row) {
                props.insert(k.to_string(), v);
            }
        }
    }

    if let Some(serde_json::Value::Object(explicit)) = obj
        .get("properties")
        .or_else(|| obj.get("expressProperties"))
    {
        for (k, v) in explicit {
            if detected_kind.is_none() {
                detected_kind = EffectorKind::from_property_key(k);
            }
            if let Some(et_val) = json_to_et_value(v) {
                props.insert(k.clone(), et_val);
            }
        }
    }

    let kind = detected_kind?;
    Some(ExpressEffectorConfig {
        kind,
        properties: props,
    })
}

/// Resolves Zoidium's container `LettersMode` from a container or `expressProperties` JSON block,
/// matching `expresstext-plugin.js:migrateLettersScope` & `expresstext-layout.js:lettersScopeOf`.
pub fn resolve_letters_mode(value: &serde_json::Value, has_content: bool) -> LettersMode {
    let ep = value
        .get("expressProperties")
        .filter(|v| v.is_object())
        .unwrap_or(value);
    let Some(map) = ep.as_object() else {
        return if has_content {
            LettersMode::AcrossTexts
        } else {
            LettersMode::PerText
        };
    };
    if let Some(raw_mode) = map.get("lettersMode") {
        if let Some(s) = raw_mode.as_str() {
            return match s.trim().to_ascii_lowercase().as_str() {
                "pertext" | "per_text" | "per text" | "0" => LettersMode::PerText,
                _ => LettersMode::AcrossTexts,
            };
        }
        if let Some(v) = json_to_et_value(raw_mode) {
            return LettersMode::from_raw(v.as_number(1.0));
        }
    }
    let has_version = map.contains_key("lettersVersion");
    let looks_pre_scope = !map.is_empty();
    if !has_version && (looks_pre_scope || has_content) {
        LettersMode::AcrossTexts
    } else {
        LettersMode::PerText
    }
}

/// Parses a Zoidium ExpressText+ object JSON or `expressProperties` JSON block into
/// container settings (`LettersMode`, `rotation_order`) and ordered `Vec<ExpressEffectorConfig>`.
pub fn parse_express_text_config(value: &serde_json::Value) -> Result<ExpressTextConfig, String> {
    if value.is_null() {
        return Ok(ExpressTextConfig::default());
    }
    if let Some(arr) = value.as_array() {
        let effectors = arr.iter().filter_map(parse_custom_section).collect();
        return Ok(ExpressTextConfig {
            letters_mode: LettersMode::PerText,
            rotation_order: "XYZ".to_string(),
            effectors,
        });
    }
    let Some(root) = value.as_object() else {
        return Err("expressProperties JSON must be an object or array".to_string());
    };

    let has_custom = root
        .get("customProperties")
        .is_some_and(|v| !json_raw_array(v).is_empty());
    let has_objects = root
        .get("objects")
        .is_some_and(|v| !json_raw_array(v).is_empty());
    let letters_mode = resolve_letters_mode(value, has_custom || has_objects);

    let ep_map = root
        .get("expressProperties")
        .and_then(serde_json::Value::as_object)
        .unwrap_or(root);

    let rotation_order = ep_map
        .get("rotationOrder")
        .and_then(json_to_et_value)
        .and_then(|v| v.as_str().map(str::to_string))
        .filter(|s| matches!(s.as_str(), "XYZ" | "XZY" | "YXZ" | "YZX" | "ZXY" | "ZYX"))
        .unwrap_or_else(|| "XYZ".to_string());

    let mut effectors = Vec::new();

    // 1. Explicit `effectors` array if present.
    if let Some(list) = root.get("effectors").or_else(|| ep_map.get("effectors")) {
        for item in json_raw_array(list) {
            if let Some(cfg) = parse_custom_section(item) {
                effectors.push(cfg);
            }
        }
    }

    // 2. Zoidium v9 inline `customProperties` effector sections (panel order).
    if let Some(customs) = root.get("customProperties") {
        for item in json_raw_array(customs) {
            if let Some(cfg) = parse_custom_section(item) {
                effectors.push(cfg);
            }
        }
    }

    // 3. Legacy child effector objects in `objects`.
    if let Some(objs) = root.get("objects") {
        for child in json_raw_array(objs) {
            if let Some(kind) = child
                .get("type")
                .and_then(serde_json::Value::as_str)
                .and_then(EffectorKind::from_name)
            {
                let mut props = BTreeMap::new();
                if let Some(child_ep) = child
                    .get("expressProperties")
                    .or_else(|| child.get("properties"))
                    .and_then(serde_json::Value::as_object)
                {
                    for (k, v) in child_ep {
                        if let Some(et_val) = json_to_et_value(v) {
                            props.insert(k.clone(), et_val);
                        }
                    }
                }
                effectors.push(ExpressEffectorConfig {
                    kind,
                    properties: props,
                });
            }
        }
    }

    // 4. Flat prefixed keys inside `expressProperties` (or root map).
    effectors.extend(parse_flat_express_props_map(ep_map));

    Ok(ExpressTextConfig {
        letters_mode,
        rotation_order,
        effectors,
    })
}

/// Parses Zoidium's `expressProperties` (or full ExpressText+ object) JSON into
/// `Vec<ExpressEffectorConfig>`.
pub fn parse_express_properties(
    value: &serde_json::Value,
) -> Result<Vec<ExpressEffectorConfig>, String> {
    parse_express_text_config(value).map(|cfg| cfg.effectors)
}

/// Parses a raw JSON string containing Zoidium's `expressProperties` into
/// `Vec<ExpressEffectorConfig>`.
pub fn parse_express_properties_str(json: &str) -> Result<Vec<ExpressEffectorConfig>, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid expressProperties JSON: {e}"))?;
    parse_express_properties(&val)
}

/// Parses a raw JSON string into [`ExpressTextConfig`].
pub fn parse_express_text_config_str(json: &str) -> Result<ExpressTextConfig, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid ExpressText+ JSON: {e}"))?;
    parse_express_text_config(&val)
}
