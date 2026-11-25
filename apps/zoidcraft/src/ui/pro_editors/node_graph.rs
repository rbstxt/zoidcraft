//! Node graph data model, state, catalogs lookups, palette, canvas and inspector.

use crate::ui::controls::{ButtonVariant, cm3_prop_button};
use crate::ui::theme::{
    BG_BUTTON, BG_PANEL, BG_SUNKEN, BG_TITLE, BORDER_CONTROL, BORDER_PANEL, Cm3Theme, DANGER_TEXT,
    FONT_COMPACT, FONT_SMALL, FONT_TITLE, TEXT_BRIGHT, TEXT_FAINT, TEXT_MUTED, TEXT_PRIMARY,
    TEXT_STRONG, color32_with_alpha,
};
use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Id, Key, Pos2, Rect, ScrollArea, Sense,
    Shape, Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use serde_json::{Map, Value, json};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use zc_session::{Command, Session};
use zc_shader::{bake_node_material_maps, compile_node_effect_to_glsl};

use super::*;

// ===========================================================================
// 1. Interactive Node Graph Editor (`nodeeffect` & `NodeMaterial`)
// ===========================================================================

/// Target graph domain for the interactive Node Graph Editor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NodeGraphTargetKind {
    #[default]
    NodeEffect,
    NodeMaterial,
}

impl NodeGraphTargetKind {
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::NodeEffect => "Node Effect",
            Self::NodeMaterial => "Node Material",
        }
    }
}

/// Typed socket classification for color coding and visual clarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PortDataType {
    Color,
    Float,
    Vec2,
    Vec3,
    Uv,
}

impl PortDataType {
    #[must_use]
    pub const fn color(self) -> Color32 {
        match self {
            Self::Color => Color32::from_rgb(235, 184, 72),
            Self::Float => Color32::from_rgb(122, 174, 232),
            Self::Vec2 => Color32::from_rgb(82, 204, 176),
            Self::Vec3 => Color32::from_rgb(168, 134, 240),
            Self::Uv => Color32::from_rgb(118, 214, 114),
        }
    }

    #[must_use]
    pub const fn short_label(self) -> &'static str {
        match self {
            Self::Color => "RGBA",
            Self::Float => "Float",
            Self::Vec2 => "Vec2",
            Self::Vec3 => "Vec3",
            Self::Uv => "UV",
        }
    }
}

/// Static socket definition on a node card.
#[derive(Debug, Clone, Copy)]
pub struct PortSpec {
    pub id: &'static str,
    pub label: &'static str,
    pub data_type: PortDataType,
}

/// Parameter widget kind in the Node Graph right-hand inspector.
#[derive(Debug, Clone, Copy)]
pub enum NodeParamKind {
    Float { min: f64, max: f64, step: f64 },
    Int { min: i64, max: i64 },
    Bool,
    Choice(&'static [&'static str]),
    ColorRgb,
    ColorRamp,
    Text,
}

/// Static parameter metadata for a node type.
#[derive(Debug, Clone, Copy)]
pub struct NodeParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub kind: NodeParamKind,
    pub default_f64: f64,
}

/// Catalog descriptor for one node type in either `NodeEffect` or `NodeMaterial`.
#[derive(Debug, Clone, Copy)]
pub struct NodeTypeSpec {
    pub kind: &'static str,
    pub title: &'static str,
    pub category: &'static str,
    pub inputs: &'static [PortSpec],
    pub outputs: &'static [PortSpec],
    pub params: &'static [NodeParamSpec],
}

impl NodeTypeSpec {
    #[must_use]
    pub fn header_color(&self) -> Color32 {
        match self.category {
            "Inputs" => Color32::from_rgb(56, 112, 152),
            "Generators" | "Procedural" => Color32::from_rgb(66, 128, 96),
            "Color" | "Color & Math" => Color32::from_rgb(148, 102, 48),
            "Math" => Color32::from_rgb(112, 84, 156),
            "UV & Filters" | "Surface" => Color32::from_rgb(52, 124, 134),
            "Output" => Color32::from_rgb(156, 62, 74),
            _ => Color32::from_rgb(88, 92, 106),
        }
    }
}

/// Editable in-memory node inside the Node Graph Editor.
#[derive(Debug, Clone, PartialEq)]
pub struct GraphNode {
    pub id: String,
    pub kind: String,
    pub title: String,
    pub pos: [f32; 2],
    pub params: Map<String, Value>,
}

/// Directed link from an output port to an input port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphLink {
    pub from_node: String,
    pub from_port: String,
    pub to_node: String,
    pub to_port: String,
}

/// Live drag state when creating a new wire between ports.
#[derive(Debug, Clone)]
pub struct PendingWireDrag {
    pub from_node: String,
    pub from_port: String,
    pub data_type: PortDataType,
}

/// Compile validation status reported in the bottom bar of the Node Graph Editor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeCompileStatus {
    Ok(String),
    Error(String),
}

impl Default for NodeCompileStatus {
    fn default() -> Self {
        Self::Ok(String::from("Compiled OK"))
    }
}

/// Full interactive state for the Node Graph Editor window.
#[derive(Debug, Clone)]
pub struct NodeGraphEditorState {
    pub open: bool,
    pub target_kind: NodeGraphTargetKind,
    pub target_id: String,
    pub property_key: String,
    pub target_label: String,
    pub nodes: Vec<GraphNode>,
    pub links: Vec<GraphLink>,
    pub selected_node: Option<String>,
    pub pan: Vec2,
    pub zoom: f32,
    pub palette_open: bool,
    pub palette_search: String,
    pub auto_apply: bool,
    pub dirty: bool,
    pub compile_status: NodeCompileStatus,
    pub next_node_seq: usize,
    pub dragging_node: Option<String>,
    pub pending_wire: Option<PendingWireDrag>,
    pub just_opened: bool,
    pub needs_initial_fit: bool,
}

impl Default for NodeGraphEditorState {
    fn default() -> Self {
        Self {
            open: false,
            target_kind: NodeGraphTargetKind::NodeEffect,
            target_id: String::new(),
            property_key: String::from("graph"),
            target_label: String::new(),
            nodes: Vec::new(),
            links: Vec::new(),
            selected_node: None,
            pan: Vec2::new(40.0, 40.0),
            zoom: 1.0,
            palette_open: false,
            palette_search: String::new(),
            auto_apply: true,
            dirty: false,
            compile_status: NodeCompileStatus::default(),
            next_node_seq: 1,
            dragging_node: None,
            pending_wire: None,
            just_opened: false,
            needs_initial_fit: false,
        }
    }
}

impl NodeGraphEditorState {
    /// Opens the Node Graph Editor bound to either a `nodeeffect` effect or a 3D `NodeMaterial`.
    pub fn open_for_target(
        &mut self,
        target_kind: NodeGraphTargetKind,
        target_id: impl Into<String>,
        property_key: impl Into<String>,
        target_label: impl Into<String>,
        graph_json: Option<&Value>,
    ) {
        self.open = true;
        self.just_opened = true;
        self.needs_initial_fit = true;
        self.target_kind = target_kind;
        self.target_id = target_id.into();
        self.property_key = property_key.into();
        self.target_label = target_label.into();
        self.pan = Vec2::new(36.0, 48.0);
        self.zoom = 1.0;
        self.palette_open = false;
        self.palette_search.clear();
        self.dragging_node = None;
        self.pending_wire = None;
        self.load_from_json(graph_json);
        self.revalidate();
        self.dirty = false;
    }

    /// Parses a Zoidium graph JSON object (or populates a clean starter graph when empty).
    pub fn load_from_json(&mut self, graph_json: Option<&Value>) {
        self.nodes.clear();
        self.links.clear();
        self.selected_node = None;

        if let Some(Value::Object(obj)) = graph_json {
            if let Some(arr) = obj.get("nodes").and_then(Value::as_array) {
                for (i, item) in arr.iter().enumerate() {
                    let Some(nobj) = item.as_object() else {
                        continue;
                    };
                    let id = nobj
                        .get("id")
                        .and_then(|v| {
                            v.as_str()
                                .map(str::to_string)
                                .or_else(|| v.as_i64().map(|n| n.to_string()))
                        })
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| format!("n{}", i + 1));
                    let kind = nobj
                        .get("type")
                        .or_else(|| nobj.get("kind"))
                        .and_then(Value::as_str)
                        .unwrap_or("output")
                        .to_string();
                    let title = nobj
                        .get("title")
                        .or_else(|| nobj.get("label"))
                        .and_then(Value::as_str)
                        .map(str::to_string)
                        .unwrap_or_else(|| {
                            find_node_spec(self.target_kind, &kind)
                                .map(|s| s.title.to_string())
                                .unwrap_or_else(|| kind.clone())
                        });
                    let x = nobj
                        .get("x")
                        .or_else(|| nobj.get("pos").and_then(|p| p.get(0)))
                        .and_then(Value::as_f64)
                        .unwrap_or(60.0 + (i as f64) * 240.0) as f32;
                    let y = nobj
                        .get("y")
                        .or_else(|| nobj.get("pos").and_then(|p| p.get(1)))
                        .and_then(Value::as_f64)
                        .unwrap_or(80.0 + ((i % 3) as f64) * 140.0)
                        as f32;
                    let mut params = Map::new();
                    for key in ["values", "inputs", "params"] {
                        if let Some(Value::Object(map)) = nobj.get(key) {
                            for (k, v) in map {
                                params.insert(k.clone(), v.clone());
                            }
                        }
                    }
                    self.nodes.push(GraphNode {
                        id,
                        kind,
                        title,
                        pos: [x, y],
                        params,
                    });
                }
            }
            if let Some(arr) = obj
                .get("links")
                .or_else(|| obj.get("edges"))
                .or_else(|| obj.get("connections"))
                .and_then(Value::as_array)
            {
                for item in arr {
                    let Some(lobj) = item.as_object() else {
                        continue;
                    };
                    let Some((from_node, from_port)) = extract_link_ep(
                        lobj,
                        "from",
                        "fromNode",
                        "source",
                        "fromPort",
                        "sourcePort",
                        "out",
                    ) else {
                        continue;
                    };
                    let Some((to_node, to_port)) = extract_link_ep(
                        lobj,
                        "to",
                        "toNode",
                        "target",
                        "toPort",
                        "targetPort",
                        "in",
                    ) else {
                        continue;
                    };
                    self.links.push(GraphLink {
                        from_node,
                        from_port,
                        to_node,
                        to_port,
                    });
                }
            }
        }

        if self.nodes.is_empty() {
            self.load_default_template();
        }
        self.refresh_next_node_seq();
        self.selected_node = self.nodes.first().map(|n| n.id.clone());
    }

    pub fn refresh_next_node_seq(&mut self) {
        let mut max_n = self.nodes.len();
        for node in &self.nodes {
            if let Some(num) = node
                .id
                .strip_prefix('n')
                .and_then(|rest| rest.parse::<usize>().ok())
            {
                max_n = max_n.max(num);
            }
        }
        self.next_node_seq = max_n + 1;
    }

    /// Loads a starter graph appropriate for the active target kind.
    pub fn load_default_template(&mut self) {
        self.nodes.clear();
        self.links.clear();
        match self.target_kind {
            NodeGraphTargetKind::NodeEffect => {
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n1",
                    "layer",
                    [40.0, 120.0],
                ));
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n2",
                    "output",
                    [360.0, 120.0],
                ));
                self.links.push(GraphLink {
                    from_node: "n1".into(),
                    from_port: "color".into(),
                    to_node: "n2".into(),
                    to_port: "color".into(),
                });
                self.links.push(GraphLink {
                    from_node: "n1".into(),
                    from_port: "alpha".into(),
                    to_node: "n2".into(),
                    to_port: "alpha".into(),
                });
            }
            NodeGraphTargetKind::NodeMaterial => {
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n1",
                    "noise",
                    [30.0, 110.0],
                ));
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n2",
                    "colorRamp",
                    [260.0, 110.0],
                ));
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n3",
                    "output",
                    [510.0, 110.0],
                ));
                self.links.push(GraphLink {
                    from_node: "n1".into(),
                    from_port: "out".into(),
                    to_node: "n2".into(),
                    to_port: "factor".into(),
                });
                self.links.push(GraphLink {
                    from_node: "n2".into(),
                    from_port: "color".into(),
                    to_node: "n3".into(),
                    to_port: "color".into(),
                });
            }
        }
        self.refresh_next_node_seq();
        self.selected_node = self.nodes.first().map(|n| n.id.clone());
    }

    /// Loads one of the built-in quick-start graph templates.
    pub fn apply_template(&mut self, template_idx: usize) {
        self.nodes.clear();
        self.links.clear();
        match (self.target_kind, template_idx) {
            (NodeGraphTargetKind::NodeEffect, 0) => self.load_default_template(),
            (NodeGraphTargetKind::NodeEffect, 1) => {
                // Glow + Vignette pipeline
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n1",
                    "uv",
                    [30.0, 120.0],
                ));
                let mut glow = make_default_node(self.target_kind, "n2", "glow", [240.0, 90.0]);
                glow.params.insert("threshold".into(), json!(0.65));
                glow.params.insert("radius".into(), json!(0.018));
                glow.params.insert("intensity".into(), json!(0.85));
                self.nodes.push(glow);
                let mut vig = make_default_node(self.target_kind, "n3", "vignette", [470.0, 90.0]);
                vig.params.insert("radius".into(), json!(0.62));
                vig.params.insert("softness".into(), json!(0.38));
                vig.params.insert("strength".into(), json!(0.45));
                self.nodes.push(vig);
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n4",
                    "output",
                    [700.0, 100.0],
                ));
                self.links.extend([
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "uv".into(),
                        to_node: "n2".into(),
                        to_port: "uv".into(),
                    },
                    GraphLink {
                        from_node: "n2".into(),
                        from_port: "color".into(),
                        to_node: "n3".into(),
                        to_port: "color".into(),
                    },
                    GraphLink {
                        from_node: "n3".into(),
                        from_port: "color".into(),
                        to_node: "n4".into(),
                        to_port: "color".into(),
                    },
                    GraphLink {
                        from_node: "n2".into(),
                        from_port: "alpha".into(),
                        to_node: "n4".into(),
                        to_port: "alpha".into(),
                    },
                ]);
            }
            (NodeGraphTargetKind::NodeEffect, 2) => {
                // Noise UV Displace + Chromatic Aberration
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n1",
                    "uv",
                    [20.0, 60.0],
                ));
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n2",
                    "time",
                    [20.0, 210.0],
                ));
                let mut noise = make_default_node(self.target_kind, "n3", "fbm", [220.0, 170.0]);
                noise.params.insert("scale".into(), json!(5.0));
                self.nodes.push(noise);
                let mut disp = make_default_node(self.target_kind, "n4", "displace", [430.0, 80.0]);
                disp.params.insert("strength".into(), json!(0.035));
                self.nodes.push(disp);
                let mut ca =
                    make_default_node(self.target_kind, "n5", "chromaticAberration", [650.0, 80.0]);
                ca.params.insert("amount".into(), json!(0.008));
                self.nodes.push(ca);
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n6",
                    "output",
                    [880.0, 90.0],
                ));
                self.links.extend([
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "uv".into(),
                        to_node: "n3".into(),
                        to_port: "uv".into(),
                    },
                    GraphLink {
                        from_node: "n2".into(),
                        from_port: "out".into(),
                        to_node: "n3".into(),
                        to_port: "time".into(),
                    },
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "uv".into(),
                        to_node: "n4".into(),
                        to_port: "uv".into(),
                    },
                    GraphLink {
                        from_node: "n3".into(),
                        from_port: "out".into(),
                        to_node: "n4".into(),
                        to_port: "map".into(),
                    },
                    GraphLink {
                        from_node: "n4".into(),
                        from_port: "uv".into(),
                        to_node: "n5".into(),
                        to_port: "uv".into(),
                    },
                    GraphLink {
                        from_node: "n5".into(),
                        from_port: "color".into(),
                        to_node: "n6".into(),
                        to_port: "color".into(),
                    },
                    GraphLink {
                        from_node: "n5".into(),
                        from_port: "alpha".into(),
                        to_node: "n6".into(),
                        to_port: "alpha".into(),
                    },
                ]);
            }
            (NodeGraphTargetKind::NodeMaterial, 0) => self.load_default_template(),
            (NodeGraphTargetKind::NodeMaterial, 1) => {
                // Procedural Marble PBR Material
                let mut marble = make_default_node(self.target_kind, "n1", "marble", [30.0, 80.0]);
                marble.params.insert("scale".into(), json!(4.5));
                self.nodes.push(marble);
                let mut ramp =
                    make_default_node(self.target_kind, "n2", "colorRamp", [260.0, 60.0]);
                ramp.params.insert(
                    "stops".into(),
                    json!([
                        {"pos": 0.0, "color": [0.12, 0.14, 0.18]},
                        {"pos": 0.45, "color": [0.85, 0.87, 0.90]},
                        {"pos": 1.0, "color": [0.98, 0.96, 0.93]}
                    ]),
                );
                self.nodes.push(ramp);
                let mut rough =
                    make_default_node(self.target_kind, "n3", "mapRange", [260.0, 230.0]);
                rough.params.insert("toMin".into(), json!(0.15));
                rough.params.insert("toMax".into(), json!(0.48));
                self.nodes.push(rough);
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n4",
                    "output",
                    [530.0, 110.0],
                ));
                self.links.extend([
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "out".into(),
                        to_node: "n2".into(),
                        to_port: "factor".into(),
                    },
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "out".into(),
                        to_node: "n3".into(),
                        to_port: "value".into(),
                    },
                    GraphLink {
                        from_node: "n2".into(),
                        from_port: "color".into(),
                        to_node: "n4".into(),
                        to_port: "color".into(),
                    },
                    GraphLink {
                        from_node: "n3".into(),
                        from_port: "out".into(),
                        to_node: "n4".into(),
                        to_port: "roughness".into(),
                    },
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "out".into(),
                        to_node: "n4".into(),
                        to_port: "bump".into(),
                    },
                ]);
            }
            (NodeGraphTargetKind::NodeMaterial, 2) => {
                // Brushed Metal + Fresnel Rim
                let mut vor = make_default_node(self.target_kind, "n1", "voronoi", [30.0, 60.0]);
                vor.params.insert("scale".into(), json!(10.0));
                self.nodes.push(vor);
                let mut fresnel =
                    make_default_node(self.target_kind, "n2", "fresnel", [30.0, 230.0]);
                fresnel.params.insert("ior".into(), json!(1.8));
                self.nodes.push(fresnel);
                let mut metal = make_default_node(self.target_kind, "n3", "float", [250.0, 230.0]);
                metal.params.insert("value".into(), json!(0.92));
                self.nodes.push(metal);
                let mut mix = make_default_node(self.target_kind, "n4", "mix", [260.0, 60.0]);
                mix.params.insert("a".into(), json!([0.48, 0.56, 0.68]));
                mix.params.insert("b".into(), json!([0.88, 0.92, 0.98]));
                self.nodes.push(mix);
                self.nodes.push(make_default_node(
                    self.target_kind,
                    "n5",
                    "output",
                    [520.0, 110.0],
                ));
                self.links.extend([
                    GraphLink {
                        from_node: "n2".into(),
                        from_port: "out".into(),
                        to_node: "n4".into(),
                        to_port: "factor".into(),
                    },
                    GraphLink {
                        from_node: "n4".into(),
                        from_port: "color".into(),
                        to_node: "n5".into(),
                        to_port: "color".into(),
                    },
                    GraphLink {
                        from_node: "n3".into(),
                        from_port: "out".into(),
                        to_node: "n5".into(),
                        to_port: "metalness".into(),
                    },
                    GraphLink {
                        from_node: "n1".into(),
                        from_port: "distance".into(),
                        to_node: "n5".into(),
                        to_port: "roughness".into(),
                    },
                ]);
            }
            _ => self.load_default_template(),
        }
        self.refresh_next_node_seq();
        self.selected_node = self.nodes.first().map(|n| n.id.clone());
        self.needs_initial_fit = true;
        self.dirty = true;
        self.revalidate();
    }

    /// Serializes the current graph into canonical Zoidium `schemaVersion: 1` JSON.
    #[must_use]
    pub fn to_json(&self) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .iter()
            .map(|n| {
                json!({
                    "id": n.id,
                    "type": n.kind,
                    "title": n.title,
                    "x": (n.pos[0] * 10.0).round() / 10.0,
                    "y": (n.pos[1] * 10.0).round() / 10.0,
                    "params": Value::Object(n.params.clone()),
                })
            })
            .collect();
        let links: Vec<Value> = self
            .links
            .iter()
            .map(|l| {
                json!({
                    "from": l.from_node,
                    "fromPort": l.from_port,
                    "to": l.to_node,
                    "toPort": l.to_port,
                })
            })
            .collect();
        json!({
            "schemaVersion": 1,
            "nodes": nodes,
            "links": links,
        })
    }

    /// Adds a new node of `kind` at `pos` and selects it.
    pub fn add_node(&mut self, kind: &str, pos: [f32; 2]) -> String {
        let id = format!("n{}", self.next_node_seq);
        self.next_node_seq += 1;
        let node = make_default_node(self.target_kind, &id, kind, pos);
        self.nodes.push(node);
        self.selected_node = Some(id.clone());
        self.dirty = true;
        self.revalidate();
        id
    }

    /// Duplicates the specified node with a slight spatial offset.
    pub fn duplicate_node(&mut self, node_id: &str) -> Option<String> {
        let src = self.nodes.iter().find(|n| n.id == node_id)?.clone();
        if src.kind == "output" {
            return None;
        }
        let id = format!("n{}", self.next_node_seq);
        self.next_node_seq += 1;
        let copy = GraphNode {
            id: id.clone(),
            kind: src.kind,
            title: format!("{} Copy", src.title),
            pos: [src.pos[0] + 32.0, src.pos[1] + 32.0],
            params: src.params,
        };
        self.nodes.push(copy);
        self.selected_node = Some(id.clone());
        self.dirty = true;
        self.revalidate();
        Some(id)
    }

    /// Deletes a node (preserving the final `output` node) and prunes incident links.
    pub fn remove_node(&mut self, node_id: &str) -> bool {
        if self
            .nodes
            .iter()
            .any(|n| n.id == node_id && n.kind == "output")
        {
            return false;
        }
        let before = self.nodes.len();
        self.nodes.retain(|n| n.id != node_id);
        if self.nodes.len() == before {
            return false;
        }
        self.links
            .retain(|l| l.from_node != node_id && l.to_node != node_id);
        if self.selected_node.as_deref() == Some(node_id) {
            self.selected_node = self.nodes.first().map(|n| n.id.clone());
        }
        self.dirty = true;
        self.revalidate();
        true
    }

    /// Attempts to connect `from_node.from_port -> to_node.to_port`.
    /// Rejects self-loops and cycles; replaces any existing wire feeding `(to_node, to_port)`.
    pub fn connect_ports(
        &mut self,
        from_node: &str,
        from_port: &str,
        to_node: &str,
        to_port: &str,
    ) -> Result<(), String> {
        if from_node == to_node {
            return Err("Cannot connect a node to itself".to_string());
        }
        if !self.nodes.iter().any(|n| n.id == from_node)
            || !self.nodes.iter().any(|n| n.id == to_node)
        {
            return Err("Unknown node endpoint".to_string());
        }
        if self.would_create_cycle(from_node, to_node) {
            return Err("Connection rejected: would create a cycle".to_string());
        }
        self.links
            .retain(|l| !(l.to_node == to_node && l.to_port == to_port));
        self.links.push(GraphLink {
            from_node: from_node.to_string(),
            from_port: from_port.to_string(),
            to_node: to_node.to_string(),
            to_port: to_port.to_string(),
        });
        self.dirty = true;
        self.revalidate();
        Ok(())
    }

    /// Removes any wire feeding `(to_node, to_port)`.
    pub fn disconnect_input(&mut self, to_node: &str, to_port: &str) -> bool {
        let before = self.links.len();
        self.links
            .retain(|l| !(l.to_node == to_node && l.to_port == to_port));
        if self.links.len() != before {
            self.dirty = true;
            self.revalidate();
            true
        } else {
            false
        }
    }

    /// Removes the link at `index`.
    pub fn remove_link_at(&mut self, index: usize) -> bool {
        if index < self.links.len() {
            self.links.remove(index);
            self.dirty = true;
            self.revalidate();
            true
        } else {
            false
        }
    }

    /// BFS check whether `from_node` is reachable from `to_node` via existing directed links.
    #[must_use]
    pub fn would_create_cycle(&self, from_node: &str, to_node: &str) -> bool {
        if from_node == to_node {
            return true;
        }
        let mut visited = BTreeSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(to_node.to_string());
        visited.insert(to_node.to_string());
        while let Some(cur) = queue.pop_front() {
            if cur == from_node {
                return true;
            }
            for link in &self.links {
                if link.from_node == cur && visited.insert(link.to_node.clone()) {
                    queue.push_back(link.to_node.clone());
                }
            }
        }
        false
    }

    /// Runs the real `zc_shader` compiler / baker on the current graph and updates `compile_status`.
    pub fn revalidate(&mut self) {
        let graph = self.to_json();
        self.compile_status = match self.target_kind {
            NodeGraphTargetKind::NodeEffect => match compile_node_effect_to_glsl(&graph) {
                Ok(compiled) => NodeCompileStatus::Ok(format!(
                    "Compiled OK ({} nodes, {} params, {} GLSL lines)",
                    self.nodes.len(),
                    compiled.parameters.len(),
                    compiled.glsl_source.lines().count(),
                )),
                Err(err) => NodeCompileStatus::Error(err),
            },
            NodeGraphTargetKind::NodeMaterial => match bake_node_material_maps(&graph, 16) {
                Ok(baked) => {
                    let map_count = usize::from(baked.color.is_some())
                        + usize::from(baked.emissive.is_some())
                        + usize::from(baked.roughness.is_some())
                        + usize::from(baked.metalness.is_some())
                        + usize::from(baked.normal.is_some())
                        + usize::from(baked.opacity.is_some());
                    let node_word = if self.nodes.len() == 1 {
                        "node"
                    } else {
                        "nodes"
                    };
                    let map_word = if map_count == 1 { "map" } else { "maps" };
                    NodeCompileStatus::Ok(format!(
                        "Compiled OK ({} {node_word}, {} baked PBR {map_word})",
                        self.nodes.len(),
                        map_count,
                    ))
                }
                Err(err) => NodeCompileStatus::Error(err),
            },
        };
    }

    /// Returns the canonical default `NodeMaterial` procedural graph JSON
    /// (`Perlin Noise -> Color Ramp -> Material Output`) so newly switched
    /// Node Materials render and serialize immediately before opening the editor.
    #[must_use]
    pub fn default_node_material_graph_json() -> Value {
        let mut st = Self {
            target_kind: NodeGraphTargetKind::NodeMaterial,
            ..Self::default()
        };
        st.load_default_template();
        st.to_json()
    }

    /// Commits the current graph JSON into `Session` via `Command::SetPropertyValue`.
    pub fn commit_to_session(&mut self, session: &mut Session) -> bool {
        if self.target_id.is_empty() || self.property_key.is_empty() {
            return false;
        }
        let value = self.to_json();
        let ok = session
            .execute(Command::SetPropertyValue {
                target_id: self.target_id.clone(),
                property: self.property_key.clone(),
                value,
            })
            .is_ok();
        if ok {
            self.dirty = false;
        }
        ok
    }
}

pub fn extract_link_ep(
    obj: &Map<String, Value>,
    nested: &str,
    node_a: &str,
    node_b: &str,
    port_a: &str,
    port_b: &str,
    default_port: &str,
) -> Option<(String, String)> {
    if let Some(Value::Object(ep)) = obj.get(nested) {
        let node = ep.get("node").or_else(|| ep.get("id")).and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })?;
        let port = ep
            .get("port")
            .or_else(|| ep.get("socket"))
            .and_then(Value::as_str)
            .unwrap_or(default_port)
            .to_string();
        return Some((node, port));
    }
    let node = obj
        .get(nested)
        .or_else(|| obj.get(node_a))
        .or_else(|| obj.get(node_b))
        .and_then(|v| {
            v.as_str()
                .map(str::to_string)
                .or_else(|| v.as_i64().map(|n| n.to_string()))
        })?;
    let port = obj
        .get(port_a)
        .or_else(|| obj.get(port_b))
        .and_then(Value::as_str)
        .unwrap_or(default_port)
        .to_string();
    Some((node, port))
}

pub fn make_default_node(
    target_kind: NodeGraphTargetKind,
    id: &str,
    kind: &str,
    pos: [f32; 2],
) -> GraphNode {
    let spec = find_node_spec(target_kind, kind);
    let title = spec
        .map(|s| s.title.to_string())
        .unwrap_or_else(|| kind.to_string());
    let mut params = Map::new();
    if let Some(spec) = spec {
        for p in spec.params {
            match p.kind {
                NodeParamKind::Float { .. } => {
                    params.insert(p.key.into(), json!(p.default_f64));
                }
                NodeParamKind::Int { .. } | NodeParamKind::Choice(_) => {
                    params.insert(p.key.into(), json!(p.default_f64.round() as i64));
                }
                NodeParamKind::Bool => {
                    params.insert(p.key.into(), json!(p.default_f64 != 0.0));
                }
                NodeParamKind::ColorRgb => {
                    params.insert(
                        p.key.into(),
                        json!([p.default_f64, p.default_f64, p.default_f64]),
                    );
                }
                NodeParamKind::ColorRamp => {
                    params.insert(
                        p.key.into(),
                        json!([
                            {"pos": 0.0, "color": [0.05, 0.08, 0.16]},
                            {"pos": 0.5, "color": [0.22, 0.62, 0.92]},
                            {"pos": 1.0, "color": [0.98, 0.96, 0.92]}
                        ]),
                    );
                }
                NodeParamKind::Text => {
                    params.insert(p.key.into(), json!("Param"));
                }
            }
        }
    }
    GraphNode {
        id: id.to_string(),
        kind: kind.to_string(),
        title,
        pos,
        params,
    }
}

pub fn show_node_palette_ui(
    ui: &mut Ui,
    state: &mut NodeGraphEditorState,
    theme: &Cm3Theme,
) -> bool {
    let mut changed = false;
    ui.label(
        egui::RichText::new("Node Palette")
            .font(theme.font(FONT_TITLE))
            .color(TEXT_STRONG),
    );
    ui.add_space(4.0);
    ui.add(
        TextEdit::singleline(&mut state.palette_search)
            .hint_text("Search nodes...")
            .desired_width(ui.available_width()),
    );
    ui.add_space(4.0);

    let query = state.palette_search.trim().to_ascii_lowercase();
    let catalog = node_catalog_for(state.target_kind);

    ScrollArea::vertical()
        .id_salt("node_palette_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let mut last_cat = "";
            for spec in catalog {
                if spec.kind == "output" {
                    continue;
                }
                if !query.is_empty()
                    && !spec.title.to_ascii_lowercase().contains(&query)
                    && !spec.kind.to_ascii_lowercase().contains(&query)
                    && !spec.category.to_ascii_lowercase().contains(&query)
                {
                    continue;
                }
                if spec.category != last_cat {
                    if !last_cat.is_empty() {
                        ui.add_space(4.0);
                    }
                    ui.label(
                        egui::RichText::new(spec.category.to_ascii_uppercase())
                            .font(theme.font(FONT_COMPACT))
                            .color(TEXT_FAINT),
                    );
                    last_cat = spec.category;
                }
                let btn = ui.add_sized(
                    Vec2::new(ui.available_width(), 22.0),
                    egui::Button::new(
                        egui::RichText::new(spec.title)
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_PRIMARY),
                    )
                    .fill(BG_BUTTON)
                    .stroke(Stroke::new(1.0, BORDER_CONTROL)),
                );
                if btn.clicked() {
                    let offset = (state.nodes.len() as f32) * 18.0;
                    let gx = (-state.pan.x + 140.0 + (offset % 160.0)) / state.zoom.max(0.2);
                    let gy =
                        (-state.pan.y + 100.0 + ((offset * 1.3) % 140.0)) / state.zoom.max(0.2);
                    state.add_node(spec.kind, [gx, gy]);
                    changed = true;
                }
            }
            let _ = theme;
        });
    changed
}

pub const NODE_BASE_WIDTH: f32 = 172.0;
pub const NODE_HEADER_H: f32 = 24.0;
pub const NODE_PORT_ROW_H: f32 = 20.0;
pub const NODE_FOOTER_PAD: f32 = 8.0;

pub fn node_size_graph_space(spec: Option<&NodeTypeSpec>) -> Vec2 {
    let rows = spec
        .map(|s| s.inputs.len().max(s.outputs.len()).max(1))
        .unwrap_or(1);
    Vec2::new(
        NODE_BASE_WIDTH,
        NODE_HEADER_H + (rows as f32) * NODE_PORT_ROW_H + NODE_FOOTER_PAD,
    )
}

pub fn graph_to_screen(canvas_rect: Rect, pan: Vec2, zoom: f32, pos: [f32; 2]) -> Pos2 {
    Pos2::new(
        canvas_rect.min.x + pan.x + pos[0] * zoom,
        canvas_rect.min.y + pan.y + pos[1] * zoom,
    )
}

pub fn show_node_canvas_ui(
    ui: &mut Ui,
    canvas_rect: Rect,
    state: &mut NodeGraphEditorState,
    theme: &Cm3Theme,
) -> bool {
    let mut changed = false;
    let painter = ui.painter().with_clip_rect(canvas_rect);

    // Canvas background & grid
    painter.rect_filled(canvas_rect, CornerRadius::same(4), BG_SUNKEN);
    let minor_step = (24.0 * state.zoom).max(10.0);
    let major_step = minor_step * 4.0;
    let grid_minor = Color32::from_rgba_unmultiplied(255, 255, 255, 10);
    let grid_major = Color32::from_rgba_unmultiplied(255, 255, 255, 22);

    let start_x = canvas_rect.min.x + state.pan.x.rem_euclid(minor_step);
    let mut gx = start_x;
    while gx < canvas_rect.max.x {
        let is_major = ((gx - canvas_rect.min.x - state.pan.x) / major_step)
            .fract()
            .abs()
            < 0.04;
        painter.line_segment(
            [
                Pos2::new(gx, canvas_rect.min.y),
                Pos2::new(gx, canvas_rect.max.y),
            ],
            Stroke::new(1.0, if is_major { grid_major } else { grid_minor }),
        );
        gx += minor_step;
    }
    let start_y = canvas_rect.min.y + state.pan.y.rem_euclid(minor_step);
    let mut gy = start_y;
    while gy < canvas_rect.max.y {
        let is_major = ((gy - canvas_rect.min.y - state.pan.y) / major_step)
            .fract()
            .abs()
            < 0.04;
        painter.line_segment(
            [
                Pos2::new(canvas_rect.min.x, gy),
                Pos2::new(canvas_rect.max.x, gy),
            ],
            Stroke::new(1.0, if is_major { grid_major } else { grid_minor }),
        );
        gy += minor_step;
    }

    // Canvas interaction sensor
    let canvas_resp = ui.interact(
        canvas_rect,
        Id::unique("pro_node_canvas_bg"),
        Sense::click_and_drag(),
    );

    // Scroll wheel zoom around cursor
    if canvas_resp.hovered() {
        let scroll_y = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll_y.abs() > 0.5 {
            let old_zoom = state.zoom;
            let factor = (1.0 + scroll_y * 0.0018).clamp(0.82, 1.22);
            let new_zoom = (old_zoom * factor).clamp(0.35, 2.2);
            if let Some(pointer) = ui.input(|i| i.pointer.hover_pos()) {
                let rel = pointer - canvas_rect.min;
                let world_before = (rel - state.pan) / old_zoom;
                state.zoom = new_zoom;
                state.pan = rel - world_before * new_zoom;
            } else {
                state.zoom = new_zoom;
            }
        }
    }

    // Keyboard delete selected node
    if canvas_resp.hovered()
        && !ui.ctx().egui_wants_keyboard_input()
        && ui.input(|i| i.key_pressed(Key::Delete) || i.key_pressed(Key::Backspace))
        && let Some(sel) = state.selected_node.clone()
        && state.remove_node(&sel)
    {
        changed = true;
    }

    // Compute screen positions of every node and every socket
    struct PortScreenPos {
        node_id: String,
        port_id: String,
        data_type: PortDataType,
        center: Pos2,
    }
    let mut node_rects: Vec<(usize, Rect)> = Vec::with_capacity(state.nodes.len());
    let mut input_sockets: Vec<PortScreenPos> = Vec::new();
    let mut output_sockets: Vec<PortScreenPos> = Vec::new();
    let mut out_pos_map: BTreeMap<(String, String), (Pos2, PortDataType)> = BTreeMap::new();
    let mut in_pos_map: BTreeMap<(String, String), (Pos2, PortDataType)> = BTreeMap::new();

    for (idx, node) in state.nodes.iter().enumerate() {
        let spec = find_node_spec(state.target_kind, &node.kind);
        let base_sz = node_size_graph_space(spec);
        let tl = graph_to_screen(canvas_rect, state.pan, state.zoom, node.pos);
        let rect = Rect::from_min_size(tl, base_sz * state.zoom);
        node_rects.push((idx, rect));

        if let Some(spec) = spec {
            for (pi, port) in spec.inputs.iter().enumerate() {
                let cy =
                    rect.min.y + (NODE_HEADER_H + (pi as f32 + 0.5) * NODE_PORT_ROW_H) * state.zoom;
                let center = Pos2::new(rect.min.x, cy);
                input_sockets.push(PortScreenPos {
                    node_id: node.id.clone(),
                    port_id: port.id.to_string(),
                    data_type: port.data_type,
                    center,
                });
                in_pos_map.insert(
                    (node.id.clone(), port.id.to_string()),
                    (center, port.data_type),
                );
            }
            for (pi, port) in spec.outputs.iter().enumerate() {
                let cy =
                    rect.min.y + (NODE_HEADER_H + (pi as f32 + 0.5) * NODE_PORT_ROW_H) * state.zoom;
                let center = Pos2::new(rect.max.x, cy);
                output_sockets.push(PortScreenPos {
                    node_id: node.id.clone(),
                    port_id: port.id.to_string(),
                    data_type: port.data_type,
                    center,
                });
                out_pos_map.insert(
                    (node.id.clone(), port.id.to_string()),
                    (center, port.data_type),
                );
            }
        }
    }

    let pointer_pos = ui.input(|i| i.pointer.interact_pos().or(i.pointer.hover_pos()));
    let primary_pressed = ui.input(|i| i.pointer.primary_pressed());
    let primary_down = ui.input(|i| i.pointer.primary_down());
    let primary_released = ui.input(|i| i.pointer.primary_released());
    let secondary_clicked = canvas_resp.secondary_clicked();
    let primary_clicked = canvas_resp.clicked();
    let middle_down = ui.input(|i| i.pointer.middle_down());
    let pointer_delta = ui.input(|i| i.pointer.delta());

    let socket_hit_r = (8.0 * state.zoom).clamp(6.0, 12.0);

    // Handle pointer start (socket wire drag, node drag, or canvas pan)
    if primary_pressed && let Some(pos) = pointer_pos.filter(|p| canvas_rect.contains(*p)) {
        // 1. Check output sockets to start wiring
        if let Some(out_s) = output_sockets
            .iter()
            .find(|s| s.center.distance(pos) <= socket_hit_r)
        {
            state.pending_wire = Some(PendingWireDrag {
                from_node: out_s.node_id.clone(),
                from_port: out_s.port_id.clone(),
                data_type: out_s.data_type,
            });
        }
        // 2. Check input sockets to disconnect existing incoming link
        else if let Some(in_s) = input_sockets
            .iter()
            .find(|s| s.center.distance(pos) <= socket_hit_r)
        {
            if state.disconnect_input(&in_s.node_id, &in_s.port_id) {
                changed = true;
            }
        }
        // 3. Check node bodies (topmost first)
        else if let Some((node_idx, _)) =
            node_rects.iter().rev().find(|(_, rect)| rect.contains(pos))
        {
            let id = state.nodes[*node_idx].id.clone();
            state.selected_node = Some(id.clone());
            state.dragging_node = Some(id);
        }
    }

    // Active dragging: node move or canvas pan
    if primary_down {
        if let Some(drag_id) = state.dragging_node.clone() {
            if let Some(node) = state.nodes.iter_mut().find(|n| n.id == drag_id) {
                node.pos[0] += pointer_delta.x / state.zoom.max(0.2);
                node.pos[1] += pointer_delta.y / state.zoom.max(0.2);
                if pointer_delta.length_sq() > 0.0 {
                    state.dirty = true;
                }
            }
        } else if state.pending_wire.is_none() && canvas_resp.dragged() {
            state.pan += pointer_delta;
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
        }
    }
    if middle_down && canvas_resp.hovered() {
        state.pan += pointer_delta;
        ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
    }

    // Complete wire drop or release node drag
    if primary_released {
        if let Some(pending) = state.pending_wire.take()
            && let Some(pos) = pointer_pos
            && let Some(in_s) = input_sockets
                .iter()
                .find(|s| s.center.distance(pos) <= socket_hit_r * 1.5)
        {
            match state.connect_ports(
                &pending.from_node,
                &pending.from_port,
                &in_s.node_id,
                &in_s.port_id,
            ) {
                Ok(()) => changed = true,
                Err(err) => {
                    state.compile_status = NodeCompileStatus::Error(err);
                }
            }
        }
        if state.dragging_node.take().is_some() && state.dirty {
            changed = true;
        }
    }

    // Draw existing cubic Bezier links + detect hover/click-to-disconnect
    let mut link_to_remove: Option<usize> = None;
    let hovered_link_idx: Option<usize> = pointer_pos.and_then(|p| {
        if !canvas_rect.contains(p) || state.pending_wire.is_some() || state.dragging_node.is_some()
        {
            return None;
        }
        state.links.iter().enumerate().find_map(|(li, link)| {
            let (p0, _) = out_pos_map.get(&(link.from_node.clone(), link.from_port.clone()))?;
            let (p3, _) = in_pos_map.get(&(link.to_node.clone(), link.to_port.clone()))?;
            if dist_to_cubic_bezier(p, *p0, *p3) <= 7.0 {
                Some(li)
            } else {
                None
            }
        })
    });

    for (li, link) in state.links.iter().enumerate() {
        let Some(&(p0, dt)) = out_pos_map.get(&(link.from_node.clone(), link.from_port.clone()))
        else {
            continue;
        };
        let Some(&(p3, _)) = in_pos_map.get(&(link.to_node.clone(), link.to_port.clone())) else {
            continue;
        };
        let is_hovered = hovered_link_idx == Some(li);
        let stroke_color = if is_hovered {
            DANGER_TEXT
        } else {
            color32_with_alpha(dt.color(), 220)
        };
        paint_cubic_wire(
            &painter,
            p0,
            p3,
            Stroke::new(if is_hovered { 3.0 } else { 2.0 }, stroke_color),
        );
    }

    if let Some(li) = hovered_link_idx {
        ui.ctx().set_cursor_icon(CursorIcon::PointingHand);
        if primary_clicked || secondary_clicked {
            link_to_remove = Some(li);
        }
    }
    if let Some(li) = link_to_remove
        && state.remove_link_at(li)
    {
        changed = true;
    }

    // Right-click on an input socket also disconnects
    if secondary_clicked
        && let Some(pos) = pointer_pos
        && let Some(in_s) = input_sockets
            .iter()
            .find(|s| s.center.distance(pos) <= socket_hit_r * 1.2)
        && state.disconnect_input(&in_s.node_id, &in_s.port_id)
    {
        changed = true;
    }

    // Draw pending wire while dragging
    if let (Some(pending), Some(cursor)) = (&state.pending_wire, pointer_pos)
        && let Some(&(p0, _)) =
            out_pos_map.get(&(pending.from_node.clone(), pending.from_port.clone()))
    {
        paint_cubic_wire(
            &painter,
            p0,
            cursor,
            Stroke::new(2.2, pending.data_type.color()),
        );
    }

    // Draw node cards
    let title_font = FontId::proportional((11.5 * state.zoom).clamp(8.0, 18.0));
    let port_font = FontId::proportional((10.0 * state.zoom).clamp(7.5, 15.0));
    let connected_inputs: BTreeSet<(&str, &str)> = state
        .links
        .iter()
        .map(|l| (l.to_node.as_str(), l.to_port.as_str()))
        .collect();
    let connected_outputs: BTreeSet<(&str, &str)> = state
        .links
        .iter()
        .map(|l| (l.from_node.as_str(), l.from_port.as_str()))
        .collect();

    for (idx, rect) in &node_rects {
        let node = &state.nodes[*idx];
        let spec = find_node_spec(state.target_kind, &node.kind);
        let selected = state.selected_node.as_deref() == Some(node.id.as_str());
        let rounding = CornerRadius::same((5.0 * state.zoom).clamp(3.0, 8.0) as u8);

        // Card drop shadow + body
        painter.rect_filled(
            rect.translate(Vec2::new(2.0, 3.0)),
            rounding,
            Color32::from_rgba_unmultiplied(0, 0, 0, 90),
        );
        painter.rect_filled(*rect, rounding, BG_PANEL);

        // Colored header bar
        let header_rect = Rect::from_min_size(
            rect.min,
            Vec2::new(rect.width(), NODE_HEADER_H * state.zoom),
        );
        let header_color = spec.map(NodeTypeSpec::header_color).unwrap_or(BG_TITLE);
        painter.rect_filled(
            header_rect,
            CornerRadius {
                nw: rounding.nw,
                ne: rounding.ne,
                sw: 0,
                se: 0,
            },
            header_color,
        );
        painter.text(
            Pos2::new(header_rect.min.x + 8.0 * state.zoom, header_rect.center().y),
            Align2::LEFT_CENTER,
            &node.title,
            title_font.clone(),
            TEXT_BRIGHT,
        );

        // Border highlight when selected
        painter.rect_stroke(
            *rect,
            rounding,
            Stroke::new(
                if selected { 2.0 } else { 1.0 },
                if selected {
                    theme.accent()
                } else {
                    BORDER_CONTROL
                },
            ),
            StrokeKind::Outside,
        );

        // Input & output port labels + sockets
        let sock_r = (4.8 * state.zoom).clamp(3.5, 8.0);
        if let Some(spec) = spec {
            for (pi, port) in spec.inputs.iter().enumerate() {
                let cy =
                    rect.min.y + (NODE_HEADER_H + (pi as f32 + 0.5) * NODE_PORT_ROW_H) * state.zoom;
                let center = Pos2::new(rect.min.x, cy);
                let is_conn = connected_inputs.contains(&(node.id.as_str(), port.id));
                painter.circle_filled(
                    center,
                    sock_r,
                    if is_conn {
                        port.data_type.color()
                    } else {
                        BG_SUNKEN
                    },
                );
                painter.circle_stroke(center, sock_r, Stroke::new(1.5, port.data_type.color()));
                painter.text(
                    Pos2::new(rect.min.x + 10.0 * state.zoom, cy),
                    Align2::LEFT_CENTER,
                    port.label,
                    port_font.clone(),
                    TEXT_PRIMARY,
                );
            }
            for (pi, port) in spec.outputs.iter().enumerate() {
                let cy =
                    rect.min.y + (NODE_HEADER_H + (pi as f32 + 0.5) * NODE_PORT_ROW_H) * state.zoom;
                let center = Pos2::new(rect.max.x, cy);
                let is_conn = connected_outputs.contains(&(node.id.as_str(), port.id));
                painter.circle_filled(
                    center,
                    sock_r,
                    if is_conn {
                        port.data_type.color()
                    } else {
                        BG_SUNKEN
                    },
                );
                painter.circle_stroke(center, sock_r, Stroke::new(1.5, port.data_type.color()));
                painter.text(
                    Pos2::new(rect.max.x - 10.0 * state.zoom, cy),
                    Align2::RIGHT_CENTER,
                    port.label,
                    port_font.clone(),
                    TEXT_PRIMARY,
                );
            }
        }
    }

    // Canvas border
    painter.rect_stroke(
        canvas_rect,
        CornerRadius::same(4),
        Stroke::new(1.0, BORDER_PANEL),
        StrokeKind::Inside,
    );

    changed
}

pub fn paint_cubic_wire(painter: &egui::Painter, p0: Pos2, p3: Pos2, stroke: Stroke) {
    let dx = ((p3.x - p0.x).abs() * 0.5).clamp(32.0, 140.0);
    let p1 = Pos2::new(p0.x + dx, p0.y);
    let p2 = Pos2::new(p3.x - dx, p3.y);
    let steps = 24;
    let mut pts = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        pts.push(eval_cubic_pos(p0, p1, p2, p3, t));
    }
    painter.add(Shape::line(pts, stroke));
}

pub fn eval_cubic_pos(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
    let u = 1.0 - t;
    let tt = t * t;
    let uu = u * u;
    let uuu = uu * u;
    let ttt = tt * t;
    Pos2::new(
        uuu * p0.x + 3.0 * uu * t * p1.x + 3.0 * u * tt * p2.x + ttt * p3.x,
        uuu * p0.y + 3.0 * uu * t * p1.y + 3.0 * u * tt * p2.y + ttt * p3.y,
    )
}

pub fn dist_to_cubic_bezier(pt: Pos2, p0: Pos2, p3: Pos2) -> f32 {
    let dx = ((p3.x - p0.x).abs() * 0.5).clamp(32.0, 140.0);
    let p1 = Pos2::new(p0.x + dx, p0.y);
    let p2 = Pos2::new(p3.x - dx, p3.y);
    let mut best = f32::INFINITY;
    let mut prev = p0;
    for i in 1..=20 {
        let cur = eval_cubic_pos(p0, p1, p2, p3, i as f32 / 20.0);
        best = best.min(point_segment_distance(pt, prev, cur));
        prev = cur;
    }
    best
}

pub fn point_segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    p.distance(proj)
}

pub fn show_node_inspector_ui(
    ui: &mut Ui,
    state: &mut NodeGraphEditorState,
    theme: &Cm3Theme,
) -> bool {
    let mut changed = false;
    let Some(sel_id) = state.selected_node.clone() else {
        ui.label(
            egui::RichText::new("Select a node to edit its parameters.")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_MUTED),
        );
        return false;
    };

    let Some(node_idx) = state.nodes.iter().position(|n| n.id == sel_id) else {
        return false;
    };

    let node_kind = state.nodes[node_idx].kind.clone();
    let is_output = node_kind == "output";
    let spec = find_node_spec(state.target_kind, &node_kind);

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Node Inspector")
                .font(theme.font(FONT_TITLE))
                .color(TEXT_STRONG),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(&node_kind)
                    .font(theme.font(FONT_COMPACT))
                    .color(TEXT_FAINT),
            );
        });
    });
    ui.separator();

    // Node Title & Actions
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new("Title")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_MUTED),
        );
        if ui
            .add(
                TextEdit::singleline(&mut state.nodes[node_idx].title)
                    .desired_width(ui.available_width()),
            )
            .changed()
        {
            state.dirty = true;
            changed = true;
        }
    });

    if !is_output {
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if cm3_prop_button(ui, "Duplicate", ButtonVariant::Normal, false, theme).clicked()
                && state.duplicate_node(&sel_id).is_some()
            {
                return;
            }
            if cm3_prop_button(ui, "Delete", ButtonVariant::Danger, false, theme).clicked() {
                state.remove_node(&sel_id);
                changed = true;
            }
        });
        // Check if node was deleted above
        if !state.nodes.iter().any(|n| n.id == sel_id) {
            return true;
        }
    }

    ui.add_space(6.0);

    // Parameters scroll area (keyed on selected node so switching nodes resets scroll to top)
    ScrollArea::vertical()
        .id_salt(("node_inspector_params", &sel_id))
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let Some(node) = state.nodes.iter_mut().find(|n| n.id == sel_id) else {
                return;
            };
            if let Some(spec) = spec {
                if spec.params.is_empty() {
                    ui.label(
                        egui::RichText::new("No editable parameters on this node.")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_FAINT),
                    );
                }
                for param in spec.params {
                    ui.add_space(3.0);
                    match param.kind {
                        NodeParamKind::Float { min, max, step } => {
                            let mut val = node
                                .params
                                .get(param.key)
                                .and_then(Value::as_f64)
                                .unwrap_or(param.default_f64);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(param.label)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut val)
                                                    .range(min..=max)
                                                    .speed(step)
                                                    .max_decimals(3),
                                            )
                                            .changed()
                                        {
                                            node.params.insert(param.key.into(), json!(val));
                                            changed = true;
                                        }
                                    },
                                );
                            });
                        }
                        NodeParamKind::Int { min, max } => {
                            let mut val = node
                                .params
                                .get(param.key)
                                .and_then(Value::as_i64)
                                .unwrap_or(param.default_f64.round() as i64);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(param.label)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut val)
                                                    .range(min..=max)
                                                    .speed(1),
                                            )
                                            .changed()
                                        {
                                            node.params.insert(param.key.into(), json!(val));
                                            changed = true;
                                        }
                                    },
                                );
                            });
                        }
                        NodeParamKind::Bool => {
                            let mut val = node
                                .params
                                .get(param.key)
                                .map(|v| {
                                    v.as_bool()
                                        .unwrap_or_else(|| v.as_f64().unwrap_or(0.0) != 0.0)
                                })
                                .unwrap_or(param.default_f64 != 0.0);
                            if ui.checkbox(&mut val, param.label).changed() {
                                node.params.insert(param.key.into(), json!(val));
                                changed = true;
                            }
                        }
                        NodeParamKind::Choice(options) => {
                            let mut idx = node
                                .params
                                .get(param.key)
                                .and_then(Value::as_u64)
                                .map(|v| (v as usize).min(options.len().saturating_sub(1)))
                                .unwrap_or(param.default_f64.round().max(0.0) as usize);
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(param.label)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                egui::ComboBox::from_id_salt(Id::unique((
                                    "node_param_choice",
                                    &sel_id,
                                    param.key,
                                )))
                                .selected_text(options.get(idx).copied().unwrap_or("—"))
                                .width(116.0)
                                .show_ui(ui, |ui| {
                                    for (oi, opt) in options.iter().enumerate() {
                                        if ui.selectable_value(&mut idx, oi, *opt).changed() {
                                            node.params.insert(param.key.into(), json!(idx));
                                            changed = true;
                                        }
                                    }
                                });
                            });
                        }
                        NodeParamKind::ColorRgb => {
                            let mut rgb = extract_rgb_f32(
                                node.params.get(param.key),
                                [param.default_f64 as f32; 3],
                            );
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(param.label)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.color_edit_button_rgb(&mut rgb).changed() {
                                            node.params.insert(
                                                param.key.into(),
                                                json!([
                                                    rgb[0] as f64,
                                                    rgb[1] as f64,
                                                    rgb[2] as f64
                                                ]),
                                            );
                                            changed = true;
                                        }
                                    },
                                );
                            });
                        }
                        NodeParamKind::ColorRamp => {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_STRONG),
                            );
                            let mut stops = parse_ramp_stops(node.params.get(param.key));
                            let mut ramp_changed = false;
                            let mut remove_idx = None;
                            let stops_len = stops.len();
                            for (si, (pos, rgb)) in stops.iter_mut().enumerate() {
                                ui.horizontal(|ui| {
                                    if ui.color_edit_button_rgb(rgb).changed() {
                                        ramp_changed = true;
                                    }
                                    if ui
                                        .add(
                                            egui::DragValue::new(pos)
                                                .range(0.0..=1.0)
                                                .speed(0.01)
                                                .max_decimals(2),
                                        )
                                        .changed()
                                    {
                                        ramp_changed = true;
                                    }
                                    if stops_len > 2 && ui.small_button("×").clicked() {
                                        remove_idx = Some(si);
                                    }
                                });
                            }
                            if let Some(ri) = remove_idx {
                                stops.remove(ri);
                                ramp_changed = true;
                            }
                            if ui.small_button("+ Add Stop").clicked() {
                                stops.push((0.5, [0.5, 0.5, 0.5]));
                                ramp_changed = true;
                            }
                            if ramp_changed {
                                stops.sort_by(|a, b| a.0.total_cmp(&b.0));
                                let serialized: Vec<Value> = stops
                                    .into_iter()
                                    .map(|(p, c)| {
                                        json!({
                                            "pos": p,
                                            "color": [c[0] as f64, c[1] as f64, c[2] as f64],
                                        })
                                    })
                                    .collect();
                                node.params
                                    .insert(param.key.into(), Value::Array(serialized));
                                changed = true;
                            }
                        }
                        NodeParamKind::Text => {
                            let mut txt = node
                                .params
                                .get(param.key)
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            ui.horizontal(|ui| {
                                ui.label(
                                    egui::RichText::new(param.label)
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_PRIMARY),
                                );
                                if ui
                                    .add(TextEdit::singleline(&mut txt).desired_width(120.0))
                                    .changed()
                                {
                                    node.params.insert(param.key.into(), json!(txt));
                                    changed = true;
                                }
                            });
                        }
                    }
                }

                // Connections summary & quick disconnect buttons
                ui.add_space(8.0);
                ui.separator();
                ui.label(
                    egui::RichText::new("Ports")
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_FAINT),
                );
                for inp in spec.inputs {
                    ui.label(
                        egui::RichText::new(format!(
                            "IN  {} ({})",
                            inp.label,
                            inp.data_type.short_label()
                        ))
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_MUTED),
                    );
                }
                for outp in spec.outputs {
                    ui.label(
                        egui::RichText::new(format!(
                            "OUT {} ({})",
                            outp.label,
                            outp.data_type.short_label()
                        ))
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_MUTED),
                    );
                }
            }
        });

    if changed {
        state.dirty = true;
        state.revalidate();
    }
    changed
}

pub fn extract_rgb_f32(val: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
    if let Some(arr) = val.and_then(Value::as_array)
        && arr.len() >= 3
    {
        return [
            arr[0].as_f64().unwrap_or(default[0] as f64) as f32,
            arr[1].as_f64().unwrap_or(default[1] as f64) as f32,
            arr[2].as_f64().unwrap_or(default[2] as f64) as f32,
        ];
    }
    default
}

pub fn parse_ramp_stops(val: Option<&Value>) -> Vec<(f64, [f32; 3])> {
    let mut out = Vec::new();
    if let Some(arr) = val.and_then(Value::as_array) {
        for item in arr {
            let pos = item
                .get("pos")
                .or_else(|| item.get("position"))
                .and_then(Value::as_f64)
                .unwrap_or(0.0);
            let rgb = extract_rgb_f32(item.get("color"), [1.0, 1.0, 1.0]);
            out.push((pos, rgb));
        }
    }
    if out.len() < 2 {
        out = vec![(0.0, [0.05, 0.08, 0.16]), (1.0, [0.95, 0.96, 1.0])];
    }
    out
}

// ---------------------------------------------------------------------------
// Node Graph Type Catalogs (Covering All zc_shader::node_graph Node Types)
// ---------------------------------------------------------------------------

pub const MIX_MODES: &[&str] = &[
    "Mix",
    "Add",
    "Subtract",
    "Multiply",
    "Screen",
    "Overlay",
    "Darken",
    "Lighten",
    "Color Dodge",
    "Color Burn",
    "Soft Light",
    "Hard Light",
    "Difference",
    "Exclusion",
    "Divide",
    "Hue",
    "Saturation",
    "Color",
    "Luminosity",
];

pub const MATH_MODES: &[&str] = &[
    "Add", "Subtract", "Multiply", "Divide", "Power", "Min", "Max", "Modulo", "Abs", "Floor",
    "Ceil", "Fract", "Sqrt", "Sin", "Cos", "Tan", "Step", "Negate", "Round", "Sign", "Log", "Exp",
];

pub const VEC_MATH_MODES: &[&str] = &[
    "Add",
    "Subtract",
    "Multiply",
    "Divide",
    "Scale",
    "Length",
    "Distance",
    "Dot",
    "Normalize",
];

pub const GRADIENT_MODES: &[&str] = &["Linear", "Radial", "Angular"];
pub const VORONOI_MODES: &[&str] = &["F1 Distance", "F2 - F1 Edge", "Cell ID"];
pub const BLUR_MODES: &[&str] = &["Box", "Gaussian", "Directional", "Radial"];
pub const MIRROR_MODES: &[&str] = &["Horizontal", "Vertical", "Quad"];
pub const TRIG_MODES: &[&str] = &["Sin", "Cos", "Tan", "Asin", "Acos", "Atan"];
pub const PARAM_KINDS: &[&str] = &["Float", "Color", "Point"];
pub const RAMP_INTERP: &[&str] = &["Linear", "Smooth", "Constant"];
pub const NORMAL_MODES: &[&str] = &["Bump Height", "Tangent Normal"];

#[must_use]
pub fn node_catalog_for(target_kind: NodeGraphTargetKind) -> &'static [NodeTypeSpec] {
    match target_kind {
        NodeGraphTargetKind::NodeEffect => EFFECT_NODE_CATALOG,
        NodeGraphTargetKind::NodeMaterial => MATERIAL_NODE_CATALOG,
    }
}

#[must_use]
pub fn find_node_spec(
    target_kind: NodeGraphTargetKind,
    kind: &str,
) -> Option<&'static NodeTypeSpec> {
    let lower = kind.trim().to_ascii_lowercase();
    node_catalog_for(target_kind)
        .iter()
        .find(|s| s.kind.eq_ignore_ascii_case(&lower))
}
