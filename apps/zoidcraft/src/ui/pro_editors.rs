//! Interactive Pro Editor floating windows for ZoidCraft:
//! - **Node Graph Editor** (`NodeGraphEditorState`, [`show_node_graph_editor_window`]):
//!   Visual node-based editor supporting both 2D `nodeeffect` graphs (GLSL ES 1.00 codegen via
//!   `zc_shader::compile_node_effect_to_glsl`) and 3D `NodeMaterial` procedural PBR graphs
//!   (texture map baking via `zc_shader::bake_node_material_maps`).
//! - **Color Looks Setup** (`LooksEditorState`, [`show_looks_editor_window`]):
//!   5-stage color grading pipeline editor (**Subject / Matte / Lens / Camera / Post**) covering
//!   all 29 Color Looks tools, presets, reordering, and live parameter editing.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use egui::{
    Align2, Color32, CornerRadius, CursorIcon, FontId, Id, Key, Pos2, Rect, ScrollArea, Sense,
    Shape, Stroke, StrokeKind, TextEdit, Ui, Vec2,
};
use serde_json::{Map, Value, json};
use zc_session::{Command, Session};
use zc_shader::{bake_node_material_maps, compile_node_effect_to_glsl};

use super::controls::{
    ButtonVariant, FloatingWindowManager, FloatingWindowSpec, cm3_floating_window, cm3_prop_button,
};
use super::theme::{
    BG_BUTTON, BG_HOVER, BG_INPUT, BG_PANEL, BG_SUNKEN, BG_TITLE, BORDER_CONTROL, BORDER_PANEL,
    Cm3Theme, DANGER_TEXT, FONT_BODY, FONT_COMPACT, FONT_SMALL, FONT_TITLE, TEXT_BRIGHT,
    TEXT_FAINT, TEXT_MUTED, TEXT_PRIMARY, TEXT_STRONG, color32_with_alpha,
};

// ===========================================================================
// Shared Entry Point
// ===========================================================================

/// Renders all open Pro Editor floating windows (Node Graph Editor and Color Looks Setup).
pub fn show_pro_editor_windows(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    node_editor: &mut NodeGraphEditorState,
    looks_editor: &mut LooksEditorState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    show_node_graph_editor_window(
        ctx,
        viewport_rect,
        floating_windows,
        node_editor,
        session,
        theme,
    );
    show_looks_editor_window(
        ctx,
        viewport_rect,
        floating_windows,
        looks_editor,
        session,
        theme,
    );
}

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
struct PendingWireDrag {
    from_node: String,
    from_port: String,
    data_type: PortDataType,
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
    next_node_seq: usize,
    dragging_node: Option<String>,
    pending_wire: Option<PendingWireDrag>,
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

    fn refresh_next_node_seq(&mut self) {
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
                    NodeCompileStatus::Ok(format!(
                        "Compiled OK ({} nodes, {} baked PBR maps)",
                        self.nodes.len(),
                        map_count,
                    ))
                }
                Err(err) => NodeCompileStatus::Error(err),
            },
        };
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

fn extract_link_ep(
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

fn make_default_node(
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

/// Renders the floating Node Graph Editor window when `state.open` is true.
pub fn show_node_graph_editor_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut NodeGraphEditorState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }

    let spec = FloatingWindowSpec::new("pro_node_graph_editor", "Node Graph Editor")
        .with_subtitle(if state.target_label.is_empty() {
            state.target_kind.label().to_string()
        } else {
            format!("{} — {}", state.target_kind.label(), state.target_label)
        })
        .with_default_size(Vec2::new(980.0, 620.0));

    let mut close_requested = false;
    let mut should_commit = false;

    let ev = cm3_floating_window(ctx, floating_windows, &spec, viewport_rect, theme, |ui| {
        // --- Top Toolbar ---
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if cm3_prop_button(
                ui,
                if state.palette_open {
                    "Close Palette"
                } else {
                    "+ Add Node"
                },
                if state.palette_open {
                    ButtonVariant::Primary
                } else {
                    ButtonVariant::Normal
                },
                false,
                theme,
            )
            .clicked()
            {
                state.palette_open = !state.palette_open;
            }

            // Template selector buttons
            let templates: &[&str] = match state.target_kind {
                NodeGraphTargetKind::NodeEffect => {
                    &["Passthrough", "Glow + Vignette", "Noise Displace"]
                }
                NodeGraphTargetKind::NodeMaterial => &["Noise Ramp", "Marble PBR", "Brushed Metal"],
            };
            egui::ComboBox::from_id_salt(Id::unique("node_graph_templates"))
                .selected_text("Templates")
                .width(120.0)
                .show_ui(ui, |ui| {
                    for (idx, name) in templates.iter().enumerate() {
                        if ui.selectable_label(false, *name).clicked() {
                            state.apply_template(idx);
                            if state.auto_apply {
                                should_commit = true;
                            }
                        }
                    }
                });

            if cm3_prop_button(ui, "Fit View", ButtonVariant::Normal, false, theme).clicked() {
                fit_node_graph_view(state);
            }

            ui.label(
                egui::RichText::new(format!("Zoom: {:.0}%", state.zoom * 100.0))
                    .font(theme.font(FONT_SMALL))
                    .color(TEXT_MUTED),
            );

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if cm3_prop_button(ui, "Close", ButtonVariant::Normal, false, theme).clicked() {
                    close_requested = true;
                }
                if cm3_prop_button(
                    ui,
                    if state.dirty { "Apply *" } else { "Apply" },
                    ButtonVariant::Primary,
                    false,
                    theme,
                )
                .clicked()
                {
                    should_commit = true;
                }
                ui.checkbox(&mut state.auto_apply, "Auto-Apply");
            });
        });

        ui.add_space(4.0);

        // --- Main 3-Column Split: Optional Palette | Canvas | Node Inspector ---
        let body_height = (ui.available_height() - 28.0).max(200.0);
        let palette_w = if state.palette_open { 200.0 } else { 0.0 };
        let inspector_w = 248.0;
        let total_w = ui.available_width();
        let canvas_w =
            (total_w - palette_w - inspector_w - if state.palette_open { 12.0 } else { 6.0 })
                .max(240.0);

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;

            // Optional Left Palette
            if state.palette_open {
                egui::Frame::NONE
                    .fill(BG_SUNKEN)
                    .stroke(Stroke::new(1.0, BORDER_PANEL))
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_width(palette_w - 14.0);
                        ui.set_height(body_height);
                        if show_node_palette_ui(ui, state, theme) && state.auto_apply {
                            should_commit = true;
                        }
                    });
            }

            // Center Interactive Graph Canvas
            let (canvas_rect, _) =
                ui.allocate_exact_size(Vec2::new(canvas_w, body_height), Sense::hover());
            if show_node_canvas_ui(ui, canvas_rect, state, theme) && state.auto_apply {
                should_commit = true;
            }

            // Right Node Inspector Sidebar
            egui::Frame::NONE
                .fill(BG_PANEL)
                .stroke(Stroke::new(1.0, BORDER_PANEL))
                .corner_radius(CornerRadius::same(4))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(inspector_w - 18.0);
                    ui.set_height(body_height);
                    if show_node_inspector_ui(ui, state, theme) && state.auto_apply {
                        should_commit = true;
                    }
                });
        });

        ui.add_space(4.0);

        // --- Bottom Compile Status Bar ---
        ui.horizontal(|ui| {
                let (badge_color, status_text) = match &state.compile_status {
                    NodeCompileStatus::Ok(msg) => (Color32::from_rgb(76, 198, 122), msg.as_str()),
                    NodeCompileStatus::Error(err) => (DANGER_TEXT, err.as_str()),
                };
                let (dot_rect, _) = ui.allocate_exact_size(Vec2::splat(8.0), Sense::hover());
                ui.painter()
                    .circle_filled(dot_rect.center(), 4.0, badge_color);
                ui.label(
                    egui::RichText::new(status_text)
                        .font(theme.font(FONT_SMALL))
                        .color(badge_color),
                );
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(
                        egui::RichText::new(
                            "Drag ports to wire • Click wire or input socket to disconnect • Scroll to zoom",
                        )
                        .font(theme.font(FONT_COMPACT))
                        .color(TEXT_FAINT),
                    );
                });
            });
    });

    if should_commit {
        state.commit_to_session(session);
    }
    if close_requested || ev.closed {
        state.open = false;
        floating_windows.close("pro_node_graph_editor");
    }
}

fn fit_node_graph_view(state: &mut NodeGraphEditorState) {
    if state.nodes.is_empty() {
        state.pan = Vec2::new(40.0, 40.0);
        state.zoom = 1.0;
        return;
    }
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    for node in &state.nodes {
        min_x = min_x.min(node.pos[0]);
        min_y = min_y.min(node.pos[1]);
    }
    state.zoom = 1.0;
    state.pan = Vec2::new(32.0 - min_x, 32.0 - min_y);
}

fn show_node_palette_ui(ui: &mut Ui, state: &mut NodeGraphEditorState, theme: &Cm3Theme) -> bool {
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

const NODE_BASE_WIDTH: f32 = 172.0;
const NODE_HEADER_H: f32 = 24.0;
const NODE_PORT_ROW_H: f32 = 20.0;
const NODE_FOOTER_PAD: f32 = 8.0;

fn node_size_graph_space(spec: Option<&NodeTypeSpec>) -> Vec2 {
    let rows = spec
        .map(|s| s.inputs.len().max(s.outputs.len()).max(1))
        .unwrap_or(1);
    Vec2::new(
        NODE_BASE_WIDTH,
        NODE_HEADER_H + (rows as f32) * NODE_PORT_ROW_H + NODE_FOOTER_PAD,
    )
}

fn graph_to_screen(canvas_rect: Rect, pan: Vec2, zoom: f32, pos: [f32; 2]) -> Pos2 {
    Pos2::new(
        canvas_rect.min.x + pan.x + pos[0] * zoom,
        canvas_rect.min.y + pan.y + pos[1] * zoom,
    )
}

fn show_node_canvas_ui(
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

fn paint_cubic_wire(painter: &egui::Painter, p0: Pos2, p3: Pos2, stroke: Stroke) {
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

fn eval_cubic_pos(p0: Pos2, p1: Pos2, p2: Pos2, p3: Pos2, t: f32) -> Pos2 {
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

fn dist_to_cubic_bezier(pt: Pos2, p0: Pos2, p3: Pos2) -> f32 {
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

fn point_segment_distance(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_sq();
    if len_sq <= 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0);
    let proj = a + ab * t;
    p.distance(proj)
}

fn show_node_inspector_ui(ui: &mut Ui, state: &mut NodeGraphEditorState, theme: &Cm3Theme) -> bool {
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

    // Parameters scroll area
    ScrollArea::vertical()
        .id_salt("node_inspector_params")
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

fn extract_rgb_f32(val: Option<&Value>, default: [f32; 3]) -> [f32; 3] {
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

fn parse_ramp_stops(val: Option<&Value>) -> Vec<(f64, [f32; 3])> {
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

const MIX_MODES: &[&str] = &[
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

const MATH_MODES: &[&str] = &[
    "Add", "Subtract", "Multiply", "Divide", "Power", "Min", "Max", "Modulo", "Abs", "Floor",
    "Ceil", "Fract", "Sqrt", "Sin", "Cos", "Tan", "Step", "Negate", "Round", "Sign", "Log", "Exp",
];

const VEC_MATH_MODES: &[&str] = &[
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

const GRADIENT_MODES: &[&str] = &["Linear", "Radial", "Angular"];
const VORONOI_MODES: &[&str] = &["F1 Distance", "F2 - F1 Edge", "Cell ID"];
const BLUR_MODES: &[&str] = &["Box", "Gaussian", "Directional", "Radial"];
const MIRROR_MODES: &[&str] = &["Horizontal", "Vertical", "Quad"];
const TRIG_MODES: &[&str] = &["Sin", "Cos", "Tan", "Asin", "Acos", "Atan"];
const PARAM_KINDS: &[&str] = &["Float", "Color", "Point"];
const RAMP_INTERP: &[&str] = &["Linear", "Smooth", "Constant"];
const NORMAL_MODES: &[&str] = &["Bump Height", "Tangent Normal"];

/// Complete catalog of 2D `nodeeffect` nodes matching `zc_shader::node_graph`.
pub const EFFECT_NODE_CATALOG: &[NodeTypeSpec] = &[
    // --- Inputs ---
    NodeTypeSpec {
        kind: "layer",
        title: "Source Layer",
        category: "Inputs",
        inputs: &[],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "uv",
        title: "UV Coords",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "pixel",
        title: "Pixel Coord",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Pixel",
            data_type: PortDataType::Vec2,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "resolution",
        title: "Resolution",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Size",
            data_type: PortDataType::Vec2,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "time",
        title: "Time",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Seconds",
            data_type: PortDataType::Float,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "parameter",
        title: "Parameter",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "label",
                label: "Label",
                kind: NodeParamKind::Text,
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "kind",
                label: "Kind",
                kind: NodeParamKind::Choice(PARAM_KINDS),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "default",
                label: "Default",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "min",
                label: "Min",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.1,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "max",
                label: "Max",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.1,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "solid",
        title: "Solid Color",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "color",
                label: "Color",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "alpha",
                label: "Alpha",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    // --- Generators ---
    NodeTypeSpec {
        kind: "noise",
        title: "Value Noise",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 100.0,
                    step: 0.2,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "time",
                label: "Evolution",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "fbm",
        title: "FBM Noise",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 100.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "lacunarity",
                label: "Lacunarity",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 2.0,
            },
            NodeParamSpec {
                key: "gain",
                label: "Gain",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 0.9,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "voronoi",
        title: "Voronoi",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 80.0,
                    step: 0.2,
                },
                default_f64: 5.0,
            },
            NodeParamSpec {
                key: "mode",
                label: "Mode",
                kind: NodeParamKind::Choice(VORONOI_MODES),
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "checker",
        title: "Checkerboard",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "a",
                label: "Color A",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "b",
                label: "Color B",
                data_type: PortDataType::Color,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Mask",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 128.0,
                    step: 0.5,
                },
                default_f64: 8.0,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.1,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.9,
            },
        ],
    },
    NodeTypeSpec {
        kind: "gradient",
        title: "Gradient",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "angle",
                label: "Angle",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Mode",
                kind: NodeParamKind::Choice(GRADIENT_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "wave",
        title: "Wave Pattern",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "scale",
                label: "Scale",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "time",
                label: "Time",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 80.0,
                    step: 0.2,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "vignette",
        title: "Vignette",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Mask",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.65,
            },
            NodeParamSpec {
                key: "softness",
                label: "Softness",
                kind: NodeParamKind::Float {
                    min: 0.01,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            NodeParamSpec {
                key: "strength",
                label: "Strength",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "grain",
        title: "Film Grain",
        category: "Generators",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "amount",
                label: "Amount",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Noise",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "amount",
                label: "Amount",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.1,
            },
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.25,
                    max: 8.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "seed",
                label: "Seed",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1000.0,
                    step: 1.0,
                },
                default_f64: 1.0,
            },
        ],
    },
    // --- Color ---
    NodeTypeSpec {
        kind: "brightness",
        title: "Brightness / Contrast",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "brightness",
                label: "Brightness",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "contrast",
                label: "Contrast",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "brightness",
                label: "Brightness",
                kind: NodeParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "hue",
        title: "HSV Adjust",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "hue",
                label: "Hue",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "saturation",
                label: "Saturation",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "value",
                label: "Value",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "hue",
                label: "Hue Shift",
                kind: NodeParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "saturation",
                label: "Saturation",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "value",
                label: "Value",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "colorBalance",
        title: "Color Balance",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "lift",
                label: "Lift",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "gain",
                label: "Gain",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "levels",
        title: "Levels",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "black",
                label: "In Black",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 0.99,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "white",
                label: "In White",
                kind: NodeParamKind::Float {
                    min: 0.01,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "outBlack",
                label: "Out Black",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "outWhite",
                label: "Out White",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "invert",
        title: "Invert",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "factor",
            label: "Factor",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "grayscale",
        title: "Grayscale",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "factor",
            label: "Factor",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "colorRamp",
        title: "Color Ramp",
        category: "Color",
        inputs: &[PortSpec {
            id: "factor",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "interpolation",
                label: "Interp",
                kind: NodeParamKind::Choice(RAMP_INTERP),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "stops",
                label: "Stops",
                kind: NodeParamKind::ColorRamp,
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "mix",
        title: "Mix Color",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "a",
                label: "Color A",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "b",
                label: "Color B",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Blend Mode",
                kind: NodeParamKind::Choice(MIX_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "factor",
                label: "Factor",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "separateRGB",
        title: "Separate RGBA",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[
            PortSpec {
                id: "r",
                label: "R",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "g",
                label: "G",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "combineRGB",
        title: "Combine RGBA",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "r",
                label: "R",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "g",
                label: "G",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "a",
            label: "Alpha",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "separateHSV",
        title: "Separate HSV",
        category: "Color",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[
            PortSpec {
                id: "h",
                label: "H",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "s",
                label: "S",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "v",
                label: "V",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "combineHSV",
        title: "Combine HSV",
        category: "Color",
        inputs: &[
            PortSpec {
                id: "h",
                label: "H",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "s",
                label: "S",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "v",
                label: "V",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[],
    },
    // --- Math ---
    NodeTypeSpec {
        kind: "math",
        title: "Scalar Math",
        category: "Math",
        inputs: &[
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Operation",
                kind: NodeParamKind::Choice(MATH_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "a",
                label: "A",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "b",
                label: "B",
                kind: NodeParamKind::Float {
                    min: -1000.0,
                    max: 1000.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "vectorMath",
        title: "Vector Math",
        category: "Math",
        inputs: &[
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Vec2,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Vec2,
            },
        ],
        outputs: &[
            PortSpec {
                id: "out",
                label: "Vector",
                data_type: PortDataType::Vec2,
            },
            PortSpec {
                id: "value",
                label: "Value",
                data_type: PortDataType::Float,
            },
        ],
        params: &[NodeParamSpec {
            key: "mode",
            label: "Operation",
            kind: NodeParamKind::Choice(VEC_MATH_MODES),
            default_f64: 0.0,
        }],
    },
    NodeTypeSpec {
        kind: "mapRange",
        title: "Map Range",
        category: "Math",
        inputs: &[PortSpec {
            id: "value",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "fromMin",
                label: "From Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "fromMax",
                label: "From Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "toMin",
                label: "To Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "toMax",
                label: "To Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "clamp",
                label: "Clamp",
                kind: NodeParamKind::Bool,
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "clamp",
        title: "Clamp",
        category: "Math",
        inputs: &[PortSpec {
            id: "value",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "min",
                label: "Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "max",
                label: "Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "smoothstep",
        title: "Smoothstep / Threshold",
        category: "Math",
        inputs: &[PortSpec {
            id: "x",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "edge0",
                label: "Edge 0",
                kind: NodeParamKind::Float {
                    min: -10.0,
                    max: 10.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            NodeParamSpec {
                key: "edge1",
                label: "Edge 1",
                kind: NodeParamKind::Float {
                    min: -10.0,
                    max: 10.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
        ],
    },
    NodeTypeSpec {
        kind: "trig",
        title: "Trigonometry",
        category: "Math",
        inputs: &[PortSpec {
            id: "x",
            label: "Angle",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "op",
            label: "Function",
            kind: NodeParamKind::Choice(TRIG_MODES),
            default_f64: 0.0,
        }],
    },
    // --- UV & Filters ---
    NodeTypeSpec {
        kind: "uvTransform",
        title: "UV Transform",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        params: &[
            NodeParamSpec {
                key: "offsetX",
                label: "Offset X",
                kind: NodeParamKind::Float {
                    min: -5.0,
                    max: 5.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "offsetY",
                label: "Offset Y",
                kind: NodeParamKind::Float {
                    min: -5.0,
                    max: 5.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "scaleX",
                label: "Scale X",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 20.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "scaleY",
                label: "Scale Y",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 20.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "displace",
        title: "UV Displace",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "map",
                label: "Map",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "strength",
                label: "Strength",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[NodeParamSpec {
            key: "strength",
            label: "Strength",
            kind: NodeParamKind::Float {
                min: -0.5,
                max: 0.5,
                step: 0.005,
            },
            default_f64: 0.04,
        }],
    },
    NodeTypeSpec {
        kind: "swirl",
        title: "Swirl",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -720.0,
                    max: 720.0,
                    step: 2.0,
                },
                default_f64: 90.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "kaleidoscope",
        title: "Kaleidoscope",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "segments",
                label: "Segments",
                kind: NodeParamKind::Float {
                    min: 2.0,
                    max: 32.0,
                    step: 1.0,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "pixelate",
        title: "Pixelate",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[NodeParamSpec {
            key: "blocks",
            label: "Blocks",
            kind: NodeParamKind::Float {
                min: 4.0,
                max: 512.0,
                step: 2.0,
            },
            default_f64: 64.0,
        }],
    },
    NodeTypeSpec {
        kind: "mirror",
        title: "Mirror",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "color",
                label: "Sampled",
                data_type: PortDataType::Color,
            },
        ],
        params: &[NodeParamSpec {
            key: "mode",
            label: "Axis",
            kind: NodeParamKind::Choice(MIRROR_MODES),
            default_f64: 0.0,
        }],
    },
    NodeTypeSpec {
        kind: "sample",
        title: "Sample Layer",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "blur",
        title: "Blur",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "radius",
                label: "Radius",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Type",
                kind: NodeParamKind::Choice(BLUR_MODES),
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 0.1,
                    step: 0.001,
                },
                default_f64: 0.01,
            },
            NodeParamSpec {
                key: "taps",
                label: "Samples",
                kind: NodeParamKind::Int { min: 3, max: 16 },
                default_f64: 9.0,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "sharpen",
        title: "Sharpen",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.0005,
                    max: 0.02,
                    step: 0.0005,
                },
                default_f64: 0.002,
            },
            NodeParamSpec {
                key: "amount",
                label: "Amount",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "edge",
        title: "Edge Detect",
        category: "UV & Filters",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Edge",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.0005,
                    max: 0.02,
                    step: 0.0005,
                },
                default_f64: 0.002,
            },
            NodeParamSpec {
                key: "strength",
                label: "Strength",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 5.0,
                    step: 0.05,
                },
                default_f64: 1.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "chromaticAberration",
        title: "Chromatic Aberration",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "amount",
                label: "Amount",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "amount",
                label: "Amount",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 0.05,
                    step: 0.001,
                },
                default_f64: 0.006,
            },
            NodeParamSpec {
                key: "angle",
                label: "Angle",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "glow",
        title: "Glow",
        category: "UV & Filters",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "intensity",
                label: "Intensity",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "threshold",
                label: "Threshold",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            NodeParamSpec {
                key: "radius",
                label: "Radius",
                kind: NodeParamKind::Float {
                    min: 0.002,
                    max: 0.08,
                    step: 0.002,
                },
                default_f64: 0.015,
            },
            NodeParamSpec {
                key: "intensity",
                label: "Intensity",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 0.8,
            },
        ],
    },
    // --- Output ---
    NodeTypeSpec {
        kind: "output",
        title: "Effect Output",
        category: "Output",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "alpha",
                label: "Alpha",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[],
        params: &[],
    },
];

/// Complete catalog of 3D `NodeMaterial` procedural PBR nodes matching `zc_shader::node_graph`.
pub const MATERIAL_NODE_CATALOG: &[NodeTypeSpec] = &[
    // --- Inputs ---
    NodeTypeSpec {
        kind: "float",
        title: "Float Constant",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "value",
            label: "Value",
            kind: NodeParamKind::Float {
                min: -100.0,
                max: 100.0,
                step: 0.02,
            },
            default_f64: 0.5,
        }],
    },
    NodeTypeSpec {
        kind: "color",
        title: "Color Constant",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "color",
            label: "Color",
            kind: NodeParamKind::ColorRgb,
            default_f64: 0.8,
        }],
    },
    NodeTypeSpec {
        kind: "randomColor",
        title: "Random Color",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "seed",
            label: "Seed",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1000.0,
                step: 1.0,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "projection",
        title: "UV Projection",
        category: "Inputs",
        inputs: &[],
        outputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        params: &[],
    },
    NodeTypeSpec {
        kind: "uvw",
        title: "UV Transform",
        category: "Inputs",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        params: &[
            NodeParamSpec {
                key: "scaleX",
                label: "Scale X",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 32.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "scaleY",
                label: "Scale Y",
                kind: NodeParamKind::Float {
                    min: 0.05,
                    max: 32.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "offsetX",
                label: "Offset X",
                kind: NodeParamKind::Float {
                    min: -4.0,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "offsetY",
                label: "Offset Y",
                kind: NodeParamKind::Float {
                    min: -4.0,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "rotation",
                label: "Rotation",
                kind: NodeParamKind::Float {
                    min: -360.0,
                    max: 360.0,
                    step: 1.0,
                },
                default_f64: 0.0,
            },
        ],
    },
    // --- Procedural ---
    NodeTypeSpec {
        kind: "noise",
        title: "Perlin Noise",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "out",
                label: "Factor",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 64.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "octaves",
                label: "Octaves",
                kind: NodeParamKind::Int { min: 1, max: 8 },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "roughness",
                label: "Detail",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 0.9,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "voronoi",
        title: "Voronoi",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "distance",
                label: "Distance",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "color",
                label: "Cell Color",
                data_type: PortDataType::Color,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 64.0,
                    step: 0.2,
                },
                default_f64: 6.0,
            },
            NodeParamSpec {
                key: "jitter",
                label: "Jitter",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "checker",
        title: "Checker",
        category: "Procedural",
        inputs: &[
            PortSpec {
                id: "uv",
                label: "UV",
                data_type: PortDataType::Uv,
            },
            PortSpec {
                id: "a",
                label: "Color A",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "b",
                label: "Color B",
                data_type: PortDataType::Color,
            },
        ],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Mask",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 64.0,
                    step: 0.5,
                },
                default_f64: 8.0,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.15,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.85,
            },
        ],
    },
    NodeTypeSpec {
        kind: "gradient",
        title: "Gradient",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "mode",
            label: "Mode",
            kind: NodeParamKind::Choice(GRADIENT_MODES),
            default_f64: 0.0,
        }],
    },
    NodeTypeSpec {
        kind: "wave",
        title: "Wave",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 64.0,
                    step: 0.2,
                },
                default_f64: 5.0,
            },
            NodeParamSpec {
                key: "distortion",
                label: "Distortion",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 10.0,
                    step: 0.1,
                },
                default_f64: 1.5,
            },
        ],
    },
    NodeTypeSpec {
        kind: "brick",
        title: "Brick",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "out",
                label: "Mortar",
                data_type: PortDataType::Float,
            },
        ],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 1.0,
                    max: 32.0,
                    step: 0.5,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "mortarSize",
                label: "Mortar",
                kind: NodeParamKind::Float {
                    min: 0.005,
                    max: 0.2,
                    step: 0.005,
                },
                default_f64: 0.03,
            },
        ],
    },
    NodeTypeSpec {
        kind: "marble",
        title: "Marble",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 32.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "turbulence",
                label: "Veins",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 12.0,
                    step: 0.2,
                },
                default_f64: 5.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "ridgedFractal",
        title: "Ridged Fractal",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "scale",
                label: "Scale",
                kind: NodeParamKind::Float {
                    min: 0.5,
                    max: 32.0,
                    step: 0.2,
                },
                default_f64: 4.0,
            },
            NodeParamSpec {
                key: "octaves",
                label: "Octaves",
                kind: NodeParamKind::Int { min: 1, max: 8 },
                default_f64: 4.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "turbulence",
        title: "Turbulence",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "scale",
            label: "Scale",
            kind: NodeParamKind::Float {
                min: 0.5,
                max: 32.0,
                step: 0.2,
            },
            default_f64: 5.0,
        }],
    },
    NodeTypeSpec {
        kind: "whiteNoise",
        title: "White Noise",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "uv",
            label: "UV",
            data_type: PortDataType::Uv,
        }],
        outputs: &[
            PortSpec {
                id: "out",
                label: "Value",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "color",
                label: "Color",
                data_type: PortDataType::Color,
            },
        ],
        params: &[],
    },
    NodeTypeSpec {
        kind: "blackbody",
        title: "Blackbody",
        category: "Procedural",
        inputs: &[PortSpec {
            id: "temperature",
            label: "Kelvin",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "temperature",
            label: "Kelvin",
            kind: NodeParamKind::Float {
                min: 1000.0,
                max: 15000.0,
                step: 100.0,
            },
            default_f64: 5500.0,
        }],
    },
    // --- Color & Math ---
    NodeTypeSpec {
        kind: "colorRamp",
        title: "Color Ramp",
        category: "Color & Math",
        inputs: &[PortSpec {
            id: "factor",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "interpolation",
                label: "Interp",
                kind: NodeParamKind::Choice(RAMP_INTERP),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "stops",
                label: "Stops",
                kind: NodeParamKind::ColorRamp,
                default_f64: 0.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "mix",
        title: "Mix",
        category: "Color & Math",
        inputs: &[
            PortSpec {
                id: "a",
                label: "Color A",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "b",
                label: "Color B",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "factor",
                label: "Factor",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Blend Mode",
                kind: NodeParamKind::Choice(MIX_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "factor",
                label: "Factor",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "a",
                label: "Color A",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.2,
            },
            NodeParamSpec {
                key: "b",
                label: "Color B",
                kind: NodeParamKind::ColorRgb,
                default_f64: 0.8,
            },
        ],
    },
    NodeTypeSpec {
        kind: "colorCorrection",
        title: "Color Correction",
        category: "Color & Math",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[
            NodeParamSpec {
                key: "brightness",
                label: "Brightness",
                kind: NodeParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "saturation",
                label: "Saturation",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 3.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: NodeParamKind::Float {
                    min: 0.1,
                    max: 4.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "invert",
        title: "Invert",
        category: "Color & Math",
        inputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        outputs: &[PortSpec {
            id: "color",
            label: "Color",
            data_type: PortDataType::Color,
        }],
        params: &[NodeParamSpec {
            key: "factor",
            label: "Factor",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "mapRange",
        title: "Map Range",
        category: "Color & Math",
        inputs: &[PortSpec {
            id: "value",
            label: "Value",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "fromMin",
                label: "From Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "fromMax",
                label: "From Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            NodeParamSpec {
                key: "toMin",
                label: "To Min",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "toMax",
                label: "To Max",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    NodeTypeSpec {
        kind: "math",
        title: "Scalar Math",
        category: "Color & Math",
        inputs: &[
            PortSpec {
                id: "a",
                label: "A",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "b",
                label: "B",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[PortSpec {
            id: "out",
            label: "Result",
            data_type: PortDataType::Float,
        }],
        params: &[
            NodeParamSpec {
                key: "mode",
                label: "Operation",
                kind: NodeParamKind::Choice(MATH_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "a",
                label: "A",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
            NodeParamSpec {
                key: "b",
                label: "B",
                kind: NodeParamKind::Float {
                    min: -100.0,
                    max: 100.0,
                    step: 0.05,
                },
                default_f64: 0.5,
            },
        ],
    },
    // --- Surface ---
    NodeTypeSpec {
        kind: "bump",
        title: "Bump / Normal Map",
        category: "Surface",
        inputs: &[PortSpec {
            id: "height",
            label: "Height",
            data_type: PortDataType::Float,
        }],
        outputs: &[PortSpec {
            id: "normal",
            label: "Normal",
            data_type: PortDataType::Vec3,
        }],
        params: &[NodeParamSpec {
            key: "strength",
            label: "Strength",
            kind: NodeParamKind::Float {
                min: 0.0,
                max: 5.0,
                step: 0.05,
            },
            default_f64: 1.0,
        }],
    },
    NodeTypeSpec {
        kind: "fresnel",
        title: "Fresnel",
        category: "Surface",
        inputs: &[],
        outputs: &[PortSpec {
            id: "out",
            label: "Factor",
            data_type: PortDataType::Float,
        }],
        params: &[NodeParamSpec {
            key: "ior",
            label: "IOR",
            kind: NodeParamKind::Float {
                min: 1.0,
                max: 4.0,
                step: 0.05,
            },
            default_f64: 1.45,
        }],
    },
    // --- Output ---
    NodeTypeSpec {
        kind: "output",
        title: "Material Output",
        category: "Output",
        inputs: &[
            PortSpec {
                id: "color",
                label: "Base Color",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "emissive",
                label: "Emissive",
                data_type: PortDataType::Color,
            },
            PortSpec {
                id: "roughness",
                label: "Roughness",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "metalness",
                label: "Metalness",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "normal",
                label: "Normal",
                data_type: PortDataType::Vec3,
            },
            PortSpec {
                id: "bump",
                label: "Bump",
                data_type: PortDataType::Float,
            },
            PortSpec {
                id: "opacity",
                label: "Opacity",
                data_type: PortDataType::Float,
            },
        ],
        outputs: &[],
        params: &[
            NodeParamSpec {
                key: "normalMode",
                label: "Normal Mode",
                kind: NodeParamKind::Choice(NORMAL_MODES),
                default_f64: 0.0,
            },
            NodeParamSpec {
                key: "normalStrength",
                label: "Normal Str",
                kind: NodeParamKind::Float {
                    min: 0.0,
                    max: 5.0,
                    step: 0.05,
                },
                default_f64: 1.0,
            },
        ],
    },
];

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

// ===========================================================================
// 2. Interactive Color Looks Setup Window (29 Tools, 5-Stage Pipeline)
// ===========================================================================

/// 5-stage Color Looks pipeline execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LooksStage {
    Subject,
    Matte,
    Lens,
    Camera,
    Post,
}

impl LooksStage {
    pub const ALL: [Self; 5] = [
        Self::Subject,
        Self::Matte,
        Self::Lens,
        Self::Camera,
        Self::Post,
    ];

    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Subject => "Subject",
            Self::Matte => "Matte",
            Self::Lens => "Lens",
            Self::Camera => "Camera",
            Self::Post => "Post",
        }
    }

    #[must_use]
    pub const fn badge_color(self) -> Color32 {
        match self {
            Self::Subject => Color32::from_rgb(76, 152, 224),
            Self::Matte => Color32::from_rgb(138, 108, 212),
            Self::Lens => Color32::from_rgb(228, 156, 68),
            Self::Camera => Color32::from_rgb(74, 186, 134),
            Self::Post => Color32::from_rgb(218, 92, 118),
        }
    }
}

/// Parameter descriptor for a Color Looks tool.
#[derive(Debug, Clone, Copy)]
pub enum LooksParamKind {
    Float { min: f64, max: f64, step: f64 },
    Bool,
    Choice(&'static [&'static str]),
    ColorRgb,
}

#[derive(Debug, Clone, Copy)]
pub struct LooksParamSpec {
    pub key: &'static str,
    pub label: &'static str,
    pub kind: LooksParamKind,
    pub default_f64: f64,
}

/// Static catalog entry for one of the 29 Color Looks tools.
#[derive(Debug, Clone, Copy)]
pub struct LooksToolSpec {
    pub tool_id: &'static str,
    pub label: &'static str,
    pub stage: LooksStage,
    pub params: &'static [LooksParamSpec],
}

/// One active tool instance in a `magiclooks` effect's `looksInstances` chain.
#[derive(Debug, Clone, PartialEq)]
pub struct LooksToolInstance {
    pub tool_id: String,
    pub enabled: bool,
    pub params: Map<String, Value>,
}

/// Interactive state for the Color Looks Setup window.
#[derive(Debug, Clone)]
pub struct LooksEditorState {
    pub open: bool,
    pub effect_id: String,
    pub effect_label: String,
    pub chain: Vec<LooksToolInstance>,
    pub selected_index: Option<usize>,
    pub picker_open: bool,
    pub picker_search: String,
    pub picker_stage_filter: Option<LooksStage>,
    pub auto_apply: bool,
    pub dirty: bool,
}

impl Default for LooksEditorState {
    fn default() -> Self {
        Self {
            open: false,
            effect_id: String::new(),
            effect_label: String::new(),
            chain: Vec::new(),
            selected_index: None,
            picker_open: false,
            picker_search: String::new(),
            picker_stage_filter: None,
            auto_apply: true,
            dirty: false,
        }
    }
}

impl LooksEditorState {
    /// Opens the Color Looks Setup window for the given `magiclooks` effect ID and `looksInstances` value.
    pub fn open_for_effect(
        &mut self,
        effect_id: impl Into<String>,
        effect_label: impl Into<String>,
        looks_instances_json: Option<&Value>,
    ) {
        self.open = true;
        self.effect_id = effect_id.into();
        self.effect_label = effect_label.into();
        self.picker_open = false;
        self.picker_search.clear();
        self.picker_stage_filter = None;
        self.load_from_json(looks_instances_json);
        self.dirty = false;
    }

    /// Parses a `looksInstances` JSON array (or JSON-encoded string).
    pub fn load_from_json(&mut self, value: Option<&Value>) {
        self.chain.clear();
        let owned;
        let arr = match value {
            Some(Value::Array(a)) => Some(a.as_slice()),
            Some(Value::String(s)) => {
                owned = serde_json::from_str::<Value>(s).ok();
                owned.as_ref().and_then(Value::as_array).map(Vec::as_slice)
            }
            _ => None,
        };
        if let Some(items) = arr {
            for item in items {
                let Some(obj) = item.as_object() else {
                    continue;
                };
                let Some(tool_id) = obj
                    .get("toolId")
                    .or_else(|| obj.get("tool_id"))
                    .or_else(|| obj.get("id"))
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                let enabled = obj.get("enabled").and_then(Value::as_bool).unwrap_or(true);
                let mut params = Map::new();
                for (k, v) in obj {
                    if !matches!(
                        k.as_str(),
                        "toolId" | "tool_id" | "id" | "enabled" | "params"
                    ) {
                        params.insert(k.clone(), v.clone());
                    }
                }
                if let Some(Value::Object(nested)) = obj.get("params") {
                    for (k, v) in nested {
                        params.insert(k.clone(), v.clone());
                    }
                }
                self.chain.push(LooksToolInstance {
                    tool_id: tool_id.to_string(),
                    enabled,
                    params,
                });
            }
        }
        self.selected_index = if self.chain.is_empty() { None } else { Some(0) };
    }

    /// Serializes the active tool chain to the canonical `looksInstances` JSON array.
    #[must_use]
    pub fn to_json(&self) -> Value {
        Value::Array(
            self.chain
                .iter()
                .map(|inst| {
                    json!({
                        "toolId": inst.tool_id,
                        "enabled": inst.enabled,
                        "params": Value::Object(inst.params.clone()),
                    })
                })
                .collect(),
        )
    }

    /// Appends a tool by canonical `tool_id` with default parameters and selects it.
    pub fn add_tool(&mut self, tool_id: &str) -> usize {
        let mut params = Map::new();
        if let Some(spec) = find_looks_tool(tool_id) {
            for p in spec.params {
                match p.kind {
                    LooksParamKind::Float { .. } => {
                        params.insert(p.key.into(), json!(p.default_f64));
                    }
                    LooksParamKind::Bool => {
                        params.insert(p.key.into(), json!(p.default_f64 != 0.0));
                    }
                    LooksParamKind::Choice(_) => {
                        params.insert(p.key.into(), json!(p.default_f64.round() as i64));
                    }
                    LooksParamKind::ColorRgb => {
                        params.insert(
                            p.key.into(),
                            json!([p.default_f64, p.default_f64, p.default_f64]),
                        );
                    }
                }
            }
        }
        self.chain.push(LooksToolInstance {
            tool_id: tool_id.to_string(),
            enabled: true,
            params,
        });
        let idx = self.chain.len() - 1;
        self.selected_index = Some(idx);
        self.dirty = true;
        idx
    }

    /// Duplicates the tool at `index`.
    pub fn duplicate_tool(&mut self, index: usize) -> Option<usize> {
        let copy = self.chain.get(index)?.clone();
        let insert_at = index + 1;
        self.chain.insert(insert_at, copy);
        self.selected_index = Some(insert_at);
        self.dirty = true;
        Some(insert_at)
    }

    /// Removes the tool at `index`.
    pub fn remove_tool(&mut self, index: usize) -> bool {
        if index >= self.chain.len() {
            return false;
        }
        self.chain.remove(index);
        if self.chain.is_empty() {
            self.selected_index = None;
        } else {
            self.selected_index = Some(index.min(self.chain.len() - 1));
        }
        self.dirty = true;
        true
    }

    /// Moves a tool one position earlier (`delta = -1`) or later (`delta = 1`).
    pub fn move_tool(&mut self, index: usize, delta: isize) -> bool {
        let target = index as isize + delta;
        if target < 0 || target as usize >= self.chain.len() {
            return false;
        }
        let target = target as usize;
        self.chain.swap(index, target);
        self.selected_index = Some(target);
        self.dirty = true;
        true
    }

    /// Applies one of the 6 built-in curated Look Presets.
    pub fn apply_preset(&mut self, preset_name: &str) {
        self.chain.clear();
        match preset_name {
            "Cinema Warm" => {
                self.push_preset_tool(
                    "warm-cool",
                    json!({"warmCool": 0.28, "tint": 0.04, "exposure": 0.08}),
                );
                self.push_preset_tool("lift-gamma-gain", json!({"strength": 0.65, "lift": [0.94, 0.96, 1.04], "gamma": [1.04, 1.0, 0.96], "gain": [1.08, 1.02, 0.92]}));
                self.push_preset_tool(
                    "diffusion",
                    json!({"size": 0.35, "grade": 0.32, "glow": 0.25, "highlightsOnly": true}),
                );
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.72, "spread": 0.45, "strength": 0.38}),
                );
                self.push_preset_tool("auto-shoulder", json!({"strength": 0.45}));
            }
            "Teal & Orange" => {
                self.push_preset_tool("mojo", json!({"mojo": 0.78, "punch": 0.35, "blueSqueeze": 0.45, "skinSqueeze": 0.55, "strength": 0.85}));
                self.push_preset_tool(
                    "color-contrast",
                    json!({"pivot": 0.48, "exposure": 0.12, "color": [1.0, 0.68, 0.42]}),
                );
                self.push_preset_tool("anamorphic-flare", json!({"size": 0.42, "boost": 0.35, "threshold": 0.78, "color": [0.35, 0.78, 1.0]}));
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.68, "spread": 0.4, "strength": 0.42}),
                );
            }
            "Neon Cyber" => {
                self.push_preset_tool(
                    "hsl-colors",
                    json!({"cyanSat": 0.45, "blueSat": 0.35, "magentaSat": 0.5, "purpleSat": 0.4}),
                );
                self.push_preset_tool(
                    "contrast",
                    json!({"contrast": 0.38, "pivot": 0.45, "exposure": 0.05}),
                );
                self.push_preset_tool(
                    "chromatic-aberration",
                    json!({"redCyan": 0.25, "greenMagenta": -0.1, "blueYellow": 0.18}),
                );
                self.push_preset_tool("star-filter", json!({"size": 0.4, "boost": 0.45, "threshold": 0.72, "angle": 30.0, "color": [0.65, 0.9, 1.0]}));
                self.push_preset_tool(
                    "diffusion",
                    json!({"size": 0.45, "glow": 0.42, "highlightsOnly": true}),
                );
            }
            "Film Noir" => {
                self.push_preset_tool("ranged-saturation", json!({"satHighlight": -1.0, "satMidtone": -1.0, "satShadow": -1.0, "exposure": 0.05}));
                self.push_preset_tool(
                    "contrast",
                    json!({"contrast": 0.55, "pivot": 0.46, "exposure": 0.0}),
                );
                self.push_preset_tool("crush", json!({"gamma": 1.25, "exposure": -0.12}));
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.58, "spread": 0.42, "strength": 0.62}),
                );
                self.push_preset_tool(
                    "edge-softness",
                    json!({"blurSize": 0.22, "radius": 0.72, "spread": 0.35}),
                );
            }
            "Pastel Dream" => {
                self.push_preset_tool("lift-gamma-gain", json!({"strength": 0.55, "lift": [1.06, 1.02, 1.08], "gamma": [1.05, 1.03, 1.06], "gain": [0.98, 0.96, 1.02]}));
                self.push_preset_tool(
                    "lightflex",
                    json!({"boost": 0.32, "exposure": 0.1, "color": [1.0, 0.88, 0.94]}),
                );
                self.push_preset_tool(
                    "diffusion",
                    json!({"size": 0.55, "grade": 0.4, "glow": 0.38}),
                );
                self.push_preset_tool(
                    "ranged-saturation",
                    json!({"satHighlight": -0.25, "satMidtone": -0.15, "satShadow": -0.2}),
                );
            }
            "Vintage Chrome" => {
                self.push_preset_tool("three-strip", json!({"strength": 0.68, "exposure": 0.06}));
                self.push_preset_tool(
                    "color-reversal",
                    json!({"strength": 0.35, "exposure": 0.04}),
                );
                self.push_preset_tool("haze-flare", json!({"spillage": 0.3, "softness": 0.55, "reach": 0.4, "exposure": 0.2, "color": [1.0, 0.86, 0.68]}));
                self.push_preset_tool(
                    "lens-distortion",
                    json!({"distortion": -0.12, "flatness": 0.5}),
                );
                self.push_preset_tool(
                    "vignette",
                    json!({"radius": 0.64, "spread": 0.48, "strength": 0.48}),
                );
            }
            _ => {}
        }
        self.selected_index = if self.chain.is_empty() { None } else { Some(0) };
        self.dirty = true;
    }

    fn push_preset_tool(&mut self, tool_id: &str, overrides: Value) {
        let idx = self.add_tool(tool_id);
        if let (Some(inst), Some(map)) = (self.chain.get_mut(idx), overrides.as_object()) {
            for (k, v) in map {
                inst.params.insert(k.clone(), v.clone());
            }
        }
    }

    /// Commits `looksInstances` into `Session` via `Command::SetPropertyValue`.
    pub fn commit_to_session(&mut self, session: &mut Session) -> bool {
        if self.effect_id.is_empty() {
            return false;
        }
        let ok = session
            .execute(Command::SetPropertyValue {
                target_id: self.effect_id.clone(),
                property: "looksInstances".to_string(),
                value: self.to_json(),
            })
            .is_ok();
        if ok {
            self.dirty = false;
        }
        ok
    }
}

pub const LOOKS_PRESET_NAMES: &[&str] = &[
    "Cinema Warm",
    "Teal & Orange",
    "Neon Cyber",
    "Film Noir",
    "Pastel Dream",
    "Vintage Chrome",
];

/// Renders the floating Color Looks Setup window when `state.open` is true.
pub fn show_looks_editor_window(
    ctx: &egui::Context,
    viewport_rect: Rect,
    floating_windows: &mut FloatingWindowManager,
    state: &mut LooksEditorState,
    session: &mut Session,
    theme: &Cm3Theme,
) {
    if !state.open {
        return;
    }

    let spec = FloatingWindowSpec::new("pro_color_looks_editor", "Color Looks Setup")
        .with_subtitle(state.effect_label.clone())
        .with_default_size(Vec2::new(920.0, 580.0));

    let mut close_requested = false;
    let mut should_commit = false;

    let ev = cm3_floating_window(ctx, floating_windows, &spec, viewport_rect, theme, |ui| {
        // --- Top Bar: Presets + Add Tool + Actions ---
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            if cm3_prop_button(
                ui,
                if state.picker_open {
                    "Hide Tool Browser"
                } else {
                    "+ Add Tool (29)"
                },
                if state.picker_open {
                    ButtonVariant::Primary
                } else {
                    ButtonVariant::Normal
                },
                false,
                theme,
            )
            .clicked()
            {
                state.picker_open = !state.picker_open;
                state.picker_stage_filter = None;
            }

            egui::ComboBox::from_id_salt(Id::unique("looks_preset_combo"))
                .selected_text("Look Presets")
                .width(132.0)
                .show_ui(ui, |ui| {
                    for preset in LOOKS_PRESET_NAMES {
                        if ui.selectable_label(false, *preset).clicked() {
                            state.apply_preset(preset);
                            if state.auto_apply {
                                should_commit = true;
                            }
                        }
                    }
                });

            if !state.chain.is_empty()
                && cm3_prop_button(ui, "Clear All", ButtonVariant::Danger, false, theme).clicked()
            {
                state.chain.clear();
                state.selected_index = None;
                state.dirty = true;
                if state.auto_apply {
                    should_commit = true;
                }
            }

            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if cm3_prop_button(ui, "Close", ButtonVariant::Normal, false, theme).clicked() {
                    close_requested = true;
                }
                if cm3_prop_button(
                    ui,
                    if state.dirty { "Apply *" } else { "Apply" },
                    ButtonVariant::Primary,
                    false,
                    theme,
                )
                .clicked()
                {
                    should_commit = true;
                }
                ui.checkbox(&mut state.auto_apply, "Auto-Apply");
            });
        });

        ui.add_space(6.0);

        // --- 5-Stage Pipeline Strip (Subject -> Matte -> Lens -> Camera -> Post) ---
        egui::Frame::NONE
            .fill(BG_SUNKEN)
            .stroke(Stroke::new(1.0, BORDER_PANEL))
            .corner_radius(CornerRadius::same(4))
            .inner_margin(egui::Margin::same(6))
            .show(ui, |ui| {
                let stage_w = ((ui.available_width() - 24.0) / 5.0).max(96.0);
                ui.horizontal(|ui| {
                    ui.spacing_mut().item_spacing.x = 6.0;
                    for stage in LooksStage::ALL {
                        ui.vertical(|ui| {
                            ui.set_width(stage_w);
                            ui.horizontal(|ui| {
                                let (dot_r, _) =
                                    ui.allocate_exact_size(Vec2::splat(7.0), Sense::hover());
                                ui.painter().circle_filled(
                                    dot_r.center(),
                                    3.5,
                                    stage.badge_color(),
                                );
                                ui.label(
                                    egui::RichText::new(stage.label())
                                        .font(theme.font(FONT_SMALL))
                                        .color(TEXT_STRONG),
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if ui.small_button("+").clicked() {
                                            state.picker_open = true;
                                            state.picker_stage_filter = Some(stage);
                                        }
                                    },
                                );
                            });
                            ui.add_space(2.0);
                            let mut stage_count = 0usize;
                            for (idx, inst) in state.chain.iter().enumerate() {
                                let Some(spec) = find_looks_tool(&inst.tool_id) else {
                                    continue;
                                };
                                if spec.stage != stage {
                                    continue;
                                }
                                stage_count += 1;
                                let is_sel = state.selected_index == Some(idx);
                                let fill = if is_sel {
                                    color32_with_alpha(stage.badge_color(), 68)
                                } else if inst.enabled {
                                    BG_BUTTON
                                } else {
                                    BG_PANEL
                                };
                                let text_col = if inst.enabled {
                                    TEXT_PRIMARY
                                } else {
                                    TEXT_FAINT
                                };
                                let resp = ui.add_sized(
                                    Vec2::new(stage_w, 20.0),
                                    egui::Button::new(
                                        egui::RichText::new(spec.label)
                                            .font(theme.font(FONT_COMPACT))
                                            .color(text_col),
                                    )
                                    .fill(fill)
                                    .stroke(Stroke::new(
                                        1.0,
                                        if is_sel {
                                            stage.badge_color()
                                        } else {
                                            BORDER_CONTROL
                                        },
                                    )),
                                );
                                if resp.clicked() {
                                    state.selected_index = Some(idx);
                                }
                            }
                            if stage_count == 0 {
                                ui.label(
                                    egui::RichText::new("Empty")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(TEXT_FAINT),
                                );
                            }
                        });
                    }
                });
            });

        ui.add_space(6.0);

        // --- Main Split: Optional Tool Picker | Active Chain List | Tool Parameters ---
        let body_h = (ui.available_height() - 4.0).max(180.0);
        let picker_w = if state.picker_open { 220.0 } else { 0.0 };
        let chain_w = 240.0;
        let params_w = (ui.available_width()
            - picker_w
            - chain_w
            - if state.picker_open { 12.0 } else { 6.0 })
        .max(220.0);

        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = 6.0;

            // Optional Tool Browser
            if state.picker_open {
                egui::Frame::NONE
                    .fill(BG_SUNKEN)
                    .stroke(Stroke::new(1.0, BORDER_PANEL))
                    .corner_radius(CornerRadius::same(4))
                    .inner_margin(egui::Margin::same(6))
                    .show(ui, |ui| {
                        ui.set_width(picker_w - 14.0);
                        ui.set_height(body_h);
                        if show_looks_picker_ui(ui, state, theme) && state.auto_apply {
                            should_commit = true;
                        }
                    });
            }

            // Active Chain Order Column
            egui::Frame::NONE
                .fill(BG_PANEL)
                .stroke(Stroke::new(1.0, BORDER_PANEL))
                .corner_radius(CornerRadius::same(4))
                .inner_margin(egui::Margin::same(6))
                .show(ui, |ui| {
                    ui.set_width(chain_w - 14.0);
                    ui.set_height(body_h);
                    if show_looks_chain_list_ui(ui, state, theme) && state.auto_apply {
                        should_commit = true;
                    }
                });

            // Selected Tool Parameter Inspector
            egui::Frame::NONE
                .fill(BG_PANEL)
                .stroke(Stroke::new(1.0, BORDER_PANEL))
                .corner_radius(CornerRadius::same(4))
                .inner_margin(egui::Margin::same(8))
                .show(ui, |ui| {
                    ui.set_width(params_w - 18.0);
                    ui.set_height(body_h);
                    if show_looks_tool_params_ui(ui, state, theme) && state.auto_apply {
                        should_commit = true;
                    }
                });
        });
    });

    if should_commit {
        state.commit_to_session(session);
    }
    if close_requested || ev.closed {
        state.open = false;
        floating_windows.close("pro_color_looks_editor");
    }
}

fn show_looks_picker_ui(ui: &mut Ui, state: &mut LooksEditorState, theme: &Cm3Theme) -> bool {
    let mut changed = false;
    ui.label(
        egui::RichText::new("Tools (29)")
            .font(theme.font(FONT_TITLE))
            .color(TEXT_STRONG),
    );
    ui.add_space(3.0);
    ui.add(
        TextEdit::singleline(&mut state.picker_search)
            .hint_text("Filter tools...")
            .desired_width(ui.available_width()),
    );
    ui.add_space(3.0);
    ui.horizontal_wrapped(|ui| {
        if ui
            .selectable_label(state.picker_stage_filter.is_none(), "All")
            .clicked()
        {
            state.picker_stage_filter = None;
        }
        for stage in LooksStage::ALL {
            if ui
                .selectable_label(state.picker_stage_filter == Some(stage), stage.label())
                .clicked()
            {
                state.picker_stage_filter = Some(stage);
            }
        }
    });
    ui.add_space(4.0);

    let q = state.picker_search.trim().to_ascii_lowercase();
    ScrollArea::vertical()
        .id_salt("looks_tool_picker_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            for stage in LooksStage::ALL {
                if let Some(filter) = state.picker_stage_filter
                    && filter != stage
                {
                    continue;
                }
                let mut header_shown = false;
                for spec in LOOKS_TOOL_CATALOG.iter().filter(|s| s.stage == stage) {
                    if !q.is_empty()
                        && !spec.label.to_ascii_lowercase().contains(&q)
                        && !spec.tool_id.to_ascii_lowercase().contains(&q)
                    {
                        continue;
                    }
                    if !header_shown {
                        ui.add_space(3.0);
                        ui.label(
                            egui::RichText::new(stage.label().to_ascii_uppercase())
                                .font(theme.font(FONT_COMPACT))
                                .color(stage.badge_color()),
                        );
                        header_shown = true;
                    }
                    if ui
                        .add_sized(
                            Vec2::new(ui.available_width(), 22.0),
                            egui::Button::new(
                                egui::RichText::new(spec.label)
                                    .font(theme.font(FONT_SMALL))
                                    .color(TEXT_PRIMARY),
                            )
                            .fill(BG_BUTTON)
                            .stroke(Stroke::new(1.0, BORDER_CONTROL)),
                        )
                        .clicked()
                    {
                        state.add_tool(spec.tool_id);
                        changed = true;
                    }
                }
            }
        });
    changed
}

fn show_looks_chain_list_ui(ui: &mut Ui, state: &mut LooksEditorState, theme: &Cm3Theme) -> bool {
    let mut changed = false;
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("Active Chain ({})", state.chain.len()))
                .font(theme.font(FONT_TITLE))
                .color(TEXT_STRONG),
        );
    });
    ui.separator();

    if state.chain.is_empty() {
        ui.label(
            egui::RichText::new("No tools active.\nPick a preset or click '+ Add Tool'.")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_FAINT),
        );
        return false;
    }

    let mut action: Option<(usize, &'static str)> = None;
    ScrollArea::vertical()
        .id_salt("looks_chain_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let count = state.chain.len();
            for idx in 0..count {
                let selected = state.selected_index == Some(idx);
                let tool_id = state.chain[idx].tool_id.clone();
                let spec = find_looks_tool(&tool_id);
                let label = spec.map(|s| s.label).unwrap_or(tool_id.as_str());
                let stage_col = spec
                    .map(|s| s.stage.badge_color())
                    .unwrap_or_else(|| theme.accent());

                egui::Frame::NONE
                    .fill(if selected { BG_HOVER } else { BG_INPUT })
                    .stroke(Stroke::new(
                        1.0,
                        if selected { stage_col } else { BORDER_CONTROL },
                    ))
                    .corner_radius(CornerRadius::same(3))
                    .inner_margin(egui::Margin::symmetric(5, 3))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            if ui.checkbox(&mut state.chain[idx].enabled, "").changed() {
                                state.dirty = true;
                                changed = true;
                            }
                            let label_resp = ui.selectable_label(
                                selected,
                                egui::RichText::new(label)
                                    .font(theme.font(FONT_SMALL))
                                    .color(if state.chain[idx].enabled {
                                        TEXT_PRIMARY
                                    } else {
                                        TEXT_FAINT
                                    }),
                            );
                            if label_resp.clicked() {
                                state.selected_index = Some(idx);
                            }
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.small_button("×").on_hover_text("Delete").clicked() {
                                        action = Some((idx, "del"));
                                    }
                                    if ui.small_button("D").on_hover_text("Duplicate").clicked() {
                                        action = Some((idx, "dup"));
                                    }
                                    if idx + 1 < count
                                        && ui.small_button("↓").on_hover_text("Move Down").clicked()
                                    {
                                        action = Some((idx, "down"));
                                    }
                                    if idx > 0
                                        && ui.small_button("↑").on_hover_text("Move Up").clicked()
                                    {
                                        action = Some((idx, "up"));
                                    }
                                },
                            );
                        });
                    });
                ui.add_space(3.0);
            }
        });

    if let Some((idx, cmd)) = action {
        match cmd {
            "del" => changed |= state.remove_tool(idx),
            "dup" => changed |= state.duplicate_tool(idx).is_some(),
            "up" => changed |= state.move_tool(idx, -1),
            "down" => changed |= state.move_tool(idx, 1),
            _ => {}
        }
    }
    changed
}

fn show_looks_tool_params_ui(ui: &mut Ui, state: &mut LooksEditorState, theme: &Cm3Theme) -> bool {
    let mut changed = false;
    let Some(idx) = state.selected_index.filter(|&i| i < state.chain.len()) else {
        ui.label(
            egui::RichText::new("Select a tool in the chain to edit its parameters.")
                .font(theme.font(FONT_SMALL))
                .color(TEXT_MUTED),
        );
        return false;
    };

    let tool_id = state.chain[idx].tool_id.clone();
    let Some(spec) = find_looks_tool(&tool_id) else {
        ui.label(
            egui::RichText::new(format!("Unknown tool: {tool_id}"))
                .font(theme.font(FONT_SMALL))
                .color(DANGER_TEXT),
        );
        return false;
    };

    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(spec.label)
                .font(theme.font(FONT_TITLE))
                .color(TEXT_STRONG),
        );
        ui.label(
            egui::RichText::new(format!("[{}]", spec.stage.label()))
                .font(theme.font(FONT_SMALL))
                .color(spec.stage.badge_color()),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if cm3_prop_button(ui, "Reset Tool", ButtonVariant::Normal, false, theme).clicked() {
                state.chain[idx].params.clear();
                for p in spec.params {
                    match p.kind {
                        LooksParamKind::Float { .. } => {
                            state.chain[idx]
                                .params
                                .insert(p.key.into(), json!(p.default_f64));
                        }
                        LooksParamKind::Bool => {
                            state.chain[idx]
                                .params
                                .insert(p.key.into(), json!(p.default_f64 != 0.0));
                        }
                        LooksParamKind::Choice(_) => {
                            state.chain[idx]
                                .params
                                .insert(p.key.into(), json!(p.default_f64.round() as i64));
                        }
                        LooksParamKind::ColorRgb => {
                            state.chain[idx].params.insert(
                                p.key.into(),
                                json!([p.default_f64, p.default_f64, p.default_f64]),
                            );
                        }
                    }
                }
                state.dirty = true;
                changed = true;
            }
        });
    });
    ui.separator();

    ScrollArea::vertical()
        .id_salt("looks_tool_params_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            let inst = &mut state.chain[idx];
            for param in spec.params {
                ui.add_space(3.0);
                match param.kind {
                    LooksParamKind::Float { min, max, step } => {
                        let mut val = inst
                            .params
                            .get(param.key)
                            .and_then(Value::as_f64)
                            .unwrap_or(param.default_f64);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_BODY))
                                    .color(TEXT_PRIMARY),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui
                                        .add(
                                            egui::Slider::new(&mut val, min..=max)
                                                .step_by(step)
                                                .max_decimals(3),
                                        )
                                        .changed()
                                    {
                                        inst.params.insert(param.key.into(), json!(val));
                                        changed = true;
                                    }
                                },
                            );
                        });
                    }
                    LooksParamKind::Bool => {
                        let mut val = inst
                            .params
                            .get(param.key)
                            .map(|v| {
                                v.as_bool()
                                    .unwrap_or_else(|| v.as_f64().unwrap_or(0.0) != 0.0)
                            })
                            .unwrap_or(param.default_f64 != 0.0);
                        if ui.checkbox(&mut val, param.label).changed() {
                            inst.params.insert(param.key.into(), json!(val));
                            changed = true;
                        }
                    }
                    LooksParamKind::Choice(options) => {
                        let mut opt_idx = inst
                            .params
                            .get(param.key)
                            .and_then(Value::as_u64)
                            .map(|v| (v as usize).min(options.len().saturating_sub(1)))
                            .unwrap_or(param.default_f64.round().max(0.0) as usize);
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_BODY))
                                    .color(TEXT_PRIMARY),
                            );
                            egui::ComboBox::from_id_salt(Id::unique((
                                "looks_choice",
                                idx,
                                param.key,
                            )))
                            .selected_text(options.get(opt_idx).copied().unwrap_or("—"))
                            .show_ui(ui, |ui| {
                                for (oi, opt) in options.iter().enumerate() {
                                    if ui.selectable_value(&mut opt_idx, oi, *opt).changed() {
                                        inst.params.insert(param.key.into(), json!(opt_idx));
                                        changed = true;
                                    }
                                }
                            });
                        });
                    }
                    LooksParamKind::ColorRgb => {
                        let mut rgb = extract_rgb_f32(
                            inst.params.get(param.key),
                            [param.default_f64 as f32; 3],
                        );
                        ui.horizontal(|ui| {
                            ui.label(
                                egui::RichText::new(param.label)
                                    .font(theme.font(FONT_BODY))
                                    .color(TEXT_PRIMARY),
                            );
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    if ui.color_edit_button_rgb(&mut rgb).changed() {
                                        inst.params.insert(
                                            param.key.into(),
                                            json!([rgb[0] as f64, rgb[1] as f64, rgb[2] as f64]),
                                        );
                                        changed = true;
                                    }
                                },
                            );
                        });
                    }
                }
            }
        });

    if changed {
        state.dirty = true;
    }
    changed
}

// ---------------------------------------------------------------------------
// All 29 Color Looks Tool Definitions (Matching zc-render/src/effects/mod.rs)
// ---------------------------------------------------------------------------

const GAMMA_SPACES: &[&str] = &["Linear", "sRGB", "Log"];
const LUT_MODES: &[&str] = &[
    "Neutral",
    "Warm Film",
    "Cool Cinema",
    "High Contrast",
    "Faded Vintage",
];

/// Complete 29-tool Color Looks catalog across all 5 execution stages.
pub const LOOKS_TOOL_CATALOG: &[LooksToolSpec] = &[
    // ==================== SUBJECT (6) ====================
    LooksToolSpec {
        tool_id: "color-contrast",
        label: "Color Contrast",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "pivot",
                label: "Pivot",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Tint Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "hsl-colors",
        label: "HSL Colors",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "redSat",
                label: "Red Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "orangeSat",
                label: "Orange Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "yellowSat",
                label: "Yellow Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "greenSat",
                label: "Green Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "cyanSat",
                label: "Cyan Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueSat",
                label: "Blue Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "purpleSat",
                label: "Purple Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "magentaSat",
                label: "Magenta Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "redLight",
                label: "Red Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "orangeLight",
                label: "Orange Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "yellowLight",
                label: "Yellow Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "greenLight",
                label: "Green Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "cyanLight",
                label: "Cyan Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueLight",
                label: "Blue Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "purpleLight",
                label: "Purple Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "magentaLight",
                label: "Magenta Light",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "color-ranges",
        label: "Color Ranges",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "highlight",
                label: "Highlight Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            LooksParamSpec {
                key: "midtone",
                label: "Midtone Center",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "shadow",
                label: "Shadow Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "highlightColor",
                label: "Highlight Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "midtoneColor",
                label: "Midtone Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "shadowColor",
                label: "Shadow Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "crush",
        label: "Shadow Crush",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: LooksParamKind::Float {
                    min: 0.2,
                    max: 3.0,
                    step: 0.02,
                },
                default_f64: 1.15,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Crush Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "contrast",
        label: "Contrast",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "pivot",
                label: "Pivot",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "pop",
        label: "Micro Contrast",
        stage: LooksStage::Subject,
        params: &[
            LooksParamSpec {
                key: "pop",
                label: "Amount",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "size",
                label: "Radius",
                kind: LooksParamKind::Float {
                    min: 0.05,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "preserveDetail",
                label: "Preserve Detail",
                kind: LooksParamKind::Bool,
                default_f64: 1.0,
            },
        ],
    },
    // ==================== MATTE (2) ====================
    LooksToolSpec {
        tool_id: "vignette",
        label: "Vignette",
        stage: LooksStage::Matte,
        params: &[
            LooksParamSpec {
                key: "centerX",
                label: "Center X",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "centerY",
                label: "Center Y",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "radius",
                label: "Radius",
                kind: LooksParamKind::Float {
                    min: 0.05,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            LooksParamSpec {
                key: "aspect",
                label: "Aspect",
                kind: LooksParamKind::Float {
                    min: 0.25,
                    max: 3.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "spread",
                label: "Spread",
                kind: LooksParamKind::Float {
                    min: 0.01,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "falloff",
                label: "Falloff",
                kind: LooksParamKind::Float {
                    min: 0.2,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 1.5,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -3.0,
                    max: 1.0,
                    step: 0.02,
                },
                default_f64: -0.6,
            },
            LooksParamSpec {
                key: "color",
                label: "Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "edge-softness",
        label: "Edge Softness",
        stage: LooksStage::Matte,
        params: &[
            LooksParamSpec {
                key: "blurSize",
                label: "Blur Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "quality",
                label: "Quality",
                kind: LooksParamKind::Float {
                    min: 1.0,
                    max: 4.0,
                    step: 1.0,
                },
                default_f64: 2.0,
            },
            LooksParamSpec {
                key: "centerX",
                label: "Center X",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "centerY",
                label: "Center Y",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "radius",
                label: "Radius",
                kind: LooksParamKind::Float {
                    min: 0.1,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.65,
            },
            LooksParamSpec {
                key: "aspect",
                label: "Aspect",
                kind: LooksParamKind::Float {
                    min: 0.25,
                    max: 3.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "spread",
                label: "Spread",
                kind: LooksParamKind::Float {
                    min: 0.01,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
        ],
    },
    // ==================== LENS (8) ====================
    LooksToolSpec {
        tool_id: "diffusion",
        label: "Lens Diffusion",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "grade",
                label: "Grade",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "glow",
                label: "Glow",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "highlightsOnly",
                label: "Highlights Only",
                kind: LooksParamKind::Bool,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "highlightBias",
                label: "Highlight Bias",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "star-filter",
        label: "Star Filter",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "threshold",
                label: "Threshold",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "angle",
                label: "Angle",
                kind: LooksParamKind::Float {
                    min: -180.0,
                    max: 180.0,
                    step: 1.0,
                },
                default_f64: 15.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Tint",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "anamorphic-flare",
        label: "Anamorphic Flare",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Streak Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.45,
            },
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "threshold",
                label: "Threshold",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "reflection",
                label: "Reflection",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "reflBoost",
                label: "Refl Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "color",
                label: "Streak Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 0.85,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "haze-flare",
        label: "Haze Flare",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "spillage",
                label: "Spillage",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "softness",
                label: "Softness",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "reach",
                label: "Reach",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "reflExposure",
                label: "Refl Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "reflection",
                label: "Reflection",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "matteSize",
                label: "Matte Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "matteShade",
                label: "Matte Shade",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "color",
                label: "Haze Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lens-distortion",
        label: "Lens Distortion",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "distortion",
                label: "Distortion",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: -0.1,
            },
            LooksParamSpec {
                key: "flatness",
                label: "Flatness",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "chromatic-aberration",
        label: "Chromatic Aberration",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "redCyan",
                label: "Red / Cyan",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.2,
            },
            LooksParamSpec {
                key: "greenMagenta",
                label: "Green / Magenta",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueYellow",
                label: "Blue / Yellow",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: -0.15,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "shutter-streak",
        label: "Shutter Streak",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Length",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "falloff",
                label: "Falloff",
                kind: LooksParamKind::Float {
                    min: 0.1,
                    max: 4.0,
                    step: 0.05,
                },
                default_f64: 1.5,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "telecine-net",
        label: "Soft Diffusion Net",
        stage: LooksStage::Lens,
        params: &[
            LooksParamSpec {
                key: "size",
                label: "Mesh Size",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.5,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    // ==================== CAMERA (6) ====================
    LooksToolSpec {
        tool_id: "warm-cool",
        label: "Warm / Cool",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "warmCool",
                label: "Temperature",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.15,
            },
            LooksParamSpec {
                key: "tint",
                label: "Tint",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lift-gamma-gain",
        label: "Lift / Gamma / Gain",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "gammaSpace",
                label: "Space",
                kind: LooksParamKind::Choice(GAMMA_SPACES),
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "lift",
                label: "Lift (Shadows)",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "gamma",
                label: "Gamma (Mids)",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "gain",
                label: "Gain (Highs)",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "four-way",
        label: "4-Way Color",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "contrast",
                label: "Contrast",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "preview",
                label: "Preview Ranges",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "shadows",
                label: "Shadows",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "midtones",
                label: "Midtones",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "highlights",
                label: "Highlights",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "global",
                label: "Global",
                kind: LooksParamKind::ColorRgb,
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lightflex",
        label: "Shadow Lift Fog",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "boost",
                label: "Boost",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "color",
                label: "Veil Color",
                kind: LooksParamKind::ColorRgb,
                default_f64: 0.9,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "deflare",
        label: "Deflare",
        stage: LooksStage::Camera,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "auto-shoulder",
        label: "Highlight Shoulder",
        stage: LooksStage::Camera,
        params: &[LooksParamSpec {
            key: "strength",
            label: "Strength",
            kind: LooksParamKind::Float {
                min: 0.0,
                max: 1.0,
                step: 0.01,
            },
            default_f64: 0.5,
        }],
    },
    // ==================== POST (7) ====================
    LooksToolSpec {
        tool_id: "mojo",
        label: "Cinema Balance",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "mojo",
                label: "Balance",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "punch",
                label: "Punch",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.25,
            },
            LooksParamSpec {
                key: "bleach",
                label: "Bleach",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "fade",
                label: "Fade",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "blueSqueeze",
                label: "Blue Squeeze",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "skinSqueeze",
                label: "Warm Protect",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.4,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "coolWarm",
                label: "Cool / Warm",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "greenMagenta",
                label: "Green / Magenta",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "skinYP",
                label: "Skin Hue",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "color-reversal",
        label: "Color Reversal",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "three-strip",
        label: "3-Strip Film",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.6,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "ranged-saturation",
        label: "Ranged Saturation",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "satHighlight",
                label: "Highlight Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "satMidtone",
                label: "Midtone Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "satShadow",
                label: "Shadow Sat",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "exposure",
                label: "Exposure",
                kind: LooksParamKind::Float {
                    min: -2.0,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "thHighlight",
                label: "High Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.7,
            },
            LooksParamSpec {
                key: "thMidtone",
                label: "Mid Center",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
            LooksParamSpec {
                key: "thShadow",
                label: "Shadow Thresh",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.3,
            },
            LooksParamSpec {
                key: "showThreshold",
                label: "Show Threshold",
                kind: LooksParamKind::Bool,
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "balance",
                label: "Balance",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "curves",
        label: "Tone Curves",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "shadowLift",
                label: "Shadow Lift",
                kind: LooksParamKind::Float {
                    min: -0.5,
                    max: 0.5,
                    step: 0.01,
                },
                default_f64: 0.0,
            },
            LooksParamSpec {
                key: "midGamma",
                label: "Mid Gamma",
                kind: LooksParamKind::Float {
                    min: 0.3,
                    max: 2.5,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "highlightGain",
                label: "Highlight Gain",
                kind: LooksParamKind::Float {
                    min: 0.5,
                    max: 1.8,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "s-curve",
        label: "S-Curve Contrast",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "strength",
                label: "S-Curve Strength",
                kind: LooksParamKind::Float {
                    min: -1.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.35,
            },
            LooksParamSpec {
                key: "pivot",
                label: "Pivot",
                kind: LooksParamKind::Float {
                    min: 0.1,
                    max: 0.9,
                    step: 0.01,
                },
                default_f64: 0.5,
            },
        ],
    },
    LooksToolSpec {
        tool_id: "lut",
        label: "Look LUT",
        stage: LooksStage::Post,
        params: &[
            LooksParamSpec {
                key: "mode",
                label: "LUT Profile",
                kind: LooksParamKind::Choice(LUT_MODES),
                default_f64: 1.0,
            },
            LooksParamSpec {
                key: "strength",
                label: "Strength",
                kind: LooksParamKind::Float {
                    min: 0.0,
                    max: 1.0,
                    step: 0.01,
                },
                default_f64: 0.75,
            },
            LooksParamSpec {
                key: "gamma",
                label: "Gamma",
                kind: LooksParamKind::Float {
                    min: 0.5,
                    max: 2.0,
                    step: 0.02,
                },
                default_f64: 1.0,
            },
        ],
    },
];

#[must_use]
pub fn find_looks_tool(tool_id: &str) -> Option<&'static LooksToolSpec> {
    let lower = tool_id.trim().to_ascii_lowercase();
    LOOKS_TOOL_CATALOG
        .iter()
        .find(|s| s.tool_id.eq_ignore_ascii_case(&lower))
}

// ===========================================================================
// Unit Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_effect_graph_add_wire_cycle_check_and_compile() {
        let mut editor = NodeGraphEditorState::default();
        editor.open_for_target(
            NodeGraphTargetKind::NodeEffect,
            "fx1",
            "graph",
            "Node Effect",
            None,
        );
        assert_eq!(editor.nodes.len(), 2);
        assert!(matches!(editor.compile_status, NodeCompileStatus::Ok(_)));

        // Insert a brightness node between n1 (layer) and n2 (output)
        let b_id = editor.add_node("brightness", [200.0, 120.0]);
        editor
            .connect_ports("n1", "color", &b_id, "color")
            .expect("wire layer -> brightness");
        editor
            .connect_ports(&b_id, "color", "n2", "color")
            .expect("wire brightness -> output");
        assert!(matches!(editor.compile_status, NodeCompileStatus::Ok(_)));

        // Cycle rejection: output -> layer must fail
        assert!(editor.connect_ports("n2", "color", "n1", "color").is_err());
        // Self-loop rejection
        assert!(
            editor
                .connect_ports(&b_id, "color", &b_id, "color")
                .is_err()
        );

        // Templates compile cleanly
        for tpl in 0..3 {
            editor.apply_template(tpl);
            assert!(
                matches!(editor.compile_status, NodeCompileStatus::Ok(_)),
                "NodeEffect template {tpl} failed: {:?}",
                editor.compile_status
            );
        }
    }

    #[test]
    fn node_material_graph_templates_and_bake_compile() {
        let mut editor = NodeGraphEditorState::default();
        editor.open_for_target(
            NodeGraphTargetKind::NodeMaterial,
            "obj1",
            "material.graph",
            "Sphere Material",
            None,
        );
        assert!(matches!(editor.compile_status, NodeCompileStatus::Ok(_)));

        for tpl in 0..3 {
            editor.apply_template(tpl);
            assert!(
                matches!(editor.compile_status, NodeCompileStatus::Ok(_)),
                "NodeMaterial template {tpl} failed: {:?}",
                editor.compile_status
            );
        }

        // Output node cannot be deleted
        let out_id = editor
            .nodes
            .iter()
            .find(|n| n.kind == "output")
            .unwrap()
            .id
            .clone();
        assert!(!editor.remove_node(&out_id));
    }

    #[test]
    fn color_looks_has_all_29_tools_and_presets_roundtrip() {
        assert_eq!(LOOKS_TOOL_CATALOG.len(), 29);

        let mut looks = LooksEditorState::default();
        looks.open_for_effect("fx_looks", "Color Looks", None);
        for preset in LOOKS_PRESET_NAMES {
            looks.apply_preset(preset);
            assert!(
                !looks.chain.is_empty(),
                "preset {preset} must populate tools"
            );
            for inst in &looks.chain {
                assert!(
                    find_looks_tool(&inst.tool_id).is_some(),
                    "unknown tool in preset {preset}: {}",
                    inst.tool_id
                );
            }
            let encoded = looks.to_json();
            let mut reloaded = LooksEditorState::default();
            reloaded.open_for_effect("fx_looks", "Color Looks", Some(&encoded));
            assert_eq!(reloaded.chain.len(), looks.chain.len());
        }

        // Reorder, duplicate, remove
        let n0 = looks.chain.len();
        assert!(looks.duplicate_tool(0).is_some());
        assert_eq!(looks.chain.len(), n0 + 1);
        assert!(looks.move_tool(0, 1));
        assert!(looks.remove_tool(0));
        assert_eq!(looks.chain.len(), n0);
    }

    #[test]
    fn zero_third_party_brand_names_in_pro_editors() {
        let forbidden = [
            "trapcode",
            "magic bullet",
            "red giant",
            "maxon",
            "element 3d",
            "videocopilot",
        ];
        for tool in LOOKS_TOOL_CATALOG {
            let lower = tool.label.to_ascii_lowercase();
            for bad in forbidden {
                assert!(
                    !lower.contains(bad),
                    "forbidden brand `{bad}` in tool label `{}`",
                    tool.label
                );
            }
        }
        for preset in LOOKS_PRESET_NAMES {
            let lower = preset.to_ascii_lowercase();
            for bad in forbidden {
                assert!(
                    !lower.contains(bad),
                    "forbidden brand `{bad}` in preset `{preset}`"
                );
            }
        }
    }
}
