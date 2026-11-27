use super::shared_widgets::{
    compute_asset_usage_counts_cached, paint_crisp_border_2px, panel_top_header,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 1: Media (project)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MediaCardIcon {
    Scene3D,
    Adjustment,
    Text,
    PresetShape,
    VectorShape,
    ShapePlus,
    Composite,
    Camera,
    Precomp,
}

struct MediaPreset {
    kind: &'static str,
    title: &'static str,
    subtitle: &'static str,
    icon: MediaCardIcon,
    required_plugin: Option<&'static str>,
}

const MEDIA_PRESETS: &[MediaPreset] = &[
    MediaPreset {
        kind: "scene",
        title: "3D Scene",
        subtitle: "3D camera, lights & meshes",
        icon: MediaCardIcon::Scene3D,
        required_plugin: None,
    },
    MediaPreset {
        kind: "adjustment",
        title: "Adjustment",
        subtitle: "Post-processing effect layer",
        icon: MediaCardIcon::Adjustment,
        required_plugin: None,
    },
    MediaPreset {
        kind: "text_2d",
        title: "Text",
        subtitle: "2D typographic layer",
        icon: MediaCardIcon::Text,
        required_plugin: None,
    },
    MediaPreset {
        kind: "preset_shape",
        title: "Preset shape",
        subtitle: "Parametric 2D preset",
        icon: MediaCardIcon::PresetShape,
        required_plugin: None,
    },
    MediaPreset {
        kind: "shape",
        title: "Shape",
        subtitle: "Custom vector shape",
        icon: MediaCardIcon::VectorShape,
        required_plugin: None,
    },
    MediaPreset {
        kind: "composite",
        title: "Composite",
        subtitle: "Nested layer composite",
        icon: MediaCardIcon::Composite,
        required_plugin: None,
    },
    MediaPreset {
        kind: "zoidium:camera-plus/camera-layer",
        title: "Camera Layer",
        subtitle: "Camera+ shared 3D camera",
        icon: MediaCardIcon::Camera,
        required_plugin: Some("camera-plus"),
    },
    MediaPreset {
        kind: "zoidium:shape-plus/shape-layer",
        title: "Shape+",
        subtitle: "Animated 2D vector layer",
        icon: MediaCardIcon::ShapePlus,
        required_plugin: Some("shape-plus"),
    },
    MediaPreset {
        kind: "zoidium:precomp-plus/precomp",
        title: "Precomp",
        subtitle: "Nested sequence composition",
        icon: MediaCardIcon::Precomp,
        required_plugin: Some("precomp-plus"),
    },
];

fn paint_media_card_icon(
    painter: &Painter,
    rect: Rect,
    icon: MediaCardIcon,
    color: Color32,
    card_bg: Color32,
) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    match icon {
        MediaCardIcon::Scene3D => paint_cm3_icon(
            painter,
            Rect::from_center_size(rect.center(), Vec2::splat((side * 1.0625).round())),
            Cm3Icon::Objects,
            color,
        ),
        MediaCardIcon::Adjustment => paint_cm3_icon(
            painter,
            Rect::from_center_size(
                Pos2::new(rect.center().x, rect.center().y + 0.5),
                Vec2::splat((side * 1.104).round()),
            ),
            Cm3Icon::Fx,
            color,
        ),
        MediaCardIcon::Camera => paint_cm3_icon(painter, r, Cm3Icon::Camera, color),
        MediaCardIcon::Precomp => paint_cm3_icon(painter, r, Cm3Icon::Sequence, color),
        MediaCardIcon::Text => {
            // Bold geometric "T" glyph matching Zoidium Media card (39x39 inside 48x48)
            let bar = Rect::from_min_max(pt(0.10, 0.10), pt(0.90, 0.28));
            let stem = Rect::from_min_max(pt(0.40, 0.26), pt(0.60, 0.90));
            painter.rect_filled(bar, CornerRadius::ZERO, color);
            painter.rect_filled(stem, CornerRadius::ZERO, color);
        }
        MediaCardIcon::PresetShape => {
            // Geometric triangle overlapping a circle for parametric Preset shapes
            painter.circle_filled(pt(0.64, 0.64), side * 0.28, color);
            let cutout = vec![pt(0.36, 0.06), pt(0.75, 0.82), pt(-0.02, 0.82)];
            painter.add(Shape::convex_polygon(
                cutout,
                card_bg,
                Stroke::new(2.0, card_bg),
            ));
            let tri = vec![pt(0.36, 0.12), pt(0.70, 0.78), pt(0.02, 0.78)];
            painter.add(Shape::convex_polygon(tri, color, Stroke::NONE));
        }
        MediaCardIcon::VectorShape => {
            // Bezier path + anchor nodes + tangent handles for custom vector Shape
            let sw = (side * 0.065).clamp(1.8, 3.2);
            let mut curve_pts = Vec::with_capacity(17);
            let p0 = pt(0.14, 0.78);
            let p1 = pt(0.16, 0.18);
            let p2 = pt(0.84, 0.82);
            let p3 = pt(0.86, 0.22);
            for i in 0..=16 {
                let t = i as f32 / 16.0;
                let u = 1.0 - t;
                let x = u * u * u * p0.x
                    + 3.0 * u * u * t * p1.x
                    + 3.0 * u * t * t * p2.x
                    + t * t * t * p3.x;
                let y = u * u * u * p0.y
                    + 3.0 * u * u * t * p1.y
                    + 3.0 * u * t * t * p2.y
                    + t * t * t * p3.y;
                curve_pts.push(Pos2::new(x, y));
            }
            painter.add(Shape::line(curve_pts, Stroke::new(sw, color)));
            // Tangent handles
            let handle_stroke = Stroke::new((sw * 0.55).max(1.2), color32_with_alpha(color, 165));
            painter.line_segment([p0, pt(0.18, 0.34)], handle_stroke);
            painter.line_segment([p3, pt(0.82, 0.66)], handle_stroke);
            painter.circle_filled(pt(0.18, 0.34), side * 0.055, color);
            painter.circle_filled(pt(0.82, 0.66), side * 0.055, color);
            // Square bezier anchor nodes
            for anchor in [p0, pt(0.50, 0.50), p3] {
                let box_r = Rect::from_center_size(anchor, Vec2::splat(side * 0.16));
                painter.rect_filled(box_r, CornerRadius::ZERO, card_bg);
                painter.rect_stroke(
                    box_r,
                    CornerRadius::ZERO,
                    Stroke::new(sw * 0.8, color),
                    StrokeKind::Inside,
                );
            }
        }
        MediaCardIcon::ShapePlus => {
            // Stacked rounded vector shapes with a crisp '+' badge in top-right for Shape+
            let sw = (side * 0.06).clamp(1.6, 3.0);
            let back = Rect::from_min_max(pt(0.10, 0.24), pt(0.62, 0.76));
            painter.rect_stroke(
                back,
                CornerRadius::same(4),
                Stroke::new(sw, color32_with_alpha(color, 180)),
                StrokeKind::Inside,
            );
            let front = Rect::from_min_max(pt(0.24, 0.38), pt(0.76, 0.90));
            painter.rect_filled(front, CornerRadius::same(4), card_bg);
            painter.rect_filled(front.shrink(sw * 0.6), CornerRadius::same(3), color);
            // Crisp '+' badge at upper-right
            let plus_c = pt(0.78, 0.22);
            let arm = side * 0.14;
            let thick = (side * 0.07).clamp(2.0, 3.5);
            painter.rect_filled(
                Rect::from_center_size(plus_c, Vec2::new(arm * 2.0, thick)),
                CornerRadius::ZERO,
                color,
            );
            painter.rect_filled(
                Rect::from_center_size(plus_c, Vec2::new(thick, arm * 2.0)),
                CornerRadius::ZERO,
                color,
            );
        }
        MediaCardIcon::Composite => {
            // 3 stacked isometric diamonds (54x48 px at card scale) matching Zoidium Composite card
            let scale = side / 48.0;
            let cr = Rect::from_center_size(rect.center(), Vec2::new(54.0 * scale, 48.0 * scale));
            let cpt = |nx: f32, ny: f32| {
                Pos2::new(cr.left() + cr.width() * nx, cr.top() + cr.height() * ny)
            };
            let sw = (5.0 * scale).clamp(2.4, 5.5);
            let top = vec![
                cpt(0.50, 0.04),
                cpt(0.98, 0.30),
                cpt(0.50, 0.56),
                cpt(0.02, 0.30),
            ];
            painter.add(Shape::convex_polygon(top, color, Stroke::NONE));
            let mid = vec![cpt(0.04, 0.48), cpt(0.50, 0.74), cpt(0.96, 0.48)];
            painter.add(Shape::line(mid, Stroke::new(sw, color)));
            let bot = vec![cpt(0.04, 0.68), cpt(0.50, 0.94), cpt(0.96, 0.68)];
            painter.add(Shape::line(bot, Stroke::new(sw, color)));
        }
    }
}

fn paint_audio_media_icon(painter: &Painter, rect: Rect, color: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let sw = (side * 0.065).clamp(1.6, 3.0);

    // Speaker body + cone on the left
    let body = Rect::from_min_max(pt(0.10, 0.36), pt(0.28, 0.64));
    painter.rect_filled(body, CornerRadius::same(1), color);
    let cone = vec![
        pt(0.26, 0.36),
        pt(0.50, 0.16),
        pt(0.50, 0.84),
        pt(0.26, 0.64),
    ];
    painter.add(Shape::convex_polygon(cone, color, Stroke::NONE));

    // Crisp 3-bar soundwave / equalizer bars on the right
    let bars = [(0.62, 0.34, 0.66), (0.74, 0.22, 0.78), (0.86, 0.30, 0.70)];
    for (x, y0, y1) in bars {
        painter.line_segment([pt(x, y0), pt(x, y1)], Stroke::new(sw, color));
    }
}

/// Paints a crisp landscape photo-frame icon (outer border, sun, and twin mountain peaks)
/// for imported image assets (`.png`/`.jpg`/`.webp`/etc.) in Project Media.
fn paint_image_media_icon(painter: &Painter, rect: Rect, color: Color32, card_bg: Color32) {
    let side = rect.width().min(rect.height()).max(1.0);
    let r = Rect::from_center_size(rect.center(), Vec2::splat(side));
    let pt = |nx: f32, ny: f32| Pos2::new(r.left() + r.width() * nx, r.top() + r.height() * ny);
    let sw = (side * 0.06).clamp(1.6, 2.8);

    // Outer landscape picture frame
    let frame = Rect::from_min_max(pt(0.08, 0.16), pt(0.92, 0.84));
    painter.rect_filled(frame, CornerRadius::same(2), card_bg);
    painter.rect_stroke(
        frame,
        CornerRadius::same(2),
        Stroke::new(sw, color),
        StrokeKind::Inside,
    );

    // Sun disc in upper-left
    painter.circle_filled(pt(0.30, 0.36), (side * 0.085).max(1.8), color);

    // Back mountain peak (right-center)
    let back_peak = vec![pt(0.42, 0.78), pt(0.67, 0.40), pt(0.87, 0.78)];
    painter.add(Shape::convex_polygon(
        back_peak,
        color32_with_alpha(color, 185),
        Stroke::NONE,
    ));

    // Front mountain peak (left-center)
    let front_peak = vec![pt(0.13, 0.78), pt(0.39, 0.48), pt(0.62, 0.78)];
    painter.add(Shape::convex_polygon(front_peak, color, Stroke::NONE));
}

pub(super) fn layout_single_line_ellipsized(
    painter: &Painter,
    text: &str,
    font: egui::FontId,
    color: Color32,
    max_width: f32,
) -> std::sync::Arc<egui::Galley> {
    let full = painter.layout_no_wrap(text.to_owned(), font.clone(), color);
    if full.size().x <= max_width || max_width <= 8.0 {
        return full;
    }
    let chars: Vec<char> = text.chars().collect();
    let mut len = chars.len();
    let mut chosen = full;
    while len > 0 {
        len -= 1;
        let cand = format!("{}…", chars[..len].iter().collect::<String>().trim_end());
        let g = painter.layout_no_wrap(cand, font.clone(), color);
        if g.size().x <= max_width || len == 0 {
            chosen = g;
            break;
        }
    }
    chosen
}

pub(super) fn clip_kind_for_asset(kind: AssetKind, filename: &str) -> &'static str {
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match kind {
        AssetKind::Image => "image",
        AssetKind::Av => {
            if matches!(ext.as_str(), "mp4" | "webm" | "mkv" | "mov" | "avi") {
                "video"
            } else {
                "audio"
            }
        }
        _ => match ext.as_str() {
            "mp4" | "webm" | "mkv" | "mov" | "avi" => "video",
            "mp3" | "wav" | "ogg" | "flac" | "m4a" | "aac" | "opus" => "audio",
            _ => "image",
        },
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AssetCategory {
    Video,
    Audio,
    Image,
    Font,
    Model3D,
    Shader,
    File,
}

impl AssetCategory {
    fn classify(kind: AssetKind, filename: &str) -> Self {
        let ext = filename
            .rsplit('.')
            .next()
            .unwrap_or("")
            .to_ascii_lowercase();
        match kind {
            AssetKind::Image => Self::Image,
            AssetKind::Font => Self::Font,
            AssetKind::Geometry => Self::Model3D,
            AssetKind::Shader => Self::Shader,
            AssetKind::Av => {
                if matches!(ext.as_str(), "mp4" | "webm" | "mkv" | "mov" | "avi" | "m4v") {
                    Self::Video
                } else {
                    Self::Audio
                }
            }
            AssetKind::Misc => match ext.as_str() {
                "mp4" | "webm" | "mkv" | "mov" | "avi" | "m4v" => Self::Video,
                "mp3" | "wav" | "ogg" | "flac" | "m4a" | "aac" | "opus" => Self::Audio,
                "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff" | "svg" => {
                    Self::Image
                }
                "ttf" | "otf" | "ttc" | "woff" | "woff2" => Self::Font,
                "obj" | "glb" | "gltf" | "fbx" | "stl" => Self::Model3D,
                "glsl" | "frag" | "vert" | "wgsl" => Self::Shader,
                _ => Self::File,
            },
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Video => "video",
            Self::Audio => "audio",
            Self::Image => "image",
            Self::Font => "font",
            Self::Model3D => "3d model",
            Self::Shader => "shader",
            Self::File => "file",
        }
    }

    const fn is_timeline_media(self) -> bool {
        matches!(self, Self::Video | Self::Audio | Self::Image)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AssetKindFilter {
    #[default]
    All,
    Media,
    Audio,
    Image,
    Font,
    Model3D,
}

impl AssetKindFilter {
    const ALL: [Self; 6] = [
        Self::All,
        Self::Media,
        Self::Audio,
        Self::Image,
        Self::Font,
        Self::Model3D,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::All => "All",
            Self::Media => "Media",
            Self::Audio => "Audio",
            Self::Image => "Image",
            Self::Font => "Font",
            Self::Model3D => "3D Model",
        }
    }

    const fn matches(self, cat: AssetCategory) -> bool {
        match self {
            Self::All => true,
            Self::Media => matches!(
                cat,
                AssetCategory::Video | AssetCategory::Image | AssetCategory::Audio
            ),
            Self::Audio => matches!(cat, AssetCategory::Audio),
            Self::Image => matches!(cat, AssetCategory::Image),
            Self::Font => matches!(cat, AssetCategory::Font),
            Self::Model3D => matches!(cat, AssetCategory::Model3D),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum AssetStatusFilter {
    #[default]
    All,
    Used,
    Unused,
    Missing,
}

#[derive(Debug, Clone)]
struct AssetsManagerUiState {
    kind_filter: AssetKindFilter,
    status_filter: AssetStatusFilter,
    folder_filter: Option<String>,
    group_by_folder: bool,
    collapsed_folders: BTreeSet<String>,
    selected_ids: BTreeSet<String>,
    new_folder_open: bool,
    new_folder_buf: String,
}

impl Default for AssetsManagerUiState {
    fn default() -> Self {
        Self {
            kind_filter: AssetKindFilter::All,
            status_filter: AssetStatusFilter::All,
            folder_filter: None,
            group_by_folder: true,
            collapsed_folders: BTreeSet::new(),
            selected_ids: BTreeSet::new(),
            new_folder_open: false,
            new_folder_buf: String::new(),
        }
    }
}

/// Key in `Project::preserved` storing persistent explicit asset folders (including empty ones).
pub(super) const ASSET_FOLDERS_KEY: &str = "assetFolders";

/// Full 16-color CM3 / Zoidium asset label palette (`0 = None`, `1..=15 = colored labels`).
pub(super) const ASSET_LABEL_PALETTE: [(&str, Color32); 16] = [
    ("None", Color32::from_rgb(0x55, 0x55, 0x55)),
    ("Red", Color32::from_rgb(0xe0, 0x52, 0x52)),
    ("Coral", Color32::from_rgb(0xeb, 0x73, 0x52)),
    ("Orange", Color32::from_rgb(0xe8, 0x94, 0x38)),
    ("Amber", Color32::from_rgb(0xeb, 0xb8, 0x3b)),
    ("Yellow", Color32::from_rgb(0xdc, 0xd4, 0x45)),
    ("Lime", Color32::from_rgb(0x98, 0xc9, 0x42)),
    ("Green", Color32::from_rgb(0x56, 0xc2, 0x71)),
    ("Emerald", Color32::from_rgb(0x3a, 0xbd, 0x99)),
    ("Teal", Color32::from_rgb(0x38, 0xb2, 0xc4)),
    ("Cyan", Color32::from_rgb(0x4a, 0xbb, 0xeb)),
    ("Blue", Color32::from_rgb(0x4f, 0x9d, 0xf5)),
    ("Indigo", Color32::from_rgb(0x6e, 0x7c, 0xf2)),
    ("Purple", Color32::from_rgb(0x9e, 0x6b, 0xf2)),
    ("Violet", Color32::from_rgb(0xc8, 0x64, 0xe8)),
    ("Pink", Color32::from_rgb(0xe8, 0x61, 0xa8)),
];

pub(super) fn asset_label_entry(label_idx: u8) -> (&'static str, Color32) {
    ASSET_LABEL_PALETTE[(label_idx as usize) % ASSET_LABEL_PALETTE.len()]
}

/// Collects all sorted, deduplicated asset folder names from both `project.preserved["assetFolders"]`
/// (preserving empty folders across frames and save/reload) and individual `AssetRecord::preserved["folder"]`.
pub(super) fn collect_project_asset_folders(project: &zc_project::Project) -> Vec<String> {
    let mut folders = BTreeSet::new();
    if let Some(arr) = project
        .preserved
        .get(ASSET_FOLDERS_KEY)
        .and_then(Value::as_array)
    {
        for val in arr {
            if let Some(name) = val.as_str().map(str::trim).filter(|s| !s.is_empty()) {
                folders.insert(name.to_owned());
            }
        }
    }
    for asset in project.assets.values() {
        if let Some(name) = asset
            .preserved
            .get("folder")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            folders.insert(name.to_owned());
        }
    }
    folders.into_iter().collect()
}

fn write_project_asset_folders(project: &mut zc_project::Project, folders: &BTreeSet<String>) {
    if folders.is_empty() {
        project.preserved.remove(ASSET_FOLDERS_KEY);
    } else {
        let arr: Vec<Value> = folders.iter().cloned().map(Value::String).collect();
        project
            .preserved
            .insert(ASSET_FOLDERS_KEY.to_owned(), Value::Array(arr));
    }
}

/// Creates a persistent folder in `project.preserved["assetFolders"]` and optionally moves
/// `move_asset_ids` into it as one undoable session transaction.
pub(super) fn create_asset_folder_undoable(
    session: &mut Session,
    folder_name: &str,
    move_asset_ids: &[String],
) -> usize {
    let trimmed = folder_name.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let mut folders: BTreeSet<String> = collect_project_asset_folders(&session.project)
        .into_iter()
        .collect();
    let folder_is_new = !folders.contains(trimmed);
    let any_asset_changes = move_asset_ids.iter().any(|aid| {
        session.project.assets.get(aid).is_some_and(|rec| {
            rec.preserved
                .get("folder")
                .and_then(Value::as_str)
                .map(str::trim)
                != Some(trimmed)
        })
    });
    if !folder_is_new && !any_asset_changes {
        return 0;
    }

    let label = if any_asset_changes {
        format!("Move assets to folder `{trimmed}`")
    } else {
        format!("Create asset folder `{trimmed}`")
    };
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction(label).is_ok()
    } else {
        false
    };
    folders.insert(trimmed.to_owned());
    write_project_asset_folders(&mut session.project, &folders);
    let mut moved = 0usize;
    for aid in move_asset_ids {
        if let Some(rec) = session.project.assets.get_mut(aid) {
            rec.preserved.insert("folder".to_owned(), json!(trimmed));
            moved += 1;
        }
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    } else if session.history.is_in_transaction() {
        session.history.record_transaction_edit();
    }
    moved
}

/// Deletes a folder from `project.preserved["assetFolders"]` and moves any assets in that
/// folder back to Root (unfiled), recorded as a single undoable transaction.
pub(super) fn delete_asset_folder_undoable(session: &mut Session, folder_name: &str) -> usize {
    let trimmed = folder_name.trim();
    if trimmed.is_empty() {
        return 0;
    }
    let mut folders: BTreeSet<String> = collect_project_asset_folders(&session.project)
        .into_iter()
        .collect();
    let had_folder = folders.remove(trimmed);
    let affected_ids: Vec<String> = session
        .project
        .assets
        .iter()
        .filter(|(_, rec)| {
            rec.preserved
                .get("folder")
                .and_then(Value::as_str)
                .map(str::trim)
                == Some(trimmed)
        })
        .map(|(id, _)| id.clone())
        .collect();
    if !had_folder && affected_ids.is_empty() {
        return 0;
    }

    let started_tx = if !session.history.is_in_transaction() {
        session
            .begin_transaction(format!("Delete asset folder `{trimmed}`"))
            .is_ok()
    } else {
        false
    };
    write_project_asset_folders(&mut session.project, &folders);
    for aid in &affected_ids {
        if let Some(rec) = session.project.assets.get_mut(aid) {
            rec.preserved.remove("folder");
        }
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    } else if session.history.is_in_transaction() {
        session.history.record_transaction_edit();
    }
    affected_ids.len()
}

/// Moves one or more assets into `folder` (or Root/Unfiled when `folder` is `None`) in a
/// single undoable transaction.
pub(super) fn move_assets_to_folder_undoable(
    session: &mut Session,
    asset_ids: &[String],
    folder: Option<&str>,
) -> usize {
    let target = folder.map(str::trim).filter(|s| !s.is_empty());
    let changing: Vec<String> = asset_ids
        .iter()
        .filter(|aid| {
            session.project.assets.get(*aid).is_some_and(|rec| {
                rec.preserved
                    .get("folder")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    != target
            })
        })
        .cloned()
        .collect();
    if changing.is_empty() {
        return 0;
    }
    let label = match target {
        Some(name) => format!("Move assets to folder `{name}`"),
        None => "Move assets to Root".to_owned(),
    };
    let started_tx = if !session.history.is_in_transaction() {
        session.begin_transaction(label).is_ok()
    } else {
        false
    };
    if let Some(name) = target {
        let mut folders: BTreeSet<String> = collect_project_asset_folders(&session.project)
            .into_iter()
            .collect();
        folders.insert(name.to_owned());
        write_project_asset_folders(&mut session.project, &folders);
    }
    for aid in &changing {
        if let Some(rec) = session.project.assets.get_mut(aid) {
            match target {
                Some(name) => {
                    rec.preserved.insert("folder".to_owned(), json!(name));
                }
                None => {
                    rec.preserved.remove("folder");
                }
            }
        }
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    } else if session.history.is_in_transaction() {
        session.history.record_transaction_edit();
    }
    changing.len()
}

/// Sets the 16-color label index on one or more assets as a single undoable transaction.
pub(super) fn set_assets_label_color_undoable(
    session: &mut Session,
    asset_ids: &[String],
    label_idx: u8,
) -> usize {
    let normalized = label_idx % (ASSET_LABEL_PALETTE.len() as u8);
    let changing: Vec<String> = asset_ids
        .iter()
        .filter(|aid| {
            session.project.assets.get(*aid).is_some_and(|rec| {
                let cur = rec
                    .preserved
                    .get("labelColor")
                    .and_then(Value::as_u64)
                    .unwrap_or(0) as u8;
                cur % (ASSET_LABEL_PALETTE.len() as u8) != normalized
            })
        })
        .cloned()
        .collect();
    if changing.is_empty() {
        return 0;
    }
    let (color_name, _) = asset_label_entry(normalized);
    let started_tx = if !session.history.is_in_transaction() {
        session
            .begin_transaction(format!("Set asset label: {color_name}"))
            .is_ok()
    } else {
        false
    };
    for aid in &changing {
        if let Some(rec) = session.project.assets.get_mut(aid) {
            if normalized == 0 {
                rec.preserved.remove("labelColor");
            } else {
                rec.preserved
                    .insert("labelColor".to_owned(), json!(normalized));
            }
        }
    }
    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    } else if session.history.is_in_transaction() {
        session.history.record_transaction_edit();
    }
    changing.len()
}

// ---------------------------------------------------------------------------
// Safe Asset ID Rewriting & Undoable Duplicate Consolidation
// ---------------------------------------------------------------------------

fn rewrite_string_asset_id(slot: &mut String, old_id: &str, new_id: &str) -> usize {
    if slot == old_id {
        *slot = new_id.to_owned();
        1
    } else if old_id.len() >= 8 && slot.contains(old_id) && !new_id.contains(old_id) {
        let replaced = slot.replace(old_id, new_id);
        if replaced != *slot {
            *slot = replaced;
            1
        } else {
            0
        }
    } else {
        0
    }
}

fn rewrite_opt_string_asset_id(slot: &mut Option<String>, old_id: &str, new_id: &str) -> usize {
    match slot.as_mut() {
        Some(s) => rewrite_string_asset_id(s, old_id, new_id),
        None => 0,
    }
}

fn rewrite_json_asset_id(val: &mut Value, old_id: &str, new_id: &str) -> usize {
    match val {
        Value::String(s) => rewrite_string_asset_id(s, old_id, new_id),
        Value::Array(arr) => arr
            .iter_mut()
            .map(|item| rewrite_json_asset_id(item, old_id, new_id))
            .sum(),
        Value::Object(map) => map
            .values_mut()
            .map(|v| rewrite_json_asset_id(v, old_id, new_id))
            .sum(),
        _ => 0,
    }
}

fn rewrite_preserved_map_asset_id(
    map: &mut BTreeMap<String, Value>,
    old_id: &str,
    new_id: &str,
) -> usize {
    map.values_mut()
        .map(|v| rewrite_json_asset_id(v, old_id, new_id))
        .sum()
}

fn rewrite_property_value_asset_id(val: &mut PropertyValue, old_id: &str, new_id: &str) -> usize {
    match val {
        PropertyValue::Asset(slot) => rewrite_opt_string_asset_id(slot, old_id, new_id),
        PropertyValue::Text(s) | PropertyValue::List(s) | PropertyValue::Shader(s) => {
            rewrite_string_asset_id(s, old_id, new_id)
        }
        PropertyValue::Json(v) | PropertyValue::Curve(v) => {
            rewrite_json_asset_id(v, old_id, new_id)
        }
        _ => 0,
    }
}

fn rewrite_property_track_asset_id(track: &mut PropertyTrack, old_id: &str, new_id: &str) -> usize {
    let mut count = rewrite_property_value_asset_id(&mut track.value, old_id, new_id);
    if let Some(expr) = track.expression.as_mut() {
        count += rewrite_string_asset_id(expr, old_id, new_id);
    }
    for kf in &mut track.keyframes {
        count += rewrite_property_value_asset_id(&mut kf.value, old_id, new_id);
    }
    for ch in &mut track.channels {
        count += rewrite_property_track_asset_id(ch, old_id, new_id);
    }
    count
}

fn rewrite_custom_property_asset_id(cp: &mut CustomProperty, old_id: &str, new_id: &str) -> usize {
    rewrite_property_track_asset_id(&mut cp.track, old_id, new_id)
        + rewrite_preserved_map_asset_id(&mut cp.metadata, old_id, new_id)
}

fn rewrite_material_asset_id(mat: &mut MaterialSpec, old_id: &str, new_id: &str) -> usize {
    match mat {
        MaterialSpec::SingleColor { color, .. } => {
            rewrite_property_track_asset_id(color, old_id, new_id)
        }
        MaterialSpec::Texture(t) => rewrite_opt_string_asset_id(&mut t.texture, old_id, new_id),
        MaterialSpec::Video(v) => rewrite_opt_string_asset_id(&mut v.video, old_id, new_id),
        MaterialSpec::CustomPbr(p) => {
            rewrite_opt_string_asset_id(&mut p.texture, old_id, new_id)
                + rewrite_opt_string_asset_id(&mut p.normal_map, old_id, new_id)
        }
        MaterialSpec::CustomShader {
            custom_properties, ..
        } => custom_properties
            .iter_mut()
            .map(|cp| rewrite_custom_property_asset_id(cp, old_id, new_id))
            .sum(),
        MaterialSpec::PbrPlus(p) => {
            let mut count = 0usize;
            for slot in [
                &mut p.base_color_map,
                &mut p.emissive_map,
                &mut p.roughness_map,
                &mut p.metalness_map,
                &mut p.normal_map,
                &mut p.alpha_map,
                &mut p.ao_texture,
            ] {
                count += rewrite_opt_string_asset_id(slot, old_id, new_id);
            }
            count + rewrite_preserved_map_asset_id(&mut p.preserved, old_id, new_id)
        }
        MaterialSpec::Matcap(m) => rewrite_opt_string_asset_id(&mut m.texture, old_id, new_id),
        MaterialSpec::ImagePlus(i) => rewrite_opt_string_asset_id(&mut i.texture, old_id, new_id),
        MaterialSpec::NodeMaterial { graph, .. } => rewrite_json_asset_id(graph, old_id, new_id),
        MaterialSpec::Unknown { payload, .. } => rewrite_json_asset_id(payload, old_id, new_id),
    }
}

fn rewrite_objects_3d_asset_id(objects: &mut [Object3D], old_id: &str, new_id: &str) -> usize {
    let mut count = 0usize;
    for obj in objects {
        if let Some(mat) = obj.material.as_mut() {
            count += rewrite_material_asset_id(mat, old_id, new_id);
        }
        for mat in obj.material_slots.values_mut() {
            count += rewrite_material_asset_id(mat, old_id, new_id);
        }
        count += rewrite_preserved_map_asset_id(&mut obj.preserved, old_id, new_id);
        match &mut obj.kind {
            Object3DKind::Group {
                custom_properties,
                children,
                ..
            } => {
                for cp in custom_properties {
                    count += rewrite_custom_property_asset_id(cp, old_id, new_id);
                }
                count += rewrite_objects_3d_asset_id(children, old_id, new_id);
            }
            Object3DKind::Text3D(t) => {
                count += rewrite_opt_string_asset_id(&mut t.font, old_id, new_id);
            }
            Object3DKind::Shape3D(zc_project::Shape3DKind::CustomGeometry { asset_id }) => {
                count += rewrite_opt_string_asset_id(asset_id, old_id, new_id);
            }
            Object3DKind::Model(m) => {
                count += rewrite_opt_string_asset_id(&mut m.asset_id, old_id, new_id);
                for tr in m.model_properties.values_mut() {
                    count += rewrite_property_track_asset_id(tr, old_id, new_id);
                }
            }
            Object3DKind::Particles(p) => {
                count += rewrite_opt_string_asset_id(&mut p.texture, old_id, new_id);
            }
            Object3DKind::Dummy { payload, .. } => {
                count += rewrite_json_asset_id(payload, old_id, new_id);
            }
            Object3DKind::Shape3D(_) | Object3DKind::Light(_) | Object3DKind::Camera(_) => {}
        }
    }
    count
}

fn rewrite_shape2d_nodes_asset_id(
    nodes: &mut [zc_project::Shape2DNode],
    old_id: &str,
    new_id: &str,
) -> usize {
    let mut count = 0usize;
    for node in nodes {
        count += rewrite_preserved_map_asset_id(&mut node.preserved, old_id, new_id);
        for op in &mut node.operations {
            match op {
                zc_project::DrawOp::Fill { color } => {
                    count += rewrite_property_track_asset_id(color, old_id, new_id);
                }
                zc_project::DrawOp::Stroke {
                    color,
                    size,
                    opacity,
                } => {
                    count += rewrite_property_track_asset_id(color, old_id, new_id);
                    count += rewrite_property_track_asset_id(size, old_id, new_id);
                    count += rewrite_property_track_asset_id(opacity, old_id, new_id);
                }
            }
        }
        match &mut node.kind {
            zc_project::Shape2DKind::Group { children } => {
                count += rewrite_shape2d_nodes_asset_id(children, old_id, new_id);
            }
            zc_project::Shape2DKind::Path { path } => {
                count += rewrite_json_asset_id(path, old_id, new_id);
            }
        }
    }
    count
}

fn rewrite_effects_asset_id(effects: &mut [EffectInstance], old_id: &str, new_id: &str) -> usize {
    let mut count = 0usize;
    for fx in effects {
        for tr in fx.params.values_mut() {
            count += rewrite_property_track_asset_id(tr, old_id, new_id);
        }
        count += rewrite_preserved_map_asset_id(&mut fx.preserved, old_id, new_id);
        match &mut fx.kind {
            EffectKind::CustomShader {
                custom_properties, ..
            } => {
                for cp in custom_properties {
                    count += rewrite_custom_property_asset_id(cp, old_id, new_id);
                }
            }
            EffectKind::EffectGroup {
                custom_properties,
                children,
            } => {
                for cp in custom_properties {
                    count += rewrite_custom_property_asset_id(cp, old_id, new_id);
                }
                count += rewrite_effects_asset_id(children, old_id, new_id);
            }
            EffectKind::Standard(_) | EffectKind::Plugin { .. } => {}
        }
    }
    count
}

fn rewrite_layer_asset_id(layer: &mut zc_project::Layer, old_id: &str, new_id: &str) -> usize {
    let mut count = rewrite_effects_asset_id(&mut layer.effects, old_id, new_id);
    count += rewrite_preserved_map_asset_id(&mut layer.preserved, old_id, new_id);
    match &mut layer.kind {
        LayerKind::Video { media } => {
            count += rewrite_opt_string_asset_id(media, old_id, new_id);
        }
        LayerKind::Image { image } => {
            count += rewrite_opt_string_asset_id(image, old_id, new_id);
        }
        LayerKind::Text2D { font, shapes, .. } => {
            count += rewrite_opt_string_asset_id(font, old_id, new_id);
            count += rewrite_shape2d_nodes_asset_id(shapes, old_id, new_id);
        }
        LayerKind::Shape2D { shapes } | LayerKind::PresetShape2D { shapes, .. } => {
            count += rewrite_shape2d_nodes_asset_id(shapes, old_id, new_id);
        }
        LayerKind::Scene3D { objects, .. } => {
            count += rewrite_objects_3d_asset_id(objects, old_id, new_id);
        }
        LayerKind::Composite { layers } => {
            for child in layers {
                count += rewrite_layer_asset_id(child, old_id, new_id);
            }
        }
        LayerKind::Plugin { payload, .. } => {
            count += rewrite_json_asset_id(payload, old_id, new_id);
        }
        LayerKind::Adjustment
        | LayerKind::Generator
        | LayerKind::CameraLayer { .. }
        | LayerKind::PrecompRef { .. } => {}
    }
    count
}

fn rewrite_sequence_asset_id(seq: &mut zc_project::Sequence, old_id: &str, new_id: &str) -> usize {
    let mut count = 0usize;
    for vt in &mut seq.video_tracks {
        count += rewrite_preserved_map_asset_id(&mut vt.preserved, old_id, new_id);
        for vc in &mut vt.clips {
            count += rewrite_opt_string_asset_id(&mut vc.media, old_id, new_id);
            count += rewrite_property_track_asset_id(&mut vc.time_curve, old_id, new_id);
            count += rewrite_layer_asset_id(&mut vc.layer, old_id, new_id);
            count += rewrite_preserved_map_asset_id(&mut vc.preserved, old_id, new_id);
        }
    }
    for at in &mut seq.audio_tracks {
        count += rewrite_preserved_map_asset_id(&mut at.preserved, old_id, new_id);
        for ac in &mut at.clips {
            count += rewrite_opt_string_asset_id(&mut ac.media, old_id, new_id);
            count += rewrite_property_track_asset_id(&mut ac.time_curve, old_id, new_id);
            count += rewrite_preserved_map_asset_id(&mut ac.audio.preserved, old_id, new_id);
            count += rewrite_preserved_map_asset_id(&mut ac.preserved, old_id, new_id);
        }
    }
    for ctl in &mut seq.controls {
        count += rewrite_property_track_asset_id(&mut ctl.track, old_id, new_id);
        count += rewrite_preserved_map_asset_id(&mut ctl.preserved, old_id, new_id);
    }
    for objs in seq.camera_layer_objects.values_mut() {
        count += rewrite_objects_3d_asset_id(objs, old_id, new_id);
    }
    for masks in seq.layer_masks.values_mut() {
        for mask in masks {
            count += rewrite_property_track_asset_id(&mut mask.path, old_id, new_id);
            count += rewrite_preserved_map_asset_id(&mut mask.preserved, old_id, new_id);
        }
    }
    count += rewrite_preserved_map_asset_id(&mut seq.preserved, old_id, new_id);
    count
}

/// Rewrites every reference to `old_id` across the entire [`zc_project::Project`] to `new_id`,
/// including `project.media`, `project.sequence`, all nested `project.compositions`, audio/video
/// clips, 2D layers & plugin payloads, effects/custom shaders/groups, 3D objects, materials, and
/// node material graphs. Returns the total number of rewritten references.
pub(super) fn rewrite_asset_id_across_project(
    project: &mut zc_project::Project,
    old_id: &str,
    new_id: &str,
) -> usize {
    if old_id.is_empty() || old_id == new_id {
        return 0;
    }
    let mut count = 0usize;
    for media in &mut project.media {
        count += rewrite_opt_string_asset_id(&mut media.asset_id, old_id, new_id);
        count += rewrite_preserved_map_asset_id(&mut media.preserved, old_id, new_id);
    }
    for (k, v) in &mut project.preserved {
        if k != ASSET_FOLDERS_KEY {
            count += rewrite_json_asset_id(v, old_id, new_id);
        }
    }
    count += rewrite_sequence_asset_id(&mut project.sequence, old_id, new_id);
    for comp in project.compositions.values_mut() {
        count += rewrite_sequence_asset_id(comp, old_id, new_id);
    }
    count
}

/// Rebuilds `session.document` directly from `session.project` (preserving the active
/// composition, undo/redo history, revision, selection, and embedded asset buffers with
/// zero media-byte copies).
fn resync_session_document_from_project(session: &mut Session) {
    if session.history.is_in_transaction() {
        return;
    }
    let active_comp = session.document().active_composition_id;
    let saved_selection = session.selection.clone();
    let saved_reqs = session.project.plugin_requirements.clone();

    let mut taken_bytes: Vec<(String, Vec<u8>)> = Vec::new();
    for (id, rec) in &mut session.project.assets {
        if let Some(bytes) = rec.bytes.take() {
            taken_bytes.push((id.clone(), bytes));
        }
    }

    let mut temp = Session::from_project(session.project.clone(), session.plugins.clone());
    if active_comp.is_some() {
        let _ = temp.execute(Command::SetActiveComposition {
            comp_id: active_comp,
        });
    }
    let mut sync_snap = temp.snapshot();
    sync_snap.project.plugin_requirements = saved_reqs;
    sync_snap.selection = saved_selection;

    if session
        .history
        .begin_transaction_with_bytes("__resync_document", sync_snap, 0)
        .is_ok()
    {
        let _ = session.cancel_transaction();
    }

    for (id, bytes) in taken_bytes {
        if let Some(rec) = session.project.assets.get_mut(&id) {
            rec.bytes = Some(bytes);
        }
    }
}

/// Safely deduplicates assets sharing identical SHA-256 content across the project, rewriting
/// every reference in clips, layers, 3D objects, materials, node graphs, and effects inside a
/// single undoable transaction, then re-syncing `session.document`.
pub(super) fn consolidate_duplicate_assets(session: &mut Session) -> usize {
    session.sync_to_project();

    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (id, rec) in &session.project.assets {
        let digest = if !rec.sha256.is_empty() {
            Some(rec.sha256.clone())
        } else {
            rec.bytes.as_deref().map(zc_project::sha256_hex)
        };
        if let Some(d) = digest {
            groups.entry(d).or_default().push(id.clone());
        }
    }

    let mut remap: BTreeMap<String, String> = BTreeMap::new();
    for ids in groups.into_values() {
        if ids.len() < 2 {
            continue;
        }
        // Prefer a non-missing asset with in-memory bytes as canonical survivor; break ties deterministically by ID.
        let canonical = ids
            .iter()
            .min_by_key(|id| {
                let rec = session.project.assets.get(*id);
                let missing_rank = u8::from(rec.is_none_or(|r| r.missing));
                let no_bytes_rank = u8::from(rec.is_none_or(|r| r.bytes.is_none()));
                (missing_rank, no_bytes_rank, (*id).clone())
            })
            .cloned()
            .expect("group has at least 2 ids");
        for id in ids {
            if id != canonical {
                remap.insert(id, canonical.clone());
            }
        }
    }

    if remap.is_empty() {
        return 0;
    }

    let merged = remap.len();
    let started_tx = if !session.history.is_in_transaction() {
        session
            .begin_transaction("Consolidate duplicate assets")
            .is_ok()
    } else {
        false
    };

    for (dup_id, keep_id) in &remap {
        let _ = rewrite_asset_id_across_project(&mut session.project, dup_id, keep_id);
        if let Some(dup_rec) = session.project.assets.remove(dup_id)
            && let Some(keep_rec) = session.project.assets.get_mut(keep_id)
        {
            if keep_rec.bytes.is_none() && dup_rec.bytes.is_some() {
                keep_rec.bytes = dup_rec.bytes;
                keep_rec.missing = dup_rec.missing;
                keep_rec.external = dup_rec.external;
                if keep_rec.size == 0 {
                    keep_rec.size = dup_rec.size;
                }
            }
            if keep_rec.url.is_none() && dup_rec.url.is_some() {
                keep_rec.url = dup_rec.url;
            }
            if !keep_rec.preserved.contains_key("folder")
                && let Some(folder) = dup_rec.preserved.get("folder")
            {
                keep_rec
                    .preserved
                    .insert("folder".to_owned(), folder.clone());
            }
            if !keep_rec.preserved.contains_key("labelColor")
                && let Some(label) = dup_rec.preserved.get("labelColor")
            {
                keep_rec
                    .preserved
                    .insert("labelColor".to_owned(), label.clone());
            }
        }
    }

    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    } else if session.history.is_in_transaction() {
        session.history.record_transaction_edit();
    }

    resync_session_document_from_project(session);
    merged
}

fn format_asset_size(bytes: u64) -> String {
    if bytes < 1024 {
        return format!("{bytes} B");
    }
    let kib = bytes as f64 / 1024.0;
    if kib < 1024.0 {
        return if kib >= 100.0 {
            format!("{:.0} KiB", kib)
        } else {
            format!("{:.1} KiB", kib)
        };
    }
    let mib = kib / 1024.0;
    if mib < 1024.0 {
        return if mib >= 100.0 {
            format!("{:.0} MiB", mib)
        } else {
            format!("{:.1} MiB", mib)
        };
    }
    format!("{:.2} GiB", mib / 1024.0)
}

fn trigger_replace_asset(
    session: &mut Session,
    aid: &str,
    fname: &str,
    actions: &mut Vec<PanelAction>,
) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (session, fname);
        actions.push(PanelAction::RelinkWebAsset(aid.to_owned()));
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = rfd::FileDialog::new()
        .set_title(format!("Replace footage: {fname}"))
        .pick_file()
        && let Ok(bytes) = std::fs::read(&path)
    {
        let new_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(fname)
            .to_owned();
        let _ = session.execute(Command::RelinkAsset {
            asset_id: aid.to_owned(),
            filename: new_name.clone(),
            bytes,
        });
        actions.push(PanelAction::StatusMessage(format!(
            "Replaced footage with `{new_name}`"
        )));
    }
}

/// Reloads an asset's bytes from its linked `rec.url` disk path via an undoable `RelinkAsset`
/// command, refreshing SHA-256, size, and missing status cleanly.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn reload_asset_from_disk(
    session: &mut Session,
    asset_id: &str,
) -> Result<String, String> {
    let (url, filename) = {
        let rec = session
            .project
            .assets
            .get(asset_id)
            .ok_or_else(|| format!("Asset `{asset_id}` not found"))?;
        let url = rec
            .url
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| {
                format!(
                    "`{}` has no linked disk path — use Replace Asset… instead",
                    rec.filename
                )
            })?
            .to_owned();
        (url, rec.filename.clone())
    };
    let bytes = std::fs::read(&url)
        .map_err(|err| format!("Failed to reload `{filename}` from `{url}`: {err}"))?;
    let size = bytes.len() as u64;
    session
        .execute(Command::RelinkAsset {
            asset_id: asset_id.to_owned(),
            filename: filename.clone(),
            bytes,
        })
        .map_err(|err| format!("Failed to update `{filename}`: {err}"))?;
    Ok(format!(
        "Reloaded `{filename}` from disk ({})",
        format_asset_size(size)
    ))
}

fn resolve_asset_export_bytes(rec: &zc_project::AssetRecord) -> Result<Vec<u8>, String> {
    if let Some(bytes) = &rec.bytes {
        return Ok(bytes.clone());
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(url) = rec.url.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        return std::fs::read(url)
            .map_err(|err| format!("Failed to read `{}` from `{url}`: {err}", rec.filename));
    }
    Err(format!(
        "Asset `{}` has no exportable binary data (missing or external)",
        rec.filename
    ))
}

#[cfg(target_arch = "wasm32")]
fn mime_for_asset_filename(filename: &str) -> &'static str {
    let ext = filename
        .rsplit('.')
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "mp4" | "m4v" => "video/mp4",
        "webm" => "video/webm",
        "mov" => "video/quicktime",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        "ttf" => "font/ttf",
        "otf" => "font/otf",
        "woff" => "font/woff",
        "woff2" => "font/woff2",
        "json" => "application/json",
        "glsl" | "frag" | "vert" | "wgsl" | "txt" | "cube" => "text/plain",
        _ => "application/octet-stream",
    }
}

#[cfg(target_arch = "wasm32")]
fn download_web_asset_bytes(filename: &str, bytes: &[u8], mime: &str) -> Result<(), String> {
    use wasm_bindgen::{JsCast, JsValue};

    if crate::web::call(
        "__zcFiles",
        "downloadBlob",
        &[
            filename.into(),
            js_sys::Uint8Array::from(bytes).into(),
            mime.into(),
        ],
    )
    .is_ok()
    {
        return Ok(());
    }

    let window = web_sys::window().ok_or_else(|| "Browser window is unavailable".to_owned())?;
    let document = window
        .document()
        .ok_or_else(|| "Browser document is unavailable".to_owned())?;
    let parts = js_sys::Array::new();
    parts.push(&js_sys::Uint8Array::from(bytes).into());
    let bag = web_sys::BlobPropertyBag::new();
    bag.set_type(mime);
    let blob = web_sys::Blob::new_with_u8_array_sequence_and_options(&parts, &bag)
        .map_err(|_| "Failed to create download Blob".to_owned())?;
    let url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|_| "Failed to create Blob object URL".to_owned())?;
    let anchor: web_sys::HtmlAnchorElement = document
        .create_element("a")
        .map_err(|_| "Failed to create download anchor".to_owned())?
        .dyn_into::<web_sys::HtmlAnchorElement>()
        .map_err(|_| "Failed to cast download anchor".to_owned())?;
    anchor.set_href(&url);
    anchor.set_download(filename);
    let anchor_js: &JsValue = anchor.as_ref();
    if let Ok(click_val) = js_sys::Reflect::get(anchor_js, &JsValue::from_str("click"))
        && let Ok(click_fn) = click_val.dyn_into::<js_sys::Function>()
    {
        let _ = click_fn.call0(anchor_js);
    }
    let _ = web_sys::Url::revoke_object_url(&url);
    Ok(())
}

fn trigger_export_asset(session: &Session, aid: &str, actions: &mut Vec<PanelAction>) {
    let Some(rec) = session.project.assets.get(aid) else {
        actions.push(PanelAction::StatusMessage("Asset not found".to_owned()));
        return;
    };
    let bytes = match resolve_asset_export_bytes(rec) {
        Ok(b) => b,
        Err(err) => {
            actions.push(PanelAction::StatusMessage(err));
            return;
        }
    };
    #[cfg(target_arch = "wasm32")]
    {
        let mime = mime_for_asset_filename(&rec.filename);
        match download_web_asset_bytes(&rec.filename, &bytes, mime) {
            Ok(()) => actions.push(PanelAction::StatusMessage(format!(
                "Exported `{}` ({})",
                rec.filename,
                format_asset_size(bytes.len() as u64)
            ))),
            Err(err) => actions.push(PanelAction::StatusMessage(format!(
                "Failed to export `{}`: {err}",
                rec.filename
            ))),
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(dest) = rfd::FileDialog::new()
        .set_title(format!("Export asset: {}", rec.filename))
        .set_file_name(&rec.filename)
        .save_file()
    {
        match zc_project::write_atomic(&dest, &bytes) {
            Ok(()) => actions.push(PanelAction::StatusMessage(format!(
                "Exported `{}` ({}) to {}",
                rec.filename,
                format_asset_size(bytes.len() as u64),
                dest.display()
            ))),
            Err(err) => actions.push(PanelAction::StatusMessage(format!(
                "Failed to export `{}`: {err}",
                rec.filename
            ))),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub(super) struct CollectFilesReport {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) project_path: PathBuf,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) report_path: PathBuf,
    pub(super) collected_count: usize,
    pub(super) missing_count: usize,
    pub(super) total_bytes: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn sanitize_collected_asset_filename(raw: &str, fallback_stem: &str) -> String {
    let base = std::path::Path::new(raw)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(raw);
    let cleaned: String = base
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | ' ') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let trimmed = cleaned.trim().trim_start_matches('.');
    if trimmed.is_empty() {
        fallback_stem.to_owned()
    } else {
        trimmed.to_owned()
    }
}

/// Collects the current project (`.zcraft`), every available asset file into `assets/`, and a
/// human-readable `collect_report.txt` into a self-contained directory `dest_dir`.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn collect_project_files_to_dir(
    session: &Session,
    dest_dir: &std::path::Path,
) -> Result<CollectFilesReport, String> {
    std::fs::create_dir_all(dest_dir)
        .map_err(|err| format!("Failed to create destination folder: {err}"))?;
    let assets_dir = dest_dir.join("assets");
    std::fs::create_dir_all(&assets_dir)
        .map_err(|err| format!("Failed to create assets/ directory: {err}"))?;

    let mut project = crate::project_io::prepare_project_for_zcraft_save(session);
    let mut used_names: BTreeSet<String> = BTreeSet::new();
    let mut collected_count = 0usize;
    let mut missing_count = 0usize;
    let mut total_bytes = 0u64;
    let mut report_lines: Vec<String> = Vec::new();

    let asset_ids: Vec<String> = project.assets.keys().cloned().collect();
    for (idx, aid) in asset_ids.iter().enumerate() {
        let Some(rec) = project.assets.get_mut(aid) else {
            continue;
        };
        let fallback = format!("asset_{}", idx + 1);
        let safe_base = sanitize_collected_asset_filename(&rec.filename, &fallback);
        let candidate_path = std::path::Path::new(&safe_base);
        let stem = candidate_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(&safe_base)
            .to_owned();
        let ext = candidate_path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| format!(".{e}"))
            .unwrap_or_default();
        let mut unique_name = safe_base.clone();
        let mut suffix = 2usize;
        while !used_names.insert(unique_name.to_ascii_lowercase()) {
            unique_name = format!("{stem}_{suffix}{ext}");
            suffix += 1;
        }

        let folder = rec
            .preserved
            .get("folder")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or("(Root)");
        let label_idx = rec
            .preserved
            .get("labelColor")
            .and_then(Value::as_u64)
            .unwrap_or(0) as u8;
        let (label_name, _) = asset_label_entry(label_idx);
        let sha_short: String = rec.sha256.chars().take(12).collect();

        match resolve_asset_export_bytes(rec) {
            Ok(bytes) => {
                let out_file = assets_dir.join(&unique_name);
                zc_project::write_atomic(&out_file, &bytes).map_err(|err| {
                    format!("Failed to write collected asset `{unique_name}`: {err}")
                })?;
                let len = bytes.len() as u64;
                rec.url = Some(format!("assets/{unique_name}"));
                rec.size = len;
                rec.missing = false;
                collected_count += 1;
                total_bytes = total_bytes.saturating_add(len);
                report_lines.push(format!(
                    "[OK]      {} -> assets/{} | {} | folder: {} | label: {} | sha256: {}",
                    rec.filename,
                    unique_name,
                    format_asset_size(len),
                    folder,
                    label_name,
                    if sha_short.is_empty() {
                        "-"
                    } else {
                        &sha_short
                    }
                ));
            }
            Err(_) => {
                missing_count += 1;
                report_lines.push(format!(
                    "[MISSING] {} | folder: {} | label: {} | id: {}",
                    rec.filename, folder, label_name, rec.id
                ));
            }
        }
    }

    let proj_stem = crate::project_io::sanitize_filename(&project.name);
    let project_path = dest_dir.join(format!("{proj_stem}.zcraft"));
    zc_project::save_zcraft(&project, &project_path)
        .map_err(|err| format!("Failed to save collected .zcraft project: {err}"))?;

    let mut report = String::new();
    report.push_str("ZoidCraft Collect Files Report\n");
    report.push_str("==============================\n");
    report.push_str(&format!("Project: {}\n", project.name));
    report.push_str(&format!(
        "Sequence: {}x{} @ {} fps ({:.0} frames)\n",
        project.sequence.resolution[0],
        project.sequence.resolution[1],
        project.sequence.fps,
        project.sequence.length_frames
    ));
    report.push_str(&format!("Archive: {}\n", project_path.display()));
    report.push_str(&format!(
        "Assets: {} total ({} collected, {} missing, {})\n\n",
        collected_count + missing_count,
        collected_count,
        missing_count,
        format_asset_size(total_bytes)
    ));
    if report_lines.is_empty() {
        report.push_str("(No assets in project)\n");
    } else {
        for line in &report_lines {
            report.push_str(line);
            report.push('\n');
        }
    }

    let report_path = dest_dir.join("collect_report.txt");
    zc_project::write_atomic(&report_path, report.as_bytes())
        .map_err(|err| format!("Failed to write collect_report.txt: {err}"))?;

    Ok(CollectFilesReport {
        project_path,
        report_path,
        collected_count,
        missing_count,
        total_bytes,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn trigger_collect_files(session: &Session, actions: &mut Vec<PanelAction>) {
    if let Some(dest_dir) = rfd::FileDialog::new()
        .set_title("Collect Project & Assets Into Folder")
        .pick_folder()
    {
        match collect_project_files_to_dir(session, &dest_dir) {
            Ok(rep) => {
                let missing_note = if rep.missing_count > 0 {
                    format!(", {} missing", rep.missing_count)
                } else {
                    String::new()
                };
                actions.push(PanelAction::StatusMessage(format!(
                    "Collected project + {} asset(s) ({}{missing_note}) to {}",
                    rep.collected_count,
                    format_asset_size(rep.total_bytes),
                    dest_dir.display()
                )));
            }
            Err(err) => actions.push(PanelAction::StatusMessage(format!(
                "Collect Files failed: {err}"
            ))),
        }
    }
}

fn reveal_asset_usage(session: &mut Session, aid: &str, actions: &mut Vec<PanelAction>) {
    let found = session
        .document()
        .video_tracks
        .iter()
        .chain(session.document().audio_tracks.iter())
        .flat_map(|t| t.clips.iter())
        .find(|c| {
            c.properties
                .get("assetId")
                .and_then(|p| p.keyframes.first())
                .and_then(|k| k.value.as_str())
                == Some(aid)
        })
        .map(|c| (c.id.clone(), c.kind.clone(), c.start));
    if let Some((cid, ckind, start_f)) = found {
        let _ = session.selection.select_clip(&cid, Some(&ckind), false);
        let _ = session.execute(Command::SetFrame { frame: start_f });
    } else {
        actions.push(PanelAction::StatusMessage(
            "Asset used in nested composition, shader, or object material".to_owned(),
        ));
    }
}

struct MediaLibraryEntry {
    name: String,
    asset_id: Option<String>,
    clip_kind: &'static str,
    missing: bool,
}

enum MediaGridItem<'a> {
    Preset(&'a MediaPreset),
    Imported(MediaLibraryEntry),
}

pub(super) fn show_media_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("media_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // Top search box: 34px tall, 4px left/right margin, 5px top/bottom margin,
            // #2a2a2a fill, crisp 2px solid #363636 border, single-frame (Frame::NONE)
            Frame::NONE
                .inner_margin(egui::Margin {
                    left: 4,
                    right: 4,
                    top: 5,
                    bottom: 5,
                })
                .show(ui, |ui| {
                    let search_id = Id::unique("media_search_box");
                    let focused = ui.memory(|m| m.has_focus(search_id));
                    let box_w = ui.available_width();
                    ui.allocate_ui_with_layout(
                        Vec2::new(box_w, 34.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            let box_rect =
                                Rect::from_min_size(ui.cursor().min, Vec2::new(box_w, 34.0));
                            ui.painter()
                                .rect_filled(box_rect, CornerRadius::ZERO, BG_PANEL);
                            paint_crisp_border_2px(
                                ui.painter(),
                                box_rect,
                                if focused {
                                    theme.accent()
                                } else {
                                    BORDER_FILTER
                                },
                            );
                            Frame::NONE
                                .inner_margin(egui::Margin::symmetric(7, 6))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.media_search)
                                            .id(search_id)
                                            .frame(Frame::NONE)
                                            .font(theme.font(FONT_BODY))
                                            .text_color(TEXT_PRIMARY)
                                            .desired_width((box_w - 14.0).max(40.0)),
                                    );
                                });
                            if state.media_search.is_empty() {
                                ui.painter().text(
                                    Pos2::new(box_rect.left() + 8.0, box_rect.center().y),
                                    Align2::LEFT_CENTER,
                                    "search media",
                                    theme.font(FONT_BODY),
                                    Color32::from_rgb(0x75, 0x75, 0x75),
                                );
                            }
                        },
                    );
                });

            // Precomp+ Compositions manager (Main + nested compositions)
            if session.plugins.is_enabled("precomp-plus")
                || !session.project.compositions.is_empty()
            {
                show_compositions_manager_section(ui, session, theme, state, actions);
            }

            // 100% edge-to-edge 30px #222222 "Project media" bar with borderless "+ Import" inside
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 30.0), Sense::hover());
            ui.painter()
                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
            let import_galley = ui.painter().layout_no_wrap(
                "Import".to_owned(),
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
            let plus_size = 16.0;
            let plus_gap = 7.0;
            let right_pad = 8.0;
            let import_w = plus_size + plus_gap + import_galley.size().x + right_pad + 8.0;
            let import_rect = Rect::from_min_max(
                Pos2::new(
                    (bar_rect.right() - import_w).max(bar_rect.left()),
                    bar_rect.top(),
                ),
                bar_rect.right_bottom(),
            );
            // Keep title cleanly clipped/shortened before "+ Import" so it never overlaps at 900x600
            let title_left = bar_rect.left() + 5.0;
            let title_max_w = (import_rect.left() - title_left - 6.0).max(16.0);
            let full_title_galley = ui.painter().layout_no_wrap(
                "Project media".to_owned(),
                theme.font(FONT_BODY),
                TEXT_PRIMARY,
            );
            let title_galley = if full_title_galley.size().x <= title_max_w {
                full_title_galley
            } else {
                let short_galley = ui.painter().layout_no_wrap(
                    "Media".to_owned(),
                    theme.font(FONT_BODY),
                    TEXT_PRIMARY,
                );
                if short_galley.size().x <= title_max_w {
                    short_galley
                } else {
                    let chars: Vec<char> = "Media".chars().collect();
                    let mut len = chars.len();
                    let mut chosen = short_galley;
                    while len > 0 {
                        len -= 1;
                        let cand = format!("{}…", chars[..len].iter().collect::<String>());
                        let g =
                            ui.painter()
                                .layout_no_wrap(cand, theme.font(FONT_BODY), TEXT_PRIMARY);
                        if g.size().x <= title_max_w || len == 0 {
                            chosen = g;
                            break;
                        }
                    }
                    chosen
                }
            };
            let title_clip = Rect::from_min_max(
                Pos2::new(title_left, bar_rect.top()),
                Pos2::new(
                    (import_rect.left() - 4.0).max(title_left),
                    bar_rect.bottom(),
                ),
            );
            ui.painter().with_clip_rect(title_clip).galley(
                Pos2::new(
                    title_left,
                    bar_rect.center().y + 1.0 - title_galley.size().y * 0.5,
                ),
                title_galley,
                TEXT_PRIMARY,
            );
            let import_resp = ui
                .interact(import_rect, Id::unique("media_import_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Import image, video, audio, or font media");
            if import_resp.hovered() {
                ui.painter()
                    .rect_filled(import_rect, CornerRadius::ZERO, BG_HOVER);
            }
            let import_color = if import_resp.hovered() {
                TEXT_BRIGHT
            } else {
                TOOLBAR_ICON_INACTIVE
            };
            let label_right = bar_rect.right() - right_pad;
            let label_left = label_right - import_galley.size().x;
            ui.painter().text(
                Pos2::new(label_right, bar_rect.center().y + 1.0),
                Align2::RIGHT_CENTER,
                "Import",
                theme.font(FONT_BODY),
                import_color,
            );
            let plus_cx = (label_left - plus_gap - plus_size * 0.5).round();
            let plus_cy = (bar_rect.center().y + 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_size(
                    Pos2::new(plus_cx - 8.0, plus_cy - 1.0),
                    Vec2::new(16.0, 2.0),
                ),
                CornerRadius::ZERO,
                import_color,
            );
            ui.painter().rect_filled(
                Rect::from_min_size(
                    Pos2::new(plus_cx - 1.0, plus_cy - 8.0),
                    Vec2::new(2.0, 16.0),
                ),
                CornerRadius::ZERO,
                import_color,
            );
            if import_resp.clicked() {
                actions.push(PanelAction::ImportMedia);
            }

            // Build unified list: built-in presets + imported media in the same 2-column grid
            let media_asset_usage = compute_asset_usage_counts_cached(session, state);
            let q = state.media_search.trim().to_ascii_lowercase();
            let mut items: Vec<MediaGridItem<'_>> = MEDIA_PRESETS
                .iter()
                .filter(|p| {
                    p.required_plugin
                        .is_none_or(|pid| session.plugins.is_enabled(pid))
                })
                .filter(|p| {
                    q.is_empty()
                        || p.title.to_ascii_lowercase().contains(&q)
                        || p.subtitle.to_ascii_lowercase().contains(&q)
                })
                .map(MediaGridItem::Preset)
                .collect();

            let mut seen_asset_ids: BTreeSet<String> = BTreeSet::new();
            let mut seen_names: BTreeSet<String> = BTreeSet::new();

            for m in &session.project.media {
                let asset_rec = m
                    .asset_id
                    .as_deref()
                    .and_then(|aid| session.project.assets.get(aid));
                if let Some(aid) = &m.asset_id {
                    seen_asset_ids.insert(aid.clone());
                }
                seen_names.insert(m.name.to_ascii_lowercase());
                let filename = asset_rec
                    .map(|a| a.filename.as_str())
                    .unwrap_or(m.name.as_str());
                let kind = asset_rec.map_or(m.kind, |a| a.kind);
                let clip_kind = clip_kind_for_asset(kind, filename);
                if !q.is_empty()
                    && !m.name.to_ascii_lowercase().contains(&q)
                    && !clip_kind.contains(&q)
                {
                    continue;
                }
                items.push(MediaGridItem::Imported(MediaLibraryEntry {
                    name: m.name.clone(),
                    asset_id: m.asset_id.clone(),
                    clip_kind,
                    missing: asset_rec.is_some_and(|a| a.missing),
                }));
            }

            for a in session.project.assets.values() {
                if seen_asset_ids.contains(&a.id)
                    || seen_names.contains(&a.filename.to_ascii_lowercase())
                {
                    continue;
                }
                let clip_kind = clip_kind_for_asset(a.kind, &a.filename);
                if !q.is_empty()
                    && !a.filename.to_ascii_lowercase().contains(&q)
                    && !clip_kind.contains(&q)
                {
                    continue;
                }
                items.push(MediaGridItem::Imported(MediaLibraryEntry {
                    name: a.filename.clone(),
                    asset_id: Some(a.id.clone()),
                    clip_kind,
                    missing: a.missing,
                }));
            }

            // 2-column grid: 4px outer side margin, 3px gap, ~155x111 flat #333333 borderless cards
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(4, 4))
                .show(ui, |ui| {
                    let cols = if ui.available_width() >= 140.0 { 2 } else { 1 };
                    let gap = 3.0;
                    let card_w =
                        ((ui.available_width() - gap * (cols - 1) as f32) / cols as f32).max(72.0);
                    let card_h = 111.5;
                    let card_hover_bg = Color32::from_rgb(0x3d, 0x3d, 0x3d);

                    for chunk in items.chunks(cols) {
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = gap;
                            for item in chunk {
                                let (rect, resp) = ui
                                    .allocate_exact_size(Vec2::new(card_w, card_h), Sense::click());
                                let bg = if resp.hovered() {
                                    card_hover_bg
                                } else {
                                    BG_HOVER
                                };
                                ui.painter().rect_filled(rect, CornerRadius::ZERO, bg);

                                let narrow_card = card_w < 96.0;
                                let icon_rect = Rect::from_center_size(
                                    Pos2::new(
                                        rect.center().x,
                                        rect.top() + if narrow_card { 42.0 } else { 45.5 },
                                    ),
                                    Vec2::splat(if narrow_card { 40.0 } else { 48.0 }),
                                );
                                let label_clip = Rect::from_min_max(
                                    Pos2::new(
                                        rect.left() + 5.0,
                                        rect.bottom() - if narrow_card { 24.0 } else { 26.0 },
                                    ),
                                    Pos2::new(rect.right() - 5.0, rect.bottom() - 3.0),
                                );
                                let card_font =
                                    theme.font(if narrow_card { FONT_COMPACT } else { FONT_BODY });

                                match item {
                                    MediaGridItem::Preset(preset) => {
                                        let resp = resp
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text(format!(
                                                "{} — {}",
                                                preset.title, preset.subtitle
                                            ));
                                        paint_media_card_icon(
                                            ui.painter(),
                                            icon_rect,
                                            preset.icon,
                                            TEXT_PRIMARY,
                                            bg,
                                        );
                                        let galley = layout_single_line_ellipsized(
                                            ui.painter(),
                                            preset.title,
                                            card_font.clone(),
                                            TEXT_PRIMARY,
                                            label_clip.width().max(24.0),
                                        );
                                        let label_y = label_clip.center().y - galley.size().y * 0.5;
                                        ui.painter().with_clip_rect(label_clip).galley(
                                            Pos2::new(label_clip.left(), label_y),
                                            galley,
                                            TEXT_PRIMARY,
                                        );
                                        if resp.clicked() {
                                            add_preset_clip(session, preset.kind, preset.title);
                                        }
                                    }
                                    MediaGridItem::Imported(entry) => {
                                        let use_count = entry
                                            .asset_id
                                            .as_deref()
                                            .and_then(|aid| media_asset_usage.get(aid).copied())
                                            .unwrap_or(0);
                                        let usage_hint = if entry.missing {
                                            "missing".to_owned()
                                        } else if use_count == 0 {
                                            "unused".to_owned()
                                        } else {
                                            format!("used {use_count}×")
                                        };
                                        let resp = resp
                                            .on_hover_cursor(CursorIcon::PointingHand)
                                            .on_hover_text(format!(
                                                "{} ({usage_hint}) — click to add {} clip to timeline",
                                                entry.name, entry.clip_kind
                                            ));
                                        let text_col = if entry.missing {
                                            DANGER_TEXT
                                        } else {
                                            TEXT_PRIMARY
                                        };
                                        if entry.clip_kind == "audio" {
                                            paint_audio_media_icon(
                                                ui.painter(),
                                                icon_rect,
                                                text_col,
                                            );
                                        } else if entry.clip_kind == "image" {
                                            paint_image_media_icon(
                                                ui.painter(),
                                                icon_rect,
                                                text_col,
                                                bg,
                                            );
                                        } else {
                                            let icon = match entry.clip_kind {
                                                "video" => Cm3Icon::Sequence,
                                                _ => Cm3Icon::Project,
                                            };
                                            paint_cm3_icon(ui.painter(), icon_rect, icon, text_col);
                                        }
                                        if entry.missing || use_count > 0 {
                                            let badge_txt = if entry.missing {
                                                "MISSING".to_owned()
                                            } else {
                                                format!("{use_count}×")
                                            };
                                            let badge_w = if entry.missing { 46.0 } else { 24.0 };
                                            let badge_r = Rect::from_min_size(
                                                Pos2::new(rect.right() - badge_w - 4.0, rect.top() + 4.0),
                                                Vec2::new(badge_w, 14.0),
                                            );
                                            let (badge_bg, badge_fg) = if entry.missing {
                                                (Color32::from_rgb(0x6e, 0x22, 0x22), TEXT_BRIGHT)
                                            } else {
                                                (color32_with_alpha(theme.accent(), 170), TEXT_BRIGHT)
                                            };
                                            paint_badge(
                                                ui.painter(),
                                                badge_r,
                                                &badge_txt,
                                                badge_bg,
                                                badge_fg,
                                            );
                                        }
                                        let galley = layout_single_line_ellipsized(
                                            ui.painter(),
                                            &entry.name,
                                            card_font.clone(),
                                            text_col,
                                            label_clip.width().max(24.0),
                                        );
                                        let label_y = label_clip.center().y - galley.size().y * 0.5;
                                        ui.painter().with_clip_rect(label_clip).galley(
                                            Pos2::new(label_clip.left(), label_y),
                                            galley,
                                            text_col,
                                        );
                                        if resp.clicked() {
                                            add_clip_with_asset(
                                                session,
                                                entry.clip_kind,
                                                &entry.name,
                                                entry.asset_id.as_deref(),
                                            );
                                        }
                                    }
                                }
                            }
                        });
                        ui.add_space(gap);
                    }
                });
        });
}

pub(super) fn add_preset_clip(session: &mut Session, kind: &str, title: &str) {
    add_clip_with_asset(session, kind, title, None);
}

fn add_clip_with_asset(session: &mut Session, kind: &str, title: &str, asset_id: Option<&str>) {
    let is_audio = kind == "audio";
    let desired_start = session.playback.current_frame.max(0.0).round();
    let length = session.active_sequence().length_frames.clamp(1.0, 150.0);

    // Place the new clip at the current playhead (`desired_start`) on the first unlocked
    // matching track with no overlapping clip in `[desired_start, desired_start + length)`.
    // If every existing unlocked track overlaps at `desired_start`, create a new track above
    // via `Command::AddVideoTrack` / `Command::AddAudioTrack` so the clip is immediately
    // visible in both the viewport and timeline instead of pushed off-screen!
    let chosen_track_id: Option<String> = {
        let tracks = if is_audio {
            &session.document().audio_tracks
        } else {
            &session.document().video_tracks
        };
        tracks
            .iter()
            .find(|t| {
                !t.locked
                    && !t.clips.iter().any(|c| {
                        c.start < desired_start + length - 1e-6
                            && desired_start < c.start + c.length - 1e-6
                    })
            })
            .map(|t| t.id.clone())
    };

    let started_tx = session.begin_transaction(format!("add {title}")).is_ok();

    let (track_id, start) = if let Some(tid) = chosen_track_id {
        (tid, desired_start)
    } else {
        let create_cmd = if is_audio {
            Command::AddAudioTrack { name: None }
        } else {
            Command::AddVideoTrack { name: None }
        };
        if let Ok(res) = session.execute(create_cmd)
            && let Some(tid) = res.get("track_id").and_then(Value::as_str)
        {
            (tid.to_owned(), desired_start)
        } else {
            if started_tx && session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }
            return;
        }
    };

    if let Ok(res) = session.execute(Command::AddClip {
        track_id,
        kind: kind.to_owned(),
        name: Some(title.to_owned()),
        start,
        length,
    }) && let Some(cid) = res.get("clip_id").and_then(Value::as_str)
    {
        if let Some(aid) = asset_id {
            let _ = session.execute(Command::SetPropertyValue {
                target_id: cid.to_owned(),
                property: "assetId".to_owned(),
                value: json!(aid),
            });
        }

        // Seed default Camera + Spot Light + 3D Text inside new 3D Scene clips so they
        // are immediately visible in the viewport.
        if kind == "scene" {
            let _ = session.execute(Command::AddObject3D {
                clip_id: cid.to_owned(),
                parent_id: None,
                kind: "camera".to_owned(),
                name: Some("Camera".to_owned()),
            });
            if let Ok(light_res) = session.execute(Command::AddObject3D {
                clip_id: cid.to_owned(),
                parent_id: None,
                kind: "spot_light".to_owned(),
                name: Some("Spot Light".to_owned()),
            }) && let Some(lid) = light_res.get("object_id").and_then(Value::as_str)
            {
                let _ = session.execute(Command::SetPropertyValue {
                    target_id: lid.to_owned(),
                    property: "position".to_owned(),
                    value: json!([0.0, 10.0, 0.0]),
                });
            }
            let _ = session.execute(Command::AddObject3D {
                clip_id: cid.to_owned(),
                parent_id: None,
                kind: "text".to_owned(),
                name: Some("Text".to_owned()),
            });
        }

        let _ = session.selection.select_clip(cid, Some(kind), false);
    }

    if started_tx && session.history.is_in_transaction() {
        let _ = session.commit_transaction(true);
    }
}

// ---------------------------------------------------------------------------
// 1. Precomp+ Compositions Manager (Media Tab)
// ---------------------------------------------------------------------------

fn count_composition_references(project: &zc_project::Project, comp_id: Uuid) -> usize {
    fn count_in_layer(layer: &zc_project::Layer, comp_id: Uuid) -> usize {
        match &layer.kind {
            LayerKind::PrecompRef { composition_id } => usize::from(*composition_id == comp_id),
            LayerKind::Composite { layers } => {
                layers.iter().map(|c| count_in_layer(c, comp_id)).sum()
            }
            _ => 0,
        }
    }
    let count_in_seq = |seq: &zc_project::Sequence| -> usize {
        seq.video_tracks
            .iter()
            .flat_map(|t| t.clips.iter())
            .map(|c| count_in_layer(&c.layer, comp_id))
            .sum()
    };
    count_in_seq(&project.sequence)
        + project
            .compositions
            .values()
            .map(count_in_seq)
            .sum::<usize>()
}

fn show_compositions_manager_section(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let active_comp = session.document().active_composition_id;
    let mut comps: Vec<(Uuid, String, [u32; 2], u32, usize)> = session
        .project
        .compositions
        .iter()
        .map(|(id, seq)| {
            let refs = count_composition_references(&session.project, *id);
            (*id, seq.name.clone(), seq.resolution, seq.fps, refs)
        })
        .collect();
    comps.sort_by(|a, b| a.1.cmp(&b.1));

    let (add_comp, _) = cm3_collapsible_section_with_action(
        ui,
        Id::unique("media_compositions_section"),
        "Compositions",
        true,
        false,
        state.prop_columns.label_ratio,
        Some((Cm3Icon::Add, "New composition")),
        theme,
        |ui| {
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing.y = 4.0;
                    // Breadcrumb / Back to Main + Pre-compose Selected actions
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if active_comp.is_some()
                            && cm3_prop_button(
                                ui,
                                "← Back to Main",
                                ButtonVariant::Primary,
                                false,
                                theme,
                            )
                            .clicked()
                        {
                            let _ =
                                session.execute(Command::SetActiveComposition { comp_id: None });
                        }
                        let selected_clip_uuids: Vec<Uuid> = session
                            .selection
                            .clip_ids
                            .iter()
                            .filter_map(|s| Uuid::parse_str(s).ok())
                            .collect();
                        if cm3_prop_button(
                            ui,
                            "Pre-compose Selected",
                            ButtonVariant::Normal,
                            selected_clip_uuids.is_empty(),
                            theme,
                        )
                        .clicked()
                        {
                            let name =
                                format!("Precomp {}", session.project.compositions.len() + 1);
                            let _ = session.execute(Command::PrecomposeClips {
                                clip_ids: selected_clip_uuids,
                                name,
                            });
                        }
                    });

                    // Main composition row
                    let main_active = active_comp.is_none();
                    ui.horizontal(|ui| {
                        let lbl = format!(
                            "Main ({}×{} @ {}fps)",
                            session.project.sequence.resolution[0],
                            session.project.sequence.resolution[1],
                            session.project.sequence.fps
                        );
                        if ui
                            .selectable_label(
                                main_active,
                                egui::RichText::new(lbl).font(theme.font(FONT_SMALL)).color(
                                    if main_active {
                                        TEXT_BRIGHT
                                    } else {
                                        TEXT_PRIMARY
                                    },
                                ),
                            )
                            .clicked()
                            && !main_active
                        {
                            let _ =
                                session.execute(Command::SetActiveComposition { comp_id: None });
                        }
                    });

                    // Nested compositions
                    for (cid, cname, cres, cfps, use_count) in &comps {
                        let is_active = active_comp == Some(*cid);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 3.0;
                            if state.renaming_comp_id == Some(*cid) {
                                let resp = ui.add(
                                    TextEdit::singleline(&mut state.renaming_comp_buf)
                                        .desired_width(110.0)
                                        .font(theme.font(FONT_SMALL)),
                                );
                                if ui.small_button("OK").clicked()
                                    || (resp.lost_focus()
                                        && ui.input(|i| i.key_pressed(Key::Enter)))
                                {
                                    let trimmed = state.renaming_comp_buf.trim().to_owned();
                                    if !trimmed.is_empty() {
                                        let _ = session.execute(Command::RenameComposition {
                                            comp_id: *cid,
                                            name: trimmed,
                                        });
                                    }
                                    state.renaming_comp_id = None;
                                }
                            } else {
                                let lbl = format!(
                                    "{cname} ({}×{} @ {cfps}fps · {use_count}×)",
                                    cres[0], cres[1]
                                );
                                let actions_w = 58.0_f32;
                                let label_w = (ui.available_width() - actions_w - 6.0).max(48.0);
                                ui.allocate_ui_with_layout(
                                    Vec2::new(label_w, 18.0),
                                    egui::Layout::left_to_right(egui::Align::Center),
                                    |ui| {
                                        let resp = ui
                                            .add(
                                                egui::Button::new(
                                                    egui::RichText::new(&lbl)
                                                        .font(theme.font(FONT_SMALL))
                                                        .color(if is_active {
                                                            TEXT_BRIGHT
                                                        } else {
                                                            TEXT_PRIMARY
                                                        }),
                                                )
                                                .fill(if is_active {
                                                    theme.accent()
                                                } else {
                                                    Color32::TRANSPARENT
                                                })
                                                .stroke(Stroke::NONE)
                                                .truncate(),
                                            )
                                            .on_hover_text(&lbl);
                                        if resp.clicked() {
                                            let _ =
                                                session.execute(Command::SetActiveComposition {
                                                    comp_id: Some(*cid),
                                                });
                                        }
                                    },
                                );
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        if cm3_icon_button(
                                            ui,
                                            Cm3Icon::Remove,
                                            13.0,
                                            TEXT_MUTED,
                                            "Delete comp",
                                            theme,
                                        ) && let Err(err) = session
                                            .execute(Command::DeleteComposition { comp_id: *cid })
                                        {
                                            actions.push(PanelAction::StatusMessage(format!(
                                                "Cannot delete composition: {err}"
                                            )));
                                        }
                                        if cm3_icon_button(
                                            ui,
                                            Cm3Icon::Duplicate,
                                            13.0,
                                            TEXT_MUTED,
                                            "Duplicate comp",
                                            theme,
                                        ) {
                                            let _ =
                                                session.execute(Command::DuplicateComposition {
                                                    comp_id: *cid,
                                                });
                                        }
                                        if cm3_icon_button(
                                            ui,
                                            Cm3Icon::Edit2,
                                            13.0,
                                            TEXT_MUTED,
                                            "Rename comp",
                                            theme,
                                        ) {
                                            state.renaming_comp_id = Some(*cid);
                                            state.renaming_comp_buf = cname.clone();
                                        }
                                    },
                                );
                            }
                        });
                    }
                });
        },
    );
    if add_comp {
        let idx = session.project.compositions.len() + 1;
        let name = format!("{} {idx}", state.new_comp_name.trim());
        let res = session.project.sequence.resolution;
        let fps = session.project.sequence.fps;
        let len = (session.project.sequence.length_frames.round() as u32).max(150);
        let _ = session.execute(Command::CreateComposition {
            name,
            resolution: res,
            fps,
            length_frames: len,
        });
    }
}

// ---------------------------------------------------------------------------
// Tab 9: Assets — Project Assets & Starter Templates
// ---------------------------------------------------------------------------

pub(super) fn show_assets_tab(
    ui: &mut Ui,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    ScrollArea::vertical()
        .id_salt("assets_tab_scroll")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;
            panel_top_header(ui, "Assets", theme);

            // Single-frame "type to filter" search box (34px tall, 4px side margin, 2px #363636)
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(4, 6))
                .show(ui, |ui| {
                    let filter_id = Id::unique("assets_filter_box");
                    let focused = ui.memory(|m| m.has_focus(filter_id));
                    let box_w = ui.available_width();
                    ui.allocate_ui_with_layout(
                        Vec2::new(box_w, 34.0),
                        egui::Layout::left_to_right(egui::Align::Center),
                        |ui| {
                            let box_rect =
                                Rect::from_min_size(ui.cursor().min, Vec2::new(box_w, 34.0));
                            ui.painter()
                                .rect_filled(box_rect, CornerRadius::ZERO, BG_PANEL);
                            ui.painter().rect_stroke(
                                box_rect,
                                CornerRadius::ZERO,
                                Stroke::new(
                                    2.0,
                                    if focused {
                                        theme.accent()
                                    } else {
                                        BORDER_FILTER
                                    },
                                ),
                                StrokeKind::Inside,
                            );
                            Frame::NONE
                                .inner_margin(egui::Margin::symmetric(8, 6))
                                .show(ui, |ui| {
                                    ui.add(
                                        TextEdit::singleline(&mut state.assets_filter)
                                            .id(filter_id)
                                            .frame(Frame::NONE)
                                            .hint_text(
                                                egui::RichText::new("type to filter")
                                                    .color(TEXT_EMPTY),
                                            )
                                            .font(theme.font(FONT_BODY))
                                            .text_color(TEXT_PRIMARY)
                                            .desired_width((box_w - 16.0).max(40.0)),
                                    );
                                });
                        },
                    );
                });

            let q = state.assets_filter.trim().to_ascii_lowercase();

            let asset_usage = compute_asset_usage_counts_cached(session, state);
            let ui_state_id = Id::unique("assets_manager_ui_state");
            let mut mgr_state = ui
                .data_mut(|d| d.get_temp::<AssetsManagerUiState>(ui_state_id))
                .unwrap_or_default();
            mgr_state
                .selected_ids
                .retain(|id| session.project.assets.contains_key(id));

            let total_asset_count = session.project.assets.len();
            let unused_ids: Vec<String> = session
                .project
                .assets
                .keys()
                .filter(|id| asset_usage.get(*id).copied().unwrap_or(0) == 0)
                .cloned()
                .collect();
            let unused_asset_count = unused_ids.len();
            let unused_total_bytes: u64 = unused_ids
                .iter()
                .filter_map(|id| session.project.assets.get(id).map(|a| a.size))
                .sum();
            let missing_asset_count = session
                .project
                .assets
                .values()
                .filter(|a| a.missing)
                .count();

            let all_folders = collect_project_asset_folders(&session.project);
            if let Some(ref active_folder) = mgr_state.folder_filter
                && !active_folder.is_empty()
                && !all_folders.iter().any(|f| f == active_folder)
            {
                mgr_state.folder_filter = None;
            }

            // Project assets bar (responsive layout preventing title/button overlap at 800x540)
            let (bar_rect, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 27.0), Sense::hover());
            ui.painter()
                .rect_filled(bar_rect, CornerRadius::ZERO, BG_TITLE);
            ui.painter().line_segment(
                [bar_rect.left_bottom(), bar_rect.right_bottom()],
                Stroke::new(1.0, BORDER_ROW_SOFT),
            );
            let narrow_assets_bar = bar_rect.width() < 235.0;
            let import_lbl = "+ Import";
            let purge_lbl = if narrow_assets_bar {
                if unused_asset_count > 0 {
                    format!("Remove ({unused_asset_count})")
                } else {
                    "Purge".to_owned()
                }
            } else if unused_asset_count > 0 {
                format!("Remove Unused ({unused_asset_count})")
            } else {
                "Remove Unused".to_owned()
            };
            let import_w = ui
                .painter()
                .layout_no_wrap(import_lbl.to_owned(), theme.font(FONT_SMALL), TEXT_PRIMARY)
                .size()
                .x
                .ceil()
                + 14.0;
            let purge_w = ui
                .painter()
                .layout_no_wrap(purge_lbl.clone(), theme.font(FONT_COMPACT), TEXT_MUTED)
                .size()
                .x
                .ceil()
                + 10.0;
            let imp_rect = Rect::from_min_max(
                Pos2::new(
                    (bar_rect.right() - import_w).max(bar_rect.left() + 80.0),
                    bar_rect.top(),
                ),
                bar_rect.right_bottom(),
            );
            let purge_rect = Rect::from_min_max(
                Pos2::new(
                    (imp_rect.left() - purge_w - 4.0).max(bar_rect.left() + 52.0),
                    bar_rect.top(),
                ),
                Pos2::new(
                    (imp_rect.left() - 4.0).max(bar_rect.left() + 56.0),
                    bar_rect.bottom(),
                ),
            );
            let title_left = bar_rect.left() + 8.0;
            let title_max_w = (purge_rect.left() - title_left - 6.0).max(24.0);
            let title_text = if bar_rect.width() < 210.0 {
                "Assets"
            } else {
                "Project assets"
            };
            let title_galley = layout_single_line_ellipsized(
                ui.painter(),
                title_text,
                theme.font(FONT_SMALL),
                TEXT_STRONG,
                title_max_w,
            );
            let title_clip = Rect::from_min_max(
                Pos2::new(title_left, bar_rect.top()),
                Pos2::new((purge_rect.left() - 4.0).max(title_left), bar_rect.bottom()),
            );
            ui.painter().with_clip_rect(title_clip).galley(
                Pos2::new(
                    title_left,
                    bar_rect.center().y - title_galley.size().y * 0.5,
                ),
                title_galley,
                TEXT_STRONG,
            );
            let purge_resp = ui
                .interact(purge_rect, Id::unique("assets_purge_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text(
                    "Remove all unused assets not referenced by any clip, object, or effect (undoable)",
                );
            ui.painter().with_clip_rect(purge_rect).text(
                Pos2::new(purge_rect.right() - 4.0, bar_rect.center().y),
                Align2::RIGHT_CENTER,
                &purge_lbl,
                theme.font(FONT_COMPACT),
                if unused_asset_count == 0 {
                    TEXT_EMPTY
                } else if purge_resp.hovered() {
                    DANGER_TEXT
                } else {
                    TEXT_MUTED
                },
            );
            if purge_resp.clicked() {
                if unused_asset_count == 0 {
                    actions.push(PanelAction::StatusMessage(
                        "No unused assets to remove".to_owned(),
                    ));
                } else if let Ok(res) = session.execute(Command::PurgeUnusedAssets) {
                    let removed = res
                        .get("removed_count")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    mgr_state
                        .selected_ids
                        .retain(|id| session.project.assets.contains_key(id));
                    actions.push(PanelAction::StatusMessage(format!(
                        "Removed {removed} unused asset(s)"
                    )));
                }
            }
            let imp_resp = ui
                .interact(imp_rect, Id::unique("assets_import_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand);
            ui.painter().with_clip_rect(imp_rect).text(
                Pos2::new(bar_rect.right() - 8.0, bar_rect.center().y),
                Align2::RIGHT_CENTER,
                import_lbl,
                theme.font(FONT_SMALL),
                if imp_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    theme.value()
                },
            );
            if imp_resp.clicked() {
                actions.push(PanelAction::ImportMedia);
            }

            // Asset kind filter chips (All / Media / Audio / Image / Font / 3D Model)
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 4))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(3.0, 3.0);
                        for kind_chip in AssetKindFilter::ALL {
                            let active = mgr_state.kind_filter == kind_chip;
                            let btn = egui::Button::new(
                                egui::RichText::new(kind_chip.label())
                                    .font(theme.font(FONT_COMPACT))
                                    .color(if active { TEXT_BRIGHT } else { TEXT_MUTED }),
                            )
                            .fill(if active { theme.accent() } else { BG_BUTTON })
                            .stroke(Stroke::new(
                                1.0,
                                if active {
                                    BORDER_OPTION_ACTIVE
                                } else {
                                    BORDER_CONTROL
                                },
                            ))
                            .corner_radius(CornerRadius::same(2));
                            if ui
                                .add(btn)
                                .on_hover_cursor(CursorIcon::PointingHand)
                                .clicked()
                            {
                                mgr_state.kind_filter = kind_chip;
                            }
                        }
                    });
                });

            // Status filter + Select Unused + Folder grouping toolbar
            Frame::NONE
                .inner_margin(egui::Margin {
                    left: 6,
                    right: 6,
                    top: 0,
                    bottom: 4,
                })
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(4.0, 3.0);
                        let status_choices: &[(AssetStatusFilter, String)] = &[
                            (AssetStatusFilter::All, "All".to_owned()),
                            (AssetStatusFilter::Used, "Used".to_owned()),
                            (
                                AssetStatusFilter::Unused,
                                if unused_asset_count > 0 {
                                    format!("Unused ({unused_asset_count})")
                                } else {
                                    "Unused".to_owned()
                                },
                            ),
                            (
                                AssetStatusFilter::Missing,
                                if missing_asset_count > 0 {
                                    format!("Missing ({missing_asset_count})")
                                } else {
                                    "Missing".to_owned()
                                },
                            ),
                        ];
                        for (sf, lbl) in status_choices {
                            if *sf == AssetStatusFilter::Missing
                                && missing_asset_count == 0
                                && mgr_state.status_filter != AssetStatusFilter::Missing
                            {
                                continue;
                            }
                            let active = mgr_state.status_filter == *sf;
                            if ui
                                .selectable_label(
                                    active,
                                    egui::RichText::new(lbl)
                                        .font(theme.font(FONT_COMPACT))
                                        .color(if active {
                                            TEXT_BRIGHT
                                        } else if *sf == AssetStatusFilter::Missing {
                                            DANGER_TEXT
                                        } else {
                                            TEXT_MUTED
                                        }),
                                )
                                .clicked()
                            {
                                mgr_state.status_filter = *sf;
                            }
                        }
                        if ui
                            .small_button("Select Unused")
                            .on_hover_text("Highlight all assets not referenced anywhere in the project")
                            .clicked()
                        {
                            mgr_state.selected_ids.clear();
                            for id in &unused_ids {
                                mgr_state.selected_ids.insert(id.clone());
                            }
                            if unused_ids.is_empty() {
                                actions.push(PanelAction::StatusMessage(
                                    "All project assets are currently in use".to_owned(),
                                ));
                            } else {
                                actions.push(PanelAction::StatusMessage(format!(
                                    "Selected {} unused asset(s)",
                                    unused_ids.len()
                                )));
                            }
                        }
                        if ui
                            .selectable_label(
                                mgr_state.new_folder_open,
                                egui::RichText::new("+ Folder")
                                    .font(theme.font(FONT_COMPACT))
                                    .color(TEXT_MUTED),
                            )
                            .on_hover_text("Create a folder or move selected assets into a folder")
                            .clicked()
                        {
                            mgr_state.new_folder_open = !mgr_state.new_folder_open;
                        }
                    });

                    if mgr_state.new_folder_open {
                        ui.add_space(3.0);
                        ui.horizontal(|ui| {
                            ui.spacing_mut().item_spacing.x = 4.0;
                            let resp = ui.add(
                                TextEdit::singleline(&mut mgr_state.new_folder_buf)
                                    .hint_text("Folder name…")
                                    .font(theme.font(FONT_COMPACT))
                                    .desired_width((ui.available_width() - 56.0).max(60.0)),
                            );
                            let commit = ui.small_button("Create").clicked()
                                || (resp.lost_focus() && ui.input(|i| i.key_pressed(Key::Enter)));
                            if commit {
                                let name = mgr_state.new_folder_buf.trim().to_owned();
                                if !name.is_empty() {
                                    let selected_vec: Vec<String> =
                                        mgr_state.selected_ids.iter().cloned().collect();
                                    let existed = all_folders.iter().any(|f| f == &name);
                                    let moved = create_asset_folder_undoable(
                                        session,
                                        &name,
                                        &selected_vec,
                                    );
                                    if moved > 0 {
                                        actions.push(PanelAction::StatusMessage(format!(
                                            "Moved {moved} asset(s) to folder `{name}`"
                                        )));
                                    } else if !existed {
                                        actions.push(PanelAction::StatusMessage(format!(
                                            "Created folder `{name}`"
                                        )));
                                    }
                                    mgr_state.new_folder_buf.clear();
                                    mgr_state.new_folder_open = false;
                                }
                            }
                        });
                    }

                    if !all_folders.is_empty() {
                        ui.add_space(3.0);
                        ui.horizontal_wrapped(|ui| {
                            ui.spacing_mut().item_spacing = Vec2::new(4.0, 2.0);
                            if ui
                                .selectable_label(
                                    mgr_state.group_by_folder,
                                    egui::RichText::new("Group")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(if mgr_state.group_by_folder {
                                            TEXT_BRIGHT
                                        } else {
                                            TEXT_MUTED
                                        }),
                                )
                                .on_hover_text("Group assets into collapsible folders")
                                .clicked()
                            {
                                mgr_state.group_by_folder = !mgr_state.group_by_folder;
                            }
                            if ui
                                .selectable_label(
                                    mgr_state.folder_filter.is_none(),
                                    egui::RichText::new("All Folders")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(if mgr_state.folder_filter.is_none() {
                                            TEXT_BRIGHT
                                        } else {
                                            TEXT_MUTED
                                        }),
                                )
                                .clicked()
                            {
                                mgr_state.folder_filter = None;
                            }
                            let root_active = mgr_state.folder_filter.as_deref() == Some("");
                            if ui
                                .selectable_label(
                                    root_active,
                                    egui::RichText::new("Root")
                                        .font(theme.font(FONT_COMPACT))
                                        .color(if root_active {
                                            TEXT_BRIGHT
                                        } else {
                                            TEXT_MUTED
                                        }),
                                )
                                .clicked()
                            {
                                mgr_state.folder_filter = Some(String::new());
                            }
                            for folder in &all_folders {
                                let active =
                                    mgr_state.folder_filter.as_deref() == Some(folder.as_str());
                                if ui
                                    .selectable_label(
                                        active,
                                        egui::RichText::new(folder)
                                            .font(theme.font(FONT_COMPACT))
                                            .color(if active { TEXT_BRIGHT } else { TEXT_MUTED }),
                                    )
                                    .clicked()
                                {
                                    mgr_state.folder_filter = Some(folder.clone());
                                }
                            }
                        });
                    }
                });

            // Sort selector + Consolidate duplicates + Collect Files bar (Asset Manager parity)
            Frame::NONE
                .inner_margin(egui::Margin::symmetric(6, 2))
                .show(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing = Vec2::new(4.0, 3.0);
                        ui.label(
                            egui::RichText::new("Sort:")
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        for (si, slbl) in ["Name", "Size", "Usage", "Type"].iter().enumerate() {
                            let active = state.assets_sort_mode == si;
                            if ui
                                .selectable_label(
                                    active,
                                    egui::RichText::new(*slbl)
                                        .font(theme.font(FONT_COMPACT))
                                        .color(if active { TEXT_BRIGHT } else { TEXT_MUTED }),
                                )
                                .clicked()
                            {
                                state.assets_sort_mode = si;
                            }
                        }
                        if ui
                            .small_button("Consolidate")
                            .on_hover_text(
                                "Merge duplicate assets with identical file content and rewrite all project references (undoable)",
                            )
                            .clicked()
                        {
                            let merged = consolidate_duplicate_assets(session);
                            if merged == 0 {
                                actions.push(PanelAction::StatusMessage(
                                    "No duplicate assets found".to_owned(),
                                ));
                            } else {
                                mgr_state
                                    .selected_ids
                                    .retain(|id| session.project.assets.contains_key(id));
                                actions.push(PanelAction::StatusMessage(format!(
                                    "Consolidated {merged} duplicate asset(s)"
                                )));
                            }
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        if ui
                            .small_button("Collect Files")
                            .on_hover_text(
                                "Collect project (.zcraft), all assets into assets/, and collect_report.txt into a folder",
                            )
                            .clicked()
                        {
                            trigger_collect_files(session, actions);
                        }
                    });
                });

            struct AssetRowItem {
                id: String,
                filename: String,
                clip_kind: &'static str,
                category: AssetCategory,
                size: u64,
                missing: bool,
                use_count: usize,
                label_idx: u8,
                folder: String,
                #[cfg(not(target_arch = "wasm32"))]
                has_disk_url: bool,
            }

            let mut project_assets: Vec<AssetRowItem> = session
                .project
                .assets
                .values()
                .filter_map(|a| {
                    let category = AssetCategory::classify(a.kind, &a.filename);
                    if !mgr_state.kind_filter.matches(category) {
                        return None;
                    }
                    let count = asset_usage.get(&a.id).copied().unwrap_or(0);
                    match mgr_state.status_filter {
                        AssetStatusFilter::All => {}
                        AssetStatusFilter::Used => {
                            if count == 0 {
                                return None;
                            }
                        }
                        AssetStatusFilter::Unused => {
                            if count > 0 {
                                return None;
                            }
                        }
                        AssetStatusFilter::Missing => {
                            if !a.missing {
                                return None;
                            }
                        }
                    }
                    let folder = a
                        .preserved
                        .get("folder")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .unwrap_or("")
                        .to_owned();
                    if let Some(ref ff) = mgr_state.folder_filter
                        && &folder != ff
                    {
                        return None;
                    }
                    if !q.is_empty() {
                        let fname_l = a.filename.to_ascii_lowercase();
                        let folder_l = folder.to_ascii_lowercase();
                        let cat_l = category.label();
                        let status_match = (a.missing && "missing".contains(&q))
                            || (count == 0 && "unused".contains(&q));
                        if !fname_l.contains(&q)
                            && !folder_l.contains(&q)
                            && !cat_l.contains(&q)
                            && !status_match
                        {
                            return None;
                        }
                    }
                    let label_idx = a
                        .preserved
                        .get("labelColor")
                        .and_then(Value::as_u64)
                        .unwrap_or(0) as u8;
                    #[cfg(not(target_arch = "wasm32"))]
                    let has_disk_url = a
                        .url
                        .as_deref()
                        .map(str::trim)
                        .is_some_and(|u| !u.is_empty());
                    Some(AssetRowItem {
                        id: a.id.clone(),
                        filename: a.filename.clone(),
                        clip_kind: clip_kind_for_asset(a.kind, &a.filename),
                        category,
                        size: a.size,
                        missing: a.missing,
                        use_count: count,
                        label_idx,
                        folder,
                        #[cfg(not(target_arch = "wasm32"))]
                        has_disk_url,
                    })
                })
                .collect();

            match state.assets_sort_mode {
                1 => project_assets
                    .sort_by(|a, b| b.size.cmp(&a.size).then_with(|| a.filename.cmp(&b.filename))),
                2 => project_assets.sort_by(|a, b| {
                    b.use_count
                        .cmp(&a.use_count)
                        .then_with(|| a.filename.cmp(&b.filename))
                }),
                3 => project_assets.sort_by(|a, b| {
                    a.category
                        .label()
                        .cmp(b.category.label())
                        .then_with(|| a.filename.cmp(&b.filename))
                }),
                _ => project_assets.sort_by_key(|a| a.filename.to_ascii_lowercase()),
            }

            let shown_asset_count = project_assets.len();
            let is_flat_view = !q.is_empty()
                || mgr_state.kind_filter != AssetKindFilter::All
                || mgr_state.status_filter != AssetStatusFilter::All
                || mgr_state.folder_filter.is_some();
            let use_folder_groups =
                mgr_state.group_by_folder && !is_flat_view && !all_folders.is_empty();

            if project_assets.is_empty() && !use_folder_groups {
                Frame::NONE
                    .inner_margin(egui::Margin::symmetric(10, 8))
                    .show(ui, |ui| {
                        let empty_msg = if total_asset_count == 0 {
                            "No imported assets in project."
                        } else {
                            "No assets match the current filter."
                        };
                        ui.label(
                            egui::RichText::new(empty_msg)
                                .font(theme.font(FONT_SMALL))
                                .color(TEXT_EMPTY),
                        );
                    });
            } else {
                let render_asset_row = |ui: &mut Ui,
                                        session: &mut Session,
                                        mgr_state: &mut AssetsManagerUiState,
                                        actions: &mut Vec<PanelAction>,
                                        item: &AssetRowItem,
                                        indent_px: f32| {
                    let aid = &item.id;
                    let fname = &item.filename;
                    let clip_kind = item.clip_kind;
                    let size = item.size;
                    let missing = item.missing;
                    let use_count = item.use_count;
                    let label_idx = item.label_idx % (ASSET_LABEL_PALETTE.len() as u8);
                    let (label_name, label_col) = asset_label_entry(label_idx);
                    let is_selected = mgr_state.selected_ids.contains(aid);

                    let row_h = 44.0;
                    let (row_rect, resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), row_h),
                        Sense::click(),
                    );
                    let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                    if is_selected {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_OPTION_ACTIVE);
                        ui.painter().rect_filled(
                            Rect::from_min_size(row_rect.min, Vec2::new(2.0, row_h)),
                            CornerRadius::ZERO,
                            theme.accent(),
                        );
                    } else if resp.hovered() {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                    }
                    ui.painter().line_segment(
                        [row_rect.left_bottom(), row_rect.right_bottom()],
                        Stroke::new(1.0, BORDER_ROW_SOFT),
                    );

                    // Clickable 16-color inline label badge dot (click cycles, right-click row picks any of 16)
                    let dot_rect = Rect::from_center_size(
                        Pos2::new(row_rect.left() + 9.0 + indent_px, row_rect.center().y),
                        Vec2::splat(12.0),
                    );
                    let dot_resp = ui
                        .interact(dot_rect, Id::unique(("asset_label", aid)), Sense::click())
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text(format!(
                            "Label: {label_name} (click to cycle, right-click row for 16-color palette)"
                        ));
                    ui.painter().circle_filled(
                        dot_rect.center(),
                        if label_idx == 0 { 3.5 } else { 4.2 },
                        label_col,
                    );
                    if label_idx != 0 {
                        ui.painter().circle_stroke(
                            dot_rect.center(),
                            4.5,
                            Stroke::new(1.0, color32_with_alpha(TEXT_BRIGHT, 90)),
                        );
                    }
                    let mut action_consumed = false;
                    if dot_resp.clicked() {
                        action_consumed = true;
                        let next = (label_idx + 1) % (ASSET_LABEL_PALETTE.len() as u8);
                        let targets = if is_selected && mgr_state.selected_ids.len() > 1 {
                            mgr_state.selected_ids.iter().cloned().collect()
                        } else {
                            vec![aid.clone()]
                        };
                        let _ = set_assets_label_color_undoable(session, &targets, next);
                    }

                    // Crisp usage badge pill on every asset row
                    let (badge_text, badge_bg, badge_fg, badge_w) = if missing {
                        (
                            "MISSING".to_owned(),
                            Color32::from_rgb(0x6e, 0x22, 0x22),
                            TEXT_BRIGHT,
                            46.0_f32,
                        )
                    } else if use_count == 0 {
                        ("Unused".to_owned(), WARNING_SURFACE, WARNING_BAR, 44.0_f32)
                    } else {
                        (
                            format!("{use_count}×"),
                            color32_with_alpha(theme.accent(), 150),
                            TEXT_BRIGHT,
                            26.0_f32,
                        )
                    };

                    let btns_w = if use_count > 0 { 114.0 } else { 66.0 };
                    let badge_rect = Rect::from_min_size(
                        Pos2::new(
                            (row_rect.right() - btns_w - badge_w - 4.0)
                                .max(row_rect.left() + 20.0 + indent_px),
                            row_rect.top() + 5.0,
                        ),
                        Vec2::new(badge_w, 15.0),
                    );
                    let text_left = row_rect.left() + 19.0 + indent_px;
                    let name_right = (badge_rect.left() - 4.0).max(text_left + 16.0);
                    let meta_right = (row_rect.right() - btns_w - 2.0).max(text_left + 16.0);
                    let name_clip = Rect::from_min_max(
                        Pos2::new(text_left, row_rect.top()),
                        Pos2::new(name_right, row_rect.top() + 23.0),
                    );
                    let meta_clip = Rect::from_min_max(
                        Pos2::new(text_left, row_rect.top() + 20.0),
                        Pos2::new(meta_right, row_rect.bottom()),
                    );

                    let fname_col = if missing { DANGER_TEXT } else { TEXT_PRIMARY };
                    let fname_galley = layout_single_line_ellipsized(
                        ui.painter(),
                        fname,
                        theme.font(FONT_SMALL),
                        fname_col,
                        (name_right - text_left).max(16.0),
                    );
                    ui.painter().with_clip_rect(name_clip).galley(
                        Pos2::new(
                            text_left,
                            row_rect.top() + 13.0 - fname_galley.size().y * 0.5,
                        ),
                        fname_galley,
                        fname_col,
                    );
                    if badge_rect.left() > text_left + 12.0 {
                        paint_badge(ui.painter(), badge_rect, &badge_text, badge_bg, badge_fg);
                    }

                    let mut meta_parts = vec![item.category.label().to_owned()];
                    if size > 0 {
                        meta_parts.push(format_asset_size(size));
                    }
                    if !item.folder.is_empty() && !use_folder_groups {
                        meta_parts.push(format!("/{}", item.folder));
                    }
                    let meta = meta_parts.join(" · ");
                    let meta_col = if missing { DANGER_TEXT } else { TEXT_MUTED };
                    let meta_galley = layout_single_line_ellipsized(
                        ui.painter(),
                        &meta,
                        theme.font(FONT_COMPACT),
                        meta_col,
                        (meta_right - text_left).max(16.0),
                    );
                    ui.painter().with_clip_rect(meta_clip).galley(
                        Pos2::new(
                            text_left,
                            row_rect.top() + 30.0 - meta_galley.size().y * 0.5,
                        ),
                        meta_galley,
                        meta_col,
                    );

                    // Reveal Usage button when referenced
                    if use_count > 0 {
                        let rev_rect = Rect::from_min_size(
                            Pos2::new(row_rect.right() - 110.0, row_rect.center().y - 10.0),
                            Vec2::new(42.0, 20.0),
                        );
                        let rev_resp = ui
                            .interact(rev_rect, Id::unique(("asset_reveal", aid)), Sense::click())
                            .on_hover_cursor(CursorIcon::PointingHand)
                            .on_hover_text("Select first clip using this asset");
                        ui.painter().rect_filled(
                            rev_rect,
                            CornerRadius::same(2),
                            if rev_resp.hovered() {
                                theme.accent()
                            } else {
                                BG_BUTTON
                            },
                        );
                        ui.painter().text(
                            rev_rect.center(),
                            Align2::CENTER_CENTER,
                            "Find",
                            theme.font(FONT_COMPACT),
                            TEXT_BRIGHT,
                        );
                        if rev_resp.clicked() {
                            action_consumed = true;
                            reveal_asset_usage(session, aid, actions);
                        }
                    }

                    // Relink / Replace Footage button (always available, both missing and loaded!)
                    let relink_rect = Rect::from_min_size(
                        Pos2::new(row_rect.right() - 64.0, row_rect.center().y - 10.0),
                        Vec2::new(58.0, 20.0),
                    );
                    let relink_resp = ui
                        .interact(
                            relink_rect,
                            Id::unique(("asset_relink", aid)),
                            Sense::click(),
                        )
                        .on_hover_cursor(CursorIcon::PointingHand)
                        .on_hover_text(if missing {
                            "Relink missing footage file"
                        } else {
                            "Replace asset file in place"
                        });
                    ui.painter().rect_filled(
                        relink_rect,
                        CornerRadius::same(2),
                        if relink_resp.hovered() {
                            theme.accent()
                        } else {
                            BG_BUTTON
                        },
                    );
                    ui.painter().rect_stroke(
                        relink_rect,
                        CornerRadius::same(2),
                        Stroke::new(1.0, BORDER_CONTROL),
                        StrokeKind::Inside,
                    );
                    ui.painter().text(
                        relink_rect.center(),
                        Align2::CENTER_CENTER,
                        if missing { "Relink…" } else { "Replace" },
                        theme.font(FONT_COMPACT),
                        TEXT_BRIGHT,
                    );
                    if relink_resp.clicked() {
                        action_consumed = true;
                        trigger_replace_asset(session, aid, fname, actions);
                    }

                    // Right-click context menu: Timeline, Find, Reload, Replace, Export, Folder, 16-Color Label Swatches
                    let target_ids: Vec<String> = if is_selected && mgr_state.selected_ids.len() > 1
                    {
                        mgr_state.selected_ids.iter().cloned().collect()
                    } else {
                        vec![aid.clone()]
                    };
                    resp.context_menu(|ui| {
                        if item.category.is_timeline_media() && ui.button("Add to Timeline").clicked()
                        {
                            add_clip_with_asset(session, clip_kind, fname, Some(aid));
                            ui.close();
                        }
                        if use_count > 0 && ui.button("Find Usage").clicked() {
                            reveal_asset_usage(session, aid, actions);
                            ui.close();
                        }
                        #[cfg(not(target_arch = "wasm32"))]
                        {
                            let reload_btn = ui.add_enabled(
                                item.has_disk_url,
                                egui::Button::new("Reload from Disk"),
                            );
                            let reload_btn = if item.has_disk_url {
                                reload_btn.on_hover_text("Re-read asset bytes from its linked file path")
                            } else {
                                reload_btn.on_disabled_hover_text(
                                    "Embedded-only asset with no disk path — use Replace Asset…",
                                )
                            };
                            if reload_btn.clicked() {
                                match reload_asset_from_disk(session, aid) {
                                    Ok(msg) | Err(msg) => {
                                        actions.push(PanelAction::StatusMessage(msg));
                                    }
                                }
                                ui.close();
                            }
                        }
                        if ui
                            .button(if missing {
                                "Relink Missing File…"
                            } else {
                                "Replace Asset…"
                            })
                            .clicked()
                        {
                            trigger_replace_asset(session, aid, fname, actions);
                            ui.close();
                        }
                        if ui.button("Export File…").clicked() {
                            trigger_export_asset(session, aid, actions);
                            ui.close();
                        }
                        if !all_folders.is_empty() || !item.folder.is_empty() {
                            ui.separator();
                            ui.label(
                                egui::RichText::new("Move to Folder")
                                    .font(theme.font(FONT_COMPACT))
                                    .color(TEXT_MUTED),
                            );
                            if !item.folder.is_empty() && ui.button("Root (Unfiled)").clicked() {
                                let moved =
                                    move_assets_to_folder_undoable(session, &target_ids, None);
                                if moved > 0 {
                                    actions.push(PanelAction::StatusMessage(format!(
                                        "Moved {moved} asset(s) to Root"
                                    )));
                                }
                                ui.close();
                            }
                            for folder in &all_folders {
                                let label = if item.folder == *folder {
                                    format!("• {folder}")
                                } else {
                                    folder.clone()
                                };
                                if ui.button(label).clicked() {
                                    let moved = move_assets_to_folder_undoable(
                                        session,
                                        &target_ids,
                                        Some(folder.as_str()),
                                    );
                                    if moved > 0 {
                                        actions.push(PanelAction::StatusMessage(format!(
                                            "Moved {moved} asset(s) to folder `{folder}`"
                                        )));
                                    }
                                    ui.close();
                                }
                            }
                        }
                        ui.separator();
                        ui.label(
                            egui::RichText::new(format!("Color Label ({label_name})"))
                                .font(theme.font(FONT_COMPACT))
                                .color(TEXT_MUTED),
                        );
                        // 2×8 16-color swatch picker grid
                        for row in 0..2usize {
                            ui.horizontal(|ui| {
                                ui.spacing_mut().item_spacing = Vec2::new(3.0, 3.0);
                                for col in 0..8usize {
                                    let idx = (row * 8 + col) as u8;
                                    let (cname, ccol) = ASSET_LABEL_PALETTE[idx as usize];
                                    let (s_rect, s_resp) =
                                        ui.allocate_exact_size(Vec2::splat(16.0), Sense::click());
                                    let s_resp = s_resp
                                        .on_hover_cursor(CursorIcon::PointingHand)
                                        .on_hover_text(cname);
                                    ui.painter().rect_filled(s_rect, CornerRadius::same(2), ccol);
                                    let is_cur = idx == label_idx;
                                    if is_cur || s_resp.hovered() {
                                        ui.painter().rect_stroke(
                                            s_rect,
                                            CornerRadius::same(2),
                                            Stroke::new(
                                                if is_cur { 1.8 } else { 1.0 },
                                                TEXT_BRIGHT,
                                            ),
                                            StrokeKind::Inside,
                                        );
                                    }
                                    if idx == 0 {
                                        // Subtle slash inside "None" swatch
                                        ui.painter().line_segment(
                                            [
                                                Pos2::new(s_rect.left() + 3.0, s_rect.bottom() - 3.0),
                                                Pos2::new(s_rect.right() - 3.0, s_rect.top() + 3.0),
                                            ],
                                            Stroke::new(1.2, TEXT_MUTED),
                                        );
                                    }
                                    if s_resp.clicked() {
                                        let _ = set_assets_label_color_undoable(
                                            session,
                                            &target_ids,
                                            idx,
                                        );
                                        ui.close();
                                    }
                                }
                            });
                            if row == 0 {
                                ui.add_space(2.0);
                            }
                        }
                    });

                    if !action_consumed && resp.clicked() {
                        let mods = ui.input(|i| i.modifiers);
                        if mods.ctrl || mods.command {
                            if !mgr_state.selected_ids.remove(aid) {
                                mgr_state.selected_ids.insert(aid.clone());
                            }
                        } else {
                            mgr_state.selected_ids.clear();
                            mgr_state.selected_ids.insert(aid.clone());
                            if item.category.is_timeline_media() {
                                add_clip_with_asset(session, clip_kind, fname, Some(aid));
                            } else if item.category == AssetCategory::Font {
                                actions.push(PanelAction::StatusMessage(format!(
                                    "Font `{fname}` — choose it in a Text or Shape+ Font picker"
                                )));
                            } else if item.category == AssetCategory::Model3D {
                                actions.push(PanelAction::StatusMessage(format!(
                                    "3D Model `{fname}` — attach it to a 3D Object in Objects"
                                )));
                            }
                        }
                    }
                };

                if use_folder_groups {
                    // Render each named folder group first (including empty persistent folders), then root/unfiled items.
                    for folder in &all_folders {
                        let group_items: Vec<&AssetRowItem> = project_assets
                            .iter()
                            .filter(|it| it.folder == *folder)
                            .collect();
                        let collapsed = mgr_state.collapsed_folders.contains(folder);
                        let (hdr_rect, hdr_resp) = ui.allocate_exact_size(
                            Vec2::new(ui.available_width(), 24.0),
                            Sense::click(),
                        );
                        let hdr_resp = hdr_resp.on_hover_cursor(CursorIcon::PointingHand);
                        ui.painter().rect_filled(
                            hdr_rect,
                            CornerRadius::ZERO,
                            if hdr_resp.hovered() { BG_HOVER } else { BG_SUNKEN },
                        );
                        ui.painter().line_segment(
                            [hdr_rect.left_bottom(), hdr_rect.right_bottom()],
                            Stroke::new(1.0, BORDER_ROW_SOFT),
                        );
                        let chevron = if collapsed { "▸" } else { "▾" };
                        let lbl = format!("{chevron} {folder} ({})", group_items.len());
                        ui.painter().text(
                            Pos2::new(hdr_rect.left() + 8.0, hdr_rect.center().y),
                            Align2::LEFT_CENTER,
                            lbl,
                            theme.font(FONT_COMPACT),
                            TEXT_STRONG,
                        );
                        hdr_resp.context_menu(|ui| {
                            if ui.button("Delete Folder (keep assets)").clicked() {
                                let unfiled = delete_asset_folder_undoable(session, folder);
                                mgr_state.collapsed_folders.remove(folder);
                                if mgr_state.folder_filter.as_deref() == Some(folder.as_str()) {
                                    mgr_state.folder_filter = None;
                                }
                                let msg = if unfiled > 0 {
                                    format!(
                                        "Deleted folder `{folder}` ({unfiled} asset(s) moved to Root)"
                                    )
                                } else {
                                    format!("Deleted folder `{folder}`")
                                };
                                actions.push(PanelAction::StatusMessage(msg));
                                ui.close();
                            }
                        });
                        if hdr_resp.clicked() && !mgr_state.collapsed_folders.remove(folder) {
                            mgr_state.collapsed_folders.insert(folder.clone());
                        }
                        if !collapsed {
                            for item in group_items {
                                render_asset_row(
                                    ui,
                                    session,
                                    &mut mgr_state,
                                    actions,
                                    item,
                                    8.0,
                                );
                            }
                        }
                    }
                    for item in project_assets.iter().filter(|it| it.folder.is_empty()) {
                        render_asset_row(ui, session, &mut mgr_state, actions, item, 0.0);
                    }
                } else {
                    for item in &project_assets {
                        render_asset_row(ui, session, &mut mgr_state, actions, item, 0.0);
                    }
                }
            }

            // Asset Manager summary footer (items · unused count & bytes · missing count)
            if total_asset_count > 0 {
                let mut summary_parts = Vec::new();
                if shown_asset_count != total_asset_count {
                    summary_parts.push(format!("{shown_asset_count} of {total_asset_count} items"));
                } else {
                    summary_parts.push(format!(
                        "{total_asset_count} {}",
                        if total_asset_count == 1 {
                            "item"
                        } else {
                            "items"
                        }
                    ));
                }
                if unused_asset_count > 0 {
                    if unused_total_bytes > 0 {
                        summary_parts.push(format!(
                            "{unused_asset_count} unused ({})",
                            format_asset_size(unused_total_bytes)
                        ));
                    } else {
                        summary_parts.push(format!("{unused_asset_count} unused"));
                    }
                }
                if missing_asset_count > 0 {
                    summary_parts.push(format!("{missing_asset_count} missing"));
                }
                let summary_line = summary_parts.join(" · ");
                let (sum_rect, _) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 22.0), Sense::hover());
                ui.painter()
                    .rect_filled(sum_rect, CornerRadius::ZERO, BG_SUNKEN);
                ui.painter().line_segment(
                    [sum_rect.left_bottom(), sum_rect.right_bottom()],
                    Stroke::new(1.0, BORDER_ROW_SOFT),
                );
                let sum_galley = layout_single_line_ellipsized(
                    ui.painter(),
                    &summary_line,
                    theme.font(FONT_COMPACT),
                    TEXT_MUTED,
                    (sum_rect.width() - 14.0).max(20.0),
                );
                ui.painter().galley(
                    Pos2::new(
                        sum_rect.left() + 8.0,
                        sum_rect.center().y - sum_galley.size().y * 0.5,
                    ),
                    sum_galley,
                    TEXT_MUTED,
                );
            }

            ui.data_mut(|d| d.insert_temp(ui_state_id, mgr_state));

            // Built-in starter templates / presets list
            let (tpl_hdr, _) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 27.0), Sense::hover());
            ui.painter()
                .rect_filled(tpl_hdr, CornerRadius::ZERO, BG_TITLE);
            ui.painter().line_segment(
                [tpl_hdr.left_bottom(), tpl_hdr.right_bottom()],
                Stroke::new(1.0, BORDER_ROW_SOFT),
            );
            ui.painter().text(
                Pos2::new(tpl_hdr.left() + 10.0, tpl_hdr.center().y),
                Align2::LEFT_CENTER,
                "Starter templates & presets",
                theme.font(FONT_SMALL),
                TEXT_STRONG,
            );

            let filtered_presets: Vec<&MediaPreset> = MEDIA_PRESETS
                .iter()
                .filter(|p| {
                    p.required_plugin
                        .is_none_or(|pid| session.plugins.is_enabled(pid))
                        && (q.is_empty()
                            || p.title.to_ascii_lowercase().contains(&q)
                            || p.subtitle.to_ascii_lowercase().contains(&q))
                })
                .collect();
            for preset in filtered_presets {
                let (row_rect, resp) =
                    ui.allocate_exact_size(Vec2::new(ui.available_width(), 44.0), Sense::click());
                let resp = resp.on_hover_cursor(CursorIcon::PointingHand);
                let row_bg = if resp.hovered() { BG_HOVER } else { BG_PANEL };
                ui.painter()
                    .rect_filled(row_rect, CornerRadius::ZERO, row_bg);
                ui.painter().line_segment(
                    [row_rect.left_bottom(), row_rect.right_bottom()],
                    Stroke::new(1.0, BORDER_ROW_SOFT),
                );
                let icon_rect = Rect::from_min_size(
                    Pos2::new(row_rect.left() + 10.0, row_rect.center().y - 11.0),
                    Vec2::splat(22.0),
                );
                paint_media_card_icon(ui.painter(), icon_rect, preset.icon, TEXT_PRIMARY, row_bg);
                let tx = icon_rect.right() + 10.0;
                let max_text_w = (row_rect.right() - tx - 8.0).max(24.0);
                let text_clip = Rect::from_min_max(
                    Pos2::new(tx, row_rect.top()),
                    Pos2::new((row_rect.right() - 6.0).max(tx), row_rect.bottom()),
                );
                let title_col = if resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_PRIMARY
                };
                let title_galley = layout_single_line_ellipsized(
                    ui.painter(),
                    preset.title,
                    theme.font(FONT_SMALL),
                    title_col,
                    max_text_w,
                );
                ui.painter().with_clip_rect(text_clip).galley(
                    Pos2::new(tx, row_rect.top() + 14.0 - title_galley.size().y * 0.5),
                    title_galley,
                    title_col,
                );
                let sub_galley = layout_single_line_ellipsized(
                    ui.painter(),
                    preset.subtitle,
                    theme.font(FONT_COMPACT),
                    TEXT_OPTION_DESC,
                    max_text_w,
                );
                ui.painter().with_clip_rect(text_clip).galley(
                    Pos2::new(tx, row_rect.top() + 31.0 - sub_galley.size().y * 0.5),
                    sub_galley,
                    TEXT_OPTION_DESC,
                );
                if resp.clicked() {
                    add_preset_clip(session, preset.kind, preset.title);
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_asset_record(id: &str, filename: &str, bytes: &[u8]) -> zc_project::AssetRecord {
        zc_project::AssetRecord {
            id: id.to_owned(),
            kind: AssetKind::Image,
            source: zc_project::AssetSource::File,
            filename: filename.to_owned(),
            url: None,
            external: false,
            bytes: Some(bytes.to_vec()),
            missing: false,
            sha256: zc_project::sha256_hex(bytes),
            size: bytes.len() as u64,
            preserved: BTreeMap::new(),
        }
    }

    fn make_test_video_clip(
        name: &str,
        media: Option<&str>,
        layer: zc_project::Layer,
    ) -> zc_project::VideoClip {
        zc_project::VideoClip {
            id: Uuid::new_v4(),
            name: name.to_owned(),
            start: 0.0,
            length: 60.0,
            link: None,
            media: media.map(str::to_owned),
            time_curve: PropertyTrack::linear_time_curve(60.0, 30),
            layer,
            preserved: BTreeMap::new(),
        }
    }

    #[test]
    fn rewrite_asset_id_across_project_updates_every_reference_kind_with_zero_dangling() {
        let old_id = "asset-old-11111111-2222-4333-8444-555555555555";
        let new_id = "asset-new-aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee";

        let mut project = zc_project::Project::new_cm3_default();
        project.media.push(zc_project::MediaItem {
            id: Uuid::new_v4(),
            name: "dup.png".to_owned(),
            asset_id: Some(old_id.to_owned()),
            kind: AssetKind::Image,
            preserved: BTreeMap::new(),
        });

        // Audio clip in main sequence
        project.sequence.audio_tracks[0]
            .clips
            .push(zc_project::AudioClip {
                id: Uuid::new_v4(),
                name: "Audio".to_owned(),
                start: 0.0,
                length: 60.0,
                link: None,
                media: Some(old_id.to_owned()),
                time_curve: PropertyTrack::linear_time_curve(60.0, 30),
                audio: zc_project::AudioClipData::default(),
                preserved: BTreeMap::new(),
            });

        // 2D Image, Video, Text2D, Shape2D, Plugin, and Composite layers + Effects (Asset, Text, CustomShader, EffectGroup)
        let mut img_layer = zc_project::Layer::new(
            "Img",
            LayerKind::Image {
                image: Some(old_id.to_owned()),
            },
            [1920, 1080],
        );
        let mut fx_params = BTreeMap::new();
        fx_params.insert(
            "lut".to_owned(),
            PropertyTrack::asset(Some(old_id.to_owned())),
        );
        fx_params.insert("overlayId".to_owned(), PropertyTrack::text(old_id));
        let shader_cp = CustomProperty {
            name: "uTex".to_owned(),
            prop_type: Some(8),
            track: PropertyTrack::asset(Some(old_id.to_owned())),
            metadata: BTreeMap::new(),
        };
        let child_shader_fx = EffectInstance {
            id: Uuid::new_v4(),
            name: "ChildShader".to_owned(),
            kind: EffectKind::CustomShader {
                frag_shader: "void main(){}".to_owned(),
                custom_properties: vec![shader_cp.clone()],
            },
            enabled: PropertyTrack::option(1),
            params: fx_params.clone(),
            preserved: BTreeMap::new(),
        };
        img_layer.effects.push(EffectInstance {
            id: Uuid::new_v4(),
            name: "GroupFx".to_owned(),
            kind: EffectKind::EffectGroup {
                custom_properties: vec![shader_cp],
                children: vec![child_shader_fx],
            },
            enabled: PropertyTrack::option(1),
            params: fx_params,
            preserved: BTreeMap::new(),
        });
        project.sequence.video_tracks[0]
            .clips
            .push(make_test_video_clip("ImgClip", Some(old_id), img_layer));

        let vid_layer = zc_project::Layer::new(
            "Vid",
            LayerKind::Video {
                media: Some(old_id.to_owned()),
            },
            [1920, 1080],
        );
        project.sequence.video_tracks[0]
            .clips
            .push(make_test_video_clip("VidClip", Some(old_id), vid_layer));

        let text_layer = zc_project::Layer::new(
            "Txt2D",
            LayerKind::Text2D {
                text: "Hi".to_owned(),
                font: Some(old_id.to_owned()),
                size: 48,
                shapes: Vec::new(),
            },
            [1920, 1080],
        );
        project.sequence.video_tracks[0]
            .clips
            .push(make_test_video_clip("TxtClip", None, text_layer));

        let shape_layer = zc_project::Layer::new(
            "Shape2D",
            LayerKind::Shape2D {
                shapes: vec![zc_project::Shape2DNode {
                    id: Uuid::new_v4(),
                    name: "Path".to_owned(),
                    kind: zc_project::Shape2DKind::Path {
                        path: json!({ "textureAsset": old_id }),
                    },
                    position: PropertyTrack::vec2([0.0, 0.0]),
                    rotation: PropertyTrack::number(0.0),
                    scale: PropertyTrack::vec2([1.0, 1.0]),
                    operations: Vec::new(),
                    preserved: BTreeMap::new(),
                }],
            },
            [1920, 1080],
        );
        let plugin_layer = zc_project::Layer::new(
            "Plugin",
            LayerKind::Plugin {
                plugin_type: "zoidium:shape-plus/shape-layer".to_owned(),
                payload: json!({ "items": [{ "font": old_id, "embedded": format!("ref:{old_id}") }] }),
            },
            [1920, 1080],
        );
        let composite_layer = zc_project::Layer::new(
            "Comp",
            LayerKind::Composite {
                layers: vec![shape_layer, plugin_layer],
            },
            [1920, 1080],
        );
        project.sequence.video_tracks[0]
            .clips
            .push(make_test_video_clip("CompClip", None, composite_layer));

        // 3D objects (nested inside Group) + ALL material types & NodeMaterial graph JSON + preserved IES
        let mut text_3d = Object3D::new(
            "Text3D",
            Object3DKind::Text3D(Box::new(zc_project::Text3DData {
                font: Some(old_id.to_owned()),
                ..Default::default()
            })),
        );
        text_3d.material = Some(MaterialSpec::Texture(Box::new(
            zc_project::TextureMaterial {
                texture: Some(old_id.to_owned()),
                ..Default::default()
            },
        )));
        text_3d.material_slots.insert(
            "side".to_owned(),
            MaterialSpec::ImagePlus(Box::new(zc_project::ImagePlusMaterial {
                texture: Some(old_id.to_owned()),
                ..Default::default()
            })),
        );
        text_3d.material_slots.insert(
            "bevel".to_owned(),
            MaterialSpec::CustomPbr(Box::new(zc_project::CustomPbrMaterial {
                texture: Some(old_id.to_owned()),
                normal_map: Some(old_id.to_owned()),
                ..Default::default()
            })),
        );

        let mut model_3d = Object3D::new(
            "Model3D",
            Object3DKind::Model(Box::new(zc_project::ModelData {
                preset_type: None,
                asset_id: Some(old_id.to_owned()),
                model_properties: BTreeMap::new(),
            })),
        );
        model_3d.material = Some(MaterialSpec::PbrPlus(Box::new(
            zc_project::PbrPlusMaterial {
                base_color_map: Some(old_id.to_owned()),
                emissive_map: Some(old_id.to_owned()),
                roughness_map: Some(old_id.to_owned()),
                metalness_map: Some(old_id.to_owned()),
                normal_map: Some(old_id.to_owned()),
                alpha_map: Some(old_id.to_owned()),
                ao_texture: Some(old_id.to_owned()),
                ..Default::default()
            },
        )));
        model_3d.material_slots.insert(
            "shader".to_owned(),
            MaterialSpec::CustomShader {
                frag_shader: "void main(){}".to_owned(),
                custom_properties: vec![CustomProperty {
                    name: "map".to_owned(),
                    prop_type: Some(8),
                    track: PropertyTrack::asset(Some(old_id.to_owned())),
                    metadata: BTreeMap::new(),
                }],
            },
        );
        model_3d.material_slots.insert(
            "nodes".to_owned(),
            MaterialSpec::NodeMaterial {
                schema_version: 1,
                resolution: 512,
                graph: json!({
                    "nodes": [
                        { "id": "n1", "type": "image_texture", "params": { "assetId": old_id } }
                    ]
                }),
            },
        );

        let mut custom_geo = Object3D::new(
            "CustomGeo",
            Object3DKind::Shape3D(zc_project::Shape3DKind::CustomGeometry {
                asset_id: Some(old_id.to_owned()),
            }),
        );
        custom_geo.material = Some(MaterialSpec::Matcap(Box::new(zc_project::MatcapMaterial {
            texture: Some(old_id.to_owned()),
            ..Default::default()
        })));
        custom_geo
            .preserved
            .insert("property.iesProfile".to_owned(), json!(old_id));

        let mut particles = Object3D::new(
            "Particles",
            Object3DKind::Particles(Box::new(zc_project::ParticlesData {
                texture: Some(old_id.to_owned()),
                ..Default::default()
            })),
        );
        particles.material = Some(MaterialSpec::Video(Box::new(zc_project::VideoMaterial {
            video: Some(old_id.to_owned()),
            ..Default::default()
        })));

        let group = Object3D::new(
            "Group",
            Object3DKind::Group {
                reflection_visibility: 0,
                custom_properties: vec![CustomProperty {
                    name: "groupTex".to_owned(),
                    prop_type: Some(8),
                    track: PropertyTrack::asset(Some(old_id.to_owned())),
                    metadata: BTreeMap::new(),
                }],
                children: vec![text_3d, model_3d, custom_geo, particles],
            },
        );

        if let LayerKind::Scene3D { objects, .. } =
            &mut project.sequence.video_tracks[0].clips[0].layer.kind
        {
            objects.push(group);
        }

        // Nested composition referencing old_id in both video and audio clips
        let mut comp = zc_project::Sequence::default();
        comp.video_tracks.push(zc_project::VideoTrack {
            id: Uuid::new_v4(),
            name: "V1".to_owned(),
            enabled: true,
            solo: false,
            locked: false,
            is_3d: false,
            clips: vec![make_test_video_clip(
                "CompImg",
                Some(old_id),
                zc_project::Layer::new(
                    "CompImg",
                    LayerKind::Image {
                        image: Some(old_id.to_owned()),
                    },
                    [1920, 1080],
                ),
            )],
            preserved: BTreeMap::new(),
        });
        comp.audio_tracks.push(zc_project::AudioTrack {
            id: Uuid::new_v4(),
            name: "A1".to_owned(),
            enabled: true,
            muted: false,
            solo: false,
            locked: false,
            clips: vec![zc_project::AudioClip {
                id: Uuid::new_v4(),
                name: "CompAudio".to_owned(),
                start: 0.0,
                length: 30.0,
                link: None,
                media: Some(old_id.to_owned()),
                time_curve: PropertyTrack::linear_time_curve(30.0, 30),
                audio: zc_project::AudioClipData::default(),
                preserved: BTreeMap::new(),
            }],
            preserved: BTreeMap::new(),
        });
        let comp_id = comp.id;
        project.compositions.insert(comp_id, comp);

        let rewritten = rewrite_asset_id_across_project(&mut project, old_id, new_id);
        assert!(
            rewritten >= 25,
            "expected all asset reference kinds to be rewritten, got {rewritten}"
        );

        // Strongest invariant: zero occurrences of `old_id` remain anywhere in serialized Project JSON.
        let serialized = serde_json::to_string(&project).expect("serialize rewritten project");
        assert!(
            !serialized.contains(old_id),
            "dangling reference to old_id survived rewrite"
        );
        assert!(serialized.contains(new_id));
        assert_eq!(project.media[0].asset_id.as_deref(), Some(new_id));
        assert_eq!(
            project.sequence.audio_tracks[0].clips[0].media.as_deref(),
            Some(new_id)
        );
        assert_eq!(
            project.compositions[&comp_id].audio_tracks[0].clips[0]
                .media
                .as_deref(),
            Some(new_id)
        );
    }

    #[test]
    fn consolidate_duplicate_assets_rewrites_project_and_document_and_undoes_cleanly() {
        let mut session = Session::new();
        let payload = b"identical-texture-payload-bytes".to_vec();
        let keep_id = "asset-a-canonical-0000000000000001".to_owned();
        let dup_id = "asset-b-duplicate-0000000000000002".to_owned();

        session.project.assets.insert(
            keep_id.clone(),
            make_test_asset_record(&keep_id, "tex.png", &payload),
        );
        session.project.assets.insert(
            dup_id.clone(),
            make_test_asset_record(&dup_id, "tex_copy.png", &payload),
        );
        session.project.media.push(zc_project::MediaItem {
            id: Uuid::new_v4(),
            name: "tex_copy.png".to_owned(),
            asset_id: Some(dup_id.clone()),
            kind: AssetKind::Image,
            preserved: BTreeMap::new(),
        });

        // Add an image clip referencing dup_id via Session commands so both document and project hold it.
        let vtrack_id = session.document().video_tracks[0].id.clone();
        let add_res = session
            .execute(Command::AddClip {
                track_id: vtrack_id,
                kind: "image".to_owned(),
                name: Some("DupImage".to_owned()),
                start: 180.0,
                length: 60.0,
            })
            .expect("add image clip");
        let clip_id = add_res["clip_id"].as_str().expect("clip_id").to_owned();
        session
            .execute(Command::SetPropertyValue {
                target_id: clip_id.clone(),
                property: "assetId".to_owned(),
                value: json!(dup_id),
            })
            .expect("assign dup_id");

        let merged = consolidate_duplicate_assets(&mut session);
        assert_eq!(merged, 1);
        assert!(session.project.assets.contains_key(&keep_id));
        assert!(!session.project.assets.contains_key(&dup_id));
        assert_eq!(
            session.project.media[0].asset_id.as_deref(),
            Some(keep_id.as_str())
        );

        // Both SessionDocument and Project must point to keep_id!
        let doc_clip = session.document().video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id == clip_id)
            .expect("doc clip");
        assert_eq!(
            doc_clip
                .properties
                .get("assetId")
                .and_then(|p| p.keyframes.first())
                .and_then(|k| k.value.as_str()),
            Some(keep_id.as_str())
        );
        let serialized = serde_json::to_string(&session.project).expect("serialize project");
        assert!(!serialized.contains(&dup_id));

        // Undo must restore dup_id (including bytes) and original reference!
        session.execute(Command::Undo).expect("undo consolidate");
        assert!(session.project.assets.contains_key(&dup_id));
        assert_eq!(
            session.project.assets[&dup_id].bytes.as_deref(),
            Some(payload.as_slice())
        );
        let doc_clip_after_undo = session.document().video_tracks[0]
            .clips
            .iter()
            .find(|c| c.id == clip_id)
            .expect("doc clip after undo");
        assert_eq!(
            doc_clip_after_undo
                .properties
                .get("assetId")
                .and_then(|p| p.keyframes.first())
                .and_then(|k| k.value.as_str()),
            Some(dup_id.as_str())
        );

        // Redo must re-apply consolidation cleanly!
        session.execute(Command::Redo).expect("redo consolidate");
        assert!(!session.project.assets.contains_key(&dup_id));
        assert!(session.project.assets.contains_key(&keep_id));
    }

    #[test]
    fn asset_label_palette_has_16_distinct_colors_and_undoable_assignment() {
        assert_eq!(ASSET_LABEL_PALETTE.len(), 16);
        let unique_names: BTreeSet<&str> = ASSET_LABEL_PALETTE.iter().map(|(n, _)| *n).collect();
        let unique_colors: BTreeSet<[u8; 4]> = ASSET_LABEL_PALETTE
            .iter()
            .map(|(_, c)| c.to_array())
            .collect();
        assert_eq!(unique_names.len(), 16);
        assert_eq!(unique_colors.len(), 16);

        let mut session = Session::new();
        let aid = session
            .project
            .insert_binary_asset(AssetKind::Image, "label.png", vec![1, 2, 3]);
        session.mark_saved();

        let updated = set_assets_label_color_undoable(&mut session, std::slice::from_ref(&aid), 13);
        assert_eq!(updated, 1);
        assert!(session.is_dirty());
        assert_eq!(
            session.project.assets[&aid]
                .preserved
                .get("labelColor")
                .and_then(Value::as_u64),
            Some(13)
        );

        session.execute(Command::Undo).expect("undo label color");
        assert!(!session.is_dirty());
        assert!(
            !session.project.assets[&aid]
                .preserved
                .contains_key("labelColor")
        );
    }

    #[test]
    fn empty_asset_folders_persist_across_save_reload_and_undo_cleanly() {
        let mut session = Session::new();
        session.mark_saved();

        // Create an empty folder (no assets moved)
        let moved = create_asset_folder_undoable(&mut session, "SFX", &[]);
        assert_eq!(moved, 0);
        assert!(session.is_dirty());
        assert_eq!(
            collect_project_asset_folders(&session.project),
            vec!["SFX".to_owned()]
        );

        // Roundtrip through .zcraft archive to prove empty folder survives save & reload
        let archive = zc_project::save_zcraft_to_bytes(&session.project).expect("save zcraft");
        let reloaded = zc_project::load_zcraft_from_bytes(&archive).expect("load zcraft");
        assert_eq!(
            collect_project_asset_folders(&reloaded),
            vec!["SFX".to_owned()]
        );

        // Undo removes the empty folder cleanly
        session.execute(Command::Undo).expect("undo create folder");
        assert!(!session.is_dirty());
        assert!(collect_project_asset_folders(&session.project).is_empty());

        // Redo restores it; deleting unfiles assets & removes registry entry undoably
        session.execute(Command::Redo).expect("redo create folder");
        let aid = session
            .project
            .insert_binary_asset(AssetKind::Av, "boom.wav", vec![9, 8, 7]);
        assert_eq!(
            move_assets_to_folder_undoable(&mut session, std::slice::from_ref(&aid), Some("SFX")),
            1
        );
        assert_eq!(
            session.project.assets[&aid]
                .preserved
                .get("folder")
                .and_then(Value::as_str),
            Some("SFX")
        );
        assert_eq!(delete_asset_folder_undoable(&mut session, "SFX"), 1);
        assert!(collect_project_asset_folders(&session.project).is_empty());
        assert!(
            !session.project.assets[&aid]
                .preserved
                .contains_key("folder")
        );

        session.execute(Command::Undo).expect("undo delete folder");
        assert_eq!(
            collect_project_asset_folders(&session.project),
            vec!["SFX".to_owned()]
        );
        assert_eq!(
            session.project.assets[&aid]
                .preserved
                .get("folder")
                .and_then(Value::as_str),
            Some("SFX")
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn reload_asset_from_disk_and_collect_files_bundle_work_end_to_end() {
        let tmp_root = std::env::temp_dir().join(format!("zc_asset_mgr_test_{}", Uuid::new_v4()));
        std::fs::create_dir_all(&tmp_root).expect("create temp root");
        let src_file = tmp_root.join("logo.png");
        std::fs::write(&src_file, b"v1-bytes").expect("write v1");

        let mut session = Session::new();
        let aid = session.project.insert_binary_asset_prehashed(
            AssetKind::Image,
            "logo.png",
            Some(src_file.to_string_lossy().into_owned()),
            b"v1-bytes".to_vec(),
            zc_project::sha256_hex(b"v1-bytes"),
        );

        // Modify source file on disk and reload
        std::fs::write(&src_file, b"v2-updated-bytes").expect("write v2");
        let msg = reload_asset_from_disk(&mut session, &aid).expect("reload from disk");
        assert!(msg.contains("logo.png"));
        assert_eq!(
            session.project.assets[&aid].bytes.as_deref(),
            Some(b"v2-updated-bytes".as_slice())
        );

        // Collect Files into a self-contained directory
        let bundle_dir = tmp_root.join("bundle");
        let report = collect_project_files_to_dir(&session, &bundle_dir).expect("collect files");
        assert_eq!(report.collected_count, 1);
        assert_eq!(report.missing_count, 0);
        assert!(report.project_path.is_file());
        assert!(report.report_path.is_file());
        assert_eq!(
            std::fs::read(bundle_dir.join("assets/logo.png")).expect("read collected asset"),
            b"v2-updated-bytes"
        );
        let report_txt = std::fs::read_to_string(&report.report_path).expect("read report");
        assert!(report_txt.contains("logo.png"));
        assert!(report_txt.contains("assets/logo.png"));

        let _ = std::fs::remove_dir_all(&tmp_root);
    }
}
