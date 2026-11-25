use super::objects_2d::show_non_scene_objects_tab;
use super::objects_inspector::show_object_inspector;
use super::property_rows::find_clip_cached;
use super::shared_widgets::{empty_panel_message, paint_crisp_panel_border};
use super::state::{
    InlineRenameState, TreeDragState, TreeDropMode, commit_inspector_tx_if_open,
    copy_pz_json_to_clipboard, header_compact_action_button, paste_pz_json_from_clipboard,
};
use super::*;

// ---------------------------------------------------------------------------
// Tab 4: Objects (objects) — 40/60 Auto-Split 3D Tree + Inspector
// ---------------------------------------------------------------------------

pub(super) fn build_object_picker_items(session: &Session) -> Vec<FuzzyItem> {
    let mut items = vec![
        // Core 10 CM3/Zoidium Objects (exact Zoidium OBJECTS 10 ordering & descriptions)
        FuzzyItem {
            id: "box".into(),
            category: "Objects".into(),
            name: "Shape".into(),
            desc: "Primitive object such as a box or sphere.".into(),
        },
        FuzzyItem {
            id: "text".into(),
            category: "Objects".into(),
            name: "Text".into(),
            desc: "Custom 3D text.".into(),
        },
        FuzzyItem {
            id: "spot_light".into(),
            category: "Objects".into(),
            name: "Light".into(),
            desc: "Creates a Spot Light with shadows in a scene.".into(),
        },
        FuzzyItem {
            id: "group".into(),
            category: "Objects".into(),
            name: "Group".into(),
            desc: "Group multiple objects together.".into(),
        },
        FuzzyItem {
            id: "camera".into(),
            category: "Objects".into(),
            name: "Camera".into(),
            desc: "Perspective or orthographic camera.".into(),
        },
        FuzzyItem {
            id: "particles".into(),
            category: "Objects".into(),
            name: "Particles".into(),
            desc: "3D particle emitter.".into(),
        },
        FuzzyItem {
            id: "model".into(),
            category: "Objects".into(),
            name: "3D Model".into(),
            desc: "Imported OBJ/GLTF mesh.".into(),
        },
        FuzzyItem {
            id: "sphere".into(),
            category: "Objects".into(),
            name: "Sphere".into(),
            desc: "UV sphere mesh.".into(),
        },
        FuzzyItem {
            id: "cylinder".into(),
            category: "Objects".into(),
            name: "Cylinder".into(),
            desc: "Cylinder or cone primitive.".into(),
        },
        FuzzyItem {
            id: "donut".into(),
            category: "Objects".into(),
            name: "Donut".into(),
            desc: "Torus ring primitive.".into(),
        },
        // Additional CM3 Primitives & Lights
        FuzzyItem {
            id: "rectangle".into(),
            category: "More Objects".into(),
            name: "Rectangle".into(),
            desc: "Flat 3D plane.".into(),
        },
        FuzzyItem {
            id: "circle".into(),
            category: "More Objects".into(),
            name: "Circle".into(),
            desc: "Flat 3D disc.".into(),
        },
        FuzzyItem {
            id: "wire".into(),
            category: "More Objects".into(),
            name: "Wire".into(),
            desc: "3D polyline tube.".into(),
        },
        FuzzyItem {
            id: "point_light".into(),
            category: "More Objects".into(),
            name: "Point Light".into(),
            desc: "Omnidirectional point light.".into(),
        },
        FuzzyItem {
            id: "directional_light".into(),
            category: "More Objects".into(),
            name: "Directional Light".into(),
            desc: "Parallel sun-like light.".into(),
        },
        FuzzyItem {
            id: "hemisphere_light".into(),
            category: "More Objects".into(),
            name: "Hemisphere Light".into(),
            desc: "Sky & ground ambient light.".into(),
        },
        FuzzyItem {
            id: "ambient_light".into(),
            category: "More Objects".into(),
            name: "Ambient Light".into(),
            desc: "Uniform omnidirectional scene fill light.".into(),
        },
    ];

    if session.plugins.is_enabled("geometry-plus") {
        for (id, name, desc) in [
            (
                "zoidium:geometry-plus/rounded-box",
                "Rounded Box",
                "Beveled procedural box",
            ),
            (
                "zoidium:geometry-plus/polyhedron",
                "Polyhedron",
                "Parametric platonic polyhedron",
            ),
            (
                "zoidium:geometry-plus/tetrahedron",
                "Tetrahedron",
                "Low-poly 4-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/octahedron",
                "Octahedron",
                "Low-poly 8-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/dodecahedron",
                "Dodecahedron",
                "Low-poly 12-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/icosahedron",
                "Icosahedron",
                "Low-poly 20-face polyhedron",
            ),
            (
                "zoidium:geometry-plus/cone",
                "Cone",
                "Procedural cone primitive",
            ),
            (
                "zoidium:geometry-plus/capsule",
                "Capsule",
                "Procedural pill mesh",
            ),
            ("zoidium:geometry-plus/tube", "Tube", "Swept 3D spline tube"),
            (
                "zoidium:geometry-plus/gear",
                "Gear",
                "Procedural involute toothed gear",
            ),
            (
                "zoidium:geometry-plus/helix",
                "Helix",
                "3D spiral coil mesh",
            ),
            (
                "zoidium:geometry-plus/spline",
                "Spline",
                "Smooth 3D control-point curve",
            ),
            (
                "zoidium:geometry-plus/torus-knot",
                "Torus Knot",
                "Procedural mathematical knot",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Geometry+".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    if session.plugins.is_enabled("light-plus") {
        items.push(FuzzyItem {
            id: "zoidium:light-plus/light_rig".into(),
            category: "Light+".into(),
            name: "Light+ Rig".into(),
            desc: "Extended Spot/Point/Directional/Hemi/Ambient chooser".into(),
        });
    }

    if session.plugins.is_enabled("repeater") {
        for (id, name, desc) in [
            (
                "zoidium:repeater/repeater",
                "Repeater",
                "Procedural child instancer",
            ),
            (
                "zoidium:repeater/linear-repeater",
                "Linear Repeater",
                "Linear array instancer",
            ),
            (
                "zoidium:repeater/random-repeater",
                "Random Repeater",
                "Seeded random instancer",
            ),
            (
                "zoidium:repeater/echo-repeater",
                "Echo Repeater",
                "Temporal ghost instancer",
            ),
            (
                "zoidium:repeater/radial",
                "Radial Repeater",
                "Circular ring child instancer",
            ),
            (
                "zoidium:repeater/grid",
                "Grid Repeater",
                "3D XYZ grid child instancer",
            ),
            (
                "zoidium:repeater/twist",
                "Twist Deformer",
                "Axial twist mesh deformer",
            ),
            (
                "zoidium:repeater/warp",
                "Warp Deformer",
                "Bend & shear mesh deformer",
            ),
            (
                "zoidium:repeater/voronoi-fracture",
                "Voronoi Fracture",
                "3D cell fracture deformer",
            ),
            (
                "zoidium:repeater/plain",
                "Plain Effector",
                "Falloff-driven transform effector",
            ),
            (
                "zoidium:repeater/delay",
                "Delay Effector",
                "Spring & temporal smoothing effector",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Effector".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    if session.plugins.is_enabled("text-plus") {
        for (id, name, desc) in [
            (
                "zoidium:text-plus/text-plus",
                "Text+ Group",
                "Per-character 3D text animator container",
            ),
            (
                "zoidium:text-plus/gradient-transform",
                "Gradient Transform",
                "Linear gradient per-character offset",
            ),
            (
                "zoidium:text-plus/character-shake",
                "Character Shake",
                "Procedural per-character shake",
            ),
            (
                "zoidium:text-plus/selected-transform",
                "Selected Transform",
                "Index-range character transform",
            ),
            (
                "zoidium:text-plus/selected-shake",
                "Selected Shake",
                "Index-range character shake",
            ),
            (
                "zoidium:text-plus/random-scatter",
                "Random Scatter",
                "Seeded 3D character scatter",
            ),
            (
                "zoidium:text-plus/character-material",
                "Character Material",
                "Per-character material override",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Text+".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    if session.plugins.is_enabled("expresstext-plus") {
        for (id, name, desc) in [
            (
                "zoidium:expresstext-plus/expresstext",
                "ExpressText+ Container",
                "Expression-driven kinetic 3D text container",
            ),
            (
                "zoidium:expresstext-plus/shake",
                "ExpressText Shake",
                "Per-character expression shake",
            ),
            (
                "zoidium:expresstext-plus/plain",
                "ExpressText Plain",
                "Per-character plain effector",
            ),
            (
                "zoidium:expresstext-plus/random",
                "ExpressText Random",
                "Seeded random kinetic modifier",
            ),
            (
                "zoidium:expresstext-plus/formula",
                "ExpressText Formula",
                "Math expression per-character modifier",
            ),
            (
                "zoidium:expresstext-plus/delay",
                "ExpressText Delay",
                "Spring & lag character modifier",
            ),
            (
                "zoidium:expresstext-plus/spline",
                "ExpressText Spline",
                "Align characters along a 3D spline",
            ),
            (
                "zoidium:expresstext-plus/orbit",
                "ExpressText Orbit",
                "Circular orbital character layout",
            ),
            (
                "zoidium:expresstext-plus/repeater",
                "ExpressText Repeater",
                "Repeat text glyph copies",
            ),
            (
                "zoidium:expresstext-plus/twist",
                "ExpressText Twist",
                "Twist character orientations",
            ),
            (
                "zoidium:expresstext-plus/wrap",
                "ExpressText Wrap",
                "Cylindrical/spherical text wrap",
            ),
            (
                "zoidium:expresstext-plus/voronoi",
                "ExpressText Voronoi",
                "Shatter & explode character clusters",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "ExpressText+".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    if session.plugins.is_enabled("camera-plus") {
        items.push(FuzzyItem {
            id: "zoidium:camera-plus/camera".into(),
            category: "Camera+".into(),
            name: "Camera+".into(),
            desc: "Film gate, DOF, motion blur & vibrate".into(),
        });
    }

    if session.plugins.is_enabled("optical-flares") {
        items.push(FuzzyItem {
            id: "zoidium:optical-flares/flare".into(),
            category: "Lens Flares".into(),
            name: "Lens Flare".into(),
            desc: "3D depth-occluded lens flare".into(),
        });
    }

    if session.plugins.is_enabled("trapcode-suite") {
        for (id, name, desc) in [
            (
                "zoidium:trapcode-suite/particular",
                "3D Particle Emitter",
                "Deterministic 3D fluid particle sim",
            ),
            (
                "zoidium:trapcode-suite/form",
                "Particle Lattice",
                "Deformable 3D particle lattice",
            ),
            (
                "zoidium:trapcode-suite/plexus",
                "Point Cloud Network",
                "Procedural point & edge cloud",
            ),
            (
                "zoidium:trapcode-suite/area-light",
                "Area Light (LTC)",
                "Rectangular soft area light",
            ),
            (
                "zoidium:trapcode-suite/ies-light",
                "IES Photometric Light",
                "LM-63 photometric light",
            ),
        ] {
            items.push(FuzzyItem {
                id: id.into(),
                category: "Particle & Light Suite".into(),
                name: name.into(),
                desc: desc.into(),
            });
        }
    }

    items
}

/// Full-height inline fuzzy picker with wrapped descriptions and fresh scroll-to-top per open.
pub(super) fn show_panel_fuzzy_picker(
    ui: &mut Ui,
    panel_rect: Rect,
    id: Id,
    open_seq: u64,
    items: &[FuzzyItem],
    state: &mut FuzzyPickerState,
    theme: &Cm3Theme,
) -> FuzzyPickerEvent {
    let mut event = FuzzyPickerEvent::None;
    let inner = panel_rect.shrink(1.0);
    ui.painter()
        .rect_filled(panel_rect, CornerRadius::ZERO, BG_PANEL);

    show_split_pane(ui, id.with("full_pane"), inner, |ui, _| {
        ui.spacing_mut().item_spacing = Vec2::ZERO;

        // 1. Full-width Cancel button
        if cm3_prop_button(ui, "cancel", ButtonVariant::Normal, true, theme).clicked() {
            event = FuzzyPickerEvent::Cancelled;
            return;
        }

        // 2. Auto-focused filter box
        let filter_id = id.with("filter_box");
        let focused = ui.memory(|m| m.has_focus(filter_id));
        Frame::NONE
            .fill(BG_PANEL)
            .stroke(Stroke::new(
                2.0,
                if focused {
                    theme.accent()
                } else {
                    BORDER_FILTER
                },
            ))
            .inner_margin(egui::Margin::symmetric(5, 4))
            .show(ui, |ui| {
                let resp = ui.add(
                    TextEdit::singleline(&mut state.query)
                        .id(filter_id)
                        .frame(Frame::NONE)
                        .hint_text("type to filter")
                        .font(theme.font(FONT_BODY))
                        .text_color(TEXT_PRIMARY)
                        .desired_width(ui.available_width()),
                );
                if state.just_opened {
                    resp.request_focus();
                    state.just_opened = false;
                }
            });

        let filtered: Vec<&FuzzyItem> = if state.query.trim().is_empty() {
            items.iter().collect()
        } else {
            let mut scored: Vec<(f32, &FuzzyItem)> = items
                .iter()
                .filter_map(|it| {
                    let s = fuzzy_score(&state.query, &it.name, &it.desc);
                    (s >= 0.45).then_some((s, it))
                })
                .collect();
            scored.sort_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().map(|(_, it)| it).collect()
        };

        if filtered.is_empty() || state.active_index >= filtered.len() {
            state.active_index = 0;
        }

        let mut key_navigated = false;
        let pointer_moved = ui.input(|i| i.pointer.delta() != Vec2::ZERO);
        ui.input(|i| {
            if i.key_pressed(Key::Escape) {
                event = FuzzyPickerEvent::Cancelled;
            } else if !filtered.is_empty() {
                if i.key_pressed(Key::ArrowDown) {
                    state.active_index = (state.active_index + 1) % filtered.len();
                    key_navigated = true;
                } else if i.key_pressed(Key::ArrowUp) {
                    state.active_index = (state.active_index + filtered.len() - 1) % filtered.len();
                    key_navigated = true;
                } else if i.key_pressed(Key::Enter) {
                    event = FuzzyPickerEvent::Selected(filtered[state.active_index].id.clone());
                }
            }
        });
        if event != FuzzyPickerEvent::None {
            return;
        }

        // 3. Scrollable options list keyed on `open_seq` so reopening ALWAYS starts at top
        ScrollArea::vertical()
            .id_salt((id, "options_scroll", open_seq))
            .auto_shrink([false, false])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing = Vec2::ZERO;
                if filtered.is_empty() {
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new("no matching items")
                            .font(theme.font(FONT_SMALL))
                            .color(TEXT_FAINT),
                    );
                    return;
                }

                let mut last_category: Option<&str> = None;
                let show_categories = state.query.trim().is_empty();

                for (idx, item) in filtered.iter().enumerate() {
                    if show_categories && last_category != Some(item.category.as_str()) {
                        last_category = Some(item.category.as_str());
                        let (cat_rect, _) = ui.allocate_exact_size(
                            Vec2::new(ui.available_width(), 24.0),
                            Sense::hover(),
                        );
                        ui.painter()
                            .rect_filled(cat_rect, CornerRadius::ZERO, BG_TITLE);
                        ui.painter()
                            .with_clip_rect(cat_rect.shrink2(Vec2::new(4.0, 0.0)))
                            .text(
                                Pos2::new(cat_rect.left() + 5.0, cat_rect.center().y),
                                Align2::LEFT_CENTER,
                                &item.category,
                                theme.font(FONT_BANNER),
                                TEXT_FAINT,
                            );
                    }

                    let avail_w = ui.available_width();
                    let text_wrap_w = (avail_w - 16.0).max(80.0);
                    let has_desc = !item.desc.is_empty();
                    let desc_galley = has_desc.then(|| {
                        ui.painter().layout(
                            item.desc.clone(),
                            theme.font(FONT_SMALL),
                            TEXT_OPTION_DESC,
                            text_wrap_w,
                        )
                    });
                    let row_h = if let Some(ref g) = desc_galley {
                        (26.0 + g.size().y + 6.0).max(42.0)
                    } else {
                        28.0
                    };
                    let (row_rect, row_resp) =
                        ui.allocate_exact_size(Vec2::new(avail_w, row_h), Sense::click());
                    let row_resp = row_resp.on_hover_cursor(CursorIcon::PointingHand);
                    let is_active = idx == state.active_index;
                    if key_navigated && is_active {
                        row_resp.scroll_to_me(None);
                    }

                    if is_active {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_OPTION_ACTIVE);
                        ui.painter().rect_stroke(
                            row_rect,
                            CornerRadius::ZERO,
                            Stroke::new(1.0, BORDER_OPTION_ACTIVE),
                            StrokeKind::Inside,
                        );
                    } else if row_resp.hovered() {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::ZERO, BG_OPTION_HOVER);
                    }

                    ui.painter().line_segment(
                        [
                            Pos2::new(row_rect.left(), row_rect.bottom() - 0.5),
                            Pos2::new(row_rect.right(), row_rect.bottom() - 0.5),
                        ],
                        Stroke::new(1.0, BORDER_ROW_SOFT),
                    );

                    let row_clip = ui
                        .painter()
                        .with_clip_rect(row_rect.shrink2(Vec2::new(4.0, 0.0)));
                    row_clip.text(
                        Pos2::new(
                            row_rect.left() + 8.0,
                            if has_desc {
                                row_rect.top() + 13.0
                            } else {
                                row_rect.center().y
                            },
                        ),
                        Align2::LEFT_CENTER,
                        &item.name,
                        theme.font(FONT_BODY),
                        if is_active { TEXT_BRIGHT } else { TEXT_PRIMARY },
                    );
                    if let Some(g) = desc_galley {
                        row_clip.galley(
                            Pos2::new(row_rect.left() + 8.0, row_rect.top() + 24.0),
                            g,
                            TEXT_OPTION_DESC,
                        );
                    }

                    if pointer_moved && row_resp.hovered() {
                        state.active_index = idx;
                    }
                    if row_resp.clicked() {
                        event = FuzzyPickerEvent::Selected(item.id.clone());
                    }
                }
            });
    });
    paint_crisp_panel_border(ui.painter(), panel_rect, BORDER_PANEL);
    event
}

pub(super) fn show_objects_tab(
    ui: &mut Ui,
    panel_rect: Rect,
    viewport_rect: Rect,
    session: &mut Session,
    theme: &Cm3Theme,
    state: &mut PanelsState,
    actions: &mut Vec<PanelAction>,
) {
    let Some(clip_id) = session.selection.clip_ids.first().cloned() else {
        state.add_object_open = false;
        empty_panel_message(ui, "Select a clip to edit its objects", theme);
        return;
    };
    let rev_before = session.revision();
    let Some(clip) = find_clip_cached(session, state, &clip_id) else {
        state.add_object_open = false;
        empty_panel_message(ui, "Select a clip to edit its objects", theme);
        return;
    };
    if clip.kind != "scene" {
        if show_non_scene_objects_tab(
            ui,
            panel_rect,
            viewport_rect,
            &clip,
            session,
            theme,
            state,
            actions,
        ) {
            return;
        }
        empty_panel_message(ui, "select a 3D scene, Shape+, or Text clip", theme);
        return;
    }

    // Expand the Add 3D Object picker over the FULL sidebar height when open
    if state.add_object_open {
        let items = build_object_picker_items(session);
        match show_panel_fuzzy_picker(
            ui,
            panel_rect,
            Id::unique("add_object_picker"),
            state.picker_open_seq,
            &items,
            &mut state.add_object_picker,
            theme,
        ) {
            FuzzyPickerEvent::Cancelled => {
                state.add_object_open = false;
            }
            FuzzyPickerEvent::Selected(kind) => {
                commit_inspector_tx_if_open(session, state);
                let parent_id = session.selection.object_3d_ids.first().and_then(|sel| {
                    clip.objects_3d
                        .iter()
                        .find(|o| o.id == *sel && is_container_object(&o.kind))
                        .map(|o| o.id.clone())
                });
                let display = items.iter().find(|i| i.id == kind).map(|i| i.name.clone());
                if let Ok(res) = session.execute(Command::AddObject3D {
                    clip_id: clip_id.clone(),
                    parent_id,
                    kind: kind.clone(),
                    name: display,
                }) && let Some(oid) = res.get("object_id").and_then(Value::as_str)
                {
                    let _ = session.selection.select_object_3d(oid, &kind, false);
                }
                state.add_object_open = false;
            }
            FuzzyPickerEvent::None => {}
        }
        return;
    }

    let (top_rect, bot_rect) = begin_split_panel(
        ui,
        Id::unique("objects_40_60_split"),
        panel_rect,
        &mut state.objects_split,
    );
    ui.input(|i| {
        if (i.pointer.primary_pressed() || i.pointer.secondary_pressed())
            && let Some(pos) = i.pointer.interact_pos()
        {
            if top_rect.contains(pos) {
                state.objects_focus_bottom = false;
            } else if bot_rect.contains(pos) {
                state.objects_focus_bottom = true;
            }
        }
    });
    let top_inner = top_rect.shrink(1.0);
    let bot_inner = bot_rect.shrink(1.0);
    ui.painter()
        .rect_filled(top_rect, CornerRadius::ZERO, BG_PANEL);
    ui.painter()
        .rect_filled(bot_rect, CornerRadius::ZERO, BG_PANEL);

    show_split_pane(
        ui,
        Id::unique("objects_40_60_split_p0"),
        top_inner,
        |ui, _top_rect| {
            ui.spacing_mut().item_spacing = Vec2::ZERO;

            // 28px #222222 header bar (27px body + 1px #242424 bottom border), 7x7 framed square glyph,
            // 16px #cccccc "3D Objects" title, and right-aligned `+`
            let (hdr_rect, hdr_resp) =
                ui.allocate_exact_size(Vec2::new(ui.available_width(), 28.0), Sense::click());
            ui.painter()
                .rect_filled(hdr_rect, CornerRadius::ZERO, BG_TITLE);
            let hdr_div_y = (hdr_rect.bottom() - 1.0).round();
            ui.painter().rect_filled(
                Rect::from_min_max(
                    Pos2::new(hdr_rect.left(), hdr_div_y),
                    Pos2::new(hdr_rect.right(), hdr_div_y + 1.0),
                ),
                CornerRadius::ZERO,
                BORDER_ROW_SOFT,
            );
            let add_icon_rect = Rect::from_min_size(
                Pos2::new(hdr_rect.right() - 24.0, (hdr_rect.center().y - 8.0).round()),
                Vec2::splat(15.0),
            );
            let add_rect = add_icon_rect.expand(3.0);
            let paste_w = (ui
                .painter()
                .layout_no_wrap("Paste".to_owned(), theme.font(FONT_COMPACT), TEXT_PRIMARY)
                .size()
                .x
                .ceil()
                + 12.0)
                .max(44.0);
            let paste_rect = Rect::from_min_size(
                Pos2::new(
                    (add_rect.left() - paste_w - 4.0).max(hdr_rect.left() + 80.0),
                    (hdr_rect.center().y - 9.0).round(),
                ),
                Vec2::new(paste_w, 18.0),
            );
            let (paste_clicked, paste_hovered) = header_compact_action_button(
                ui,
                Id::unique("objects_paste_btn"),
                paste_rect,
                "Paste",
                "Paste 3D Object JSON (Cmd+V)",
                theme,
            );
            let add_resp = ui
                .interact(add_rect, Id::unique("objects_add_btn"), Sense::click())
                .on_hover_cursor(CursorIcon::PointingHand)
                .on_hover_text("Add 3D object");
            if add_resp.hovered() {
                ui.painter().rect_stroke(
                    add_rect,
                    CornerRadius::ZERO,
                    Stroke::new(2.0, theme.accent()),
                    StrokeKind::Inside,
                );
            }
            paint_cm3_tree_box(
                ui.painter(),
                Pos2::new(hdr_rect.left() + 10.5, hdr_rect.center().y + 1.0),
                state.objects_list_open,
                if hdr_resp.hovered() && !add_resp.hovered() && !paste_hovered {
                    TEXT_BRIGHT
                } else {
                    TEXT_MUTED
                },
            );
            ui.painter()
                .with_clip_rect(Rect::from_min_max(
                    Pos2::new(hdr_rect.left() + 20.0, hdr_rect.top()),
                    Pos2::new(
                        (paste_rect.left() - 4.0).max(hdr_rect.left() + 24.0),
                        hdr_rect.bottom(),
                    ),
                ))
                .text(
                    Pos2::new(hdr_rect.left() + 21.0, hdr_rect.center().y),
                    Align2::LEFT_CENTER,
                    "3D Objects",
                    theme.font(FONT_BODY),
                    TEXT_PRIMARY,
                );
            paint_cm3_icon(
                ui.painter(),
                add_icon_rect,
                Cm3Icon::Add,
                if add_resp.hovered() {
                    TEXT_BRIGHT
                } else {
                    TEXT_PRIMARY
                },
            );
            if add_resp.clicked() {
                state.picker_open_seq = state.picker_open_seq.wrapping_add(1);
                state.add_object_open = true;
                state.add_object_picker = FuzzyPickerState {
                    just_opened: true,
                    ..FuzzyPickerState::default()
                };
            } else if paste_clicked {
                let selected_group_id = session
                    .selection
                    .object_3d_ids
                    .first()
                    .and_then(|sel_id| clip.objects_3d.iter().find(|o| o.id == *sel_id))
                    .filter(|o| o.kind.eq_ignore_ascii_case("group"))
                    .map(|o| o.id.clone());
                paste_pz_json_from_clipboard(
                    session,
                    state,
                    actions,
                    "objects3d",
                    Some(clip_id.clone()),
                    selected_group_id,
                    None,
                    None,
                    "3D object JSON",
                );
            } else if hdr_resp.clicked() {
                state.objects_list_open = !state.objects_list_open;
            }

            if !state.objects_list_open {
                return;
            }

            // Hierarchical 3D object tree
            ScrollArea::vertical()
                .id_salt("objects_tree_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    if clip.objects_3d.is_empty() {
                        return;
                    }

                    let flat_tree = flatten_object_tree(&clip.objects_3d, &state.collapsed_objects);
                    let mut drop_request: Option<(String, String, TreeDropMode, bool)> = None;

                    for (obj_idx, depth, has_children) in flat_tree {
                        let obj = &clip.objects_3d[obj_idx];
                        let selected = session.selection.object_3d_ids.contains(&obj.id);
                        let is_group = is_container_object(&obj.kind);

                        let row_w = ui.available_width();
                        let (row_rect, row_resp) =
                            ui.allocate_exact_size(Vec2::new(row_w, 27.0), Sense::click_and_drag());
                        let pointer_in_row = ui
                            .input(|i| i.pointer.hover_pos().is_some_and(|p| row_rect.contains(p)));

                        if selected {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, theme.accent());
                        } else if row_resp.hovered() || pointer_in_row {
                            ui.painter()
                                .rect_filled(row_rect, CornerRadius::ZERO, BG_HOVER);
                        }
                        ui.painter().line_segment(
                            [
                                Pos2::new(row_rect.left(), row_rect.bottom() - 0.5),
                                Pos2::new(row_rect.right(), row_rect.bottom() - 0.5),
                            ],
                            Stroke::new(1.0, BORDER_ROW_SOFT),
                        );

                        // Top-level row name indent: 36px from panel inner left
                        let label_left = row_rect.left() + 36.0 + depth as f32 * 15.0;

                        // Collapse triangle for containers (positioned left of label)
                        if has_children || is_group {
                            let tri_rect = Rect::from_min_size(
                                Pos2::new(
                                    (label_left - 20.0).max(row_rect.left() + 4.0),
                                    row_rect.top() + 3.0,
                                ),
                                Vec2::splat(20.0),
                            );
                            let collapsed = state.collapsed_objects.contains(&obj.id);
                            let tri_resp = ui
                                .interact(
                                    tri_rect,
                                    Id::unique(("obj_collapse", &obj.id)),
                                    Sense::click(),
                                )
                                .on_hover_cursor(CursorIcon::PointingHand);
                            paint_cm3_icon(
                                ui.painter(),
                                tri_rect.shrink(4.0),
                                if collapsed {
                                    Cm3Icon::ChevronRight
                                } else {
                                    Cm3Icon::ChevronDown
                                },
                                TEXT_MUTED,
                            );
                            if tri_resp.clicked() {
                                if collapsed {
                                    state.collapsed_objects.remove(&obj.id);
                                } else {
                                    state.collapsed_objects.insert(obj.id.clone());
                                }
                            }
                        }

                        // Label or Inline Rename input (Duplicate & Remove live in Inspector header)
                        let label_right = (row_rect.right() - 8.0).max(label_left);
                        let label_rect = Rect::from_min_max(
                            Pos2::new(label_left, row_rect.top()),
                            Pos2::new(label_right, row_rect.bottom()),
                        );

                        let is_renaming = state
                            .rename_state
                            .as_ref()
                            .is_some_and(|r| r.target_id == obj.id);
                        if is_renaming {
                            if let Some(rename) = state.rename_state.as_mut() {
                                let edit_id = Id::unique(("obj_rename", &obj.id));
                                let mut commit = false;
                                let mut cancel = false;
                                ui.input(|i| {
                                    if i.key_pressed(Key::Enter) {
                                        commit = true;
                                    } else if i.key_pressed(Key::Escape) {
                                        cancel = true;
                                    }
                                });
                                let mut rename_ui =
                                    ui.new_child(egui::UiBuilder::new().max_rect(label_rect));
                                let resp = Frame::NONE
                                    .fill(BG_INPUT)
                                    .stroke(Stroke::new(1.0, theme.accent()))
                                    .inner_margin(egui::Margin::symmetric(3, 1))
                                    .show(&mut rename_ui, |ui| {
                                        ui.add(
                                            TextEdit::singleline(&mut rename.buffer)
                                                .id(edit_id)
                                                .frame(Frame::NONE)
                                                .font(theme.font(FONT_BODY))
                                                .text_color(theme.value())
                                                .desired_width(label_rect.width()),
                                        )
                                    })
                                    .inner;
                                if rename.just_started {
                                    resp.request_focus();
                                    rename.just_started = false;
                                }
                                if cancel {
                                    state.rename_state = None;
                                } else if commit || resp.lost_focus() {
                                    let trimmed = rename.buffer.trim().to_owned();
                                    if !trimmed.is_empty() && trimmed != obj.name {
                                        let _ = session.execute(Command::SetPropertyValue {
                                            target_id: obj.id.clone(),
                                            property: "name".to_owned(),
                                            value: json!(trimmed),
                                        });
                                    }
                                    state.rename_state = None;
                                }
                            }
                        } else {
                            ui.painter().with_clip_rect(label_rect).text(
                                Pos2::new(label_rect.left(), label_rect.center().y),
                                Align2::LEFT_CENTER,
                                &obj.name,
                                theme.font(FONT_BODY),
                                if selected { TEXT_BRIGHT } else { TEXT_PRIMARY },
                            );
                        }

                        // Click / Shift+Click selection with same-type guard (Spec 01 §4.7)
                        if row_resp.clicked() {
                            commit_inspector_tx_if_open(session, state);
                            let shift = ui.input(|i| i.modifiers.shift);
                            if let Err(err) = session
                                .selection
                                .select_object_3d(&obj.id, &obj.kind, shift)
                            {
                                actions.push(PanelAction::StatusMessage(err.to_string()));
                            }
                        }

                        // Right-click context menu (Rename, Copy/Paste PZ JSON, Duplicate, Remove)
                        let ctx_obj_id = obj.id.clone();
                        let ctx_obj_name = obj.name.clone();
                        let ctx_parent_group_id = if obj.kind.eq_ignore_ascii_case("group") {
                            Some(obj.id.clone())
                        } else {
                            obj.parent_id.clone().filter(|pid| {
                                clip.objects_3d
                                    .iter()
                                    .any(|p| p.id == *pid && p.kind.eq_ignore_ascii_case("group"))
                            })
                        };
                        row_resp.context_menu(|ui| {
                            if ui.button("Rename").clicked() {
                                state.rename_state = Some(InlineRenameState {
                                    target_id: ctx_obj_id.clone(),
                                    buffer: ctx_obj_name.clone(),
                                    just_started: true,
                                });
                                ui.close();
                            }
                            if ui.button("Copy JSON").clicked() {
                                let ids = if selected && session.selection.object_3d_ids.len() > 1 {
                                    session.selection.object_3d_ids.clone()
                                } else {
                                    vec![ctx_obj_id.clone()]
                                };
                                let count = ids.len();
                                let res = session.copy_objects_3d_as_pz_json(&clip_id, &ids);
                                copy_pz_json_to_clipboard(
                                    ui.ctx(),
                                    res,
                                    &format!(
                                        "Copied {count} 3D object{} JSON to clipboard",
                                        if count == 1 { "" } else { "s" }
                                    ),
                                    actions,
                                );
                                ui.close();
                            }
                            if ui.button("Paste JSON").clicked() {
                                paste_pz_json_from_clipboard(
                                    session,
                                    state,
                                    actions,
                                    "objects3d",
                                    Some(clip_id.clone()),
                                    ctx_parent_group_id.clone(),
                                    None,
                                    None,
                                    "3D object JSON",
                                );
                                ui.close();
                            }
                            ui.separator();
                            if ui.button("Duplicate").clicked() {
                                let _ = session.execute(Command::DuplicateObject3D {
                                    clip_id: clip_id.clone(),
                                    object_id: ctx_obj_id.clone(),
                                });
                                ui.close();
                            }
                            if ui.button("Remove").clicked() {
                                commit_inspector_tx_if_open(session, state);
                                let _ = session.execute(Command::DeleteObject3D {
                                    clip_id: clip_id.clone(),
                                    object_id: ctx_obj_id.clone(),
                                });
                                ui.close();
                            }
                        });

                        // 5 px vertical drag-to-reorder & nest-inside (Spec 01 §4.7)
                        if row_resp.drag_started()
                            && let Some(pos) = ui.input(|i| i.pointer.interact_pos())
                        {
                            state.object_drag = Some(TreeDragState {
                                dragging_id: obj.id.clone(),
                                start_y: pos.y,
                                active: false,
                            });
                        }

                        if let Some(drag) = state.object_drag.as_mut()
                            && let Some(pos) = ui.input(|i| i.pointer.latest_pos())
                        {
                            if !drag.active && (pos.y - drag.start_y).abs() >= 5.0 {
                                drag.active = true;
                            }
                            if drag.active && drag.dragging_id != obj.id && row_rect.contains(pos) {
                                let n = ((pos.y - row_rect.top()) / row_rect.height().max(1.0))
                                    .clamp(0.0, 1.0);
                                let t = if is_group { 1.0 } else { 0.0 };
                                let s = 1.0 / (t + 2.0);
                                let mode = if n < s {
                                    TreeDropMode::Before
                                } else if 1.0 - n < s {
                                    TreeDropMode::After
                                } else if is_group {
                                    TreeDropMode::Inside
                                } else {
                                    TreeDropMode::After
                                };

                                let indicator_color = Color32::from_white_alpha(64); // #fff @ 0.25
                                match mode {
                                    TreeDropMode::Before => {
                                        let bar = Rect::from_min_size(
                                            row_rect.left_top(),
                                            Vec2::new(row_rect.width(), 3.0),
                                        );
                                        ui.painter().rect_filled(
                                            bar,
                                            CornerRadius::ZERO,
                                            indicator_color,
                                        );
                                    }
                                    TreeDropMode::After => {
                                        let bar = Rect::from_min_size(
                                            Pos2::new(row_rect.left(), row_rect.bottom() - 3.0),
                                            Vec2::new(row_rect.width(), 3.0),
                                        );
                                        ui.painter().rect_filled(
                                            bar,
                                            CornerRadius::ZERO,
                                            indicator_color,
                                        );
                                    }
                                    TreeDropMode::Inside => {
                                        ui.painter().rect_filled(
                                            row_rect,
                                            CornerRadius::ZERO,
                                            indicator_color,
                                        );
                                    }
                                }

                                if !ui.input(|i| i.pointer.primary_down()) {
                                    let alt = ui.input(|i| i.modifiers.alt);
                                    drop_request =
                                        Some((drag.dragging_id.clone(), obj.id.clone(), mode, alt));
                                }
                            }
                        }
                    }

                    // Finalize drag on release
                    if !ui.input(|i| i.pointer.primary_down()) {
                        if let Some((src_id, dst_id, mode, alt_clone)) = drop_request {
                            apply_object_drop(session, &clip, &src_id, &dst_id, mode, alt_clone);
                        }
                        state.object_drag = None;
                    }
                });
        },
    );
    // Crisp single 1px #333333 border around top pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), top_rect, BORDER_PANEL);
    // Only re-fetch clip snapshot if pane 0 actually mutated the session (avoids a second full
    // clip clone on every 60Hz frame).
    let clip_for_inspector = if session.revision() != rev_before {
        find_clip_cached(session, state, &clip_id).unwrap_or(clip)
    } else {
        clip
    };
    show_split_pane(
        ui,
        Id::unique("objects_40_60_split_p1"),
        bot_inner,
        |ui, bot_rect| {
            // Pane 1 (60%): Selected 3D Object Property Inspector
            show_object_inspector(
                ui,
                bot_rect,
                viewport_rect,
                &clip_for_inspector,
                session,
                theme,
                state,
                actions,
            );
        },
    );
    // Crisp single 1px #333333 border around bottom pane (layout.rs paints the focus accent only when SideInspector is focused)
    paint_crisp_panel_border(ui.painter(), bot_rect, BORDER_PANEL);
}

pub(super) fn is_container_object(kind: &str) -> bool {
    let lower = kind.to_ascii_lowercase();
    lower == "group"
        || lower.starts_with("zoidium:repeater/")
        || lower.starts_with("zoidcraft:repeater/")
        || lower.starts_with("zoidium:text-plus/")
        || lower.starts_with("zoidium:text_plus/")
        || lower.starts_with("zoidcraft:text-plus/")
        || lower.starts_with("zoidium:expresstext-plus/")
        || lower.starts_with("zoidium:expresstext_plus/")
        || lower.starts_with("zoidcraft:expresstext-plus/")
}

fn flatten_object_tree(
    objects: &[SessionObject3D],
    collapsed: &BTreeSet<String>,
) -> Vec<(usize, usize, bool)> {
    let mut out = Vec::with_capacity(objects.len());
    fn visit(
        objects: &[SessionObject3D],
        parent_id: Option<&str>,
        depth: usize,
        collapsed: &BTreeSet<String>,
        out: &mut Vec<(usize, usize, bool)>,
    ) {
        for (idx, obj) in objects.iter().enumerate() {
            if obj.parent_id.as_deref() == parent_id {
                let has_children = objects
                    .iter()
                    .any(|child| child.parent_id.as_deref() == Some(obj.id.as_str()));
                out.push((idx, depth, has_children));
                if has_children && !collapsed.contains(&obj.id) {
                    visit(objects, Some(obj.id.as_str()), depth + 1, collapsed, out);
                }
            }
        }
    }
    visit(objects, None, 0, collapsed, &mut out);
    out
}

pub(super) fn apply_object_drop(
    session: &mut Session,
    clip: &SessionClip,
    src_id: &str,
    dst_id: &str,
    mode: TreeDropMode,
    alt_clone: bool,
) {
    let Some(src_idx) = clip.objects_3d.iter().position(|o| o.id == src_id) else {
        return;
    };
    let Some(dst_idx) = clip.objects_3d.iter().position(|o| o.id == dst_id) else {
        return;
    };
    let dst = &clip.objects_3d[dst_idx];
    let (new_parent, mut target_idx) = match mode {
        TreeDropMode::Inside => (Some(dst.id.clone()), clip.objects_3d.len()),
        TreeDropMode::Before => (dst.parent_id.clone(), dst_idx),
        TreeDropMode::After => (dst.parent_id.clone(), dst_idx + 1),
    };
    if !alt_clone && src_idx < target_idx {
        target_idx -= 1;
    }

    if alt_clone {
        let started_tx = session.begin_transaction("clone object").is_ok();
        if let Ok(res) = session.execute(Command::DuplicateObject3D {
            clip_id: clip.id.clone(),
            object_id: src_id.to_owned(),
        }) && let Some(cloned_id) = res.get("object_id").and_then(Value::as_str)
        {
            let _ = session.execute(Command::ReorderObject3D {
                clip_id: clip.id.clone(),
                object_id: cloned_id.to_owned(),
                parent_id: new_parent,
                index: target_idx,
            });
        }
        if started_tx && session.history.is_in_transaction() {
            let _ = session.commit_transaction(true);
        }
    } else {
        let _ = session.execute(Command::ReorderObject3D {
            clip_id: clip.id.clone(),
            object_id: src_id.to_owned(),
            parent_id: new_parent,
            index: target_idx,
        });
    }
}
