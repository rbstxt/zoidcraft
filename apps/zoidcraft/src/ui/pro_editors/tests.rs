//! Unit tests for the Pro Editor windows.

use super::*;
use crate::ui::controls::{FloatingWindowManager, FloatingWindowSpec};
use crate::ui::theme::Cm3Theme;
use egui::{Id, Key, Pos2, Rect, Vec2};
use zc_session::Session;

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

#[test]
fn clamped_editor_window_rect_always_stays_inside_screen() {
    for (sw, sh) in [(1280.0_f32, 752.0_f32), (900.0, 600.0), (1920.0, 1080.0)] {
        let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(sw, sh));
        let (pos, size) = compute_clamped_editor_window_rect(screen, 860.0, 560.0);
        assert!(pos.x >= 40.0, "x={} must be >= 40 on {sw}x{sh}", pos.x);
        assert!(pos.y >= 40.0, "y={} must be >= 40 on {sw}x{sh}", pos.y);
        assert!(size.x <= 860.0 && size.x <= sw - 80.0);
        assert!(size.y <= 560.0 && size.y <= sh - 80.0);
        assert!(
            pos.x + size.x <= sw - 8.0,
            "right edge clipped on {sw}x{sh}"
        );
        assert!(
            pos.y + size.y <= sh - 8.0,
            "bottom edge clipped on {sw}x{sh}"
        );
    }
}

#[test]
fn node_graph_editor_window_opens_centered_and_unclipped_on_1280x752() {
    let ctx = egui::Context::default();
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 752.0));
    let mut session = Session::new();
    let theme = Cm3Theme::default();
    let mut fw = FloatingWindowManager::default();
    let mut state = NodeGraphEditorState::default();
    state.open_for_target(
        NodeGraphTargetKind::NodeMaterial,
        "obj1",
        "material.graph",
        "Sphere Material",
        None,
    );

    let raw = egui::RawInput {
        screen_rect: Some(screen),
        ..Default::default()
    };
    let mut out = ctx.run_ui(raw, |ui| {
        show_node_graph_editor_window(ui.ctx(), screen, &mut fw, &mut state, &mut session, &theme);
    });
    out.textures_delta.clear();

    let win = fw
        .windows
        .get("pro_node_graph_editor")
        .expect("node graph editor floating window state");
    assert!(
        win.pos.x >= 40.0,
        "window spawned off left: x={}",
        win.pos.x
    );
    assert!(win.pos.y >= 40.0, "window spawned off top: y={}", win.pos.y);
    assert!(
        win.pos.x + win.size.x <= screen.right() - 8.0,
        "window clipped right: {}",
        win.pos.x + win.size.x
    );
    assert!(
        win.pos.y + win.size.y <= screen.bottom() - 8.0,
        "window clipped bottom: {}",
        win.pos.y + win.size.y
    );
}

#[test]
fn node_material_editor_auto_fits_all_default_nodes_inside_canvas_on_first_open() {
    let mut state = NodeGraphEditorState::default();
    state.open_for_target(
        NodeGraphTargetKind::NodeMaterial,
        "obj1",
        "material.graph",
        "Sphere Material",
        None,
    );
    assert!(state.needs_initial_fit);
    assert_eq!(state.nodes.len(), 3);

    // Simulate realistic canvas inside 860x560 floating window (minus 248px inspector + padding)
    let canvas_rect = Rect::from_min_size(Pos2::ZERO, Vec2::new(580.0, 440.0));
    fit_node_graph_view(&mut state, canvas_rect.size());
    assert!(
        state.zoom >= 0.35 && state.zoom <= 1.0,
        "unexpected zoom: {}",
        state.zoom
    );
    for node in &state.nodes {
        let spec = find_node_spec(state.target_kind, &node.kind);
        let sz = node_size_graph_space(spec) * state.zoom;
        let tl = graph_to_screen(canvas_rect, state.pan, state.zoom, node.pos);
        let nr = Rect::from_min_size(tl, sz);
        assert!(
            nr.left() >= canvas_rect.left() + 12.0,
            "node `{}` clipped left: {:?}",
            node.title,
            nr
        );
        assert!(
            nr.right() <= canvas_rect.right() - 12.0,
            "node `{}` clipped right: {:?} vs canvas right {}",
            node.title,
            nr,
            canvas_rect.right()
        );
        assert!(
            nr.top() >= canvas_rect.top() + 12.0 && nr.bottom() <= canvas_rect.bottom() - 12.0,
            "node `{}` clipped vertically: {:?}",
            node.title,
            nr
        );
    }
}

#[test]
fn node_status_bar_layout_prevents_text_collision_and_hides_hint_when_narrow() {
    // Wide window: shows hint and reserves >= 12px gap between status and hint
    let (left_w, show_hint) = compute_node_status_bar_layout(820.0, 210.0);
    assert!(show_hint);
    assert!(16.0 + left_w + 12.0 + 210.0 <= 820.0 + 0.01);

    // Narrow window: hides hint so left status gets full width without overlap
    let (narrow_left_w, narrow_show_hint) = compute_node_status_bar_layout(340.0, 210.0);
    assert!(!narrow_show_hint);
    assert!(narrow_left_w >= 300.0);
}

#[test]
fn clamp_floating_window_keeps_titlebar_buttons_inside_screen_after_oversized_resize() {
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(900.0, 600.0));
    let spec = FloatingWindowSpec::new("expression_editor", "Expression Editor")
        .with_default_size(Vec2::new(460.0, 340.0));
    let mut fw = FloatingWindowManager::default();
    fw.open_or_focus(&spec, screen);

    // Simulate aggressive east/SE resize pushing titlebar close button off-screen right & bottom
    let win = fw.windows.get_mut("expression_editor").unwrap();
    win.pos = Pos2::new(620.0, 420.0);
    win.size = Vec2::new(1400.0, 900.0);
    win.dragging = true;

    clamp_floating_window_in_screen(&mut fw, &spec, screen);
    let win = fw.windows.get("expression_editor").unwrap();
    assert!(win.pos.x >= screen.left() + 8.0);
    assert!(win.pos.y >= screen.top() + 8.0);
    assert!(
        win.pos.x + win.size.x <= screen.right() - 8.0,
        "right edge exceeded screen: {} + {} > {}",
        win.pos.x,
        win.size.x,
        screen.right() - 8.0
    );
    assert!(
        win.pos.y + win.size.y <= screen.bottom() - 8.0,
        "bottom edge exceeded screen: {} + {} > {}",
        win.pos.y,
        win.size.y,
        screen.bottom() - 8.0
    );
}

#[test]
fn default_node_material_graph_and_singular_map_status() {
    let graph = NodeGraphEditorState::default_node_material_graph_json();
    let nodes = graph["nodes"].as_array().expect("nodes array");
    let links = graph["links"].as_array().expect("links array");
    assert_eq!(nodes.len(), 3);
    assert_eq!(nodes[0]["type"], "noise");
    assert_eq!(nodes[1]["type"], "colorRamp");
    assert_eq!(nodes[2]["type"], "output");
    assert_eq!(links.len(), 2);

    let mut st = NodeGraphEditorState::default();
    st.open_for_target(
        NodeGraphTargetKind::NodeMaterial,
        "obj_1",
        "material.graph",
        "Cube",
        Some(&graph),
    );
    match &st.compile_status {
        NodeCompileStatus::Ok(msg) => {
            assert!(
                msg.contains("1 baked PBR map)") && !msg.contains("1 baked PBR maps"),
                "expected singular '1 baked PBR map', got: {msg}"
            );
        }
        other => panic!("expected Ok compile status, got {other:?}"),
    }
}

#[test]
fn escape_closes_only_topmost_floating_editor_and_respects_popover_flag() {
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 800.0));
    let mut fw = FloatingWindowManager::default();
    let spec_a = FloatingWindowSpec::new("pro_node_graph_editor", "Node Graph Editor");
    let spec_b = FloatingWindowSpec::new("pro_color_looks_editor", "Color Looks Setup");
    fw.open_or_focus(&spec_a, screen);
    fw.open_or_focus(&spec_b, screen);

    let esc_input = || egui::RawInput {
        events: vec![egui::Event::Key {
            key: Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        ..Default::default()
    };

    // 1. When a transient popover already consumed Escape, neither floating editor closes.
    let ctx = egui::Context::default();
    let mut out = ctx.run_ui(esc_input(), |ui| {
        let ctx = ui.ctx();
        ctx.data_mut(|d| {
            d.insert_temp(Id::unique("zc_transient_popover_consumed_esc"), true);
        });
        assert!(!consume_escape_for_floating_editor(
            ctx,
            &fw,
            "pro_color_looks_editor"
        ));
        assert!(!consume_escape_for_floating_editor(
            ctx,
            &fw,
            "pro_node_graph_editor"
        ));
    });
    out.textures_delta.clear();

    // 2. Without popover flag, Escape closes ONLY the topmost window and consumes the key.
    let ctx2 = egui::Context::default();
    let mut out2 = ctx2.run_ui(esc_input(), |ui| {
        let ctx = ui.ctx();
        assert!(!consume_escape_for_floating_editor(
            ctx,
            &fw,
            "pro_node_graph_editor"
        ));
        assert!(consume_escape_for_floating_editor(
            ctx,
            &fw,
            "pro_color_looks_editor"
        ));
        fw.close("pro_color_looks_editor");
        // Same frame: key is consumed so lower editor does NOT close on the same Escape press.
        assert!(!consume_escape_for_floating_editor(
            ctx,
            &fw,
            "pro_node_graph_editor"
        ));
        assert!(!ctx.input(|i| i.key_pressed(Key::Escape)));
    });
    out2.textures_delta.clear();
    assert!(!fw.is_open("pro_color_looks_editor"));
    assert!(fw.is_open("pro_node_graph_editor"));

    // 3. Even if focused_id was cleared, next Escape closes the remaining open editor.
    fw.clear_focus();
    let ctx3 = egui::Context::default();
    let mut out3 = ctx3.run_ui(esc_input(), |ui| {
        let ctx = ui.ctx();
        assert!(consume_escape_for_floating_editor(
            ctx,
            &fw,
            "pro_node_graph_editor"
        ));
        fw.close("pro_node_graph_editor");
    });
    out3.textures_delta.clear();
    assert!(!fw.is_open("pro_node_graph_editor"));
}
