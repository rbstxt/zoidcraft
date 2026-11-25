//! Unit tests for CM3 property widgets and floating windows.

use super::*;
use crate::ui::theme::Cm3Theme;
use egui::{Frame, Id, Key, Pos2, Rect, Ui, Vec2};
use zc_color::{Rgb, Rgba};
use zc_project::GradientStop;

fn run_frame(ctx: &egui::Context, events: Vec<egui::Event>, f: impl FnMut(&mut Ui)) {
    let mut raw = egui::RawInput {
        screen_rect: Some(Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0))),
        events,
        ..Default::default()
    };
    raw.focused = true;
    let mut f = f;
    let mut out = ctx.run_ui(raw, |ui| {
        egui::CentralPanel::default()
            .frame(Frame::NONE)
            .show(ui, |ui| {
                f(ui);
            });
    });
    out.textures_delta.clear();
}

#[test]
fn default_swatches_order_and_css_roundtrip() {
    let sw = default_recent_swatches();
    assert_eq!(sw.len(), 36);
    // Row 0 col 0 is pure red (h = 0)
    assert_eq!(sw[0].to_u8(), [255, 0, 0]);
    // Row 3 (18..24) is white -> black grayscale
    assert_eq!(sw[18].to_u8(), [255, 255, 255]);
    assert_eq!(sw[19].to_u8(), [204, 204, 204]);
    assert_eq!(sw[20].to_u8(), [153, 153, 153]);
    assert_eq!(sw[21].to_u8(), [102, 102, 102]);
    assert_eq!(sw[22].to_u8(), [51, 51, 51]);
    assert_eq!(sw[23].to_u8(), [0, 0, 0]);

    let black = Rgba::new(0.0, 0.0, 0.0, 1.0);
    assert_eq!(format_picker_css(black, false), "rgb(0, 0, 0)");
    assert_eq!(format_picker_css(black, true), "rgba(0, 0, 0, 1)");
    let semi = Rgba::new(1.0, 0.0, 0.5, 0.5);
    let css = format_picker_css(semi, true);
    assert!(css.starts_with("rgba(255, 0, 128, 0.5"));
    let parsed_css = parse_picker_color("rgb(12, 34, 56)").expect("rgb parse");
    assert_eq!(parsed_css.to_u8(), [12, 34, 56, 255]);
    let parsed_hex = parse_picker_color("#112233").expect("hex parse");
    assert_eq!(parsed_hex.to_u8(), [0x11, 0x22, 0x33, 255]);
    assert!(parse_picker_color("not-a-color").is_none());
}

#[test]
fn scrub_pending_drag_to_text_edit_cancels_transaction_and_commits_cleanly() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_scrub_tx");
    let cfg = ScrubConfig::new(0.0, 100.0, 1.0, 2);
    let mut val = 10.0_f64;

    // Frame 1: layout to discover hit rect
    let mut hit_center = Pos2::ZERO;
    run_frame(&ctx, vec![], |ui| {
        let start = ui.cursor().min;
        let ev = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
        assert_eq!(ev, ScrubEvent::default());
        hit_center = Pos2::new(start.x + 8.0, start.y + 10.0);
    });

    // Frame 2: pointer down -> PendingDrag, ev.started = true
    let mut ev2 = ScrubEvent::default();
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(hit_center),
            egui::Event::PointerButton {
                pos: hit_center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            ev2 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
        },
    );
    assert!(ev2.started && !ev2.cancelled && !ev2.committed);

    // Frame 3: pointer up without dragging -> immediately cancels PendingDrag tx & enters TextEdit
    let mut ev3 = ScrubEvent::default();
    run_frame(
        &ctx,
        vec![egui::Event::PointerButton {
            pos: hit_center,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
        |ui| {
            ev3 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
        },
    );
    assert!(ev3.cancelled && !ev3.started && !ev3.committed && !ev3.changed);

    // Frame 4: TextEdit renders and requests focus
    let mut ev4 = ScrubEvent::default();
    run_frame(&ctx, vec![], |ui| {
        ev4 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
    });
    assert_eq!(ev4, ScrubEvent::default());

    // Frame 5: Escape in TextEdit -> reverts and emits NO events (no open tx)
    let mut ev5 = ScrubEvent::default();
    run_frame(
        &ctx,
        vec![egui::Event::Key {
            key: Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        |ui| {
            ev5 = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
        },
    );
    assert_eq!(ev5, ScrubEvent::default());
    assert!((val - 10.0).abs() < 1e-9);
}

#[test]
fn scrub_read_only_mid_pending_drag_emits_cancel() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_scrub_ro");
    let cfg = ScrubConfig::new(0.0, 100.0, 1.0, 2);
    let mut val = 25.0_f64;

    let mut hit_center = Pos2::ZERO;
    run_frame(&ctx, vec![], |ui| {
        let start = ui.cursor().min;
        let _ = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
        hit_center = Pos2::new(start.x + 8.0, start.y + 10.0);
    });
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(hit_center),
            egui::Event::PointerButton {
                pos: hit_center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            let ev = cm3_scrub_number(ui, id, &mut val, cfg, &theme);
            assert!(ev.started);
        },
    );

    let mut ev_ro = ScrubEvent::default();
    run_frame(&ctx, vec![], |ui| {
        ev_ro = cm3_scrub_number(ui, id, &mut val, cfg.with_read_only(true), &theme);
    });
    assert!(ev_ro.cancelled);
    assert!((val - 25.0).abs() < 1e-9);
}

#[test]
fn vector2_text_edit_commit_scales_linked_axis_proportionally() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_vec2_link");
    let cfg = ScrubConfig::new(0.0, 100.0, 1.0, 2);
    let mut vals = [2.0_f64, 6.0_f64];

    // Enter TextEdit on axis 0 via temp memory (simulating pointer-down -> pointer-up -> typed "5.00")
    ctx.data_mut(|d| {
        d.insert_temp(
            id.with(0usize),
            ScrubMemory {
                mode: ScrubMode::TextEdit {
                    buffer: "5.00".to_owned(),
                    initial_display: 2.0,
                    start_value: 2.0,
                    just_entered: false,
                    alt_at_start: false,
                },
            },
        );
    });
    ctx.memory_mut(|m| m.request_focus(id.with(0usize).with("text_edit")));

    // Frame to settle focus
    run_frame(&ctx, vec![], |ui| {
        let _ = cm3_vector2_with_sep(ui, id, &mut vals, &cfg, true, ",", &theme);
    });

    // Press Enter to commit 5.00 on axis 0 -> axis 1 should scale 6.0 * (5.0 / 2.0) = 15.0
    let mut ev = ScrubEvent::default();
    run_frame(
        &ctx,
        vec![egui::Event::Key {
            key: Key::Enter,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        |ui| {
            ev = cm3_vector2_with_sep(ui, id, &mut vals, &cfg, true, ",", &theme);
        },
    );
    assert!(ev.changed && ev.committed && !ev.started && !ev.cancelled);
    assert!((vals[0] - 5.0).abs() < 1e-6);
    assert!((vals[1] - 15.0).abs() < 1e-6);
}

#[test]
fn property_row_col2_aligns_identically_for_short_and_long_labels() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let col_state = PropertyColumnState::default();
    let mut short_x = 0.0_f32;
    let mut long_x = 0.0_f32;

    run_frame(&ctx, vec![], |ui| {
        ui.set_width(321.0);
        cm3_property_row_aligned(
            ui,
            "Time",
            true,
            col_state,
            &theme,
            |_| {},
            |ui| {
                short_x = ui.cursor().left();
            },
        );
        cm3_property_row_aligned(
            ui,
            "Multisample motion blur",
            true,
            col_state,
            &theme,
            |_| {},
            |ui| {
                long_x = ui.cursor().left();
            },
        );
    });
    assert!(
        (short_x - long_x).abs() < 1e-3,
        "Column 2 misaligned: short={short_x}, long={long_x}"
    );
}

#[test]
fn gradient_bar_stop_cancel_and_switch_events() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_grad_bar");
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let mut stops = vec![
        GradientStop {
            position: 0.0,
            color: "rgba(255,0,0,1.000)".to_owned(),
        },
        GradientStop {
            position: 1.0,
            color: "rgba(0,0,255,1.000)".to_owned(),
        },
    ];

    // Open stop color picker popup in dirty state with pre-drag initial = red, current = green
    let stop_popup_id = id.with("stop_color").with("hsva_popup");
    stops[0].color = "rgba(0,255,0,1.000)".to_owned();
    ctx.data_mut(|d| {
        d.insert_temp(
            stop_popup_id,
            ColorPopupState {
                open: true,
                initial: Rgba::new(1.0, 0.0, 0.0, 1.0),
                hsv: Rgb::from_u8(0, 255, 0).to_hsv(),
                alpha: 1.0,
                hex_buf: "rgba(0, 255, 0, 1)".to_owned(),
                dirty: true,
            },
        );
    });

    // Press Escape -> restores red, emits changed=true, cancelled=true, committed=false
    let mut ev = GradientBarEvent::default();
    run_frame(
        &ctx,
        vec![egui::Event::Key {
            key: Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }],
        |ui| {
            ev = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
        },
    );
    assert!(ev.changed && ev.cancelled && !ev.committed);
    assert_eq!(parse_css_rgba(&stops[0].color).to_u8(), [255, 0, 0, 255]);
}

#[test]
fn hsva_color_picker_read_only_mid_open_cancels_and_restores() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_color_ro_cancel");
    let popup_id = id.with("hsva_popup");
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let initial = Rgba::new(0.2, 0.4, 0.8, 1.0);
    let mut color = Rgba::new(0.9, 0.1, 0.1, 1.0);

    // Case 1: popup is open and dirty, then read_only flips true -> restores initial, cancels, changed=true
    ctx.data_mut(|d| {
        d.insert_temp(
            popup_id,
            ColorPopupState {
                open: true,
                initial,
                hsv: color.rgb().to_hsv(),
                alpha: 1.0,
                hex_buf: format_picker_css(color, false),
                dirty: true,
            },
        );
    });
    let mut ev = ColorPickerEvent::default();
    run_frame(&ctx, vec![], |ui| {
        ev = cm3_hsva_color_picker(ui, id, &mut color, false, true, vp, &theme);
    });
    assert!(ev.cancelled && ev.changed && !ev.committed && !ev.started);
    assert_eq!(color.to_u8(), initial.to_u8());
    let st = ctx
        .data_mut(|d| d.get_temp::<ColorPopupState>(popup_id))
        .expect("persisted state");
    assert!(!st.open && !st.dirty);

    // Case 2: popup is open and clean (not dirty), read_only flips true -> cancels, changed=false
    ctx.data_mut(|d| {
        d.insert_temp(
            popup_id,
            ColorPopupState {
                open: true,
                initial,
                hsv: initial.rgb().to_hsv(),
                alpha: 1.0,
                hex_buf: format_picker_css(initial, false),
                dirty: false,
            },
        );
    });
    let mut ev_clean = ColorPickerEvent::default();
    run_frame(&ctx, vec![], |ui| {
        ev_clean = cm3_hsva_color_picker(ui, id, &mut color, false, true, vp, &theme);
    });
    assert!(ev_clean.cancelled && !ev_clean.changed && !ev_clean.committed);
    let st2 = ctx
        .data_mut(|d| d.get_temp::<ColorPopupState>(popup_id))
        .expect("persisted state");
    assert!(!st2.open && !st2.dirty);
}

#[test]
fn gradient_bar_clean_stop_switch_emits_no_spurious_commit() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_grad_clean_switch");
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let mut stops = vec![
        GradientStop {
            position: 0.0,
            color: "rgba(255,0,0,1.000)".to_owned(),
        },
        GradientStop {
            position: 1.0,
            color: "rgba(0,0,255,1.000)".to_owned(),
        },
    ];

    // Frame 1: layout to discover stop 1 rect (right end of 245px lane)
    let mut lane_origin = Pos2::ZERO;
    run_frame(&ctx, vec![], |ui| {
        ui.set_width(245.0);
        lane_origin = ui.cursor().min;
        let _ = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
    });

    // Open stop 0's color picker in clean (dirty=false) state
    let stop_popup_id = id.with("stop_color").with("hsva_popup");
    ctx.data_mut(|d| {
        d.insert_temp(
            stop_popup_id,
            ColorPopupState {
                open: true,
                initial: Rgba::new(1.0, 0.0, 0.0, 1.0),
                hsv: Rgb::from_u8(255, 0, 0).to_hsv(),
                alpha: 1.0,
                hex_buf: "rgba(255, 0, 0, 1)".to_owned(),
                dirty: false,
            },
        );
    });

    // Click stop 1 (at x = lane_origin.x + 245.0, y = lane_origin.y + 30.0 + 3.0 + 7.5)
    let stop1_pos = Pos2::new(lane_origin.x + 244.0, lane_origin.y + 40.5);
    let mut ev_down = GradientBarEvent::default();
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(stop1_pos),
            egui::Event::PointerButton {
                pos: stop1_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            ui.set_width(245.0);
            ev_down = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
        },
    );
    let mut ev_up = GradientBarEvent::default();
    run_frame(
        &ctx,
        vec![egui::Event::PointerButton {
            pos: stop1_pos,
            button: egui::PointerButton::Primary,
            pressed: false,
            modifiers: egui::Modifiers::NONE,
        }],
        |ui| {
            ui.set_width(245.0);
            ev_up = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
        },
    );
    assert!(!ev_down.committed && !ev_up.committed);
    let st = ctx
        .data_mut(|d| d.get_temp::<ColorPopupState>(stop_popup_id))
        .expect("stop popup state");
    assert!(!st.open);
}

#[test]
fn gradient_bar_reverted_stop_switch_emits_cancel_to_close_tx() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_grad_reverted_switch");
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let mut stops = vec![
        GradientStop {
            position: 0.0,
            color: "rgba(255,0,0,1.000)".to_owned(),
        },
        GradientStop {
            position: 1.0,
            color: "rgba(0,0,255,1.000)".to_owned(),
        },
    ];

    // Frame 1: layout to discover stop 1 rect (right end of 245px lane)
    let mut lane_origin = Pos2::ZERO;
    run_frame(&ctx, vec![], |ui| {
        ui.set_width(245.0);
        lane_origin = ui.cursor().min;
        let _ = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
    });

    // Stop 0's color picker is open and clean (e.g. after reverting via comparison bar)
    let stop_popup_id = id.with("stop_color").with("hsva_popup");
    ctx.data_mut(|d| {
        d.insert_temp(
            stop_popup_id,
            ColorPopupState {
                open: true,
                initial: Rgba::new(1.0, 0.0, 0.0, 1.0),
                hsv: Rgb::from_u8(255, 0, 0).to_hsv(),
                alpha: 1.0,
                hex_buf: "rgba(255, 0, 0, 1)".to_owned(),
                dirty: false,
            },
        );
    });

    // Switch to stop 1 in a single click frame so `stop_changed` fires while `st.open && !st.dirty`
    let stop1_pos = Pos2::new(lane_origin.x + 244.0, lane_origin.y + 40.5);
    let mut ev = GradientBarEvent::default();
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(stop1_pos),
            egui::Event::PointerButton {
                pos: stop1_pos,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: stop1_pos,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            ui.set_width(245.0);
            ev = cm3_gradient_bar(ui, id, &mut stops, vp, &theme);
        },
    );
    assert!(ev.cancelled && !ev.committed && !ev.changed, "ev={ev:?}");
    let bar_st = ctx
        .data_mut(|d| d.get_temp::<GradientBarState>(id))
        .expect("gradient bar state");
    assert_eq!(bar_st.selected_stop, 1);
    let st = ctx
        .data_mut(|d| d.get_temp::<ColorPopupState>(stop_popup_id))
        .expect("stop popup state");
    assert!(!st.open);
}

#[test]
fn scrub_event_merge_preserves_same_frame_text_commit_and_pending_drag_start() {
    let a = ScrubEvent {
        started: false,
        changed: true,
        committed: true,
        cancelled: false,
    };
    let b = ScrubEvent {
        started: true,
        changed: false,
        committed: false,
        cancelled: false,
    };
    let merged = a.merge(b);
    assert!(merged.started && merged.changed && merged.committed && !merged.cancelled);
    assert_eq!(b.merge(a), merged);
}

#[test]
fn hsv_color_picker_single_click_in_sv_square_picks_immediately() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_hsv_sv_single_click");
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let mut color = Rgba::new(1.0, 1.0, 1.0, 1.0); // white: S=0, V=1

    // Frame 1: measure swatch position.
    let mut swatch_center = Pos2::ZERO;
    run_frame(&ctx, vec![], |ui| {
        let start = ui.cursor().min;
        let _ = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
        swatch_center = Pos2::new(start.x + 15.0, start.y + 12.0);
    });

    // Frame 2: click swatch to open HSVA popup.
    let mut open_ev = ColorPickerEvent::default();
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(swatch_center),
            egui::Event::PointerButton {
                pos: swatch_center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: swatch_center,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            open_ev = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
        },
    );
    assert!(open_ev.started);

    // Frame 3: render open popup once so layout settles.
    run_frame(&ctx, vec![], |ui| {
        let _ = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
    });

    // Frame 4: single pointer-down inside the lower-right of the 250x250 SV square (no drag!).
    let popup_pos = clamp_popup_pos(
        Rect::from_min_size(
            Pos2::new(swatch_center.x - 15.0, swatch_center.y - 10.0),
            Vec2::new(30.0, 20.0),
        ),
        Vec2::new(482.0, 262.0),
        vp,
    );
    let sv_click = Pos2::new(popup_pos.x + 6.0 + 200.0, popup_pos.y + 6.0 + 125.0);
    let mut pick_ev = ColorPickerEvent::default();
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(sv_click),
            egui::Event::PointerButton {
                pos: sv_click,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            pick_ev = cm3_hsva_color_picker(ui, id, &mut color, false, false, vp, &theme);
        },
    );
    assert!(
        pick_ev.changed,
        "single click press in SV square must pick immediately without requiring a drag"
    );
    let hsv = color.rgb().to_hsv();
    assert!(
        hsv.s > 0.6 && hsv.v < 0.7,
        "expected high saturation / mid value from SV click, got hsv={hsv:?}"
    );
}

#[test]
fn dropdown_dynamic_option_count_reanchors_flush_to_button() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let id = Id::unique("test_dd_dynamic_opts");
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let mut selected = 0usize;
    let opts_5 = ["Basic", "Lambert", "Phong", "Standard", "Physical"];
    let opts_9 = [
        "Basic", "Lambert", "Phong", "Standard", "Physical", "Toon", "Matcap", "Normal", "Depth",
    ];

    // Position button near bottom of viewport so the 9-option popup flips above cleanly.
    let mut btn_center = Pos2::ZERO;
    run_frame(&ctx, vec![], |ui| {
        ui.add_space(430.0);
        let top_left = ui.cursor().min;
        let _ = cm3_dropdown(ui, id, &mut selected, &opts_5, false, vp, &theme);
        btn_center = Pos2::new(top_left.x + 30.0, top_left.y + 12.0);
    });

    // Open with 5 options.
    run_frame(
        &ctx,
        vec![
            egui::Event::PointerMoved(btn_center),
            egui::Event::PointerButton {
                pos: btn_center,
                button: egui::PointerButton::Primary,
                pressed: true,
                modifiers: egui::Modifiers::NONE,
            },
            egui::Event::PointerButton {
                pos: btn_center,
                button: egui::PointerButton::Primary,
                pressed: false,
                modifiers: egui::Modifiers::NONE,
            },
        ],
        |ui| {
            ui.add_space(430.0);
            let _ = cm3_dropdown(ui, id, &mut selected, &opts_5, false, vp, &theme);
        },
    );
    let popup_id = id.with("dropdown_popup");
    let st1 = ctx
        .data_mut(|d| d.get_temp::<DropdownPopupState>(popup_id))
        .expect("popup state after 5 options");
    assert!(st1.open);
    assert_eq!(st1.last_option_count, 5);
    let epoch1 = st1.open_epoch;

    // Grow options dynamically from 5 to 9 mid-session (e.g. Material+ enabled).
    run_frame(&ctx, vec![], |ui| {
        ui.add_space(430.0);
        let _ = cm3_dropdown(ui, id, &mut selected, &opts_9, false, vp, &theme);
    });
    let st2 = ctx
        .data_mut(|d| d.get_temp::<DropdownPopupState>(popup_id))
        .expect("popup state after 9 options");
    assert_eq!(st2.last_option_count, 9);
    assert_ne!(
        st2.open_epoch, epoch1,
        "growing option count must bump open_epoch so egui never reuses stale cached Area height"
    );
}

#[test]
fn vector3_fits_within_default_property_column_without_clipping() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let col_state = PropertyColumnState::default();
    let id = Id::unique("test_vec3_fit");
    let cfg = ScrubConfig::new(-10000.0, 10000.0, 1.0, 2);
    let mut vals = [100.0_f64, 100.0, 100.0];
    let mut row_right = 0.0_f32;
    let mut used_right = 0.0_f32;

    run_frame(&ctx, vec![], |ui| {
        ui.allocate_ui(Vec2::new(324.0, 120.0), |ui| {
            row_right = ui.max_rect().right();
            cm3_property_row_aligned(
                ui,
                "Scale",
                true,
                col_state,
                &theme,
                |_| {},
                |ui| {
                    let _ = cm3_vector3(ui, id, &mut vals, cfg, false, &theme);
                    used_right = ui.min_rect().right();
                },
            );
        });
    });
    assert!(
        used_right <= row_right + 1.0,
        "Vec3 (100.00, 100.00, 100.00) must fit inside default inspector column without clipping: used_right={used_right}, row_right={row_right}"
    );
}

#[test]
fn vector4_fits_inline_and_wraps_to_2x2_when_narrow_without_clipping() {
    let ctx = egui::Context::default();
    crate::ui::theme::install_cm3_fonts(&ctx);
    let theme = Cm3Theme::default();
    let col_state = PropertyColumnState::default();
    let cfg = ScrubConfig::new(-10000.0, 10000.0, 0.5, 1);

    // 1. Standard 1280px inspector column (~330px total -> ~198px Col 2) fits 4 components cleanly.
    let mut vals = [12.0_f64, 12.0, 40.0, 20.0];
    let mut row_right = 0.0_f32;
    let mut used_right = 0.0_f32;
    run_frame(&ctx, vec![], |ui| {
        ui.allocate_ui(Vec2::new(330.0, 120.0), |ui| {
            row_right = ui.max_rect().right();
            cm3_property_row_aligned(
                ui,
                "Amplitude",
                true,
                col_state,
                &theme,
                |_| {},
                |ui| {
                    let _ =
                        cm3_vector4(ui, Id::unique("test_vec4_fit_std"), &mut vals, cfg, &theme);
                    used_right = ui.min_rect().right();
                },
            );
        });
    });
    assert!(
        used_right <= row_right + 1.0,
        "Vec4 [12.0, 12.0, 40.0, 20.0] must fit inside default inspector column without clipping: used_right={used_right}, row_right={row_right}"
    );

    // 2. Narrow column (~200px total -> ~120px Col 2) with wide 2-decimal numbers wraps into
    //    a neat 2x2 grid (height >= 44px) and NEVER clips past row_right.
    let mut wide_vals = [120.25_f64, 120.50, 400.75, 200.00];
    let wide_cfg = ScrubConfig::new(-10000.0, 10000.0, 0.1, 2);
    let mut narrow_row_right = 0.0_f32;
    let mut narrow_used_right = 0.0_f32;
    let mut narrow_used_h = 0.0_f32;
    run_frame(&ctx, vec![], |ui| {
        ui.allocate_ui(Vec2::new(200.0, 160.0), |ui| {
            narrow_row_right = ui.max_rect().right();
            cm3_property_row_aligned(
                ui,
                "Noise",
                true,
                col_state,
                &theme,
                |_| {},
                |ui| {
                    let top = ui.cursor().top();
                    let _ = cm3_vector4(
                        ui,
                        Id::unique("test_vec4_fit_narrow"),
                        &mut wide_vals,
                        wide_cfg,
                        &theme,
                    );
                    narrow_used_right = ui.min_rect().right();
                    narrow_used_h = ui.min_rect().bottom() - top;
                },
            );
        });
    });
    assert!(
        narrow_used_right <= narrow_row_right + 1.0,
        "Wrapped 2x2 Vec4 must never clip right edge: used_right={narrow_used_right}, row_right={narrow_row_right}"
    );
    assert!(
        narrow_used_h >= 40.0,
        "Narrow Vec4 should wrap cleanly into 2 rows (2x2 grid), got height={narrow_used_h}"
    );
}

#[test]
fn dropdown_reserves_chevron_padding_and_ellipsizes_when_constrained() {
    let ctx = egui::Context::default();
    crate::ui::theme::install_cm3_fonts(&ctx);
    let theme = Cm3Theme::default();
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));
    let opts = ["Classic Dark", "High Contrast", "Ocean Slate"];
    let mut selected = 0usize;

    // Roomy container: dropdown allocates max_opt_w + 28px so full label never touches chevron.
    let mut full_box_w = 0.0_f32;
    let mut classic_w = 0.0_f32;
    run_frame(&ctx, vec![], |ui| {
        classic_w = ui
            .painter()
            .layout_no_wrap("Classic Dark".to_owned(), theme.font(13.3), theme.value())
            .size()
            .x;
        ui.allocate_ui(Vec2::new(240.0, 40.0), |ui| {
            let start = ui.cursor().left();
            let _ = cm3_dropdown(
                ui,
                Id::unique("test_dd_roomy"),
                &mut selected,
                &opts,
                false,
                vp,
                &theme,
            );
            full_box_w = ui.min_rect().right() - start;
        });
    });
    // Label starts at +5px and clips at right - 19px => available label width = full_box_w - 24px.
    assert!(
        classic_w <= full_box_w - 24.0,
        "roomy dropdown ({full_box_w}px) must leave full clearance for 'Classic Dark' ({classic_w}px) before chevron"
    );

    // Constrained container: helper produces clean ellipsis fitting strictly within max_width.
    run_frame(&ctx, vec![], |ui| {
        let max_w = (classic_w * 0.65).max(24.0);
        let g = layout_truncated_dropdown_label(
            ui.painter(),
            "Classic Dark",
            theme.font(13.3),
            theme.value(),
            max_w,
        );
        assert!(
            g.text().ends_with('…'),
            "expected ellipsis, got '{}'",
            g.text()
        );
        assert!(
            g.size().x <= max_w + 0.5,
            "truncated label ({}px) must fit strictly within {}px before chevron",
            g.size().x,
            max_w
        );
    });
}

#[test]
fn property_column_default_ratio_rebalanced_and_usable_at_220px() {
    let col = PropertyColumnState::default();
    assert!(
        (col.label_ratio - 0.48).abs() < 1e-6,
        "default PropertyColumnState::label_ratio must be 0.48, got {}",
        col.label_ratio
    );

    let ctx = egui::Context::default();
    crate::ui::theme::install_cm3_fonts(&ctx);
    let theme = Cm3Theme::default();
    let cfg = ScrubConfig::new(-10000.0, 10000.0, 1.0, 2);
    let mut vals = [100.0_f64, 100.0, 100.0];
    let mut row_right = 0.0_f32;
    let mut col2_left = 0.0_f32;
    let mut used_right = 0.0_f32;

    run_frame(&ctx, vec![], |ui| {
        ui.allocate_ui(Vec2::new(220.0, 120.0), |ui| {
            row_right = ui.max_rect().right();
            cm3_property_row_aligned(
                ui,
                "Position",
                true,
                col,
                &theme,
                |_| {},
                |ui| {
                    col2_left = ui.cursor().left();
                    let _ = cm3_vector3(
                        ui,
                        Id::unique("test_vec3_220px"),
                        &mut vals,
                        cfg,
                        false,
                        &theme,
                    );
                    used_right = ui.min_rect().right();
                },
            );
        });
    });

    let col2_w = row_right - col2_left;
    assert!(
        col2_w >= 100.0,
        "220px panel must preserve >=100px value column, got {col2_w}"
    );
    assert!(
        used_right <= row_right + 1.0,
        "220px panel Vec3 must not clip past right border: used_right={used_right}, row_right={row_right}"
    );
}

#[test]
fn floating_windows_and_color_picker_dismiss_in_strict_lifo_order_on_escape() {
    let ctx = egui::Context::default();
    let theme = Cm3Theme::default();
    let vp = Rect::from_min_size(Pos2::ZERO, Vec2::new(1280.0, 720.0));
    let picker_id = Id::unique("test_lifo_color_picker");
    let mut color = Rgba::new(1.0, 1.0, 1.0, 1.0); // top-left white corner (QA6-01 #5)

    let mut mgr = FloatingWindowManager::default();
    let spec_a = FloatingWindowSpec::new("expr_editor", "Expression Editor");
    let spec_b = FloatingWindowSpec::new("shader_editor", "Shader Editor");
    mgr.open_or_focus(&spec_a, vp);
    mgr.open_or_focus(&spec_b, vp);
    assert_eq!(mgr.focused_id.as_deref(), Some("shader_editor"));

    // Open color picker popover on top of both floating windows.
    ctx.data_mut(|d| {
        d.insert_temp(
            picker_id.with("hsva_popup"),
            ColorPopupState {
                open: true,
                initial: color,
                hsv: color.rgb().to_hsv(),
                alpha: 1.0,
                hex_buf: "rgb(255, 255, 255)".to_owned(),
                dirty: false,
            },
        );
    });

    let esc_event = || {
        vec![egui::Event::Key {
            key: Key::Escape,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::NONE,
        }]
    };

    // 1st Escape: closes ONLY the color picker; both floating windows stay open.
    let mut picker_ev = ColorPickerEvent::default();
    let mut ev_b = FloatingWindowEvent::default();
    let mut ev_a = FloatingWindowEvent::default();
    reset_transient_popover_escape_flag(&ctx);
    run_frame(&ctx, esc_event(), |ui| {
        picker_ev = cm3_hsva_color_picker(ui, picker_id, &mut color, false, false, vp, &theme);
        ev_b = cm3_floating_window(ui.ctx(), &mut mgr, &spec_b, vp, &theme, |_| {});
        ev_a = cm3_floating_window(ui.ctx(), &mut mgr, &spec_a, vp, &theme, |_| {});
    });
    assert!(picker_ev.cancelled, "1st Escape must dismiss color picker");
    assert!(
        !ev_b.closed && mgr.is_open("shader_editor"),
        "Shader Editor must stay open when color picker consumed Escape"
    );
    assert!(
        !ev_a.closed && mgr.is_open("expr_editor"),
        "Expression Editor must stay open when color picker consumed Escape"
    );

    // 2nd Escape: closes ONLY the topmost focused window (Shader Editor), transferring focus to Expression Editor.
    reset_transient_popover_escape_flag(&ctx);
    run_frame(&ctx, esc_event(), |ui| {
        ev_b = cm3_floating_window(ui.ctx(), &mut mgr, &spec_b, vp, &theme, |_| {});
        ev_a = cm3_floating_window(ui.ctx(), &mut mgr, &spec_a, vp, &theme, |_| {});
    });
    assert!(ev_b.closed && !mgr.is_open("shader_editor"));
    assert!(
        !ev_a.closed && mgr.is_open("expr_editor"),
        "Expression Editor must NOT close on the same Escape press that closed Shader Editor"
    );
    assert_eq!(mgr.focused_id.as_deref(), Some("expr_editor"));

    // 3rd Escape: closes Expression Editor.
    reset_transient_popover_escape_flag(&ctx);
    run_frame(&ctx, esc_event(), |ui| {
        ev_a = cm3_floating_window(ui.ctx(), &mut mgr, &spec_a, vp, &theme, |_| {});
    });
    assert!(ev_a.closed && !mgr.is_open("expr_editor"));
}
