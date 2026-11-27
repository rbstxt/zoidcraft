//! 3D translate / rotate / scale gizmo math, interaction, and picking.

use super::*;

pub(crate) fn write_object_3d_vec3(
    session: &mut Session,
    object_id: &str,
    property: &str,
    local_frame: f64,
    value: Vec3,
) {
    let is_animated = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .flat_map(|c| c.objects_3d.iter())
        .find(|o| o.id == object_id)
        .and_then(|o| o.properties.get(property))
        .is_some_and(|p| p.animated);

    let clean_zero = |v: f32| if v == 0.0 { 0.0_f64 } else { f64::from(v) };
    let json_val = json!([
        clean_zero(value.x),
        clean_zero(value.y),
        clean_zero(value.z)
    ]);
    if is_animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: object_id.to_owned(),
            property: property.to_owned(),
            frame: local_frame.round(),
            value: json_val,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: object_id.to_owned(),
            property: property.to_owned(),
            value: json_val,
        });
    }
}

pub(crate) fn write_object_3d_scalar(
    session: &mut Session,
    object_id: &str,
    property: &str,
    local_frame: f64,
    value: f32,
) {
    let is_animated = session
        .document()
        .video_tracks
        .iter()
        .flat_map(|t| t.clips.iter())
        .flat_map(|c| c.objects_3d.iter())
        .find(|o| o.id == object_id)
        .and_then(|o| o.properties.get(property))
        .is_some_and(|p| p.animated);

    let clean_zero = |v: f32| if v == 0.0 { 0.0_f64 } else { f64::from(v) };
    let json_val = json!(clean_zero(value));
    if is_animated {
        let _ = session.execute(Command::SetKeyframe {
            target_id: object_id.to_owned(),
            property: property.to_owned(),
            frame: local_frame.round(),
            value: json_val,
            tween: None,
        });
    } else {
        let _ = session.execute(Command::SetPropertyValue {
            target_id: object_id.to_owned(),
            property: property.to_owned(),
            value: json_val,
        });
    }
}

pub(crate) fn ray_plane_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    plane_point: Vec3,
    plane_normal: Vec3,
) -> Option<Vec3> {
    let denom = ray_dir.dot(plane_normal);
    if denom.abs() <= 1e-5 {
        return None;
    }
    let t = (plane_point - ray_origin).dot(plane_normal) / denom;
    if t < 0.0 || !t.is_finite() {
        return None;
    }
    Some(ray_origin + ray_dir * t)
}

pub(crate) fn dist_point_to_segment_2d(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let l2 = ab.length_sq();
    if l2 <= 1e-6 {
        return p.distance(a);
    }
    let t = ((p - a).dot(ab) / l2).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

pub(crate) fn point_in_quad_2d(p: Pos2, quad: [Pos2; 4]) -> bool {
    let mut pos_sign = 0;
    let mut neg_sign = 0;
    for i in 0..4 {
        let a = quad[i];
        let b = quad[(i + 1) % 4];
        let cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if cross > 0.0 {
            pos_sign += 1;
        } else if cross < 0.0 {
            neg_sign += 1;
        }
    }
    pos_sign == 0 || neg_sign == 0
}

pub(crate) fn gizmo_3d_axes_for_space(world: Mat4, space: Gizmo3DSpace) -> (Vec3, Vec3, Vec3) {
    match space {
        Gizmo3DSpace::World => (Vec3::X, Vec3::Y, Vec3::Z),
        Gizmo3DSpace::Local => (
            normalized_or(world.x_axis.truncate(), Vec3::X),
            normalized_or(world.y_axis.truncate(), Vec3::Y),
            normalized_or(world.z_axis.truncate(), Vec3::Z),
        ),
    }
}

/// Computes updated local 3D position for a Translate gizmo axis/plane/center drag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_gizmo_3d_translate(
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    handle: Gizmo3DHandle,
    origin_world: Vec3,
    origin_screen: Pos2,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    parent_world: Mat4,
    init_pos: Vec3,
    init_plane_hit: Option<Vec3>,
    start_pointer: Pos2,
    cur_pointer: Pos2,
    snap_unit: bool,
) -> Vec3 {
    let delta_screen = cur_pointer - start_pointer;
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);
    let project_axis_delta = |axis: Vec3| -> Vec3 {
        if let Some(tip_s) = cam.project_world_point(origin_world + axis * world_len, canvas_rect) {
            let s_vec = tip_s - origin_screen;
            let s_len = s_vec.length();
            if s_len > 3.0 {
                let s_dir = s_vec / s_len;
                let px_along = delta_screen.dot(s_dir);
                let mut amount = px_along * (world_len / s_len);
                if snap_unit {
                    amount = amount.round();
                }
                return axis * amount;
            }
        }
        let w_delta = cam.unproject_delta_at_depth(delta_screen, origin_world, canvas_rect);
        let mut amount = w_delta.dot(axis);
        if snap_unit {
            amount = amount.round();
        }
        axis * amount
    };

    let world_delta = match handle {
        Gizmo3DHandle::AxisX => project_axis_delta(axis_x),
        Gizmo3DHandle::AxisY => project_axis_delta(axis_y),
        Gizmo3DHandle::AxisZ => project_axis_delta(axis_z),
        Gizmo3DHandle::PlaneXY | Gizmo3DHandle::PlaneXZ | Gizmo3DHandle::PlaneYZ => {
            let normal = match handle {
                Gizmo3DHandle::PlaneXY => axis_z,
                Gizmo3DHandle::PlaneXZ => axis_y,
                _ => axis_x,
            };
            let hit_delta = init_plane_hit.and_then(|start_hit| {
                let (ro, rd) = cam.screen_ray(cur_pointer, canvas_rect)?;
                let cur_hit = ray_plane_intersect(ro, rd, origin_world, normal)?;
                Some(cur_hit - start_hit)
            });
            let mut d = hit_delta.unwrap_or_else(|| {
                let raw = cam.unproject_delta_at_depth(delta_screen, origin_world, canvas_rect);
                raw - normal * raw.dot(normal)
            });
            if snap_unit {
                d = Vec3::new(d.x.round(), d.y.round(), d.z.round());
            }
            d
        }
        _ => {
            let mut d = cam.unproject_delta_at_depth(delta_screen, origin_world, canvas_rect);
            if snap_unit {
                d = Vec3::new(d.x.round(), d.y.round(), d.z.round());
            }
            d
        }
    };

    let local_delta = if parent_world.determinant().abs() > 1e-8 {
        parent_world.inverse().transform_vector3(world_delta)
    } else {
        world_delta
    };
    init_pos + local_delta
}

/// Computes updated local 3D Euler rotation (radians) for a Rotate gizmo drag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_gizmo_3d_rotate(
    handle: Gizmo3DHandle,
    space: Gizmo3DSpace,
    cam_forward: Vec3,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    parent_world: Mat4,
    init_rot: Vec3,
    euler_order: GeomEulerOrder,
    accum_angle: f32,
    snap_15_deg: bool,
) -> Vec3 {
    let mut angle = accum_angle;
    if snap_15_deg {
        let step = 15.0_f32.to_radians();
        angle = (angle / step).round() * step;
    }
    let world_axis = match handle {
        Gizmo3DHandle::AxisX => axis_x,
        Gizmo3DHandle::AxisY => axis_y,
        Gizmo3DHandle::AxisZ => axis_z,
        _ => -cam_forward,
    };
    // Flip sign if viewer is looking along +axis so visual ring drag follows pointer CW/CCW.
    let signed_angle = if world_axis.dot(cam_forward) > 0.0 {
        -angle
    } else {
        angle
    };
    let init_q = zc_geom::euler_to_quat(init_rot, euler_order);
    let new_q = match space {
        Gizmo3DSpace::Local => {
            let local_axis = match handle {
                Gizmo3DHandle::AxisX => Vec3::X,
                Gizmo3DHandle::AxisY => Vec3::Y,
                Gizmo3DHandle::AxisZ => Vec3::Z,
                _ => {
                    let parent_inv_q = parent_world.to_scale_rotation_translation().1.inverse();
                    (init_q.inverse() * (parent_inv_q * world_axis)).normalize_or_zero()
                }
            };
            init_q * Quat::from_axis_angle(normalized_or(local_axis, Vec3::Z), signed_angle)
        }
        Gizmo3DSpace::World => {
            let parent_inv_q = if parent_world.determinant().abs() > 1e-8 {
                parent_world.to_scale_rotation_translation().1.inverse()
            } else {
                Quat::IDENTITY
            };
            let parent_local_axis = normalized_or(parent_inv_q * world_axis, world_axis);
            Quat::from_axis_angle(parent_local_axis, signed_angle) * init_q
        }
    };
    let raw = zc_geom::quat_to_euler(new_q.normalize(), euler_order);
    let clean = |v: f32| if v.abs() < 1e-5 { 0.0 } else { v };
    Vec3::new(clean(raw.x), clean(raw.y), clean(raw.z))
}

/// Computes updated local 3D scale for a Scale gizmo drag.
#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_gizmo_3d_scale(
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    handle: Gizmo3DHandle,
    origin_world: Vec3,
    origin_screen: Pos2,
    axis_x: Vec3,
    axis_y: Vec3,
    axis_z: Vec3,
    init_scale: Vec3,
    start_pointer: Pos2,
    cur_pointer: Pos2,
    snap_tenth: bool,
) -> Vec3 {
    let delta_screen = cur_pointer - start_pointer;
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);
    let axis_factor = |axis: Vec3| -> f32 {
        let px = if let Some(tip_s) =
            cam.project_world_point(origin_world + axis * world_len, canvas_rect)
        {
            let s_vec = tip_s - origin_screen;
            let s_len = s_vec.length();
            if s_len > 3.0 {
                delta_screen.dot(s_vec / s_len)
            } else {
                delta_screen.x - delta_screen.y
            }
        } else {
            delta_screen.x - delta_screen.y
        };
        let mut factor = (1.0 + px / GIZMO_SCREEN_LEN_PX).clamp(0.01, 100.0);
        if snap_tenth {
            factor = (factor * 10.0).round() / 10.0;
        }
        factor.max(0.01)
    };

    match handle {
        Gizmo3DHandle::AxisX => Vec3::new(
            init_scale.x * axis_factor(axis_x),
            init_scale.y,
            init_scale.z,
        ),
        Gizmo3DHandle::AxisY => Vec3::new(
            init_scale.x,
            init_scale.y * axis_factor(axis_y),
            init_scale.z,
        ),
        Gizmo3DHandle::AxisZ => Vec3::new(
            init_scale.x,
            init_scale.y,
            init_scale.z * axis_factor(axis_z),
        ),
        _ => {
            let mut factor =
                (1.0 + (delta_screen.x - delta_screen.y) / GIZMO_SCREEN_LEN_PX).clamp(0.01, 100.0);
            if snap_tenth {
                factor = (factor * 10.0).round() / 10.0;
            }
            init_scale * factor.max(0.01)
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn handle_and_paint_gizmo_3d(
    ui: &mut Ui,
    session: &mut Session,
    state: &mut ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    panel_resp: &egui::Response,
    seq_w: f32,
    seq_h: f32,
    pointer_in_hud: bool,
    action: &mut ViewportAction,
) -> bool {
    let Some((clip, obj, locked)) = active_scene_3d_selected_object(session) else {
        reconcile_stale_field_edit_drag(ui, session, None);
        return false;
    };
    let single_selected = session.selection.object_3d_ids.len() == 1;
    let clip_id = clip.id.clone();
    let clip_start = clip.start;
    let obj_id = obj.id.clone();
    let local_frame = session.playback.current_frame - clip_start;
    if !is_object_3d_enabled(&clip.objects_3d, &obj_id, local_frame) {
        reconcile_stale_field_edit_drag(ui, session, None);
        return false;
    }

    let cam =
        resolve_active_3d_camera(session, state, clip, local_frame, canvas_rect, seq_w, seq_h);
    let parent_world = obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
        compute_object_3d_world_matrix(&clip.objects_3d, pid, local_frame)
    });
    let init_pos = obj
        .properties
        .get("position")
        .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
    let init_rot = obj
        .properties
        .get("rotation")
        .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
    let init_scale = obj
        .properties
        .get("scale")
        .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
    let euler_order = eval_object_3d_euler_order(obj, local_frame);
    let world = compute_object_3d_world_matrix(&clip.objects_3d, &obj_id, local_frame);
    let origin_world = world.w_axis.truncate();
    let origin_screen = cam.project_world_point(origin_world, canvas_rect);
    let visible_canvas = canvas_rect.intersect(panel_rect);

    let repeater_points_world = if single_selected {
        compute_repeater_copy_origins_world(obj, world, local_frame)
    } else {
        Vec::new()
    };
    let evaluated_field = if single_selected {
        evaluate_falloff_field(obj, local_frame)
    } else {
        None
    };

    // 1. Editor-only Repeater copy origin helper points (scene-repeater-points parity).
    if !repeater_points_world.is_empty() && visible_canvas.is_positive() {
        let helper_painter = ui.painter().with_clip_rect(visible_canvas);
        paint_repeater_helper_points(
            &helper_painter,
            &cam,
            canvas_rect,
            visible_canvas,
            &repeater_points_world,
        );
    }

    // 2. Interactive Falloff Field 3D wireframe & drag handles (scene-field-edit parity).
    if handle_and_paint_falloff_field(
        ui,
        session,
        state,
        &cam,
        panel_rect,
        canvas_rect,
        visible_canvas,
        panel_resp,
        &clip_id,
        &obj_id,
        local_frame,
        world,
        origin_screen,
        locked,
        pointer_in_hud,
        evaluated_field.as_ref(),
        action,
    ) {
        return true;
    }

    let Some(origin_screen) = origin_screen else {
        return false;
    };

    let (axis_x, axis_y, axis_z) = gizmo_3d_axes_for_space(world, state.gizmo_3d_space);
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);

    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (pointer_pos, press_origin, primary_pressed, primary_down, modifiers, space_down) = ui
        .input(|i| {
            (
                i.pointer.latest_pos(),
                i.pointer.press_origin(),
                i.pointer.primary_pressed(),
                i.pointer.primary_down(),
                i.modifiers,
                !wants_kb && i.key_down(egui::Key::Space),
            )
        });
    // On `primary_pressed`, hit-test `press_origin` before `latest_pos` so brisk flick-drags on
    // 3D gizmo axes/rings still grab the handle even if the pointer already moved away in the press frame.
    let hit_pos = if primary_pressed {
        press_origin.or(pointer_pos)
    } else {
        pointer_pos
    };

    // Hit-test active mode handles.
    let mode = state.gizmo_3d_mode;
    let tip_x = cam.project_world_point(origin_world + axis_x * world_len, canvas_rect);
    let tip_y = cam.project_world_point(origin_world + axis_y * world_len, canvas_rect);
    let tip_z = cam.project_world_point(origin_world + axis_z * world_len, canvas_rect);

    let make_plane_quad = |a: Vec3, b: Vec3| -> Option<[Pos2; 4]> {
        let p0 = cam.project_world_point(
            origin_world + (a * 0.22 + b * 0.22) * world_len,
            canvas_rect,
        )?;
        let p1 = cam.project_world_point(
            origin_world + (a * 0.42 + b * 0.22) * world_len,
            canvas_rect,
        )?;
        let p2 = cam.project_world_point(
            origin_world + (a * 0.42 + b * 0.42) * world_len,
            canvas_rect,
        )?;
        let p3 = cam.project_world_point(
            origin_world + (a * 0.22 + b * 0.42) * world_len,
            canvas_rect,
        )?;
        Some([p0, p1, p2, p3])
    };
    let quad_xy = make_plane_quad(axis_x, axis_y);
    let quad_xz = make_plane_quad(axis_x, axis_z);
    let quad_yz = make_plane_quad(axis_y, axis_z);

    let ring_points = |u_ax: Vec3, v_ax: Vec3, r: f32| -> Vec<Pos2> {
        let segs = 48usize;
        let mut pts = Vec::with_capacity(segs + 1);
        for i in 0..=segs {
            let th = (i as f32 / segs as f32) * std::f32::consts::TAU;
            let (s, c) = th.sin_cos();
            let wp = origin_world + (u_ax * c + v_ax * s) * r;
            if let Some(sp) = cam.project_world_point(wp, canvas_rect) {
                pts.push(sp);
            }
        }
        pts
    };

    let ring_x = if mode == Gizmo3DMode::Rotate {
        ring_points(axis_y, axis_z, world_len)
    } else {
        Vec::new()
    };
    let ring_y = if mode == Gizmo3DMode::Rotate {
        ring_points(axis_x, axis_z, world_len)
    } else {
        Vec::new()
    };
    let ring_z = if mode == Gizmo3DMode::Rotate {
        ring_points(axis_x, axis_y, world_len)
    } else {
        Vec::new()
    };
    let ring_screen = if mode == Gizmo3DMode::Rotate {
        ring_points(cam.right, cam.up, world_len * 1.15)
    } else {
        Vec::new()
    };

    let hovered_handle = if !pointer_in_hud
        && !space_down
        && let Some(pos) = hit_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
        && visible_canvas.contains(pos)
    {
        match mode {
            Gizmo3DMode::Translate => {
                if pos.distance(origin_screen) <= 9.0 {
                    Some(Gizmo3DHandle::Center)
                } else if quad_xy.is_some_and(|q| point_in_quad_2d(pos, q)) {
                    Some(Gizmo3DHandle::PlaneXY)
                } else if quad_xz.is_some_and(|q| point_in_quad_2d(pos, q)) {
                    Some(Gizmo3DHandle::PlaneXZ)
                } else if quad_yz.is_some_and(|q| point_in_quad_2d(pos, q)) {
                    Some(Gizmo3DHandle::PlaneYZ)
                } else {
                    let mut best: Option<(Gizmo3DHandle, f32)> = None;
                    for (h, tip) in [
                        (Gizmo3DHandle::AxisX, tip_x),
                        (Gizmo3DHandle::AxisY, tip_y),
                        (Gizmo3DHandle::AxisZ, tip_z),
                    ] {
                        if let Some(t) = tip {
                            let d = dist_point_to_segment_2d(pos, origin_screen, t);
                            if d <= 9.0 && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((h, d));
                            }
                        }
                    }
                    best.map(|(h, _)| h)
                }
            }
            Gizmo3DMode::Scale => {
                if pos.distance(origin_screen) <= 10.0 {
                    Some(Gizmo3DHandle::Center)
                } else {
                    let mut best: Option<(Gizmo3DHandle, f32)> = None;
                    for (h, tip) in [
                        (Gizmo3DHandle::AxisX, tip_x),
                        (Gizmo3DHandle::AxisY, tip_y),
                        (Gizmo3DHandle::AxisZ, tip_z),
                    ] {
                        if let Some(t) = tip {
                            let d = dist_point_to_segment_2d(pos, origin_screen, t)
                                .min(pos.distance(t));
                            if d <= 10.0 && best.is_none_or(|(_, bd)| d < bd) {
                                best = Some((h, d));
                            }
                        }
                    }
                    best.map(|(h, _)| h)
                }
            }
            Gizmo3DMode::Rotate => {
                let min_poly_dist = |pts: &[Pos2]| -> f32 {
                    pts.windows(2)
                        .map(|w| dist_point_to_segment_2d(pos, w[0], w[1]))
                        .fold(f32::INFINITY, f32::min)
                };
                let mut best: Option<(Gizmo3DHandle, f32)> = None;
                for (h, pts) in [
                    (Gizmo3DHandle::AxisX, &ring_x),
                    (Gizmo3DHandle::AxisY, &ring_y),
                    (Gizmo3DHandle::AxisZ, &ring_z),
                    (Gizmo3DHandle::ScreenRing, &ring_screen),
                ] {
                    let d = min_poly_dist(pts);
                    if d <= 9.0 && best.is_none_or(|(_, bd)| d < bd) {
                        best = Some((h, d));
                    }
                }
                best.map(|(h, _)| h)
            }
        }
    } else {
        None
    };

    let active_handle = match &state.drag_state {
        ViewportDragState::Gizmo3D(d) if d.object_id == obj_id => Some(d.handle),
        _ => hovered_handle,
    };

    // Paint gizmo visuals clipped to `visible_canvas`.
    let painter = ui.painter().with_clip_rect(visible_canvas);
    let col_for = |h: Gizmo3DHandle, base: Color32| -> Color32 {
        if active_handle == Some(h) {
            COLOR_GIZMO_HOVER
        } else {
            base
        }
    };

    match mode {
        Gizmo3DMode::Translate => {
            for (h, quad, base) in [
                (Gizmo3DHandle::PlaneXY, quad_xy, COLOR_GIZMO_Z),
                (Gizmo3DHandle::PlaneXZ, quad_xz, COLOR_GIZMO_Y),
                (Gizmo3DHandle::PlaneYZ, quad_yz, COLOR_GIZMO_X),
            ] {
                if let Some(q) = quad {
                    let c = col_for(h, base);
                    let fill = Color32::from_rgba_unmultiplied(
                        c.r(),
                        c.g(),
                        c.b(),
                        if active_handle == Some(h) { 125 } else { 58 },
                    );
                    painter.add(Shape::convex_polygon(q.to_vec(), fill, Stroke::new(1.0, c)));
                }
            }
            for (h, tip, base, label) in [
                (Gizmo3DHandle::AxisX, tip_x, COLOR_GIZMO_X, "X"),
                (Gizmo3DHandle::AxisY, tip_y, COLOR_GIZMO_Y, "Y"),
                (Gizmo3DHandle::AxisZ, tip_z, COLOR_GIZMO_Z, "Z"),
            ] {
                if let Some(t) = tip {
                    let c = col_for(h, base);
                    painter.line_segment([origin_screen, t], Stroke::new(2.2, c));
                    let dir = (t - origin_screen).normalized();
                    let perp = vec2(-dir.y, dir.x);
                    let arrow_tip = t + dir * 2.0;
                    let arrow_base = t - dir * 10.0;
                    painter.add(Shape::convex_polygon(
                        vec![arrow_tip, arrow_base + perp * 4.5, arrow_base - perp * 4.5],
                        c,
                        Stroke::NONE,
                    ));
                    painter.text(
                        t + dir * 10.0,
                        Align2::CENTER_CENTER,
                        label,
                        FontId::monospace(10.0),
                        c,
                    );
                }
            }
            let c_center = col_for(Gizmo3DHandle::Center, COLOR_GIZMO_SCREEN);
            painter.circle(
                origin_screen,
                5.5,
                Color32::from_rgba_unmultiplied(c_center.r(), c_center.g(), c_center.b(), 90),
                Stroke::new(1.5, c_center),
            );
        }
        Gizmo3DMode::Rotate => {
            for (h, pts, base) in [
                (Gizmo3DHandle::ScreenRing, &ring_screen, COLOR_GIZMO_SCREEN),
                (Gizmo3DHandle::AxisX, &ring_x, COLOR_GIZMO_X),
                (Gizmo3DHandle::AxisY, &ring_y, COLOR_GIZMO_Y),
                (Gizmo3DHandle::AxisZ, &ring_z, COLOR_GIZMO_Z),
            ] {
                if pts.len() >= 2 {
                    let c = col_for(h, base);
                    let width = if active_handle == Some(h) {
                        2.8
                    } else if h == Gizmo3DHandle::ScreenRing {
                        1.4
                    } else {
                        2.0
                    };
                    painter.add(Shape::line(pts.clone(), Stroke::new(width, c)));
                }
            }
            painter.circle_filled(origin_screen, 3.0, COLOR_GIZMO_SCREEN);
        }
        Gizmo3DMode::Scale => {
            for (h, tip, base, label) in [
                (Gizmo3DHandle::AxisX, tip_x, COLOR_GIZMO_X, "X"),
                (Gizmo3DHandle::AxisY, tip_y, COLOR_GIZMO_Y, "Y"),
                (Gizmo3DHandle::AxisZ, tip_z, COLOR_GIZMO_Z, "Z"),
            ] {
                if let Some(t) = tip {
                    let c = col_for(h, base);
                    painter.line_segment([origin_screen, t], Stroke::new(2.2, c));
                    let cap = Rect::from_center_size(t, vec2(8.0, 8.0));
                    painter.rect_filled(cap, CornerRadius::same(1), c);
                    let dir = (t - origin_screen).normalized();
                    painter.text(
                        t + dir * 11.0,
                        Align2::CENTER_CENTER,
                        label,
                        FontId::monospace(10.0),
                        c,
                    );
                }
            }
            let c_center = col_for(Gizmo3DHandle::Center, COLOR_GIZMO_SCREEN);
            let center_box = Rect::from_center_size(origin_screen, vec2(10.0, 10.0));
            painter.rect(
                center_box,
                CornerRadius::same(1),
                Color32::from_rgba_unmultiplied(c_center.r(), c_center.g(), c_center.b(), 100),
                Stroke::new(1.5, c_center),
                StrokeKind::Inside,
            );
        }
    }

    // Start gizmo drag on primary press over handle (preserving double-click reset for editing camera).
    if !locked
        && !space_down
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(h) = hovered_handle
        && let Some(pos) = hit_pos
    {
        ui.ctx().set_cursor_icon(match mode {
            Gizmo3DMode::Translate => CursorIcon::Move,
            Gizmo3DMode::Rotate => CursorIcon::Grab,
            Gizmo3DMode::Scale => CursorIcon::ResizeNeSw,
        });
        let is_primary_double_click = panel_resp.double_clicked_by(PointerButton::Primary)
            || ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary));
        if primary_pressed
            && !(is_editing_camera_active(session) && is_primary_double_click)
            && session.begin_transaction("3D Transform Gizmo").is_ok()
        {
            let init_plane_hit = match h {
                Gizmo3DHandle::PlaneXY | Gizmo3DHandle::PlaneXZ | Gizmo3DHandle::PlaneYZ => {
                    let normal = match h {
                        Gizmo3DHandle::PlaneXY => axis_z,
                        Gizmo3DHandle::PlaneXZ => axis_y,
                        _ => axis_x,
                    };
                    cam.screen_ray(pos, canvas_rect)
                        .and_then(|(ro, rd)| ray_plane_intersect(ro, rd, origin_world, normal))
                }
                _ => None,
            };
            state.drag_state = ViewportDragState::Gizmo3D(Gizmo3DDrag {
                clip_id,
                object_id: obj_id,
                mode,
                space: state.gizmo_3d_space,
                handle: h,
                start_pointer: pos,
                prev_pointer: pos,
                origin_world,
                origin_screen,
                parent_world,
                axis_x,
                axis_y,
                axis_z,
                init_pos,
                init_rot,
                init_scale,
                euler_order,
                accum_angle: 0.0,
                init_plane_hit,
                changed: false,
            });
            // Fall through into the active drag update below so coalesced press+move flicks apply
            // immediately from `press_origin` -> `latest_pos`.
        } else {
            return false;
        }
    }

    // Active gizmo drag update.
    let current = std::mem::take(&mut state.drag_state);
    if let ViewportDragState::Gizmo3D(mut drag) = current {
        if let Some(pos) = pointer_pos {
            let delta = pos - drag.start_pointer;
            if !drag.changed && delta.length() > 0.5 {
                drag.changed = true;
            }
            if drag.changed {
                match drag.mode {
                    Gizmo3DMode::Translate => {
                        let next_pos = compute_gizmo_3d_translate(
                            &cam,
                            canvas_rect,
                            drag.handle,
                            drag.origin_world,
                            drag.origin_screen,
                            drag.axis_x,
                            drag.axis_y,
                            drag.axis_z,
                            drag.parent_world,
                            drag.init_pos,
                            drag.init_plane_hit,
                            drag.start_pointer,
                            pos,
                            modifiers.shift,
                        );
                        write_object_3d_vec3(
                            session,
                            &drag.object_id,
                            "position",
                            local_frame,
                            next_pos,
                        );
                        action.transform_modified = true;
                    }
                    Gizmo3DMode::Rotate => {
                        let prev_vec = drag.prev_pointer - drag.origin_screen;
                        let cur_vec = pos - drag.origin_screen;
                        if let Some(d_angle) = unwrapped_rotation_delta(prev_vec, cur_vec) {
                            drag.accum_angle += d_angle;
                            drag.prev_pointer = pos;
                        }
                        let next_rot = compute_gizmo_3d_rotate(
                            drag.handle,
                            drag.space,
                            cam.forward,
                            drag.axis_x,
                            drag.axis_y,
                            drag.axis_z,
                            drag.parent_world,
                            drag.init_rot,
                            drag.euler_order,
                            drag.accum_angle,
                            modifiers.shift,
                        );
                        write_object_3d_vec3(
                            session,
                            &drag.object_id,
                            "rotation",
                            local_frame,
                            next_rot,
                        );
                        action.transform_modified = true;
                    }
                    Gizmo3DMode::Scale => {
                        let next_scale = compute_gizmo_3d_scale(
                            &cam,
                            canvas_rect,
                            drag.handle,
                            drag.origin_world,
                            drag.origin_screen,
                            drag.axis_x,
                            drag.axis_y,
                            drag.axis_z,
                            drag.init_scale,
                            drag.start_pointer,
                            pos,
                            modifiers.shift,
                        );
                        write_object_3d_vec3(
                            session,
                            &drag.object_id,
                            "scale",
                            local_frame,
                            next_scale,
                        );
                        action.transform_modified = true;
                    }
                }
            }
        }

        if primary_down {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            state.drag_state = ViewportDragState::Gizmo3D(drag);
            return true;
        }
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        return drag.changed;
    }

    state.drag_state = current;
    false
}

pub(crate) fn ray_aabb_intersect(origin: Vec3, dir: Vec3, min_b: Vec3, max_b: Vec3) -> Option<f32> {
    let mut t_min = 0.0_f32;
    let mut t_max = f32::INFINITY;
    for i in 0..3 {
        let o = origin[i];
        let d = dir[i];
        if d.abs() <= 1e-7 {
            if o < min_b[i] || o > max_b[i] {
                return None;
            }
        } else {
            let inv_d = 1.0 / d;
            let mut t1 = (min_b[i] - o) * inv_d;
            let mut t2 = (max_b[i] - o) * inv_d;
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
            if t_min > t_max {
                return None;
            }
        }
    }
    if t_max >= 0.0 { Some(t_min) } else { None }
}

pub(crate) fn ray_sphere_intersect(
    origin: Vec3,
    dir: Vec3,
    center: Vec3,
    radius: f32,
) -> Option<f32> {
    let oc = origin - center;
    let b = oc.dot(dir);
    let c = oc.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let t0 = -b - sqrt_d;
    if t0 >= 0.0 {
        Some(t0)
    } else {
        let t1 = -b + sqrt_d;
        (t1 >= 0.0).then_some(t1)
    }
}

pub(crate) fn estimate_object_3d_local_half_extents(
    obj: &SessionObject3D,
    local_frame: f64,
) -> Vec3 {
    let kind_low = obj.kind.to_ascii_lowercase();
    if let Some(p) = obj.properties.get("size") {
        let s = eval_prop_vec3(p, local_frame, Vec3::splat(10.0)).abs() * 0.5;
        if s.length_squared() > 1e-3 {
            return s.max(Vec3::splat(1.0));
        }
    }
    if let Some(p) = obj.properties.get("radius") {
        let r = (eval_prop_scalar(p, local_frame, 5.0) as f32)
            .abs()
            .max(1.0);
        let h = obj.properties.get("height").map_or(r * 2.0, |hp| {
            (eval_prop_scalar(hp, local_frame, f64::from(r * 2.0)) as f32).abs()
        });
        return Vec3::new(r, (h * 0.5).max(1.0), r);
    }
    if kind_low.contains("text") {
        let height = obj.properties.get("height").map_or(8.0, |p| {
            (eval_prop_scalar(p, local_frame, 8.0) as f32)
                .abs()
                .max(2.0)
        });
        return Vec3::new(height * 2.2, height * 0.7, 3.0);
    }
    if is_camera_3d_kind(&obj.kind) || kind_low.contains("light") {
        return Vec3::splat(4.5);
    }
    Vec3::splat(5.0)
}

pub(crate) fn raycast_pick_3d_object(
    session: &Session,
    state: &ViewportState,
    click_pos: Pos2,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) -> Option<(String, String)> {
    let (clip, local_frame, _) = active_scene_3d_clip(session)?;
    let cam =
        resolve_active_3d_camera(session, state, clip, local_frame, canvas_rect, seq_w, seq_h);
    let (ray_origin, ray_dir) = cam.screen_ray(click_pos, canvas_rect)?;

    let obj_index = build_object_3d_index(&clip.objects_3d);
    let mut closest: Option<(f32, String, String)> = None;
    for obj in &clip.objects_3d {
        if !is_object_3d_enabled_indexed(&clip.objects_3d, &obj_index, &obj.id, local_frame) {
            continue;
        }
        // Do not pick the active camera we are currently viewing through.
        if cam.camera_object_id.as_deref() == Some(obj.id.as_str()) {
            continue;
        }
        let world = compute_object_3d_world_matrix_indexed(
            &clip.objects_3d,
            &obj_index,
            &obj.id,
            local_frame,
        );
        let half = estimate_object_3d_local_half_extents(obj, local_frame);

        let hit_t = if world.determinant().abs() > 1e-8 {
            let inv = world.inverse();
            let lo = inv.transform_point3(ray_origin);
            let ld = inv.transform_vector3(ray_dir);
            ray_aabb_intersect(lo, ld, -half, half).map(|t_local| {
                let hit_world = world.transform_point3(lo + ld * t_local);
                (hit_world - ray_origin).dot(ray_dir).max(0.0)
            })
        } else {
            let center = world.w_axis.truncate();
            let max_s = world
                .x_axis
                .truncate()
                .length()
                .max(world.y_axis.truncate().length())
                .max(world.z_axis.truncate().length())
                .max(0.5);
            ray_sphere_intersect(ray_origin, ray_dir, center, half.length() * max_s)
        };

        if let Some(t) = hit_t
            && closest.as_ref().is_none_or(|(best_t, _, _)| t < *best_t)
        {
            closest = Some((t, obj.id.clone(), obj.kind.clone()));
        }
    }
    closest.map(|(_, id, kind)| (id, kind))
}

// ---------------------------------------------------------------------------
// Repeater Copy Origin Helper Points (Editor-Only Viewport Overlay)
// ---------------------------------------------------------------------------

const MAX_REPEATER_HELPER_POINTS: usize = zc_scene::MAX_REPEATS as usize;
const COLOR_REPEATER_POINT_FILL: Color32 = Color32::from_rgb(0x3b, 0x82, 0xf6);
const COLOR_REPEATER_POINT_OUTLINE: Color32 = Color32::from_rgb(0xe0, 0xf2, 0xfe);
const REPEATER_POINT_RADIUS_PX: f32 = 3.5;

/// Repeater variant supported by the editor-only copy-origin helper overlay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RepeaterHelperKind {
    Step,
    Linear,
    Radial,
    Grid,
    Random,
    Echo,
}

/// Detects whether a 3D object kind is one of the 6 Repeater classes.
#[must_use]
pub(crate) fn detect_repeater_helper_kind(kind: &str) -> Option<RepeaterHelperKind> {
    let mut normalized = kind.trim().to_ascii_lowercase().replace('_', "-");
    if let Some(rest) = normalized.strip_prefix("zoidcraft:") {
        normalized = format!("zoidium:{rest}");
    }
    if let Some(rest) = normalized.strip_prefix("zoidium:effector-plus/") {
        normalized = format!("zoidium:repeater/{rest}");
    } else if let Some(rest) = normalized.strip_prefix("effector-plus/") {
        normalized = format!("zoidium:repeater/{rest}");
    } else if let Some(rest) = normalized.strip_prefix("repeater/") {
        normalized = format!("zoidium:repeater/{rest}");
    }
    match normalized.as_str() {
        "zoidium:repeater/repeater" | "repeater" | "step-repeater" => {
            Some(RepeaterHelperKind::Step)
        }
        "zoidium:repeater/linear-repeater" | "linear-repeater" => Some(RepeaterHelperKind::Linear),
        "zoidium:repeater/radial"
        | "zoidium:repeater/radial-repeater"
        | "radial-repeater"
        | "radial" => Some(RepeaterHelperKind::Radial),
        "zoidium:repeater/grid" | "zoidium:repeater/grid-repeater" | "grid-repeater" | "grid" => {
            Some(RepeaterHelperKind::Grid)
        }
        "zoidium:repeater/random-repeater" | "random-repeater" => Some(RepeaterHelperKind::Random),
        "zoidium:repeater/echo-repeater" | "echo-repeater" => Some(RepeaterHelperKind::Echo),
        _ => None,
    }
}

fn eval_prop_str_from_keys(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
) -> Option<String> {
    for &key in keys {
        if let Some(prop) = props.get(key) {
            let val = if prop.keyframes.is_empty() {
                &prop.default_value
            } else if !prop.animated
                || prop.keyframes.len() == 1
                || local_frame <= prop.keyframes[0].frame
            {
                &prop.keyframes[0].value
            } else {
                let idx = prop
                    .keyframes
                    .partition_point(|k| k.frame <= local_frame)
                    .saturating_sub(1);
                &prop.keyframes[idx].value
            };
            if let Some(s) = val.as_str() {
                return Some(s.to_owned());
            }
        }
    }
    None
}

fn eval_prop_bool_from_keys(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: bool,
) -> bool {
    if let Some(s) = eval_prop_str_from_keys(props, keys, local_frame) {
        return match s.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "no" | "off" => false,
            "1" | "true" | "yes" | "on" => true,
            _ => default,
        };
    }
    let (v, _, found) =
        eval_prop_from_keys_scalar(props, keys, local_frame, if default { 1.0 } else { 0.0 });
    if found { v >= 0.5 } else { default }
}

fn eval_prop_u32_from_keys(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: u32,
) -> u32 {
    let (v, _, _) = eval_prop_from_keys_scalar(props, keys, local_frame, f64::from(default));
    if v.is_finite() {
        v.round().clamp(0.0, f64::from(u32::MAX)) as u32
    } else {
        default
    }
}

fn eval_prop_i32_from_keys(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: i32,
) -> i32 {
    let (v, _, _) = eval_prop_from_keys_scalar(props, keys, local_frame, f64::from(default));
    if v.is_finite() {
        v.round().clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
    } else {
        default
    }
}

fn eval_prop_vec3_or_scalar(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: Vec3,
) -> Vec3 {
    for &key in keys {
        if let Some(prop) = props.get(key) {
            let sample = if prop.keyframes.is_empty() {
                &prop.default_value
            } else {
                &prop.keyframes[0].value
            };
            if sample.is_array() {
                return eval_prop_vec3(prop, local_frame, default);
            }
            if sample.is_number() {
                let s = eval_prop_scalar(prop, local_frame, f64::from(default.x)) as f32;
                return Vec3::splat(s);
            }
        }
    }
    default
}

/// Computes every repeater copy's local-space origin (up to 128 copies, matching `MAX_REPEATS`).
#[must_use]
pub(crate) fn compute_repeater_copy_origins_local(
    obj: &SessionObject3D,
    local_frame: f64,
) -> Vec<Vec3> {
    let Some(base_kind) = detect_repeater_helper_kind(&obj.kind) else {
        return Vec::new();
    };
    let props = &obj.properties;
    let kind = match base_kind {
        RepeaterHelperKind::Radial | RepeaterHelperKind::Grid => base_kind,
        other => match eval_prop_str_from_keys(props, &["mode", "repeaterMode"], local_frame)
            .map(|s| s.trim().to_ascii_lowercase())
            .as_deref()
        {
            Some("linear" | "1") => RepeaterHelperKind::Linear,
            Some("random" | "2") => RepeaterHelperKind::Random,
            Some("echo" | "3") => RepeaterHelperKind::Echo,
            Some("step" | "0") => RepeaterHelperKind::Step,
            _ => other,
        },
    };

    match kind {
        RepeaterHelperKind::Step => {
            let count =
                eval_prop_u32_from_keys(props, &["count", "copies", "repeats"], local_frame, 5)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let step = eval_prop_from_keys_vec3(
                props,
                &[
                    "positionStep",
                    "position_step",
                    "offsetPosition",
                    "offset_position",
                    "offset",
                    "positionOffset",
                ],
                local_frame,
                Vec3::new(10.0, 0.0, 0.0),
            );
            (0..count).map(|i| step * (i as f32)).collect()
        }
        RepeaterHelperKind::Linear => {
            let count =
                eval_prop_u32_from_keys(props, &["count", "copies", "repeats"], local_frame, 5)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let end_pos = eval_prop_from_keys_vec3(
                props,
                &[
                    "positionEnd",
                    "position_end",
                    "endPosition",
                    "offsetPosition",
                    "offset_position",
                    "offset",
                ],
                local_frame,
                Vec3::new(40.0, 0.0, 0.0),
            );
            (0..count)
                .map(|i| {
                    let amount = if count > 1 {
                        i as f32 / (count - 1) as f32
                    } else {
                        0.0
                    };
                    end_pos * amount
                })
                .collect()
        }
        RepeaterHelperKind::Radial => {
            let count =
                eval_prop_u32_from_keys(props, &["count", "copies", "repeats"], local_frame, 8)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let radius = eval_prop_from_keys_scalar(props, &["radius"], local_frame, 28.0).0 as f32;
            let has_sweep = props.contains_key("sweep");
            let has_end_angle = props.contains_key("endAngle") || props.contains_key("end_angle");
            let raw_start =
                eval_prop_from_keys_scalar(props, &["startAngle", "start_angle"], local_frame, 0.0)
                    .0 as f32;

            let (start_rad, span_rad) = if has_sweep && !has_end_angle {
                let raw_sweep = eval_prop_from_keys_scalar(
                    props,
                    &["sweep"],
                    local_frame,
                    f64::from(std::f32::consts::TAU),
                )
                .0 as f32;
                let degrees_mode = raw_sweep.abs() > std::f32::consts::TAU * 2.0 + 1e-3;
                if degrees_mode {
                    (raw_start.to_radians(), raw_sweep.to_radians())
                } else {
                    (raw_start, raw_sweep)
                }
            } else if has_end_angle {
                let raw_end = eval_prop_from_keys_scalar(
                    props,
                    &["endAngle", "end_angle"],
                    local_frame,
                    360.0,
                )
                .0 as f32;
                let s_rad = raw_start.to_radians();
                (s_rad, raw_end.to_radians() - s_rad)
            } else {
                let s_rad = if raw_start.abs() > std::f32::consts::TAU + 1e-3 {
                    raw_start.to_radians()
                } else {
                    raw_start
                };
                (s_rad, std::f32::consts::TAU)
            };

            // Plane indices match Zoidium: 0 = XY, 1 = XZ, 2 = YZ.
            // Axis indices match ZoidCraft defaults: 0 = X (YZ plane), 1 = Y (XZ plane), 2 = Z (XY plane).
            let plane = if props.contains_key("plane") {
                if let Some(s) = eval_prop_str_from_keys(props, &["plane"], local_frame) {
                    match s.trim().to_ascii_lowercase().as_str() {
                        "xy" | "0" => 0u32,
                        "yz" | "2" => 2u32,
                        _ => 1u32,
                    }
                } else {
                    eval_prop_u32_from_keys(props, &["plane"], local_frame, 1).min(2)
                }
            } else if props.contains_key("axis") {
                if let Some(s) = eval_prop_str_from_keys(props, &["axis"], local_frame) {
                    match s.trim().to_ascii_lowercase().as_str() {
                        "x" | "0" => 2u32,
                        "y" | "1" => 1u32,
                        _ => 0u32,
                    }
                } else {
                    match eval_prop_u32_from_keys(props, &["axis"], local_frame, 2) {
                        0 => 2u32,
                        1 => 1u32,
                        _ => 0u32,
                    }
                }
            } else {
                0u32
            };

            let full_circle = (span_rad.abs() - std::f32::consts::TAU).abs() < 1e-3;
            (0..count)
                .map(|i| {
                    let t = if count <= 1 {
                        0.0
                    } else if full_circle {
                        i as f32 / count as f32
                    } else {
                        i as f32 / (count - 1) as f32
                    };
                    let angle = start_rad + span_rad * t;
                    let (s, c) = angle.sin_cos();
                    match plane {
                        0 => Vec3::new(c * radius, s * radius, 0.0),
                        2 => Vec3::new(0.0, c * radius, s * radius),
                        _ => Vec3::new(c * radius, 0.0, s * radius),
                    }
                })
                .collect()
        }
        RepeaterHelperKind::Grid => {
            let nx =
                eval_prop_u32_from_keys(props, &["countX", "count_x", "xCount"], local_frame, 3)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let ny =
                eval_prop_u32_from_keys(props, &["countY", "count_y", "yCount"], local_frame, 3)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let nz =
                eval_prop_u32_from_keys(props, &["countZ", "count_z", "zCount"], local_frame, 1)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let spacing_v = eval_prop_vec3_or_scalar(
                props,
                &["spacing", "gridSpacing", "step"],
                local_frame,
                Vec3::splat(16.0),
            );
            let sp_x = eval_prop_from_keys_scalar(
                props,
                &["spacingX", "spacing_x"],
                local_frame,
                f64::from(spacing_v.x),
            )
            .0 as f32;
            let sp_y = eval_prop_from_keys_scalar(
                props,
                &["spacingY", "spacing_y"],
                local_frame,
                f64::from(spacing_v.y),
            )
            .0 as f32;
            let sp_z = eval_prop_from_keys_scalar(
                props,
                &["spacingZ", "spacing_z"],
                local_frame,
                f64::from(spacing_v.z),
            )
            .0 as f32;
            let centered = eval_prop_bool_from_keys(props, &["centered"], local_frame, true);
            let center_x = if centered {
                (nx as f32 - 1.0) * 0.5
            } else {
                0.0
            };
            let center_y = if centered {
                (ny as f32 - 1.0) * 0.5
            } else {
                0.0
            };
            let center_z = if centered {
                (nz as f32 - 1.0) * 0.5
            } else {
                0.0
            };

            let total_cap = (nx as usize)
                .saturating_mul(ny as usize)
                .saturating_mul(nz as usize)
                .min(MAX_REPEATER_HELPER_POINTS);
            let mut out = Vec::with_capacity(total_cap);
            'grid: for iz in 0..nz {
                let oz = (iz as f32 - center_z) * sp_z;
                for iy in 0..ny {
                    let oy = (iy as f32 - center_y) * sp_y;
                    for ix in 0..nx {
                        if out.len() >= total_cap {
                            break 'grid;
                        }
                        let ox = (ix as f32 - center_x) * sp_x;
                        out.push(Vec3::new(ox, oy, oz));
                    }
                }
            }
            out
        }
        RepeaterHelperKind::Random => {
            let count =
                eval_prop_u32_from_keys(props, &["count", "copies", "repeats"], local_frame, 5)
                    .clamp(1, zc_scene::MAX_REPEATS);
            let position_min = eval_prop_from_keys_vec3(
                props,
                &["positionMin", "position_min"],
                local_frame,
                Vec3::splat(-10.0),
            );
            let position_max = eval_prop_from_keys_vec3(
                props,
                &["positionMax", "position_max"],
                local_frame,
                Vec3::splat(10.0),
            );
            let seed = eval_prop_i32_from_keys(props, &["seed"], local_frame, 1);
            let params = zc_scene::RepeaterParams {
                mode: zc_scene::RepeaterMode::Random,
                count,
                position_min,
                position_max,
                seed,
                ..zc_scene::RepeaterParams::default()
            };
            (0..count)
                .map(|i| zc_scene::repeater_instance_transform(&params, i).position)
                .collect()
        }
        RepeaterHelperKind::Echo => {
            let count =
                eval_prop_u32_from_keys(props, &["count", "copies", "repeats"], local_frame, 5)
                    .clamp(1, zc_scene::MAX_REPEATS);
            vec![Vec3::ZERO; count as usize]
        }
    }
}

/// Computes every repeater copy's world-space origin by transforming local copy origins
/// through the repeater's world matrix.
#[must_use]
pub(crate) fn compute_repeater_copy_origins_world(
    obj: &SessionObject3D,
    world: Mat4,
    local_frame: f64,
) -> Vec<Vec3> {
    compute_repeater_copy_origins_local(obj, local_frame)
        .into_iter()
        .map(|p| world.transform_point3(p))
        .collect()
}

pub(crate) fn paint_repeater_helper_points(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    visible_canvas: Rect,
    world_points: &[Vec3],
) {
    let bounds = visible_canvas.expand(8.0);
    let halo_col = Color32::from_rgba_unmultiplied(11, 18, 32, 185);
    let outline_col = Color32::from_rgba_unmultiplied(
        COLOR_REPEATER_POINT_OUTLINE.r(),
        COLOR_REPEATER_POINT_OUTLINE.g(),
        COLOR_REPEATER_POINT_OUTLINE.b(),
        235,
    );
    let fill_col = Color32::from_rgba_unmultiplied(
        COLOR_REPEATER_POINT_FILL.r(),
        COLOR_REPEATER_POINT_FILL.g(),
        COLOR_REPEATER_POINT_FILL.b(),
        245,
    );
    for &wp in world_points.iter().take(MAX_REPEATER_HELPER_POINTS) {
        if let Some(sp) = cam.project_world_point(wp, canvas_rect)
            && bounds.contains(sp)
        {
            painter.circle_filled(sp, REPEATER_POINT_RADIUS_PX + 1.1, halo_col);
            painter.circle(
                sp,
                REPEATER_POINT_RADIUS_PX,
                fill_col,
                Stroke::new(1.0, outline_col),
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Interactive Falloff Field 3D Wireframe & Drag Handles
// ---------------------------------------------------------------------------

const COLOR_FIELD_ACTIVE: Color32 = Color32::from_rgb(0x50, 0xe0, 0xd0);
const FIELD_HANDLE_RADIUS_PX: f32 = 5.0;
const FIELD_HANDLE_HIT_RADIUS_PX: f32 = 10.0;
const FIELD_RING_SEGMENTS: usize = 32;

/// Active geometric Falloff Field shape rendered in the 3D editor viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldWireframeKind {
    Linear,
    Box,
    Sphere,
    Cylinder,
}

/// Which session property controls a falloff field's size/radius so drag handles
/// write back to the exact schema property present on the effector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldExtentProperty {
    Radius(&'static str),
    Size(&'static str),
    Scale(&'static str),
}

/// Interactive handle identifier for direct viewport manipulation of a falloff field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldEditHandle {
    Center,
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

/// Evaluated falloff field properties in effector-local space.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct EvaluatedFalloffField {
    pub kind: FieldWireframeKind,
    pub position_prop: &'static str,
    pub extent_prop: FieldExtentProperty,
    pub local_position: Vec3,
    pub rotation_deg: Vec3,
    pub local_direction: Option<Vec3>,
    pub raw_extent: Vec3,
    pub outer_half: Vec3,
    pub inner_half: Option<Vec3>,
    pub falloff_frac: f32,
}

/// World-space wireframe line segments and interactive drag handle positions for a falloff field.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FieldWireframeGeometry {
    pub kind: FieldWireframeKind,
    /// Primary solid segments (inner full-strength core when falloff is partial, else outer boundary).
    pub primary_segments: Vec<(Vec3, Vec3)>,
    /// Secondary dashed/dimmer falloff boundary segments (outer falloff shell or Linear end plane).
    pub falloff_segments: Vec<(Vec3, Vec3)>,
    pub center_world: Vec3,
    pub handles: Vec<(FieldEditHandle, Vec3)>,
}

#[derive(Debug, Clone)]
struct FieldEditDrag {
    clip_id: String,
    object_id: String,
    handle: FieldEditHandle,
    kind: FieldWireframeKind,
    position_prop: &'static str,
    extent_prop: FieldExtentProperty,
    start_pointer: Pos2,
    init_center_world: Vec3,
    init_plane_hit: Option<Vec3>,
    effector_world: Mat4,
    drag_axis_world: Vec3,
    init_handle_screen: Pos2,
    init_center_screen: Pos2,
    init_local_pos: Vec3,
    init_raw_extent: Vec3,
    init_outer_half: Vec3,
    changed: bool,
}

impl Default for FieldEditDrag {
    fn default() -> Self {
        Self {
            clip_id: String::new(),
            object_id: String::new(),
            handle: FieldEditHandle::Center,
            kind: FieldWireframeKind::Sphere,
            position_prop: "fieldPosition",
            extent_prop: FieldExtentProperty::Radius("fieldRadius"),
            start_pointer: Pos2::ZERO,
            init_center_world: Vec3::ZERO,
            init_plane_hit: None,
            effector_world: Mat4::IDENTITY,
            drag_axis_world: Vec3::X,
            init_handle_screen: Pos2::ZERO,
            init_center_screen: Pos2::ZERO,
            init_local_pos: Vec3::ZERO,
            init_raw_extent: Vec3::ONE,
            init_outer_half: Vec3::ONE,
            changed: false,
        }
    }
}

fn field_edit_drag_id() -> egui::Id {
    egui::Id::unique("zc_viewport_field_edit_drag")
}

fn reconcile_stale_field_edit_drag(
    ui: &Ui,
    session: &mut Session,
    valid_target: Option<(&str, &str)>,
) {
    let drag_id = field_edit_drag_id();
    let Some(drag) = ui.ctx().data(|d| d.get_temp::<FieldEditDrag>(drag_id)) else {
        return;
    };
    let still_valid =
        valid_target.is_some_and(|(cid, oid)| drag.clip_id == cid && drag.object_id == oid);
    if !still_valid {
        ui.ctx()
            .data_mut(|d| d.remove_temp::<FieldEditDrag>(drag_id));
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
    }
}

/// Returns `true` if `kind` matches an Effector / Repeater class that can carry a Falloff Field.
#[must_use]
pub(crate) fn is_field_effector_kind(kind: &str) -> bool {
    let low = kind.trim().to_ascii_lowercase().replace('_', "-");
    low.starts_with("zoidium:repeater/")
        || low.starts_with("zoidcraft:repeater/")
        || low.starts_with("zoidium:effector-plus/")
        || low.starts_with("zoidcraft:effector-plus/")
        || low.starts_with("effector-plus/")
        || low.starts_with("repeater/")
        || contains_ascii_case_insensitive(&low, b"effector")
        || contains_ascii_case_insensitive(&low, b"expresstext")
        || matches!(
            low.as_str(),
            "twist"
                | "warp"
                | "bend"
                | "shear"
                | "voronoi-fracture"
                | "fracture"
                | "plain"
                | "plain-effector"
                | "delay"
                | "delay-effector"
        )
}

/// Evaluates an effector object's active Falloff Field properties at `local_frame`.
/// Returns `None` when the object is not a field effector or its field is Infinite/None/Noise.
#[must_use]
pub(crate) fn evaluate_falloff_field(
    obj: &SessionObject3D,
    local_frame: f64,
) -> Option<EvaluatedFalloffField> {
    let props = &obj.properties;
    let has_field_prop = props.contains_key("field")
        || props.contains_key("fieldType")
        || props.contains_key("field_type");
    if !has_field_prop && !is_field_effector_kind(&obj.kind) {
        return None;
    }

    let kind = if let Some(s) =
        eval_prop_str_from_keys(props, &["field", "fieldType", "field_type"], local_frame)
    {
        match s.trim().to_ascii_lowercase().as_str() {
            "linear" | "1" => Some(FieldWireframeKind::Linear),
            "box" | "cube" | "2" => Some(FieldWireframeKind::Box),
            "sphere" | "radial" | "3" => Some(FieldWireframeKind::Sphere),
            "cylinder" | "4" => Some(FieldWireframeKind::Cylinder),
            _ => None,
        }
    } else {
        let (v, _, found) = eval_prop_from_keys_scalar(
            props,
            &["field", "fieldType", "field_type"],
            local_frame,
            0.0,
        );
        if !found {
            None
        } else {
            match v.round() as i32 {
                1 => Some(FieldWireframeKind::Linear),
                2 => Some(FieldWireframeKind::Box),
                3 => Some(FieldWireframeKind::Sphere),
                4 => Some(FieldWireframeKind::Cylinder),
                _ => None,
            }
        }
    }?;

    let position_prop =
        if props.contains_key("field_position") && !props.contains_key("fieldPosition") {
            "field_position"
        } else {
            "fieldPosition"
        };
    let local_position = eval_prop_from_keys_vec3(
        props,
        &["fieldPosition", "field_position"],
        local_frame,
        Vec3::ZERO,
    );
    let rotation_deg = eval_prop_from_keys_vec3(
        props,
        &["fieldRotation", "field_rotation"],
        local_frame,
        Vec3::ZERO,
    );
    let local_direction =
        if props.contains_key("fieldDirection") || props.contains_key("field_direction") {
            let dir = eval_prop_from_keys_vec3(
                props,
                &["fieldDirection", "field_direction"],
                local_frame,
                Vec3::ZERO,
            );
            (dir.length_squared() > 1e-6).then(|| dir.normalize())
        } else {
            None
        };

    let (extent_prop, raw_extent, outer_half) = if props.contains_key("fieldRadius")
        || props.contains_key("field_radius")
    {
        let key = if props.contains_key("fieldRadius") {
            "fieldRadius"
        } else {
            "field_radius"
        };
        let r = (eval_prop_from_keys_scalar(props, &[key], local_frame, 10.0).0 as f32)
            .abs()
            .max(0.5);
        (
            FieldExtentProperty::Radius(key),
            Vec3::splat(r),
            Vec3::splat(r),
        )
    } else if props.contains_key("fieldSize") || props.contains_key("field_size") {
        let key = if props.contains_key("fieldSize") {
            "fieldSize"
        } else {
            "field_size"
        };
        let size = eval_prop_from_keys_vec3(props, &[key], local_frame, Vec3::splat(20.0)).abs();
        let half = (size * 0.5).max(Vec3::splat(0.5));
        (
            FieldExtentProperty::Size(key),
            size.max(Vec3::splat(1.0)),
            half,
        )
    } else {
        let key = if props.contains_key("field_scale") && !props.contains_key("fieldScale") {
            "field_scale"
        } else {
            "fieldScale"
        };
        let scale = eval_prop_from_keys_vec3(
            props,
            &["fieldScale", "field_scale"],
            local_frame,
            Vec3::splat(20.0),
        )
        .abs();
        let half = (scale * 0.5).max(Vec3::splat(2.0));
        (
            FieldExtentProperty::Scale(key),
            scale.max(Vec3::splat(1.0)),
            half,
        )
    };

    let falloff_raw = eval_prop_from_keys_scalar(
        props,
        &["fieldFalloff", "field_falloff"],
        local_frame,
        100.0,
    )
    .0 as f32;
    let falloff_frac = if falloff_raw.is_finite() {
        (falloff_raw / 100.0).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let inner_half = if falloff_frac > 0.01 && falloff_frac < 0.99 {
        Some((outer_half * (1.0 - falloff_frac)).max(Vec3::splat(0.25)))
    } else {
        None
    };

    Some(EvaluatedFalloffField {
        kind,
        position_prop,
        extent_prop,
        local_position,
        rotation_deg,
        local_direction,
        raw_extent,
        outer_half,
        inner_half,
        falloff_frac,
    })
}

fn rotate_field_local_vec(v: Vec3, rotation_deg: Vec3) -> Vec3 {
    let out = zc_scene::rotate_xyz_deg(
        [f64::from(v.x), f64::from(v.y), f64::from(v.z)],
        [
            f64::from(rotation_deg.x),
            f64::from(rotation_deg.y),
            f64::from(rotation_deg.z),
        ],
        false,
    );
    Vec3::new(out[0] as f32, out[1] as f32, out[2] as f32)
}

fn field_local_basis(field: &EvaluatedFalloffField) -> (Vec3, Vec3, Vec3) {
    if field.kind == FieldWireframeKind::Linear
        && let Some(dir) = field.local_direction
    {
        let lx = normalized_or(rotate_field_local_vec(dir, field.rotation_deg), Vec3::X);
        let ref_up = if lx.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
        let lz = normalized_or(lx.cross(ref_up), Vec3::Z);
        let ly = normalized_or(lz.cross(lx), Vec3::Y);
        return (lx, ly, lz);
    }
    let lx = normalized_or(rotate_field_local_vec(Vec3::X, field.rotation_deg), Vec3::X);
    let ly = normalized_or(rotate_field_local_vec(Vec3::Y, field.rotation_deg), Vec3::Y);
    let lz = normalized_or(rotate_field_local_vec(Vec3::Z, field.rotation_deg), Vec3::Z);
    (lx, ly, lz)
}

/// Builds world-space 3D wireframe segments and interactive drag handle positions for an active
/// Falloff Field (Sphere, Box, Linear, or Cylinder).
#[must_use]
pub(crate) fn build_field_wireframe_geometry(
    field: &EvaluatedFalloffField,
    world: Mat4,
) -> FieldWireframeGeometry {
    let (lx, ly, lz) = field_local_basis(field);
    let to_world = |u: f32, v: f32, w: f32| -> Vec3 {
        let local_pt = field.local_position + lx * u + ly * v + lz * w;
        world.transform_point3(local_pt)
    };
    let center_world = to_world(0.0, 0.0, 0.0);

    let mut primary_segments = Vec::new();
    let mut falloff_segments = Vec::new();
    let mut handles = vec![(FieldEditHandle::Center, center_world)];

    let push_ellipse = |dst: &mut Vec<(Vec3, Vec3)>,
                        a_axis: usize,
                        b_axis: usize,
                        radii: Vec3,
                        fixed: Option<(usize, f32)>| {
        let mut pts = [Vec3::ZERO; FIELD_RING_SEGMENTS + 1];
        for (i, slot) in pts.iter_mut().enumerate() {
            let theta = (i as f32 / FIELD_RING_SEGMENTS as f32) * std::f32::consts::TAU;
            let (s, c) = theta.sin_cos();
            let mut coords = [0.0_f32; 3];
            coords[a_axis] = radii[a_axis] * c;
            coords[b_axis] = radii[b_axis] * s;
            if let Some((fa, fv)) = fixed {
                coords[fa] = fv;
            }
            *slot = to_world(coords[0], coords[1], coords[2]);
        }
        for w in pts.windows(2) {
            dst.push((w[0], w[1]));
        }
    };

    let push_box = |dst: &mut Vec<(Vec3, Vec3)>, half: Vec3| {
        let corners = [
            to_world(-half.x, -half.y, -half.z),
            to_world(half.x, -half.y, -half.z),
            to_world(half.x, half.y, -half.z),
            to_world(-half.x, half.y, -half.z),
            to_world(-half.x, -half.y, half.z),
            to_world(half.x, -half.y, half.z),
            to_world(half.x, half.y, half.z),
            to_world(-half.x, half.y, half.z),
        ];
        const EDGES: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ];
        for (i, j) in EDGES {
            dst.push((corners[i], corners[j]));
        }
    };

    match field.kind {
        FieldWireframeKind::Sphere => {
            if let Some(inner) = field.inner_half {
                push_ellipse(&mut primary_segments, 0, 1, inner, None);
                push_ellipse(&mut primary_segments, 0, 2, inner, None);
                push_ellipse(&mut primary_segments, 1, 2, inner, None);
                push_ellipse(&mut falloff_segments, 0, 1, field.outer_half, None);
                push_ellipse(&mut falloff_segments, 0, 2, field.outer_half, None);
                push_ellipse(&mut falloff_segments, 1, 2, field.outer_half, None);
            } else {
                push_ellipse(&mut primary_segments, 0, 1, field.outer_half, None);
                push_ellipse(&mut primary_segments, 0, 2, field.outer_half, None);
                push_ellipse(&mut primary_segments, 1, 2, field.outer_half, None);
            }
            handles.push((
                FieldEditHandle::PosX,
                to_world(field.outer_half.x, 0.0, 0.0),
            ));
            handles.push((
                FieldEditHandle::PosY,
                to_world(0.0, field.outer_half.y, 0.0),
            ));
            handles.push((
                FieldEditHandle::PosZ,
                to_world(0.0, 0.0, field.outer_half.z),
            ));
        }
        FieldWireframeKind::Box => {
            if let Some(inner) = field.inner_half {
                push_box(&mut primary_segments, inner);
                push_box(&mut falloff_segments, field.outer_half);
            } else {
                push_box(&mut primary_segments, field.outer_half);
            }
            handles.push((
                FieldEditHandle::PosX,
                to_world(field.outer_half.x, 0.0, 0.0),
            ));
            handles.push((
                FieldEditHandle::NegX,
                to_world(-field.outer_half.x, 0.0, 0.0),
            ));
            handles.push((
                FieldEditHandle::PosY,
                to_world(0.0, field.outer_half.y, 0.0),
            ));
            handles.push((
                FieldEditHandle::NegY,
                to_world(0.0, -field.outer_half.y, 0.0),
            ));
            handles.push((
                FieldEditHandle::PosZ,
                to_world(0.0, 0.0, field.outer_half.z),
            ));
            handles.push((
                FieldEditHandle::NegZ,
                to_world(0.0, 0.0, -field.outer_half.z),
            ));
        }
        FieldWireframeKind::Cylinder => {
            let push_cylinder = |dst: &mut Vec<(Vec3, Vec3)>, half: Vec3| {
                let r = Vec3::new(half.x.max(half.z), half.y, half.x.max(half.z));
                push_ellipse(dst, 0, 2, r, Some((1, half.y)));
                push_ellipse(dst, 0, 2, r, Some((1, -half.y)));
                for (dx, dz) in [(r.x, 0.0), (-r.x, 0.0), (0.0, r.z), (0.0, -r.z)] {
                    dst.push((to_world(dx, -half.y, dz), to_world(dx, half.y, dz)));
                }
            };
            if let Some(inner) = field.inner_half {
                push_cylinder(&mut primary_segments, inner);
                push_cylinder(&mut falloff_segments, field.outer_half);
            } else {
                push_cylinder(&mut primary_segments, field.outer_half);
            }
            let r = field.outer_half.x.max(field.outer_half.z);
            handles.push((FieldEditHandle::PosX, to_world(r, 0.0, 0.0)));
            handles.push((
                FieldEditHandle::PosY,
                to_world(0.0, field.outer_half.y, 0.0),
            ));
            handles.push((FieldEditHandle::PosZ, to_world(0.0, 0.0, r)));
        }
        FieldWireframeKind::Linear => {
            let span = field.outer_half.x.max(10.0);
            let plane_h = field.outer_half.y.clamp(5.0, 24.0);
            let plane_d = field.outer_half.z.clamp(5.0, 24.0);

            // Direction arrow along local +X from -span to +span with 3D arrowhead.
            let tail = to_world(-span, 0.0, 0.0);
            let tip = to_world(span, 0.0, 0.0);
            primary_segments.push((tail, tip));
            let head_len = (span * 0.18).clamp(2.0, 6.0);
            let head_w = head_len * 0.45;
            for (dy, dz) in [(head_w, 0.0), (-head_w, 0.0), (0.0, head_w), (0.0, -head_w)] {
                primary_segments.push((tip, to_world(span - head_len, dy, dz)));
            }

            let push_plane = |dst: &mut Vec<(Vec3, Vec3)>, x: f32| {
                let c0 = to_world(x, -plane_h, -plane_d);
                let c1 = to_world(x, plane_h, -plane_d);
                let c2 = to_world(x, plane_h, plane_d);
                let c3 = to_world(x, -plane_h, plane_d);
                dst.push((c0, c1));
                dst.push((c1, c2));
                dst.push((c2, c3));
                dst.push((c3, c0));
            };

            if field.falloff_frac > 0.01 {
                let falloff_half = span * field.falloff_frac;
                // Start plane (full weight) + End falloff plane (zero weight).
                push_plane(&mut primary_segments, -falloff_half);
                push_plane(&mut falloff_segments, falloff_half);
            } else {
                // Hard-edge cut plane at center.
                push_plane(&mut primary_segments, 0.0);
            }

            handles.push((FieldEditHandle::PosX, tip));
        }
    }

    FieldWireframeGeometry {
        kind: field.kind,
        primary_segments,
        falloff_segments,
        center_world,
        handles,
    }
}

/// Computes updated effector-local `fieldPosition` when dragging the field center handle on the
/// camera viewing plane (matching `scene-field-edit-math.js:viewPlanePoint` + `localDelta`).
#[allow(clippy::too_many_arguments)]
#[must_use]
pub(crate) fn compute_field_center_drag(
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    init_center_world: Vec3,
    init_plane_hit: Option<Vec3>,
    effector_world: Mat4,
    init_local_pos: Vec3,
    start_pointer: Pos2,
    cur_pointer: Pos2,
    snap_unit: bool,
) -> Vec3 {
    let world_delta = init_plane_hit
        .and_then(|start_hit| {
            let (ro, rd) = cam.screen_ray(cur_pointer, canvas_rect)?;
            let cur_hit = ray_plane_intersect(ro, rd, init_center_world, cam.forward)?;
            Some(cur_hit - start_hit)
        })
        .unwrap_or_else(|| {
            cam.unproject_delta_at_depth(
                cur_pointer - start_pointer,
                init_center_world,
                canvas_rect,
            )
        });
    let local_delta = if effector_world.determinant().abs() > 1e-8 {
        effector_world.inverse().transform_vector3(world_delta)
    } else {
        world_delta
    };
    let mut next = init_local_pos + local_delta;
    if snap_unit {
        next = Vec3::new(next.x.round(), next.y.round(), next.z.round());
    }
    next
}

/// Computes updated raw extent (`fieldRadius` splatted, `fieldSize`, or `fieldScale`) when
/// dragging a falloff field extent/radius endpoint handle.
#[allow(clippy::too_many_arguments)]
#[must_use]
pub(crate) fn compute_field_extent_drag(
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    kind: FieldWireframeKind,
    handle: FieldEditHandle,
    extent_prop: FieldExtentProperty,
    init_center_world: Vec3,
    init_center_screen: Pos2,
    init_handle_screen: Pos2,
    effector_world: Mat4,
    drag_axis_world: Vec3,
    init_raw_extent: Vec3,
    init_outer_half: Vec3,
    start_pointer: Pos2,
    cur_pointer: Pos2,
    snap_half_unit: bool,
    uniform_mod: bool,
) -> Vec3 {
    let axis_idx = match handle {
        FieldEditHandle::PosX | FieldEditHandle::NegX => 0usize,
        FieldEditHandle::PosY | FieldEditHandle::NegY => 1usize,
        FieldEditHandle::PosZ | FieldEditHandle::NegZ => 2usize,
        FieldEditHandle::Center => return init_raw_extent,
    };
    let init_axis_half = match kind {
        FieldWireframeKind::Cylinder if axis_idx != 1 => {
            init_outer_half.x.max(init_outer_half.z).max(0.5)
        }
        FieldWireframeKind::Linear => init_outer_half.x.max(10.0),
        _ => init_outer_half[axis_idx].max(0.5),
    };

    let delta_screen = cur_pointer - start_pointer;
    let handle_vec = init_handle_screen - init_center_screen;
    let handle_len = handle_vec.length();
    let delta_half_local = if handle_len > 4.0 {
        let px_along = delta_screen.dot(handle_vec / handle_len);
        px_along * (init_axis_half / handle_len)
    } else {
        let w_delta = cam.unproject_delta_at_depth(delta_screen, init_center_world, canvas_rect);
        if effector_world.determinant().abs() > 1e-8 {
            let inv = effector_world.inverse();
            let l_delta = inv.transform_vector3(w_delta);
            let l_axis = normalized_or(inv.transform_vector3(drag_axis_world), Vec3::X);
            l_delta.dot(l_axis)
        } else {
            w_delta.dot(drag_axis_world)
        }
    };

    let mut new_half = (init_axis_half + delta_half_local).clamp(0.5, 10_000.0);
    if snap_half_unit {
        new_half = (new_half * 2.0).round().max(1.0) * 0.5;
    }

    if matches!(extent_prop, FieldExtentProperty::Radius(_)) {
        return Vec3::splat(new_half);
    }

    let base = init_raw_extent.abs().max(Vec3::splat(1.0));
    match kind {
        FieldWireframeKind::Sphere => {
            let factor = (new_half / init_axis_half.max(0.01)).clamp(0.01, 100.0);
            (base * factor).max(Vec3::splat(1.0))
        }
        FieldWireframeKind::Box => {
            if uniform_mod {
                let factor = (new_half / init_axis_half.max(0.01)).clamp(0.01, 100.0);
                (base * factor).max(Vec3::splat(1.0))
            } else {
                let mut next = base;
                next[axis_idx] = (new_half * 2.0).max(1.0);
                next
            }
        }
        FieldWireframeKind::Cylinder => {
            let mut next = base;
            if axis_idx == 1 {
                next.y = (new_half * 2.0).max(1.0);
            } else {
                let d = (new_half * 2.0).max(1.0);
                next.x = d;
                next.z = d;
            }
            next
        }
        FieldWireframeKind::Linear => {
            let mut next = base;
            next.x = (new_half * 2.0).max(1.0);
            next
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn handle_and_paint_falloff_field(
    ui: &mut Ui,
    session: &mut Session,
    state: &ViewportState,
    cam: &ResolvedCamera3D,
    panel_rect: Rect,
    canvas_rect: Rect,
    visible_canvas: Rect,
    panel_resp: &egui::Response,
    clip_id: &str,
    obj_id: &str,
    local_frame: f64,
    world: Mat4,
    origin_screen: Option<Pos2>,
    locked: bool,
    pointer_in_hud: bool,
    field: Option<&EvaluatedFalloffField>,
    action: &mut ViewportAction,
) -> bool {
    let drag_id = field_edit_drag_id();
    let mut active_drag = ui.ctx().data(|d| d.get_temp::<FieldEditDrag>(drag_id));
    if let Some(ref drag) = active_drag {
        let still_valid =
            !locked && field.is_some() && drag.clip_id == clip_id && drag.object_id == obj_id;
        if !still_valid {
            let changed = drag.changed;
            ui.ctx()
                .data_mut(|d| d.remove_temp::<FieldEditDrag>(drag_id));
            if session.history.is_in_transaction() {
                let _ = session.commit_transaction(changed);
            }
            active_drag = None;
        }
    }

    let Some(field) = field else {
        return false;
    };

    let geom = build_field_wireframe_geometry(field, world);
    let wants_kb = ui.ctx().egui_wants_keyboard_input();
    let (
        pointer_pos,
        press_origin,
        primary_pressed,
        primary_down,
        modifiers,
        space_down,
        esc_pressed,
    ) = ui.input(|i| {
        (
            i.pointer.latest_pos(),
            i.pointer.press_origin(),
            i.pointer.primary_pressed(),
            i.pointer.primary_down(),
            i.modifiers,
            !wants_kb && i.key_down(egui::Key::Space),
            !wants_kb && i.key_pressed(egui::Key::Escape),
        )
    });
    let hit_pos = if primary_pressed {
        press_origin.or(pointer_pos)
    } else {
        pointer_pos
    };

    let dragging_center = active_drag
        .as_ref()
        .is_some_and(|d| d.handle == FieldEditHandle::Center);

    // Project interactive handles; offset Center slightly with a leader line when it sits
    // directly on top of the main 3D transform gizmo origin so both remain independently grabbable.
    let mut projected_handles: Vec<(FieldEditHandle, Pos2, Vec3, Option<Pos2>)> =
        Vec::with_capacity(geom.handles.len());
    for &(handle, world_pos) in &geom.handles {
        let Some(raw_sp) = cam.project_world_point(world_pos, canvas_rect) else {
            continue;
        };
        let (sp, leader) = if handle == FieldEditHandle::Center
            && !dragging_center
            && origin_screen.is_some_and(|orig| raw_sp.distance(orig) < 18.0)
        {
            (raw_sp + vec2(-18.0, -18.0), Some(raw_sp))
        } else {
            (raw_sp, None)
        };
        projected_handles.push((handle, sp, world_pos, leader));
    }

    let hovered_handle = if active_drag.is_none()
        && matches!(state.drag_state, ViewportDragState::None)
        && !pointer_in_hud
        && !space_down
        && let Some(pos) = hit_pos
        && pointer_hits_viewport(ui, pos, panel_rect)
        && visible_canvas.contains(pos)
    {
        let mut best: Option<(FieldEditHandle, f32)> = None;
        for &(handle, sp, _, _) in &projected_handles {
            let d = pos.distance(sp);
            if d <= FIELD_HANDLE_HIT_RADIUS_PX && best.is_none_or(|(_, bd)| d < bd) {
                best = Some((handle, d));
            }
        }
        best.map(|(h, _)| h)
    } else {
        None
    };

    let is_dragging = active_drag.is_some();
    let active_handle = active_drag.as_ref().map(|d| d.handle).or(hovered_handle);

    // Paint the 3D wireframe and drag handles clipped strictly to `visible_canvas`.
    if visible_canvas.is_positive() {
        let painter = ui.painter().with_clip_rect(visible_canvas);
        let base_col = if is_dragging {
            COLOR_FIELD_ACTIVE
        } else {
            COLOR_WIRE_FIELD_AMBER
        };
        let primary_stroke = Stroke::new(
            1.7,
            Color32::from_rgba_unmultiplied(base_col.r(), base_col.g(), base_col.b(), 235),
        );
        let falloff_stroke = Stroke::new(
            1.2,
            Color32::from_rgba_unmultiplied(base_col.r(), base_col.g(), base_col.b(), 155),
        );

        for &(a, b) in &geom.primary_segments {
            draw_3d_line(&painter, cam, canvas_rect, a, b, primary_stroke);
        }

        match geom.kind {
            FieldWireframeKind::Box | FieldWireframeKind::Linear => {
                for &(a, b) in &geom.falloff_segments {
                    draw_3d_dashed_line(&painter, cam, canvas_rect, a, b, falloff_stroke, 8);
                }
            }
            FieldWireframeKind::Sphere => {
                for (idx, &(a, b)) in geom.falloff_segments.iter().enumerate() {
                    if idx.is_multiple_of(2) {
                        draw_3d_line(&painter, cam, canvas_rect, a, b, falloff_stroke);
                    }
                }
            }
            FieldWireframeKind::Cylinder => {
                let cap_segs = FIELD_RING_SEGMENTS * 2;
                for (idx, &(a, b)) in geom.falloff_segments.iter().enumerate() {
                    if idx < cap_segs {
                        if idx.is_multiple_of(2) {
                            draw_3d_line(&painter, cam, canvas_rect, a, b, falloff_stroke);
                        }
                    } else {
                        draw_3d_dashed_line(&painter, cam, canvas_rect, a, b, falloff_stroke, 8);
                    }
                }
            }
        }

        // Paint subtle 3D cross at the field center when editing or when full-range falloff collapses inner core.
        if is_dragging || field.inner_half.is_none() {
            let cross_r = (10.0 * cam.world_units_per_px(geom.center_world, canvas_rect)).max(0.5);
            let (lx, ly, lz) = field_local_basis(field);
            let cx = normalized_or(world.transform_vector3(lx), Vec3::X) * cross_r;
            let cy = normalized_or(world.transform_vector3(ly), Vec3::Y) * cross_r;
            let cz = normalized_or(world.transform_vector3(lz), Vec3::Z) * cross_r;
            for axis in [cx, cy, cz] {
                draw_3d_line(
                    &painter,
                    cam,
                    canvas_rect,
                    geom.center_world - axis,
                    geom.center_world + axis,
                    falloff_stroke,
                );
            }
        }

        // Paint interactive drag handles.
        let dark_outline = Color32::from_rgba_unmultiplied(15, 20, 28, 225);
        for &(handle, sp, _, leader) in &projected_handles {
            if let Some(anchor) = leader {
                painter.line_segment([anchor, sp], Stroke::new(1.0, falloff_stroke.color));
            }
            let is_hot = active_handle == Some(handle);
            let col = if is_dragging && is_hot {
                COLOR_FIELD_ACTIVE
            } else if is_hot {
                COLOR_GIZMO_HOVER
            } else {
                base_col
            };
            if handle == FieldEditHandle::Center {
                let r = if is_hot {
                    FIELD_HANDLE_RADIUS_PX + 1.5
                } else {
                    FIELD_HANDLE_RADIUS_PX + 0.5
                };
                let diamond = vec![
                    pos2(sp.x, sp.y - r),
                    pos2(sp.x + r, sp.y),
                    pos2(sp.x, sp.y + r),
                    pos2(sp.x - r, sp.y),
                ];
                painter.add(Shape::convex_polygon(
                    diamond,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 230),
                    Stroke::new(1.3, dark_outline),
                ));
                if leader.is_some() {
                    painter.text(
                        sp + vec2(0.0, -(r + 6.0)),
                        Align2::CENTER_CENTER,
                        "F",
                        FontId::monospace(9.0),
                        col,
                    );
                }
            } else {
                let r = if is_hot {
                    FIELD_HANDLE_RADIUS_PX + 0.8
                } else {
                    FIELD_HANDLE_RADIUS_PX - 0.5
                };
                painter.circle_filled(sp, r + 1.1, dark_outline);
                painter.circle(
                    sp,
                    r,
                    Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 240),
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(255, 255, 255, 210)),
                );
            }
        }
    }

    // Start a new falloff field handle drag on primary press.
    if active_drag.is_none()
        && !locked
        && !space_down
        && matches!(state.drag_state, ViewportDragState::None)
        && let Some(h) = hovered_handle
        && let Some(pos) = hit_pos
    {
        ui.ctx().set_cursor_icon(match h {
            FieldEditHandle::Center => CursorIcon::Move,
            _ => CursorIcon::Grab,
        });
        let is_primary_double_click = panel_resp.double_clicked_by(PointerButton::Primary)
            || ui.input(|i| i.pointer.button_double_clicked(PointerButton::Primary));
        if primary_pressed
            && !(is_editing_camera_active(session) && is_primary_double_click)
            && session.begin_transaction("Edit Falloff Field").is_ok()
        {
            let init_plane_hit = if h == FieldEditHandle::Center {
                cam.screen_ray(pos, canvas_rect).and_then(|(ro, rd)| {
                    ray_plane_intersect(ro, rd, geom.center_world, cam.forward)
                })
            } else {
                None
            };
            let (init_handle_screen, handle_world) = projected_handles
                .iter()
                .find(|(hh, _, _, _)| *hh == h)
                .map_or((pos, geom.center_world), |(_, sp, wp, _)| (*sp, *wp));
            let init_center_screen = cam
                .project_world_point(geom.center_world, canvas_rect)
                .unwrap_or(pos);
            let drag_axis_world = if h == FieldEditHandle::Center {
                Vec3::ZERO
            } else {
                normalized_or(handle_world - geom.center_world, Vec3::X)
            };
            active_drag = Some(FieldEditDrag {
                clip_id: clip_id.to_owned(),
                object_id: obj_id.to_owned(),
                handle: h,
                kind: field.kind,
                position_prop: field.position_prop,
                extent_prop: field.extent_prop,
                start_pointer: pos,
                init_center_world: geom.center_world,
                init_plane_hit,
                effector_world: world,
                drag_axis_world,
                init_handle_screen,
                init_center_screen,
                init_local_pos: field.local_position,
                init_raw_extent: field.raw_extent,
                init_outer_half: field.outer_half,
                changed: false,
            });
        } else {
            return false;
        }
    }

    // Active falloff field handle drag update.
    if let Some(mut drag) = active_drag {
        if esc_pressed {
            ui.ctx()
                .data_mut(|d| d.remove_temp::<FieldEditDrag>(drag_id));
            if session.history.is_in_transaction() {
                let _ = session.cancel_transaction();
            }
            action.transform_modified = true;
            return true;
        }

        if let Some(pos) = pointer_pos {
            if !drag.changed && (pos - drag.start_pointer).length() > 0.5 {
                drag.changed = true;
            }
            if drag.changed {
                if drag.handle == FieldEditHandle::Center {
                    let next_pos = compute_field_center_drag(
                        cam,
                        canvas_rect,
                        drag.init_center_world,
                        drag.init_plane_hit,
                        drag.effector_world,
                        drag.init_local_pos,
                        drag.start_pointer,
                        pos,
                        modifiers.shift,
                    );
                    write_object_3d_vec3(
                        session,
                        &drag.object_id,
                        drag.position_prop,
                        local_frame,
                        next_pos,
                    );
                    action.transform_modified = true;
                } else {
                    let next_extent = compute_field_extent_drag(
                        cam,
                        canvas_rect,
                        drag.kind,
                        drag.handle,
                        drag.extent_prop,
                        drag.init_center_world,
                        drag.init_center_screen,
                        drag.init_handle_screen,
                        drag.effector_world,
                        drag.drag_axis_world,
                        drag.init_raw_extent,
                        drag.init_outer_half,
                        drag.start_pointer,
                        pos,
                        modifiers.shift,
                        modifiers.alt,
                    );
                    match drag.extent_prop {
                        FieldExtentProperty::Radius(prop) => {
                            write_object_3d_scalar(
                                session,
                                &drag.object_id,
                                prop,
                                local_frame,
                                next_extent.x,
                            );
                        }
                        FieldExtentProperty::Size(prop) | FieldExtentProperty::Scale(prop) => {
                            write_object_3d_vec3(
                                session,
                                &drag.object_id,
                                prop,
                                local_frame,
                                next_extent,
                            );
                        }
                    }
                    action.transform_modified = true;
                }
            }
        }

        if primary_down {
            ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
            ui.ctx().data_mut(|d| d.insert_temp(drag_id, drag));
            return true;
        }
        ui.ctx()
            .data_mut(|d| d.remove_temp::<FieldEditDrag>(drag_id));
        if session.history.is_in_transaction() {
            let _ = session.commit_transaction(drag.changed);
        }
        return true;
    }

    false
}

#[cfg(test)]
mod scene_plus_helper_tests {
    use super::*;

    fn static_prop(val: Value) -> SessionProperty {
        SessionProperty {
            animated: false,
            expression: None,
            default_value: val,
            keyframes: Vec::new(),
        }
    }

    fn make_obj(kind: &str, props: &[(&str, Value)]) -> SessionObject3D {
        let mut properties = BTreeMap::new();
        for (k, v) in props {
            properties.insert((*k).to_owned(), static_prop(v.clone()));
        }
        SessionObject3D {
            id: "obj_1".to_owned(),
            parent_id: None,
            kind: kind.to_owned(),
            name: "TestObj".to_owned(),
            source_id: None,
            properties,
        }
    }

    #[test]
    fn repeater_linear_and_step_points_match_expected_origins_and_world_transform() {
        let lin = make_obj(
            "zoidium:repeater/linear-repeater",
            &[
                ("count", json!(5)),
                ("positionEnd", json!([40.0, 20.0, 0.0])),
            ],
        );
        let local = compute_repeater_copy_origins_local(&lin, 0.0);
        assert_eq!(local.len(), 5);
        assert!((local[0] - Vec3::ZERO).length() < 1e-5);
        assert!((local[2] - Vec3::new(20.0, 10.0, 0.0)).length() < 1e-5);
        assert!((local[4] - Vec3::new(40.0, 20.0, 0.0)).length() < 1e-5);

        // World transform translates and scales copy origins cleanly.
        let world = Mat4::from_translation(Vec3::new(5.0, -3.0, 2.0));
        let world_pts = compute_repeater_copy_origins_world(&lin, world, 0.0);
        assert!((world_pts[0] - Vec3::new(5.0, -3.0, 2.0)).length() < 1e-5);
        assert!((world_pts[4] - Vec3::new(45.0, 17.0, 2.0)).length() < 1e-5);

        // Step repeater caps at MAX_REPEATS (128).
        let step = make_obj(
            "zoidium:repeater/repeater",
            &[
                ("count", json!(500)),
                ("positionStep", json!([2.0, 0.0, 0.0])),
            ],
        );
        let step_pts = compute_repeater_copy_origins_local(&step, 0.0);
        assert_eq!(step_pts.len(), 128);
        assert!((step_pts[127] - Vec3::new(254.0, 0.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn repeater_radial_points_cover_xy_xz_yz_planes_and_partial_sweeps() {
        // XY plane (plane = 0), 4 copies around full circle -> (R,0,0), (0,R,0), (-R,0,0), (0,-R,0).
        let xy = make_obj(
            "zoidium:repeater/radial-repeater",
            &[
                ("count", json!(4)),
                ("radius", json!(10.0)),
                ("plane", json!(0)),
                ("startAngle", json!(0.0)),
                ("sweep", json!(std::f64::consts::TAU)),
            ],
        );
        let pts_xy = compute_repeater_copy_origins_local(&xy, 0.0);
        assert_eq!(pts_xy.len(), 4);
        assert!((pts_xy[0] - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4);
        assert!((pts_xy[1] - Vec3::new(0.0, 10.0, 0.0)).length() < 1e-4);
        assert!((pts_xy[2] - Vec3::new(-10.0, 0.0, 0.0)).length() < 1e-4);
        assert!((pts_xy[3] - Vec3::new(0.0, -10.0, 0.0)).length() < 1e-4);

        // XZ plane (plane = 1), 4 copies around full circle -> (R,0,0), (0,0,R), (-R,0,0), (0,0,-R).
        let xz = make_obj(
            "radial-repeater",
            &[
                ("count", json!(4)),
                ("radius", json!(12.0)),
                ("plane", json!(1)),
                ("sweep", json!(std::f64::consts::TAU)),
            ],
        );
        let pts_xz = compute_repeater_copy_origins_local(&xz, 0.0);
        assert!((pts_xz[0] - Vec3::new(12.0, 0.0, 0.0)).length() < 1e-4);
        assert!((pts_xz[1] - Vec3::new(0.0, 0.0, 12.0)).length() < 1e-4);
        assert!((pts_xz[2] - Vec3::new(-12.0, 0.0, 0.0)).length() < 1e-4);
        assert!((pts_xz[3] - Vec3::new(0.0, 0.0, -12.0)).length() < 1e-4);

        // YZ plane (plane = 2), half-circle sweep (PI) with 3 copies includes both endpoints.
        let yz = make_obj(
            "zoidium:repeater/radial",
            &[
                ("count", json!(3)),
                ("radius", json!(8.0)),
                ("plane", json!(2)),
                ("startAngle", json!(0.0)),
                ("sweep", json!(std::f64::consts::PI)),
            ],
        );
        let pts_yz = compute_repeater_copy_origins_local(&yz, 0.0);
        assert_eq!(pts_yz.len(), 3);
        assert!((pts_yz[0] - Vec3::new(0.0, 8.0, 0.0)).length() < 1e-4);
        assert!((pts_yz[1] - Vec3::new(0.0, 0.0, 8.0)).length() < 1e-4);
        assert!((pts_yz[2] - Vec3::new(0.0, -8.0, 0.0)).length() < 1e-4);
    }

    #[test]
    fn repeater_grid_points_centered_uncentered_and_capped() {
        let centered = make_obj(
            "zoidium:repeater/grid-repeater",
            &[
                ("countX", json!(3)),
                ("countY", json!(2)),
                ("countZ", json!(1)),
                ("spacing", json!([10.0, 20.0, 5.0])),
                ("centered", json!(1)),
            ],
        );
        let pts_c = compute_repeater_copy_origins_local(&centered, 0.0);
        assert_eq!(pts_c.len(), 6);
        // X varies fastest, then Y: first row y = -10, second row y = +10.
        assert!((pts_c[0] - Vec3::new(-10.0, -10.0, 0.0)).length() < 1e-4);
        assert!((pts_c[1] - Vec3::new(0.0, -10.0, 0.0)).length() < 1e-4);
        assert!((pts_c[2] - Vec3::new(10.0, -10.0, 0.0)).length() < 1e-4);
        assert!((pts_c[5] - Vec3::new(10.0, 10.0, 0.0)).length() < 1e-4);
        let centroid = pts_c.iter().copied().sum::<Vec3>() / (pts_c.len() as f32);
        assert!(centroid.length() < 1e-4);

        let uncentered = make_obj(
            "grid-repeater",
            &[
                ("countX", json!(3)),
                ("countY", json!(2)),
                ("countZ", json!(1)),
                ("spacing", json!([10.0, 20.0, 5.0])),
                ("centered", json!(0)),
            ],
        );
        let pts_u = compute_repeater_copy_origins_local(&uncentered, 0.0);
        assert_eq!(pts_u.len(), 6);
        assert!((pts_u[0] - Vec3::ZERO).length() < 1e-4);
        assert!((pts_u[2] - Vec3::new(20.0, 0.0, 0.0)).length() < 1e-4);
        assert!((pts_u[5] - Vec3::new(20.0, 20.0, 0.0)).length() < 1e-4);

        let oversized = make_obj(
            "zoidium:repeater/grid",
            &[
                ("countX", json!(10)),
                ("countY", json!(10)),
                ("countZ", json!(2)),
            ],
        );
        assert_eq!(
            compute_repeater_copy_origins_local(&oversized, 0.0).len(),
            128
        );
    }

    #[test]
    fn repeater_random_and_echo_points_match_runtime() {
        let rand_obj = make_obj(
            "zoidium:repeater/random-repeater",
            &[
                ("count", json!(6)),
                ("seed", json!(42)),
                ("positionMin", json!([-5.0, -2.0, 0.0])),
                ("positionMax", json!([5.0, 2.0, 10.0])),
            ],
        );
        let rand_pts = compute_repeater_copy_origins_local(&rand_obj, 0.0);
        assert_eq!(rand_pts.len(), 6);
        assert_eq!(rand_pts[0], Vec3::ZERO);
        for p in &rand_pts[1..] {
            assert!((-5.0..=5.0).contains(&p.x));
            assert!((-2.0..=2.0).contains(&p.y));
            assert!((0.0..=10.0).contains(&p.z));
        }

        let echo_obj = make_obj("zoidium:repeater/echo-repeater", &[("count", json!(4))]);
        let echo_pts = compute_repeater_copy_origins_local(&echo_obj, 0.0);
        assert_eq!(echo_pts, vec![Vec3::ZERO; 4]);
    }

    #[test]
    fn field_wireframe_sphere_box_and_linear_geometry_and_handles() {
        // 1. Sphere field with partial falloff produces 3 inner rings + 3 outer falloff rings.
        let sphere_obj = make_obj(
            "zoidium:repeater/warp",
            &[
                ("field", json!(3)),
                ("fieldPosition", json!([2.0, 4.0, -6.0])),
                ("fieldRadius", json!(20.0)),
                ("fieldFalloff", json!(50.0)),
            ],
        );
        let sphere_field = evaluate_falloff_field(&sphere_obj, 0.0).expect("sphere field");
        assert_eq!(sphere_field.kind, FieldWireframeKind::Sphere);
        assert_eq!(sphere_field.outer_half, Vec3::splat(20.0));
        assert_eq!(sphere_field.inner_half, Some(Vec3::splat(10.0)));
        let sphere_geom = build_field_wireframe_geometry(&sphere_field, Mat4::IDENTITY);
        assert_eq!(sphere_geom.primary_segments.len(), FIELD_RING_SEGMENTS * 3);
        assert_eq!(sphere_geom.falloff_segments.len(), FIELD_RING_SEGMENTS * 3);
        assert!((sphere_geom.center_world - Vec3::new(2.0, 4.0, -6.0)).length() < 1e-4);
        // Every primary vertex sits at inner radius 10; every falloff vertex sits at outer radius 20.
        for &(a, _) in &sphere_geom.primary_segments {
            assert!(((a - sphere_geom.center_world).length() - 10.0).abs() < 1e-3);
        }
        for &(a, _) in &sphere_geom.falloff_segments {
            assert!(((a - sphere_geom.center_world).length() - 20.0).abs() < 1e-3);
        }
        assert_eq!(sphere_geom.handles.len(), 4);

        // 2. Box field with partial falloff produces 12 inner edges + 12 outer falloff edges.
        let box_obj = make_obj(
            "zoidium:repeater/twist",
            &[
                ("fieldType", json!("box")),
                ("fieldPosition", json!([0.0, 0.0, 0.0])),
                ("fieldSize", json!([20.0, 10.0, 30.0])),
                ("fieldFalloff", json!(25.0)),
            ],
        );
        let box_field = evaluate_falloff_field(&box_obj, 0.0).expect("box field");
        let box_world = Mat4::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let box_geom = build_field_wireframe_geometry(&box_field, box_world);
        assert_eq!(box_geom.primary_segments.len(), 12);
        assert_eq!(box_geom.falloff_segments.len(), 12);
        assert!((box_geom.center_world - Vec3::new(10.0, 0.0, 0.0)).length() < 1e-4);
        // +X handle sits at center.x + outer_half.x = 10 + 10 = 20.
        let pos_x = box_geom
            .handles
            .iter()
            .find_map(|&(h, p)| (h == FieldEditHandle::PosX).then_some(p))
            .expect("+X handle");
        assert!((pos_x - Vec3::new(20.0, 0.0, 0.0)).length() < 1e-4);

        // 3. Linear field with direction arrow + start/end falloff planes.
        let lin_obj = make_obj(
            "zoidium:repeater/shear",
            &[
                ("field", json!(1)),
                ("fieldScale", json!([40.0, 12.0, 12.0])),
                ("fieldFalloff", json!(50.0)),
                ("fieldDirection", json!([0.0, 1.0, 0.0])),
            ],
        );
        let lin_field = evaluate_falloff_field(&lin_obj, 0.0).expect("linear field");
        let lin_geom = build_field_wireframe_geometry(&lin_field, Mat4::IDENTITY);
        // 1 shaft + 4 arrowhead wings + 4 start plane edges = 9 primary segments; 4 end plane edges.
        assert_eq!(lin_geom.primary_segments.len(), 9);
        assert_eq!(lin_geom.falloff_segments.len(), 4);
        let tip = lin_geom
            .handles
            .iter()
            .find_map(|&(h, p)| (h == FieldEditHandle::PosX).then_some(p))
            .expect("linear tip handle");
        assert!((tip - Vec3::new(0.0, 20.0, 0.0)).length() < 1e-4);

        // 4. Infinite (0) and Noise (5) produce no field wireframe.
        let inf_obj = make_obj("zoidium:repeater/voronoi-fracture", &[("field", json!(0))]);
        assert!(evaluate_falloff_field(&inf_obj, 0.0).is_none());
        let noise_obj = make_obj("zoidium:repeater/warp", &[("field", json!(5))]);
        assert!(evaluate_falloff_field(&noise_obj, 0.0).is_none());
    }

    #[test]
    fn field_center_and_extent_drag_math_respects_world_inverse_and_schema() {
        let cam = ResolvedCamera3D::from_editor_camera(&EditorCameraState::new(), 16.0 / 9.0);
        let canvas = Rect::from_min_size(pos2(0.0, 0.0), vec2(960.0, 540.0));
        let center_s = cam
            .project_world_point(Vec3::ZERO, canvas)
            .expect("origin projects");
        let init_hit = cam
            .screen_ray(center_s, canvas)
            .and_then(|(ro, rd)| ray_plane_intersect(ro, rd, Vec3::ZERO, cam.forward));
        // Dragging right on screen moves local fieldPosition along +X (scaled by inverse of 2x effector scale).
        let eff_world = Mat4::from_scale(Vec3::splat(2.0));
        let moved = compute_field_center_drag(
            &cam,
            canvas,
            Vec3::ZERO,
            init_hit,
            eff_world,
            Vec3::ZERO,
            center_s,
            center_s + vec2(60.0, 0.0),
            false,
        );
        assert!(moved.x > 1.0);
        assert!(moved.y.abs() < 1e-2);

        let handle_s = pos2(580.0, 270.0); // 100 px along +X from (480, 270)
        // Dragging +50px outward increases outer_half from 20 -> 30 (50% increase).
        let next_radius = compute_field_extent_drag(
            &cam,
            canvas,
            FieldWireframeKind::Sphere,
            FieldEditHandle::PosX,
            FieldExtentProperty::Radius("fieldRadius"),
            Vec3::ZERO,
            pos2(480.0, 270.0),
            handle_s,
            Mat4::IDENTITY,
            Vec3::X,
            Vec3::splat(20.0),
            Vec3::splat(20.0),
            handle_s,
            pos2(630.0, 270.0),
            false,
            false,
        );
        assert!((next_radius.x - 30.0).abs() < 1e-3);

        // Box X-face drag updates only X full size (40 -> 60), leaving Y and Z unchanged.
        let next_box = compute_field_extent_drag(
            &cam,
            canvas,
            FieldWireframeKind::Box,
            FieldEditHandle::PosX,
            FieldExtentProperty::Scale("fieldScale"),
            Vec3::ZERO,
            pos2(480.0, 270.0),
            handle_s,
            Mat4::IDENTITY,
            Vec3::X,
            Vec3::new(40.0, 20.0, 20.0),
            Vec3::new(20.0, 10.0, 10.0),
            handle_s,
            pos2(630.0, 270.0),
            false,
            false,
        );
        assert!((next_box.x - 60.0).abs() < 1e-3);
        assert!((next_box.y - 20.0).abs() < 1e-3);
        assert!((next_box.z - 20.0).abs() < 1e-3);
    }
}

// ---------------------------------------------------------------------------
// Interactive Shape+ & LayerMask Bezier Canvas Authoring
// ---------------------------------------------------------------------------
