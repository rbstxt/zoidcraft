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
        return false;
    };
    let clip_id = clip.id.clone();
    let clip_start = clip.start;
    let obj_id = obj.id.clone();
    let local_frame = session.playback.current_frame - clip_start;
    if !is_object_3d_enabled(&clip.objects_3d, &obj_id, local_frame) {
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
    let Some(origin_screen) = cam.project_world_point(origin_world, canvas_rect) else {
        return false;
    };

    let (axis_x, axis_y, axis_z) = gizmo_3d_axes_for_space(world, state.gizmo_3d_space);
    let world_len =
        (GIZMO_SCREEN_LEN_PX * cam.world_units_per_px(origin_world, canvas_rect)).max(0.05);
    let visible_canvas = canvas_rect.intersect(panel_rect);

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
// Interactive Shape+ & LayerMask Bezier Canvas Authoring
// ---------------------------------------------------------------------------
