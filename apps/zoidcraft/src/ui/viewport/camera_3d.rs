//! 3D editing camera resolution, object transforms, and 3D helper wireframes.

use super::*;

/// Returns `true` if a 3D object kind represents any scene camera (standard or Camera+).
pub(crate) fn is_camera_3d_kind(kind: &str) -> bool {
    let k = kind.trim().to_ascii_lowercase();
    matches!(
        k.as_str(),
        "camera"
            | "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera_plus"
            | "zoidium:camera-plus/camera-plus"
            | "zoidium:camera_plus/camera"
            | "zoidium:camera_plus/camera_plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

/// Returns `true` if a 3D object kind represents a Camera+ camera using film gate / focal length.
pub(crate) fn is_camera_plus_3d_kind(kind: &str) -> bool {
    let k = kind.trim().to_ascii_lowercase();
    matches!(
        k.as_str(),
        "zoidium:camera-plus/camera"
            | "zoidium:camera-plus/camera_plus"
            | "zoidium:camera-plus/camera-plus"
            | "zoidium:camera_plus/camera"
            | "zoidium:camera_plus/camera_plus"
            | "zoidcraft:camera-plus/camera"
            | "zoidcraft:camera-plus/camera-plus"
    )
}

pub(crate) fn vibrate_hash_noise(index: f64, seed: f64) -> f64 {
    let r = ((index + 1.0) * 12.9898 + seed * 78.233).sin() * 43_758.545_3;
    (r - r.floor()) * 2.0 - 1.0
}

pub(crate) fn vibrate_sample_noise(x: f64, seed: f64, stepped: bool) -> f64 {
    let i = x.floor();
    let mut a = x - i;
    if stepped {
        a = (a + 0.5).floor();
    }
    a = a * a * (3.0 - 2.0 * a);
    vibrate_hash_noise(i, seed) * (1.0 - a) + vibrate_hash_noise(i + 1.0, seed) * a
}

pub(crate) fn find_scene_camera<'a>(
    clip: &'a SessionClip,
    local_frame: f64,
    active_camera_id: Option<&str>,
) -> Option<&'a SessionObject3D> {
    if let Some(id) = active_camera_id
        && let Some(cam) = clip
            .objects_3d
            .iter()
            .find(|o| o.id == id && is_camera_3d_kind(&o.kind))
    {
        return Some(cam);
    }
    let enabled_cams: Vec<&SessionObject3D> = clip
        .objects_3d
        .iter()
        .filter(|o| {
            is_camera_3d_kind(&o.kind) && is_object_3d_enabled(&clip.objects_3d, &o.id, local_frame)
        })
        .collect();
    if let Some(cam) = enabled_cams.iter().copied().find(|o| {
        is_camera_plus_3d_kind(&o.kind)
            && o.properties
                .get("active")
                .is_none_or(|p| eval_prop_scalar(p, local_frame, 1.0) >= 0.5)
    }) {
        return Some(cam);
    }
    if let Some(cam) = enabled_cams.iter().copied().find(|o| {
        o.properties
            .get("active")
            .is_some_and(|p| eval_prop_scalar(p, local_frame, 0.0) >= 0.5)
    }) {
        return Some(cam);
    }
    enabled_cams.into_iter().next()
}

pub(crate) fn eval_prop_from_keys_scalar(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: f64,
) -> (f64, bool, bool) {
    for &key in keys {
        if let Some(prop) = props.get(key) {
            let val = eval_prop_scalar(prop, local_frame, default);
            let animated = prop.animated || prop.expression.is_some() || prop.keyframes.len() > 1;
            return (val, animated, true);
        }
    }
    (default, false, false)
}

pub(crate) fn eval_prop_from_keys_vec3(
    props: &BTreeMap<String, SessionProperty>,
    keys: &[&str],
    local_frame: f64,
    default: Vec3,
) -> Vec3 {
    for &key in keys {
        if let Some(prop) = props.get(key) {
            return eval_prop_vec3(prop, local_frame, default);
        }
    }
    default
}

pub(crate) fn eval_object_3d_euler_order(
    obj: &SessionObject3D,
    local_frame: f64,
) -> GeomEulerOrder {
    let Some(prop) = obj.properties.get("eulerOrder") else {
        return GeomEulerOrder::YXZ;
    };
    let val = if prop.keyframes.is_empty() {
        &prop.default_value
    } else {
        &prop.keyframes[0].value
    };
    if let Some(s) = val.as_str() {
        return match s.trim().to_ascii_uppercase().as_str() {
            "XYZ" => GeomEulerOrder::XYZ,
            "YZX" => GeomEulerOrder::YZX,
            "ZXY" => GeomEulerOrder::ZXY,
            "XZY" => GeomEulerOrder::XZY,
            "ZYX" => GeomEulerOrder::ZYX,
            _ => GeomEulerOrder::YXZ,
        };
    }
    match eval_prop_scalar(prop, local_frame, 4.0).round() as i32 {
        0 => GeomEulerOrder::XYZ,
        1 => GeomEulerOrder::YZX,
        2 => GeomEulerOrder::ZXY,
        3 => GeomEulerOrder::XZY,
        5 => GeomEulerOrder::ZYX,
        _ => GeomEulerOrder::YXZ,
    }
}

/// Active 3D projection camera (either the interactive `EditorCameraState` or the scene's
/// evaluated 3D camera / default CM3 scene camera) used for accurate 3D overlays, gizmos & picking.
pub(crate) struct ResolvedCamera3D {
    pub(crate) view_proj: Mat4,
    pub(crate) eye: Vec3,
    pub(crate) right: Vec3,
    pub(crate) up: Vec3,
    pub(crate) forward: Vec3,
    pub(crate) fov_deg: f32,
    pub(crate) ortho_world_h: Option<f32>,
    pub(crate) camera_object_id: Option<String>,
}

impl ResolvedCamera3D {
    pub(crate) fn from_editor_camera(cam: &EditorCameraState, aspect: f32) -> Self {
        let (right, up, forward) = cam.basis();
        let view_proj = cam.projection_matrix(aspect) * cam.view_matrix();
        Self {
            view_proj,
            eye: cam.position,
            right,
            up,
            forward,
            fov_deg: cam.fov_deg,
            ortho_world_h: None,
            camera_object_id: None,
        }
    }

    pub(crate) fn from_scene_clip(
        clip: &SessionClip,
        local_frame: f64,
        seq_w: f32,
        seq_h: f32,
    ) -> Self {
        Self::from_scene_clip_with_override(clip, local_frame, seq_w, seq_h, None)
    }

    pub(crate) fn from_scene_clip_with_override(
        clip: &SessionClip,
        local_frame: f64,
        seq_w: f32,
        seq_h: f32,
        active_camera_id: Option<&str>,
    ) -> Self {
        let w = seq_w.max(1.0);
        let h = seq_h.max(1.0);
        let aspect = (w / h).max(0.01);

        if let Some(cam_obj) = find_scene_camera(clip, local_frame, active_camera_id) {
            let parent_world = cam_obj.parent_id.as_deref().map_or(Mat4::IDENTITY, |pid| {
                compute_object_3d_world_matrix(&clip.objects_3d, pid, local_frame)
            });
            let mut local_pos = cam_obj
                .properties
                .get("position")
                .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
            let mut local_euler = cam_obj
                .properties
                .get("rotation")
                .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
            let local_scale = cam_obj
                .properties
                .get("scale")
                .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
            let euler_order = eval_object_3d_euler_order(cam_obj, local_frame);

            // Apply Camera+ Vibrate offsets matching `zc_render::engine`.
            let (vib_enabled, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["vibrate.enabled", "vibrateEnabled"],
                local_frame,
                0.0,
            );
            if vib_enabled >= 0.5 {
                let (seed, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.seed", "vibrateSeed"],
                    local_frame,
                    1.0,
                );
                let (stepped_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.regularPulse", "vibrate.regular_pulse"],
                    local_frame,
                    0.0,
                );
                let stepped = stepped_v >= 0.5;
                let (rel_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.relative"],
                    local_frame,
                    0.0,
                );
                let relative = rel_v >= 0.5;
                let (en_pos_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.enablePosition", "vibrate.enable_position"],
                    local_frame,
                    0.0,
                );
                let (en_rot_v, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.enableRotation", "vibrate.enable_rotation"],
                    local_frame,
                    0.0,
                );
                let pos_amp = eval_prop_from_keys_vec3(
                    &cam_obj.properties,
                    &["vibrate.positionAmplitude", "vibrate.position_amplitude"],
                    local_frame,
                    Vec3::ZERO,
                );
                let rot_amp_deg = eval_prop_from_keys_vec3(
                    &cam_obj.properties,
                    &["vibrate.rotationAmplitude", "vibrate.rotation_amplitude"],
                    local_frame,
                    Vec3::ZERO,
                );
                let (fallback_freq, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.frequency"],
                    local_frame,
                    1.0,
                );
                let (pos_freq, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.positionFrequency", "vibrate.position_frequency"],
                    local_frame,
                    fallback_freq,
                );
                let (rot_freq, _, _) = eval_prop_from_keys_scalar(
                    &cam_obj.properties,
                    &["vibrate.rotationFrequency", "vibrate.rotation_frequency"],
                    local_frame,
                    fallback_freq,
                );
                let t = local_frame;
                if en_pos_v >= 0.5 {
                    let base = local_pos.to_array();
                    let mut out = base;
                    for h_idx in 0..3 {
                        let mut sc = f64::from(pos_amp[h_idx]);
                        if relative {
                            sc *= 0.01 * f64::from(base[h_idx].abs());
                        }
                        out[h_idx] += (vibrate_sample_noise(
                            t * pos_freq.max(0.0),
                            seed + 17.0 * (h_idx as f64),
                            stepped,
                        ) * sc) as f32;
                    }
                    local_pos = Vec3::from_array(out);
                }
                if en_rot_v >= 0.5 {
                    let base = local_euler.to_array();
                    let mut out = base;
                    for f_idx in 0..3 {
                        let mut sc = f64::from(rot_amp_deg[f_idx]);
                        if relative {
                            sc *= 0.01 * f64::from(base[f_idx].abs());
                        }
                        out[f_idx] += (vibrate_sample_noise(
                            t * rot_freq.max(0.0),
                            seed + 101.0 + 17.0 * (f_idx as f64),
                            stepped,
                        ) * sc
                            * (std::f64::consts::PI / 180.0))
                            as f32;
                    }
                    local_euler = Vec3::from_array(out);
                }
            }

            let local_tf = Transform3D {
                position: local_pos,
                rotation: local_euler,
                scale: local_scale,
                euler_order,
            };
            let world = parent_world * local_tf.to_matrix();
            let view = if world.determinant().abs() > 1e-8 {
                world.inverse()
            } else {
                Mat4::IDENTITY
            };
            let eye = world.w_axis.truncate();
            let right = normalized_or(world.x_axis.truncate(), Vec3::X);
            let up = normalized_or(world.y_axis.truncate(), Vec3::Y);
            // Right-handed Three.js / CM3 cameras look down local -Z.
            let forward = normalized_or(-world.z_axis.truncate(), Vec3::NEG_Z);

            let cam_kind = cam_obj
                .properties
                .get("camera.kind")
                .map_or(0, |p| eval_prop_scalar(p, local_frame, 0.0).round() as i32);
            let zoom = cam_obj
                .properties
                .get("zoom")
                .map_or(1.0, |p| eval_prop_scalar(p, local_frame, 1.0) as f32)
                .max(1e-4);
            let near = cam_obj
                .properties
                .get("near")
                .map_or(0.1, |p| eval_prop_scalar(p, local_frame, 0.1) as f32)
                .max(1e-4);
            let far = cam_obj
                .properties
                .get("far")
                .map_or(5000.0, |p| eval_prop_scalar(p, local_frame, 5000.0) as f32)
                .max(near + 1.0);
            let (film_offset_x_f64, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["filmOffsetX", "film_offset_x"],
                local_frame,
                0.0,
            );
            let (film_offset_y_f64, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["filmOffsetY", "film_offset_y"],
                local_frame,
                0.0,
            );
            let film_offset_x = film_offset_x_f64 as f32;
            let film_offset_y = film_offset_y_f64 as f32;

            if cam_kind == 1 {
                let half_h = (0.05 * h) / zoom;
                let proj = Mat4::from_cols_array_2d(&mat4_orthographic_film_offset(
                    w,
                    h,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                ));
                return Self {
                    view_proj: proj * view,
                    eye,
                    right,
                    up,
                    forward,
                    fov_deg: 60.0,
                    ortho_world_h: Some(2.0 * half_h),
                    camera_object_id: Some(cam_obj.id.clone()),
                };
            }

            let is_camera_plus = is_camera_plus_3d_kind(&cam_obj.kind);
            let (focal_f64, focal_anim, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["focalLength", "focal_length"],
                local_frame,
                35.0,
            );
            let (gate_f64, _, _) = eval_prop_from_keys_scalar(
                &cam_obj.properties,
                &["filmGate", "film_gate"],
                local_frame,
                36.0,
            );
            let focal_length = (focal_f64 as f32).max(1e-3);
            let film_gate = (gate_f64 as f32).max(1e-3);
            let uses_film_gate = is_camera_plus
                || (film_gate - 36.0).abs() > 1e-4
                || focal_anim
                || (focal_length - 35.0).abs() > 1e-4;
            let has_film_offset = film_offset_x.abs() > 1e-6 || film_offset_y.abs() > 1e-6;

            let (proj, fov_deg) = if uses_film_gate {
                let p = Mat4::from_cols_array_2d(&mat4_perspective_film_gate(
                    focal_length,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                ));
                let film_height = film_gate / aspect.max(1.0);
                let half_tan = (film_height * 0.5) / focal_length / zoom;
                let eff_fov = (2.0 * half_tan.atan()).to_degrees().clamp(1.0, 179.0);
                (p, eff_fov)
            } else if has_film_offset {
                let base_fov = cam_obj
                    .properties
                    .get("fov")
                    .map_or(60.0, |p| eval_prop_scalar(p, local_frame, 60.0) as f32)
                    .clamp(1.0, 179.0);
                let fov_rad = base_fov.to_radians();
                let film_height = film_gate / aspect.max(1.0);
                let equiv_focal = (film_height * 0.5) / (fov_rad * 0.5).tan().max(1e-5);
                let p = Mat4::from_cols_array_2d(&mat4_perspective_film_gate(
                    equiv_focal,
                    film_gate,
                    aspect,
                    zoom,
                    film_offset_x,
                    film_offset_y,
                    near,
                    far,
                ));
                let eff_fov = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
                    .to_degrees()
                    .clamp(1.0, 179.0);
                (p, eff_fov)
            } else {
                let base_fov = cam_obj
                    .properties
                    .get("fov")
                    .map_or(60.0, |p| eval_prop_scalar(p, local_frame, 60.0) as f32)
                    .clamp(1.0, 179.0);
                let fov_rad = base_fov.to_radians();
                let eff_rad = (2.0 * ((fov_rad * 0.5).tan() / zoom).atan())
                    .clamp(1.0_f32.to_radians(), 179.0_f32.to_radians());
                let p = Mat4::from_cols_array_2d(&mat4_perspective_rh(eff_rad, aspect, near, far));
                (p, eff_rad.to_degrees())
            };

            return Self {
                view_proj: proj * view,
                eye,
                right,
                up,
                forward,
                fov_deg,
                ortho_world_h: None,
                camera_object_id: Some(cam_obj.id.clone()),
            };
        }

        // Default CM3 scene camera matching `zc-render`: `(0, 0, 80) -> (0, 0, 0)`, FOV 60 deg.
        let eye = Vec3::new(0.0, 0.0, 80.0);
        let view = camera_look_at_view(eye, Vec3::ZERO, Vec3::Y);
        let proj = CameraProjection::Perspective {
            fov_deg: 60.0,
            aspect,
            near: 0.1,
            far: 5000.0,
        }
        .to_wgpu_matrix();
        Self {
            view_proj: proj * view,
            eye,
            right: Vec3::X,
            up: Vec3::Y,
            forward: Vec3::NEG_Z,
            fov_deg: 60.0,
            ortho_world_h: None,
            camera_object_id: None,
        }
    }

    pub(crate) fn project_world_point(&self, world: Vec3, canvas_rect: Rect) -> Option<Pos2> {
        project_point_with_vp(self.view_proj, world, canvas_rect)
    }

    /// World-space units per CSS pixel at `world_point`'s depth.
    pub(crate) fn world_units_per_px(&self, world_point: Vec3, canvas_rect: Rect) -> f32 {
        let world_h = if let Some(oh) = self.ortho_world_h {
            oh
        } else {
            let depth = (world_point - self.eye).dot(self.forward).max(0.5);
            2.0 * depth * (0.5 * self.fov_deg.to_radians()).tan()
        };
        world_h / canvas_rect.height().max(1.0)
    }

    pub(crate) fn unproject_delta_at_depth(
        &self,
        screen_delta: Vec2,
        world_point: Vec3,
        canvas_rect: Rect,
    ) -> Vec3 {
        let units_per_px = self.world_units_per_px(world_point, canvas_rect);
        self.right * (screen_delta.x * units_per_px) - self.up * (screen_delta.y * units_per_px)
    }

    /// Casts a world-space ray `(origin, normalized_dir)` through `screen_pos` on `canvas_rect`.
    /// Works identically for perspective (including film offsets) and orthographic projections.
    pub(crate) fn screen_ray(&self, screen_pos: Pos2, canvas_rect: Rect) -> Option<(Vec3, Vec3)> {
        if self.view_proj.determinant().abs() <= 1e-10 {
            return None;
        }
        let inv_vp = self.view_proj.inverse();
        let ndc_x =
            ((screen_pos.x - canvas_rect.left()) / canvas_rect.width().max(1.0)) * 2.0 - 1.0;
        let ndc_y =
            1.0 - ((screen_pos.y - canvas_rect.top()) / canvas_rect.height().max(1.0)) * 2.0;

        let p_near_h = inv_vp * Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
        let p_far_h = inv_vp * Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
        if p_near_h.w.abs() <= 1e-6 || p_far_h.w.abs() <= 1e-6 {
            return None;
        }
        let p_near = p_near_h.truncate() / p_near_h.w;
        let p_far = p_far_h.truncate() / p_far_h.w;
        let dir = (p_far - p_near).normalize_or_zero();
        if dir.length_squared() < 1e-6 {
            return None;
        }
        Some((p_near, dir))
    }
}

pub(crate) fn normalized_or(v: Vec3, fallback: Vec3) -> Vec3 {
    if v.length_squared() < 1e-6 {
        fallback
    } else {
        v.normalize()
    }
}

pub(crate) fn project_point_with_vp(vp: Mat4, world: Vec3, canvas_rect: Rect) -> Option<Pos2> {
    let clip: Vec4 = vp * Vec4::new(world.x, world.y, world.z, 1.0);
    if clip.w <= 1e-4 {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    if ndc.z < 0.0 || ndc.z > 1.0 {
        return None;
    }
    let sx = canvas_rect.left() + (ndc.x * 0.5 + 0.5) * canvas_rect.width();
    let sy = canvas_rect.top() + (0.5 - ndc.y * 0.5) * canvas_rect.height();
    Some(pos2(sx, sy))
}

pub(crate) fn build_object_3d_index(objects: &[SessionObject3D]) -> HashMap<&str, usize> {
    let mut index = HashMap::with_capacity(objects.len());
    for (idx, obj) in objects.iter().enumerate() {
        index.entry(obj.id.as_str()).or_insert(idx);
    }
    index
}

pub(crate) fn is_object_3d_enabled_indexed(
    objects: &[SessionObject3D],
    index: &HashMap<&str, usize>,
    id: &str,
    local_frame: f64,
) -> bool {
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(&obj_idx) = index.get(cid) else {
            return false;
        };
        let obj = &objects[obj_idx];
        let enabled = obj
            .properties
            .get("enabled")
            .is_none_or(|p| eval_prop_scalar(p, local_frame, 1.0) >= 0.5);
        if !enabled {
            return false;
        }
        cur_id = obj.parent_id.as_deref();
    }
    true
}

pub(crate) fn is_object_3d_enabled(
    objects: &[SessionObject3D],
    id: &str,
    local_frame: f64,
) -> bool {
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(obj) = objects.iter().find(|o| o.id == cid) else {
            return false;
        };
        let enabled = obj
            .properties
            .get("enabled")
            .is_none_or(|p| eval_prop_scalar(p, local_frame, 1.0) >= 0.5);
        if !enabled {
            return false;
        }
        cur_id = obj.parent_id.as_deref();
    }
    true
}

pub(crate) fn compute_object_3d_world_matrix_indexed(
    objects: &[SessionObject3D],
    index: &HashMap<&str, usize>,
    id: &str,
    local_frame: f64,
) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(&obj_idx) = index.get(cid) else {
            break;
        };
        let obj = &objects[obj_idx];
        chain.push(obj);
        cur_id = obj.parent_id.as_deref();
    }

    let mut world = Mat4::IDENTITY;
    for obj in chain.into_iter().rev() {
        let pos = obj
            .properties
            .get("position")
            .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
        let rot = obj
            .properties
            .get("rotation")
            .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
        let scale = obj
            .properties
            .get("scale")
            .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
        let euler_order = eval_object_3d_euler_order(obj, local_frame);
        world *= Transform3D {
            position: pos,
            rotation: rot,
            scale,
            euler_order,
        }
        .to_matrix();
    }
    world
}

pub(crate) fn compute_object_3d_world_matrix(
    objects: &[SessionObject3D],
    id: &str,
    local_frame: f64,
) -> Mat4 {
    let mut chain = Vec::new();
    let mut cur_id = Some(id);
    for _ in 0..32 {
        let Some(cid) = cur_id else {
            break;
        };
        let Some(obj) = objects.iter().find(|o| o.id == cid) else {
            break;
        };
        chain.push(obj);
        cur_id = obj.parent_id.as_deref();
    }

    let mut world = Mat4::IDENTITY;
    for obj in chain.into_iter().rev() {
        let pos = obj
            .properties
            .get("position")
            .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
        let rot = obj
            .properties
            .get("rotation")
            .map_or(Vec3::ZERO, |p| eval_prop_vec3(p, local_frame, Vec3::ZERO));
        let scale = obj
            .properties
            .get("scale")
            .map_or(Vec3::ONE, |p| eval_prop_vec3(p, local_frame, Vec3::ONE));
        let euler_order = eval_object_3d_euler_order(obj, local_frame);
        world *= Transform3D {
            position: pos,
            rotation: rot,
            scale,
            euler_order,
        }
        .to_matrix();
    }
    world
}

// ---------------------------------------------------------------------------
// 3D Gizmo Modes, Bezier Authoring State & 2D Transform State
// ---------------------------------------------------------------------------

pub(crate) fn active_scene_3d_clip(session: &Session) -> Option<(&SessionClip, f64, bool)> {
    if session.selection.clip_ids.len() != 1 {
        return None;
    }
    let clip_id = &session.selection.clip_ids[0];
    let (track, clip) = session
        .document()
        .video_tracks
        .iter()
        .find_map(|t| t.clips.iter().find(|c| &c.id == clip_id).map(|c| (t, c)))?;
    if clip.kind != "scene" {
        return None;
    }
    let frame = session.playback.current_frame;
    if frame < clip.start || frame > clip.start + clip.length {
        return None;
    }
    Some((clip, frame - clip.start, track.locked))
}

pub(crate) fn active_scene_3d_selected_object(
    session: &Session,
) -> Option<(&SessionClip, &SessionObject3D, bool)> {
    let (clip, _, locked) = active_scene_3d_clip(session)?;
    let obj_id = session.selection.object_3d_ids.first()?;
    let obj = clip.objects_3d.iter().find(|o| &o.id == obj_id)?;
    Some((clip, obj, locked))
}

pub(crate) fn resolve_active_3d_camera(
    session: &Session,
    state: &ViewportState,
    clip: &SessionClip,
    local_frame: f64,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) -> ResolvedCamera3D {
    if state.active_camera_id.is_some() {
        ResolvedCamera3D::from_scene_clip_with_override(
            clip,
            local_frame,
            seq_w,
            seq_h,
            state.active_camera_id.as_deref(),
        )
    } else if is_editing_camera_active(session) {
        ResolvedCamera3D::from_editor_camera(
            &state.camera,
            canvas_rect.width() / canvas_rect.height().max(1.0),
        )
    } else {
        ResolvedCamera3D::from_scene_clip_with_override(clip, local_frame, seq_w, seq_h, None)
    }
}

pub(crate) fn draw_3d_line(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    a: Vec3,
    b: Vec3,
    stroke: Stroke,
) {
    if let (Some(pa), Some(pb)) = (
        cam.project_world_point(a, canvas_rect),
        cam.project_world_point(b, canvas_rect),
    ) {
        painter.line_segment([pa, pb], stroke);
    }
}

pub(crate) fn draw_3d_dashed_line(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    a: Vec3,
    b: Vec3,
    stroke: Stroke,
    dashes: usize,
) {
    let n = dashes.max(4);
    for i in 0..n {
        if i % 2 != 0 {
            continue;
        }
        let t0 = i as f32 / n as f32;
        let t1 = ((i + 1) as f32 / n as f32).min(1.0);
        let p0 = a.lerp(b, t0);
        let p1 = a.lerp(b, t1);
        draw_3d_line(painter, cam, canvas_rect, p0, p1, stroke);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn draw_3d_circle(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    center: Vec3,
    u_axis: Vec3,
    v_axis: Vec3,
    radius: f32,
    segments: usize,
    stroke: Stroke,
) {
    let segs = segments.max(12);
    let mut prev: Option<Pos2> = None;
    for i in 0..=segs {
        let theta = (i as f32 / segs as f32) * std::f32::consts::TAU;
        let (s, c) = theta.sin_cos();
        let wp = center + (u_axis * c + v_axis * s) * radius;
        let cur = cam.project_world_point(wp, canvas_rect);
        if let (Some(p0), Some(p1)) = (prev, cur) {
            painter.line_segment([p0, p1], stroke);
        }
        prev = cur;
    }
}

pub(crate) fn draw_3d_box_wireframe(
    painter: &egui::Painter,
    cam: &ResolvedCamera3D,
    canvas_rect: Rect,
    transform: Mat4,
    half_extents: Vec3,
    stroke: Stroke,
) {
    let hx = half_extents.x;
    let hy = half_extents.y;
    let hz = half_extents.z;
    let corners = [
        Vec3::new(-hx, -hy, -hz),
        Vec3::new(hx, -hy, -hz),
        Vec3::new(hx, hy, -hz),
        Vec3::new(-hx, hy, -hz),
        Vec3::new(-hx, -hy, hz),
        Vec3::new(hx, -hy, hz),
        Vec3::new(hx, hy, hz),
        Vec3::new(-hx, hy, hz),
    ]
    .map(|p| transform.transform_point3(p));
    let edges = [
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
    for (i, j) in edges {
        draw_3d_line(painter, cam, canvas_rect, corners[i], corners[j], stroke);
    }
}

pub(crate) fn contains_ascii_case_insensitive(haystack: &str, needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    haystack
        .as_bytes()
        .windows(needle.len())
        .any(|w| w.eq_ignore_ascii_case(needle))
}

pub(crate) fn paint_3d_helper_wireframes(
    ui: &Ui,
    session: &Session,
    state: &ViewportState,
    panel_rect: Rect,
    canvas_rect: Rect,
    seq_w: f32,
    seq_h: f32,
) {
    let Some((clip, local_frame, _)) = active_scene_3d_clip(session) else {
        return;
    };
    let cam =
        resolve_active_3d_camera(session, state, clip, local_frame, canvas_rect, seq_w, seq_h);
    let visible_canvas = canvas_rect.intersect(panel_rect);
    let painter = ui.painter().with_clip_rect(visible_canvas);
    let mut obj_index: Option<HashMap<&str, usize>> = None;

    for obj in &clip.objects_3d {
        let is_light = contains_ascii_case_insensitive(&obj.kind, b"light");
        let is_non_active_camera = is_camera_3d_kind(&obj.kind)
            && cam.camera_object_id.as_deref() != Some(obj.id.as_str());
        let has_field_prop = obj.properties.contains_key("field")
            || obj.properties.contains_key("fieldType")
            || obj.properties.contains_key("field_type");
        let has_constraint_prop = obj.properties.contains_key("constraint.trackTo.enabled")
            || obj.properties.contains_key("constraint.follow.enabled");

        if !is_light && !is_non_active_camera && !has_field_prop && !has_constraint_prop {
            continue;
        }

        let index = obj_index.get_or_insert_with(|| build_object_3d_index(&clip.objects_3d));
        if !is_object_3d_enabled_indexed(&clip.objects_3d, index, &obj.id, local_frame) {
            continue;
        }
        let is_selected = session.selection.object_3d_ids.contains(&obj.id);
        let world =
            compute_object_3d_world_matrix_indexed(&clip.objects_3d, index, &obj.id, local_frame);
        let origin = world.w_axis.truncate();
        let right = normalized_or(world.x_axis.truncate(), Vec3::X);
        let up = normalized_or(world.y_axis.truncate(), Vec3::Y);
        let forward = normalized_or(-world.z_axis.truncate(), Vec3::NEG_Z);

        // 1. Lights: Directional ray arrow, Point starburst, Spot cone.
        if is_light {
            let lkind = obj
                .properties
                .get("light.kind")
                .map_or(1, |p| eval_prop_scalar(p, local_frame, 1.0).round() as i32);
            let alpha = if is_selected { 235 } else { 145 };
            let col = Color32::from_rgba_unmultiplied(
                COLOR_WIRE_LIGHT.r(),
                COLOR_WIRE_LIGHT.g(),
                COLOR_WIRE_LIGHT.b(),
                alpha,
            );
            let stroke = Stroke::new(if is_selected { 1.6 } else { 1.1 }, col);
            let target_dir = obj
                .properties
                .get("target")
                .map(|p| eval_prop_vec3(p, local_frame, Vec3::ZERO) - origin)
                .filter(|d| d.length_squared() > 1e-4)
                .map_or(forward, Vec3::normalize);

            match lkind {
                // Spot light cone
                0 => {
                    let cone_len = 22.0_f32;
                    let half_angle = obj
                        .properties
                        .get("angle")
                        .map_or(0.45, |p| eval_prop_scalar(p, local_frame, 0.45) as f32)
                        .clamp(0.08, 1.35);
                    let base_center = origin + target_dir * cone_len;
                    let radius = cone_len * half_angle.tan();
                    let ortho_u = normalized_or(
                        target_dir.cross(if target_dir.y.abs() < 0.9 {
                            Vec3::Y
                        } else {
                            Vec3::X
                        }),
                        Vec3::X,
                    );
                    let ortho_v = normalized_or(target_dir.cross(ortho_u), Vec3::Y);
                    draw_3d_circle(
                        &painter,
                        &cam,
                        canvas_rect,
                        base_center,
                        ortho_u,
                        ortho_v,
                        radius,
                        24,
                        stroke,
                    );
                    for dir_offset in [ortho_u, -ortho_u, ortho_v, -ortho_v] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin,
                            base_center + dir_offset * radius,
                            stroke,
                        );
                    }
                    draw_3d_dashed_line(
                        &painter,
                        &cam,
                        canvas_rect,
                        origin,
                        base_center,
                        stroke,
                        8,
                    );
                }
                // Directional light rays + arrow
                2 => {
                    let ray_len = 20.0_f32;
                    let tip = origin + target_dir * ray_len;
                    let ortho_u = normalized_or(
                        target_dir.cross(if target_dir.y.abs() < 0.9 {
                            Vec3::Y
                        } else {
                            Vec3::X
                        }),
                        Vec3::X,
                    );
                    let ortho_v = normalized_or(target_dir.cross(ortho_u), Vec3::Y);
                    draw_3d_circle(
                        &painter,
                        &cam,
                        canvas_rect,
                        origin,
                        ortho_u,
                        ortho_v,
                        4.0,
                        18,
                        stroke,
                    );
                    draw_3d_line(&painter, &cam, canvas_rect, origin, tip, stroke);
                    for offset in [ortho_u * 3.0, -ortho_u * 3.0, ortho_v * 3.0, -ortho_v * 3.0] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin + offset,
                            tip + offset,
                            Stroke::new(0.9, col),
                        );
                    }
                    let head_base = tip - target_dir * 3.0;
                    draw_3d_line(
                        &painter,
                        &cam,
                        canvas_rect,
                        tip,
                        head_base + ortho_u * 1.6,
                        stroke,
                    );
                    draw_3d_line(
                        &painter,
                        &cam,
                        canvas_rect,
                        tip,
                        head_base - ortho_u * 1.6,
                        stroke,
                    );
                }
                // Point / Hemisphere / Ambient starburst
                _ => {
                    let r = 5.5_f32;
                    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin - axis * r,
                            origin + axis * r,
                            stroke,
                        );
                    }
                    let d = r * 0.65;
                    for diag in [
                        Vec3::new(1.0, 1.0, 1.0),
                        Vec3::new(-1.0, 1.0, 1.0),
                        Vec3::new(1.0, -1.0, 1.0),
                        Vec3::new(1.0, 1.0, -1.0),
                    ] {
                        let nd = diag.normalize() * d;
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            origin - nd,
                            origin + nd,
                            Stroke::new(0.9, col),
                        );
                    }
                    if let Some(sp) = cam.project_world_point(origin, canvas_rect) {
                        painter.circle_filled(sp, 3.0, col);
                    }
                }
            }
        }

        // 2. Cameras: frustum pyramid wireframe (skip the active view camera).
        if is_non_active_camera {
            let alpha = if is_selected { 235 } else { 150 };
            let col = Color32::from_rgba_unmultiplied(
                COLOR_WIRE_CAMERA.r(),
                COLOR_WIRE_CAMERA.g(),
                COLOR_WIRE_CAMERA.b(),
                alpha,
            );
            let stroke = Stroke::new(if is_selected { 1.6 } else { 1.1 }, col);
            let aspect = (seq_w / seq_h.max(1.0)).max(0.1);
            let frustum_len = 18.0_f32;
            let half_h = frustum_len * 0.42;
            let half_w = half_h * aspect;
            let far_c = origin + forward * frustum_len;
            let tl = far_c - right * half_w + up * half_h;
            let tr = far_c + right * half_w + up * half_h;
            let br = far_c + right * half_w - up * half_h;
            let bl = far_c - right * half_w - up * half_h;
            for corner in [tl, tr, br, bl] {
                draw_3d_line(&painter, &cam, canvas_rect, origin, corner, stroke);
            }
            draw_3d_line(&painter, &cam, canvas_rect, tl, tr, stroke);
            draw_3d_line(&painter, &cam, canvas_rect, tr, br, stroke);
            draw_3d_line(&painter, &cam, canvas_rect, br, bl, stroke);
            draw_3d_line(&painter, &cam, canvas_rect, bl, tl, stroke);
            // Top up-indicator triangle.
            let up_apex = far_c + up * (half_h * 1.38);
            draw_3d_line(
                &painter,
                &cam,
                canvas_rect,
                far_c - right * (half_w * 0.35) + up * (half_h * 1.06),
                up_apex,
                stroke,
            );
            draw_3d_line(
                &painter,
                &cam,
                canvas_rect,
                far_c + right * (half_w * 0.35) + up * (half_h * 1.06),
                up_apex,
                stroke,
            );
            draw_3d_line(
                &painter,
                &cam,
                canvas_rect,
                far_c - right * (half_w * 0.35) + up * (half_h * 1.06),
                far_c + right * (half_w * 0.35) + up * (half_h * 1.06),
                stroke,
            );
        }

        // 3. Effector+ Falloff Fields (sphere / box / linear / cylinder).
        let field_type_token = obj
            .properties
            .get("field")
            .or_else(|| obj.properties.get("fieldType"))
            .or_else(|| obj.properties.get("field_type"))
            .and_then(|p| {
                let v = if p.keyframes.is_empty() {
                    &p.default_value
                } else {
                    &p.keyframes[0].value
                };
                if let Some(s) = v.as_str() {
                    Some(s.trim().to_ascii_lowercase())
                } else {
                    let n = eval_prop_scalar(p, local_frame, 0.0).round() as i32;
                    match n {
                        1 => Some("linear".to_owned()),
                        2 => Some("box".to_owned()),
                        3 => Some("sphere".to_owned()),
                        4 => Some("cylinder".to_owned()),
                        _ => None,
                    }
                }
            });
        let is_effector_obj = contains_ascii_case_insensitive(&obj.kind, b"effector")
            || contains_ascii_case_insensitive(&obj.kind, b"expresstext")
            || field_type_token.is_some();
        if is_effector_obj && let Some(ref ftype) = field_type_token {
            let f_pos = eval_prop_from_keys_vec3(
                &obj.properties,
                &["fieldPosition", "field_position"],
                local_frame,
                Vec3::ZERO,
            );
            let f_rot_deg = eval_prop_from_keys_vec3(
                &obj.properties,
                &["fieldRotation", "field_rotation"],
                local_frame,
                Vec3::ZERO,
            );
            let f_scale = eval_prop_from_keys_vec3(
                &obj.properties,
                &["fieldScale", "field_scale"],
                local_frame,
                Vec3::splat(20.0),
            );
            let f_local = Transform3D::new(
                f_pos,
                Vec3::new(
                    f_rot_deg.x.to_radians(),
                    f_rot_deg.y.to_radians(),
                    f_rot_deg.z.to_radians(),
                ),
                Vec3::ONE,
            )
            .to_matrix();
            let f_world = world * f_local;
            let f_center = f_world.w_axis.truncate();
            let fx = normalized_or(f_world.x_axis.truncate(), Vec3::X);
            let fy = normalized_or(f_world.y_axis.truncate(), Vec3::Y);
            let fz = normalized_or(f_world.z_axis.truncate(), Vec3::Z);
            let base_col = if is_selected {
                COLOR_WIRE_FIELD_AMBER
            } else {
                COLOR_WIRE_FIELD_CYAN
            };
            let stroke = Stroke::new(
                if is_selected { 1.6 } else { 1.0 },
                Color32::from_rgba_unmultiplied(
                    base_col.r(),
                    base_col.g(),
                    base_col.b(),
                    if is_selected { 230 } else { 125 },
                ),
            );
            let half = (f_scale.abs() * 0.5).max(Vec3::splat(2.0));
            match ftype.as_str() {
                "sphere" => {
                    let r = half.x.max(half.y).max(half.z);
                    draw_3d_circle(&painter, &cam, canvas_rect, f_center, fx, fy, r, 32, stroke);
                    draw_3d_circle(&painter, &cam, canvas_rect, f_center, fx, fz, r, 32, stroke);
                    draw_3d_circle(&painter, &cam, canvas_rect, f_center, fy, fz, r, 32, stroke);
                }
                "box" => {
                    draw_3d_box_wireframe(&painter, &cam, canvas_rect, f_world, half, stroke);
                }
                "cylinder" => {
                    let r = half.x.max(half.z);
                    let top_c = f_center + fy * half.y;
                    let bot_c = f_center - fy * half.y;
                    draw_3d_circle(&painter, &cam, canvas_rect, top_c, fx, fz, r, 28, stroke);
                    draw_3d_circle(&painter, &cam, canvas_rect, bot_c, fx, fz, r, 28, stroke);
                    for dir in [fx, -fx, fz, -fz] {
                        draw_3d_line(
                            &painter,
                            &cam,
                            canvas_rect,
                            top_c + dir * r,
                            bot_c + dir * r,
                            stroke,
                        );
                    }
                }
                "linear" => {
                    let span = half.x.max(10.0);
                    let p_start = f_center - fx * span;
                    let p_end = f_center + fx * span;
                    draw_3d_line(&painter, &cam, canvas_rect, p_start, p_end, stroke);
                    let plane_h = half.y.clamp(5.0, 20.0);
                    let plane_d = half.z.clamp(5.0, 20.0);
                    for center_p in [p_start, p_end] {
                        let c0 = center_p - fy * plane_h - fz * plane_d;
                        let c1 = center_p + fy * plane_h - fz * plane_d;
                        let c2 = center_p + fy * plane_h + fz * plane_d;
                        let c3 = center_p - fy * plane_h + fz * plane_d;
                        draw_3d_line(&painter, &cam, canvas_rect, c0, c1, stroke);
                        draw_3d_line(&painter, &cam, canvas_rect, c1, c2, stroke);
                        draw_3d_line(&painter, &cam, canvas_rect, c2, c3, stroke);
                        draw_3d_line(&painter, &cam, canvas_rect, c3, c0, stroke);
                    }
                }
                _ => {}
            }
        }

        // 4. Constraints: TrackTo & Follow dashed connector lines to target objects.
        for (en_key, target_key, col) in [
            (
                "constraint.trackTo.enabled",
                "constraint.trackTo.targetId",
                COLOR_WIRE_TRACK_TO,
            ),
            (
                "constraint.follow.enabled",
                "constraint.follow.targetId",
                COLOR_WIRE_FOLLOW,
            ),
        ] {
            let enabled = obj
                .properties
                .get(en_key)
                .is_some_and(|p| eval_prop_scalar(p, local_frame, 0.0) >= 0.5);
            if !enabled {
                continue;
            }
            let target_id = obj.properties.get(target_key).and_then(|p| {
                let v = if p.keyframes.is_empty() {
                    &p.default_value
                } else {
                    &p.keyframes[0].value
                };
                v.as_str()
            });
            if let Some(tid) = target_id
                && !tid.is_empty()
                && index.contains_key(tid)
            {
                let target_world = compute_object_3d_world_matrix_indexed(
                    &clip.objects_3d,
                    index,
                    tid,
                    local_frame,
                );
                let target_pos = target_world.w_axis.truncate();
                let stroke = Stroke::new(
                    if is_selected { 1.6 } else { 1.1 },
                    Color32::from_rgba_unmultiplied(
                        col.r(),
                        col.g(),
                        col.b(),
                        if is_selected { 235 } else { 150 },
                    ),
                );
                draw_3d_dashed_line(&painter, &cam, canvas_rect, origin, target_pos, stroke, 14);
                if let Some(tp) = cam.project_world_point(target_pos, canvas_rect) {
                    let r = 4.0;
                    painter.add(Shape::closed_line(
                        vec![
                            pos2(tp.x, tp.y - r),
                            pos2(tp.x + r, tp.y),
                            pos2(tp.x, tp.y + r),
                            pos2(tp.x - r, tp.y),
                        ],
                        stroke,
                    ));
                }
            }
        }
    }
}
