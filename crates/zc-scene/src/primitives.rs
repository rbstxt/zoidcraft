//! Geometry+ procedural 3D primitives (Tube, Gear, Helix, Spline, extended RoundedBox),
//! flat-shading normal helper, Material+ normalized UV layout, and CM3 preset 3D models.

use std::collections::{HashMap, HashSet};
use std::f32::consts::{FRAC_PI_2, PI, TAU};

use zc_geom::{Mat4, Vec2, Vec3};

use crate::mesh::{MeshData, Vertex3D, build_box, build_cylinder};

/// Splits shared vertices per triangle and assigns exact geometric flat normals.
pub fn apply_flat_normals(mesh: &mut MeshData) {
    mesh.compute_flat_normals();
}

/// Geometry+ `Tube` (`zoidium:geometry-plus/tube`): hollow cylinder along Y centered at origin.
#[must_use]
pub fn build_tube(
    outer_radius: f32,
    inner_radius: f32,
    height: f32,
    radial_segments: u32,
    end_caps: bool,
    smooth: bool,
) -> MeshData {
    let outer_r = outer_radius.max(1e-4);
    let inner_r = inner_radius.clamp(1e-5, (outer_r - 1e-5).max(1e-5));
    let h = height.max(1e-4);
    let half_h = h * 0.5;
    let segs = radial_segments.max(3);
    let stride = segs + 1;

    let mut vertices = Vec::with_capacity((stride * if end_caps { 8 } else { 4 }) as usize);
    let mut indices = Vec::with_capacity((segs * if end_caps { 24 } else { 12 }) as usize);

    // Outer wall (outward normal: +[sin, 0, cos])
    let outer_base = vertices.len() as u32;
    for row in 0..2_u32 {
        let v = row as f32;
        let y = if row == 0 { -half_h } else { half_h };
        for i in 0..=segs {
            let u = i as f32 / segs as f32;
            let theta = u * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            vertices.push(Vertex3D::new(
                [outer_r * sin_t, y, outer_r * cos_t],
                [sin_t, 0.0, cos_t],
                [u, v],
                0,
            ));
        }
    }
    for i in 0..segs {
        let b0 = outer_base + i;
        let b1 = outer_base + i + 1;
        let t0 = outer_base + stride + i;
        let t1 = outer_base + stride + i + 1;
        indices.extend_from_slice(&[b0, b1, t1, b0, t1, t0]);
    }

    // Inner wall (inward normal: -[sin, 0, cos])
    let inner_base = vertices.len() as u32;
    for row in 0..2_u32 {
        let v = row as f32;
        let y = if row == 0 { -half_h } else { half_h };
        for i in 0..=segs {
            let u = i as f32 / segs as f32;
            let theta = u * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            vertices.push(Vertex3D::new(
                [inner_r * sin_t, y, inner_r * cos_t],
                [-sin_t, 0.0, -cos_t],
                [1.0 - u, v],
                0,
            ));
        }
    }
    for i in 0..segs {
        let b0 = inner_base + i;
        let b1 = inner_base + i + 1;
        let t0 = inner_base + stride + i;
        let t1 = inner_base + stride + i + 1;
        indices.extend_from_slice(&[b0, t1, b1, b0, t0, t1]);
    }

    if end_caps {
        let ratio = (inner_r / outer_r).clamp(0.0, 1.0);

        // Top annular cap (+Y)
        let top_outer = vertices.len() as u32;
        for i in 0..=segs {
            let theta = (i as f32 / segs as f32) * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            vertices.push(Vertex3D::new(
                [outer_r * sin_t, half_h, outer_r * cos_t],
                [0.0, 1.0, 0.0],
                [0.5 + 0.5 * sin_t, 0.5 + 0.5 * cos_t],
                0,
            ));
        }
        let top_inner = vertices.len() as u32;
        for i in 0..=segs {
            let theta = (i as f32 / segs as f32) * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            vertices.push(Vertex3D::new(
                [inner_r * sin_t, half_h, inner_r * cos_t],
                [0.0, 1.0, 0.0],
                [0.5 + 0.5 * ratio * sin_t, 0.5 + 0.5 * ratio * cos_t],
                0,
            ));
        }
        for i in 0..segs {
            let ot0 = top_outer + i;
            let ot1 = top_outer + i + 1;
            let it0 = top_inner + i;
            let it1 = top_inner + i + 1;
            indices.extend_from_slice(&[ot0, ot1, it1, ot0, it1, it0]);
        }

        // Bottom annular cap (-Y)
        let bot_outer = vertices.len() as u32;
        for i in 0..=segs {
            let theta = (i as f32 / segs as f32) * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            vertices.push(Vertex3D::new(
                [outer_r * sin_t, -half_h, outer_r * cos_t],
                [0.0, -1.0, 0.0],
                [0.5 + 0.5 * sin_t, 0.5 - 0.5 * cos_t],
                0,
            ));
        }
        let bot_inner = vertices.len() as u32;
        for i in 0..=segs {
            let theta = (i as f32 / segs as f32) * TAU;
            let (sin_t, cos_t) = theta.sin_cos();
            vertices.push(Vertex3D::new(
                [inner_r * sin_t, -half_h, inner_r * cos_t],
                [0.0, -1.0, 0.0],
                [0.5 + 0.5 * ratio * sin_t, 0.5 - 0.5 * ratio * cos_t],
                0,
            ));
        }
        for i in 0..segs {
            let ob0 = bot_outer + i;
            let ob1 = bot_outer + i + 1;
            let ib0 = bot_inner + i;
            let ib1 = bot_inner + i + 1;
            indices.extend_from_slice(&[ob0, ib1, ob1, ob0, ib0, ib1]);
        }
    }

    let mut mesh = MeshData::new(vertices, indices);
    if !smooth {
        apply_flat_normals(&mut mesh);
    }
    mesh
}

/// Extrudes a closed CCW star-shaped 2D contour (around origin) along Z with optional front/back
/// bevels, outward CCW winding, clean UVs, and optional smooth or flat shading.
fn extrude_star_contour_z(
    contour: &[Vec2],
    depth: f32,
    bevel_size: f32,
    bevel_segments: u32,
    smooth: bool,
) -> MeshData {
    let n = contour.len();
    if n < 3 {
        return MeshData::empty();
    }

    let d = depth.max(1e-4);
    let half_z = d * 0.5;

    // Ensure CCW order and compute bounds + cumulative perimeter + minimum edge length.
    let mut signed_area = 0.0_f32;
    for i in 0..n {
        let a = contour[i];
        let b = contour[(i + 1) % n];
        signed_area += a.x * b.y - b.x * a.y;
    }
    let pts: Vec<Vec2> = if signed_area < 0.0 {
        contour.iter().rev().copied().collect()
    } else {
        contour.to_vec()
    };

    let mut min_xy = Vec2::splat(f32::INFINITY);
    let mut max_xy = Vec2::splat(f32::NEG_INFINITY);
    let mut min_edge = f32::INFINITY;
    let mut min_r = f32::INFINITY;
    let mut cum_u = Vec::with_capacity(n + 1);
    cum_u.push(0.0_f32);
    let mut total_perim = 0.0_f32;
    let mut edge_normals = Vec::with_capacity(n);

    for i in 0..n {
        let a = pts[i];
        let b = pts[(i + 1) % n];
        min_xy = min_xy.min(a);
        max_xy = max_xy.max(a);
        min_r = min_r.min(a.length());
        let edge = b - a;
        let len = edge.length();
        total_perim += len;
        cum_u.push(total_perim);
        if len > 1e-6 {
            min_edge = min_edge.min(len);
            edge_normals.push(Vec2::new(edge.y / len, -edge.x / len));
        } else {
            edge_normals.push(a.normalize_or_zero());
        }
    }
    let inv_perim = if total_perim > 1e-6 {
        1.0 / total_perim
    } else {
        1.0
    };
    let span_xy = (max_xy - min_xy).max(Vec2::splat(1e-5));

    // Per-vertex outward miter offset (clamped for sharp/concave vertices so bevels never invert).
    let mut miters = Vec::with_capacity(n);
    for i in 0..n {
        let n_prev = edge_normals[(i + n - 1) % n];
        let n_curr = edge_normals[i];
        let bisector = n_prev + n_curr;
        let b_len = bisector.length();
        if b_len > 1e-5 {
            let dir = bisector / b_len;
            let denom = dir.dot(n_curr).max(0.4);
            let scale = (1.0 / denom).min(1.6);
            miters.push(dir * scale);
        } else {
            miters.push(n_curr);
        }
    }

    let max_bevel = (half_z * 0.98)
        .min((min_r * 0.45).max(0.0))
        .min((min_edge * 0.4).max(0.0));
    let bevel = bevel_size.max(0.0).min(max_bevel);
    let b_segs = if bevel > 1e-5 { bevel_segments } else { 0 };

    // Build profile rings (z, inset) from front cap (+half_z) to back cap (-half_z).
    let mut rings_spec: Vec<(f32, f32)> = Vec::new();
    if b_segs > 0 {
        let straight_half = (half_z - bevel).max(0.0);
        for k in 0..=b_segs {
            let a = FRAC_PI_2 * (k as f32 / b_segs as f32);
            let z = straight_half + bevel * a.cos();
            let inset = bevel * (1.0 - a.sin());
            rings_spec.push((z, inset));
        }
        // Only add separate back-start ring if straight_half > eps (avoid duplicate zero-height ring).
        let back_start = if straight_half > 1e-5 { 0 } else { 1 };
        for k in back_start..=b_segs {
            let a = FRAC_PI_2 * (k as f32 / b_segs as f32);
            let z = -straight_half - bevel * a.sin();
            let inset = bevel * (1.0 - a.cos());
            rings_spec.push((z, inset));
        }
    } else {
        rings_spec.push((half_z, 0.0));
        rings_spec.push((-half_z, 0.0));
    }

    let mut vertices: Vec<Vertex3D> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    // 1. Side + bevel strip rings.
    let strip_vert_start = vertices.len();
    let stride = (n + 1) as u32;
    let ring_count = rings_spec.len() as u32;

    for &(z, inset) in &rings_spec {
        let v_coord = (z + half_z) / d;
        for (j, &u_raw) in cum_u.iter().enumerate() {
            let idx = j % n;
            let p2 = pts[idx] - miters[idx] * inset;
            let u_coord = u_raw * inv_perim;
            let fallback_n = edge_normals[idx];
            vertices.push(Vertex3D::new(
                [p2.x, p2.y, z],
                [fallback_n.x, fallback_n.y, 0.0],
                [u_coord, v_coord],
                1,
            ));
        }
    }

    let strip_idx_start = indices.len();
    for r in 0..(ring_count - 1) {
        let r0 = r * stride;
        let r1 = (r + 1) * stride;
        for j in 0..(n as u32) {
            let a = r0 + j;
            let b = r1 + j;
            let c = r1 + j + 1;
            let d_idx = r0 + j + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d_idx]);
        }
    }
    let strip_vert_end = vertices.len();

    // Compute smooth area-weighted normals for the strip (stitching seam j=0 <-> j=n).
    let strip_count = strip_vert_end - strip_vert_start;
    let mut accum = vec![Vec3::ZERO; strip_count];
    for tri in indices[strip_idx_start..].as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(vertices[i0].position);
        let p1 = Vec3::from_array(vertices[i1].position);
        let p2 = Vec3::from_array(vertices[i2].position);
        let wn = (p1 - p0).cross(p2 - p0);
        accum[i0 - strip_vert_start] += wn;
        accum[i1 - strip_vert_start] += wn;
        accum[i2 - strip_vert_start] += wn;
    }
    for r in 0..ring_count as usize {
        let first = r * stride as usize;
        let last = first + n;
        let combined = accum[first] + accum[last];
        accum[first] = combined;
        accum[last] = combined;
    }
    for (v, n_vec) in vertices[strip_vert_start..strip_vert_end]
        .iter_mut()
        .zip(accum)
    {
        if n_vec.length_squared() > 1e-12 {
            v.normal = n_vec.normalize().to_array();
        }
    }

    // 2. Front cap (+Z) & Back cap (-Z) using inner cap contour.
    let cap_inset = rings_spec[0].1;
    let mut cap_pts = Vec::with_capacity(n);
    let mut cap_center = Vec2::ZERO;
    for i in 0..n {
        let p = pts[i] - miters[i] * cap_inset;
        cap_pts.push(p);
        cap_center += p;
    }
    cap_center /= n as f32;

    let add_cap = |vertices: &mut Vec<Vertex3D>, indices: &mut Vec<u32>, z: f32, is_front: bool| {
        let normal = [0.0, 0.0, if is_front { 1.0 } else { -1.0 }];
        let c_idx = vertices.len() as u32;
        let c_uv = [
            (cap_center.x - min_xy.x) / span_xy.x,
            (cap_center.y - min_xy.y) / span_xy.y,
        ];
        vertices.push(Vertex3D::new(
            [cap_center.x, cap_center.y, z],
            normal,
            c_uv,
            0,
        ));
        let ring_start = vertices.len() as u32;
        for &p in &cap_pts {
            let uv = [(p.x - min_xy.x) / span_xy.x, (p.y - min_xy.y) / span_xy.y];
            vertices.push(Vertex3D::new([p.x, p.y, z], normal, uv, 0));
        }
        for i in 0..n as u32 {
            let a = ring_start + i;
            let b = ring_start + ((i + 1) % n as u32);
            if is_front {
                indices.extend_from_slice(&[c_idx, a, b]);
            } else {
                indices.extend_from_slice(&[c_idx, b, a]);
            }
        }
    };

    add_cap(&mut vertices, &mut indices, half_z, true);
    add_cap(&mut vertices, &mut indices, -half_z, false);

    let mut mesh = MeshData::new(vertices, indices);
    mesh.center_at_origin();
    if !smooth {
        apply_flat_normals(&mut mesh);
    }
    mesh
}

/// Geometry+ `RoundedBox` (`zoidium:geometry-plus/rounded-box`) with independent XY corner radius,
/// corner segments, Z-edge bevel size, bevel segments, and smooth/flat shading toggle.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_rounded_box_ext(
    width: f32,
    height: f32,
    depth: f32,
    corner_radius: f32,
    corner_segments: u32,
    bevel_size: f32,
    bevel_segments: u32,
    smooth: bool,
) -> MeshData {
    let w = width.max(0.1);
    let h = height.max(0.1);
    let d = depth.max(0.1);
    let hw = w * 0.5;
    let hh = h * 0.5;
    let radius = corner_radius.max(0.0).min(hw).min(hh);
    let c_segs = corner_segments.max(1);

    let mut contour: Vec<Vec2> = Vec::new();
    if radius <= 1e-5 {
        contour.push(Vec2::new(-hw, -hh));
        contour.push(Vec2::new(hw, -hh));
        contour.push(Vec2::new(hw, hh));
        contour.push(Vec2::new(-hw, hh));
    } else {
        // 4 CCW corner arcs: bottom-right, top-right, top-left, bottom-left.
        let corners = [
            (Vec2::new(hw - radius, -hh + radius), -FRAC_PI_2, 0.0_f32),
            (Vec2::new(hw - radius, hh - radius), 0.0_f32, FRAC_PI_2),
            (Vec2::new(-hw + radius, hh - radius), FRAC_PI_2, PI),
            (Vec2::new(-hw + radius, -hh + radius), PI, PI + FRAC_PI_2),
        ];
        for (center, a0, a1) in corners {
            for k in 0..=c_segs {
                let t = k as f32 / c_segs as f32;
                let angle = a0 + (a1 - a0) * t;
                let pt = center + Vec2::new(angle.cos() * radius, angle.sin() * radius);
                if contour
                    .last()
                    .is_none_or(|prev| prev.distance_squared(pt) > 1e-8)
                {
                    contour.push(pt);
                }
            }
        }
        while contour.len() > 3
            && contour[0].distance_squared(*contour.last().unwrap_or(&contour[0])) <= 1e-8
        {
            contour.pop();
        }
    }

    extrude_star_contour_z(&contour, d, bevel_size, bevel_segments, smooth)
}

/// Geometry+ `Gear` (`zoidium:geometry-plus/gear`): extruded toothed gear with optional bevels.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_gear(
    teeth: u32,
    inner_radius: f32,
    outer_radius: f32,
    thickness: f32,
    tooth_width: f32,
    bevel_size: f32,
    bevel_segments: u32,
    smooth: bool,
) -> MeshData {
    let teeth = teeth.max(3);
    let outer_r = outer_radius.max(0.1);
    let inner_r = inner_radius.clamp(0.05, (outer_r - 0.05).max(0.05));
    let thick = thickness.max(0.05);
    let tw = tooth_width.clamp(0.05, 0.95);

    let step = TAU / teeth as f32;
    let tooth_half = step * tw * 0.5;
    let valley_half = (step * (1.0 - tw) * 0.25).min(step * 0.45);

    let point_at = |angle: f32, r: f32| Vec2::new(angle.cos() * r, angle.sin() * r);

    let mut contour = Vec::with_capacity((teeth * 4) as usize);
    for i in 0..teeth {
        let center = i as f32 * step;
        let p0 = point_at(center - step * 0.5 + valley_half * 0.15, inner_r);
        let p1 = point_at(center - tooth_half, outer_r);
        let p2 = point_at(center + tooth_half, outer_r);
        let p3 = point_at(center + step * 0.5 - valley_half * 0.15, inner_r);
        for p in [p0, p1, p2, p3] {
            if contour
                .last()
                .is_none_or(|prev: &Vec2| prev.distance_squared(p) > 1e-8)
            {
                contour.push(p);
            }
        }
    }
    while contour.len() > 3
        && contour[0].distance_squared(*contour.last().unwrap_or(&contour[0])) <= 1e-8
    {
        contour.pop();
    }

    let bevel = bevel_size
        .max(0.0)
        .min(thick * 0.5)
        .min((outer_r - inner_r) * 0.45);
    extrude_star_contour_z(&contour, thick, bevel, bevel_segments, smooth)
}

/// Computes parallel-transport (Bishop) frames `(tangents, normals, binormals)` along a 3D path
/// without Frenet inflection flips.
fn compute_parallel_transport_frames(
    centerline: &[Vec3],
    closed_loop: bool,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
    let count = centerline.len();
    let mut diffs = Vec::with_capacity(count.saturating_sub(1));
    for i in 0..count.saturating_sub(1) {
        let d = centerline[i + 1] - centerline[i];
        diffs.push(if d.length_squared() > 1e-12 {
            d.normalize()
        } else {
            Vec3::ZERO
        });
    }

    let mut tangents = Vec::with_capacity(count);
    for i in 0..count {
        let t = if i == 0 {
            if !diffs.is_empty() && diffs[0].length_squared() > 1e-12 {
                diffs[0]
            } else {
                Vec3::X
            }
        } else if i == count - 1 {
            if diffs[i - 1].length_squared() > 1e-12 {
                diffs[i - 1]
            } else {
                tangents[i - 1]
            }
        } else {
            let sum = diffs[i - 1] + diffs[i];
            if sum.length_squared() > 1e-12 {
                sum.normalize()
            } else if diffs[i].length_squared() > 1e-12 {
                diffs[i]
            } else if diffs[i - 1].length_squared() > 1e-12 {
                diffs[i - 1]
            } else {
                tangents[i - 1]
            }
        };
        tangents.push(t);
    }

    if closed_loop && count >= 3 {
        let avg = tangents[0] + tangents[count - 1];
        if avg.length_squared() > 1e-12 {
            let unit = avg.normalize();
            tangents[0] = unit;
            tangents[count - 1] = unit;
        }
    }

    let mut normals = Vec::with_capacity(count);
    let mut binormals = Vec::with_capacity(count);

    let t0 = tangents[0];
    let reference = if t0.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
    let mut prev_n = t0.cross(reference).normalize_or_zero();
    if prev_n.length_squared() < 1e-12 {
        prev_n = Vec3::X;
    }
    normals.push(prev_n);
    binormals.push(t0.cross(prev_n).normalize_or_zero());

    for i in 1..count {
        let prev_t = tangents[i - 1];
        let curr_t = tangents[i];
        let rot_axis = prev_t.cross(curr_t);
        let rotated = if rot_axis.length_squared() > 1e-10 {
            let axis = rot_axis.normalize();
            let angle = prev_t.dot(curr_t).clamp(-1.0, 1.0).acos();
            Mat4::from_axis_angle(axis, angle)
                .transform_vector3(prev_n)
                .normalize_or_zero()
        } else {
            prev_n
        };
        let ortho = rotated - curr_t * rotated.dot(curr_t);
        let n = if ortho.length_squared() > 1e-12 {
            ortho.normalize()
        } else {
            prev_n
        };
        let b = curr_t.cross(n).normalize_or_zero();
        normals.push(n);
        binormals.push(b);
        prev_n = n;
    }

    // Distribute closing twist evenly around a full closed loop so the seam matches smoothly.
    if closed_loop && count >= 3 {
        let n_start = normals[0];
        let n_end = normals[count - 1];
        let b_start = binormals[0];
        let twist = n_end.dot(b_start).atan2(n_end.dot(n_start));
        if twist.abs() > 1e-5 {
            let denom = (count - 1) as f32;
            for i in 0..count {
                let frac = i as f32 / denom;
                let angle = -twist * frac;
                let (s, c) = angle.sin_cos();
                let n = (normals[i] * c + binormals[i] * s).normalize_or_zero();
                let b = tangents[i].cross(n).normalize_or_zero();
                normals[i] = n;
                binormals[i] = b;
            }
        }
    }

    (tangents, normals, binormals)
}

/// Sweeps a circular tube along `centerline` with per-ring `radii`, parallel-transport frames,
/// and optional flat or hemispherical end caps.
fn sweep_tube_mesh(
    centerline: &[Vec3],
    radii: &[f32],
    radial_segments: u32,
    closed_loop: bool,
    caps: bool,
    cap_style: u32,
    smooth: bool,
) -> MeshData {
    let ring_count = centerline.len();
    if ring_count < 2 || radii.len() != ring_count {
        return MeshData::empty();
    }

    let r_segs = radial_segments.max(3);
    let stride = r_segs + 1;
    let (tangents, normals, binormals) = compute_parallel_transport_frames(centerline, closed_loop);

    let mut vertices = Vec::with_capacity(ring_count * stride as usize);
    let mut indices = Vec::with_capacity((ring_count - 1) * r_segs as usize * 6);

    let denom_u = (ring_count - 1) as f32;
    for i in 0..ring_count {
        let u = i as f32 / denom_u;
        let center = centerline[i];
        let r = radii[i].max(1e-4);
        let n_vec = normals[i];
        let b_vec = binormals[i];
        for j in 0..=r_segs {
            let v = j as f32 / r_segs as f32;
            let theta = v * TAU;
            let radial = (theta.cos() * n_vec + theta.sin() * b_vec).normalize_or_zero();
            let pos = center + radial * r;
            vertices.push(Vertex3D::new(pos.to_array(), radial.to_array(), [u, v], 0));
        }
    }

    for i in 0..(ring_count as u32 - 1) {
        let r0 = i * stride;
        let r1 = (i + 1) * stride;
        for j in 0..r_segs {
            let a = r0 + j;
            let b = r0 + j + 1;
            let c = r1 + j + 1;
            let d = r1 + j;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    let tube_vert_end = vertices.len();
    let tube_idx_end = indices.len();

    // Compute smooth area-weighted normals for the swept tube wall (stitching the UV seam
    // j = 0 <-> j = r_segs) so tapered splines with `thickness_stops` have accurate tilted normals.
    let mut accum = vec![Vec3::ZERO; tube_vert_end];
    for tri in indices[..tube_idx_end].as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let p0 = Vec3::from_array(vertices[i0].position);
        let p1 = Vec3::from_array(vertices[i1].position);
        let p2 = Vec3::from_array(vertices[i2].position);
        let wn = (p1 - p0).cross(p2 - p0);
        accum[i0] += wn;
        accum[i1] += wn;
        accum[i2] += wn;
    }
    for i in 0..ring_count {
        let first = i * stride as usize;
        let last = first + r_segs as usize;
        let combined = accum[first] + accum[last];
        accum[first] = combined;
        accum[last] = combined;
    }
    if closed_loop && ring_count >= 2 {
        let last_ring = (ring_count - 1) * stride as usize;
        for j in 0..stride as usize {
            let combined = accum[j] + accum[last_ring + j];
            accum[j] = combined;
            accum[last_ring + j] = combined;
        }
    }
    for (v, n_vec) in vertices[..tube_vert_end].iter_mut().zip(accum) {
        if n_vec.length_squared() > 1e-12 {
            v.normal = n_vec.normalize().to_array();
        }
    }

    if caps && !closed_loop {
        let add_flat_cap = |vertices: &mut Vec<Vertex3D>,
                            indices: &mut Vec<u32>,
                            ring_idx: usize,
                            is_end: bool| {
            let center = centerline[ring_idx];
            let r = radii[ring_idx].max(1e-4);
            let n_vec = normals[ring_idx];
            let b_vec = binormals[ring_idx];
            let cap_n = if is_end {
                tangents[ring_idx]
            } else {
                -tangents[ring_idx]
            };
            let n_arr = cap_n.to_array();
            let c_idx = vertices.len() as u32;
            vertices.push(Vertex3D::new(center.to_array(), n_arr, [0.5, 0.5], 0));
            let ring_start = vertices.len() as u32;
            for j in 0..=r_segs {
                let theta = (j as f32 / r_segs as f32) * TAU;
                let (sin_t, cos_t) = theta.sin_cos();
                let radial = (cos_t * n_vec + sin_t * b_vec).normalize_or_zero();
                let pos = center + radial * r;
                vertices.push(Vertex3D::new(
                    pos.to_array(),
                    n_arr,
                    [0.5 + 0.5 * cos_t, 0.5 + 0.5 * sin_t],
                    0,
                ));
            }
            for j in 0..r_segs {
                let r0 = ring_start + j;
                let r1 = ring_start + j + 1;
                if is_end {
                    indices.extend_from_slice(&[c_idx, r0, r1]);
                } else {
                    indices.extend_from_slice(&[c_idx, r1, r0]);
                }
            }
        };

        let add_round_cap = |vertices: &mut Vec<Vertex3D>,
                             indices: &mut Vec<u32>,
                             ring_idx: usize,
                             direction: f32| {
            let base_center = centerline[ring_idx];
            let cap_r = radii[ring_idx].max(1e-4);
            let t_vec = tangents[ring_idx];
            let n_vec = normals[ring_idx];
            let b_vec = binormals[ring_idx];
            let steps: u32 = 6;

            // Emit ring 0 (base) through ring `steps - 1`, then apex pole.
            let cap_base = vertices.len() as u32;
            for step in 0..steps {
                let alpha = FRAC_PI_2 * (step as f32 / steps as f32);
                let (sin_a, cos_a) = alpha.sin_cos();
                let ring_r = cap_r * cos_a;
                let offset = cap_r * sin_a * direction;
                let ring_center = base_center + t_vec * offset;
                let u_coord = if direction < 0.0 { 0.0 } else { 1.0 };
                for j in 0..=r_segs {
                    let v_coord = j as f32 / r_segs as f32;
                    let theta = v_coord * TAU;
                    let radial = (theta.cos() * n_vec + theta.sin() * b_vec).normalize_or_zero();
                    let pos = ring_center + radial * ring_r;
                    let normal = (radial * cos_a + t_vec * (sin_a * direction)).normalize_or_zero();
                    vertices.push(Vertex3D::new(
                        pos.to_array(),
                        normal.to_array(),
                        [u_coord, v_coord],
                        0,
                    ));
                }
            }

            for step in 0..(steps - 1) {
                let prev = cap_base + step * stride;
                let curr = cap_base + (step + 1) * stride;
                for j in 0..r_segs {
                    if direction < 0.0 {
                        let a = prev + j;
                        let b = curr + j;
                        let c = curr + j + 1;
                        let d = prev + j + 1;
                        indices.extend_from_slice(&[a, b, c, a, c, d]);
                    } else {
                        let a = prev + j;
                        let b = prev + j + 1;
                        let c = curr + j + 1;
                        let d = curr + j;
                        indices.extend_from_slice(&[a, b, c, a, c, d]);
                    }
                }
            }

            let pole_idx = vertices.len() as u32;
            let pole_pos = base_center + t_vec * (cap_r * direction);
            let pole_n = (t_vec * direction).normalize_or_zero();
            vertices.push(Vertex3D::new(
                pole_pos.to_array(),
                pole_n.to_array(),
                [0.5, 0.5],
                0,
            ));
            let last_ring = cap_base + (steps - 1) * stride;
            for j in 0..r_segs {
                let r0 = last_ring + j;
                let r1 = last_ring + j + 1;
                if direction < 0.0 {
                    indices.extend_from_slice(&[pole_idx, r1, r0]);
                } else {
                    indices.extend_from_slice(&[pole_idx, r0, r1]);
                }
            }
        };

        if cap_style != 0 {
            add_round_cap(&mut vertices, &mut indices, 0, -1.0);
            add_round_cap(&mut vertices, &mut indices, ring_count - 1, 1.0);
        } else {
            add_flat_cap(&mut vertices, &mut indices, 0, false);
            add_flat_cap(&mut vertices, &mut indices, ring_count - 1, true);
        }
    }

    let mut mesh = MeshData::new(vertices, indices);
    if !smooth {
        apply_flat_normals(&mut mesh);
    }
    mesh
}

/// Geometry+ `Helix` (`zoidium:geometry-plus/helix`): helical tube centered around Y.
#[must_use]
pub fn build_helix(
    radius: f32,
    height: f32,
    turns: f32,
    tube_radius: f32,
    path_segments: u32,
    radial_segments: u32,
    smooth: bool,
) -> MeshData {
    let r = radius.max(0.1);
    let h = height.max(0.1);
    let turns = turns.max(0.25);
    let tr = tube_radius.max(0.01);
    let p_segs = path_segments.max(8) as usize;
    let r_segs = radial_segments.max(3);

    let mut centerline = Vec::with_capacity(p_segs + 1);
    let mut radii = Vec::with_capacity(p_segs + 1);
    for k in 0..=p_segs {
        let t = k as f32 / p_segs as f32;
        let angle = TAU * turns * t;
        centerline.push(Vec3::new(r * angle.cos(), h * (t - 0.5), r * angle.sin()));
        radii.push(tr);
    }

    sweep_tube_mesh(&centerline, &radii, r_segs, false, false, 0, smooth)
}

fn sample_linear_spline(points: &[Vec3], amount: f32, closed: bool) -> Vec3 {
    let count = points.len();
    let spans = if closed { count } else { count - 1 };
    let scaled = amount.clamp(0.0, 1.0) * spans as f32;
    let index = (scaled.floor() as usize).min(spans - 1);
    let t = scaled - index as f32;
    let a = points[index % count];
    let b = points[(index + 1) % count];
    a.lerp(b, t)
}

fn sample_hermite_span(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32, tension: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let m1 = (p2 - p0) * (tension * 0.5);
    let m2 = (p3 - p1) * (tension * 0.5);
    p1 * h00 + m1 * h10 + p2 * h01 + m2 * h11
}

fn sample_catmull_rom_spline(points: &[Vec3], amount: f32, closed: bool, tension: f32) -> Vec3 {
    let count = points.len();
    let spans = if closed { count } else { count - 1 };
    let scaled = amount.clamp(0.0, 1.0) * spans as f32;
    let index = (scaled.floor() as usize).min(spans - 1);
    let t = scaled - index as f32;
    let p1 = points[index % count];
    let p2 = points[(index + 1) % count];
    let p0 = if closed {
        points[(index + count - 1) % count]
    } else {
        points[index.saturating_sub(1)]
    };
    let p3 = if closed {
        points[(index + 2) % count]
    } else {
        points[(index + 2).min(count - 1)]
    };
    sample_hermite_span(p0, p1, p2, p3, t, tension)
}

fn sample_akima_spline(points: &[Vec3], amount: f32, closed: bool) -> Vec3 {
    let count = points.len();
    let spans = if closed { count } else { count - 1 };
    let scaled = amount.clamp(0.0, 1.0) * spans as f32;
    let index = (scaled.floor() as usize).min(spans - 1);
    let t = scaled - index as f32;
    let t2 = t * t;
    let t3 = t2 * t;

    let mut out = [0.0_f32; 3];
    for (axis, out_slot) in out.iter_mut().enumerate() {
        let values: Vec<f32> = points.iter().map(|p| p[axis]).collect();
        let mut tangents = Vec::with_capacity(count);
        if closed {
            let mut slopes = Vec::with_capacity(count);
            for k in 0..count {
                slopes.push(values[(k + 1) % count] - values[k]);
            }
            for k in 0..count {
                let s0 = slopes[(k + count - 2) % count];
                let s1 = slopes[(k + count - 1) % count];
                let s2 = slopes[k];
                let s3 = slopes[(k + 1) % count];
                let w1 = (s3 - s2).abs();
                let w2 = (s1 - s0).abs();
                let m = if w1 + w2 < 1e-12 {
                    0.5 * (s1 + s2)
                } else {
                    (w1 * s1 + w2 * s2) / (w1 + w2)
                };
                tangents.push(m);
            }
        } else {
            let mut slopes = vec![0.0_f32; count + 4];
            for k in 0..(count - 1) {
                slopes[k + 2] = values[k + 1] - values[k];
            }
            slopes[1] = 2.0 * slopes[2] - slopes[3];
            slopes[0] = 2.0 * slopes[1] - slopes[2];
            slopes[count + 1] = 2.0 * slopes[count] - slopes[count - 1];
            slopes[count + 2] = 2.0 * slopes[count + 1] - slopes[count];
            for k in 0..count {
                let w1 = (slopes[k + 3] - slopes[k + 2]).abs();
                let w2 = (slopes[k + 1] - slopes[k]).abs();
                let m = if w1 + w2 < 1e-12 {
                    0.5 * (slopes[k + 1] + slopes[k + 2])
                } else {
                    (w1 * slopes[k + 1] + w2 * slopes[k + 2]) / (w1 + w2)
                };
                tangents.push(m);
            }
        }
        let y0 = values[index];
        let y1 = values[(index + 1) % count];
        let m0 = tangents[index];
        let m1 = tangents[(index + 1) % count];
        *out_slot = (2.0 * t3 - 3.0 * t2 + 1.0) * y0
            + (t3 - 2.0 * t2 + t) * m0
            + (-2.0 * t3 + 3.0 * t2) * y1
            + (t3 - t2) * m1;
    }
    Vec3::from_array(out)
}

fn sample_bspline_spline(points: &[Vec3], amount: f32, closed: bool) -> Vec3 {
    let count = points.len();
    if closed {
        let spans = count;
        let scaled = amount.clamp(0.0, 1.0) * spans as f32;
        let index = (scaled.floor() as usize).min(spans - 1);
        let t = scaled - index as f32;
        let t2 = t * t;
        let t3 = t2 * t;
        let p0 = points[index % count];
        let p1 = points[(index + 1) % count];
        let p2 = points[(index + 2) % count];
        let p3 = points[(index + 3) % count];
        let omt = 1.0 - t;
        let b0 = omt * omt * omt;
        let b1 = 3.0 * t3 - 6.0 * t2 + 4.0;
        let b2 = -3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0;
        let b3 = t3;
        return (p0 * b0 + p1 * b1 + p2 * b2 + p3 * b3) / 6.0;
    }

    let degree = 3.min(count - 1);
    let spans = count - degree;
    let knot = |j: usize| -> f32 {
        if j <= degree {
            0.0
        } else if j >= count {
            spans as f32
        } else {
            (j - degree) as f32
        }
    };
    let mut t = amount.clamp(0.0, 1.0) * spans as f32;
    if t >= spans as f32 {
        t = spans as f32 - 1e-6;
    }
    let mut span = degree;
    while span < count - 1 && t >= knot(span + 1) {
        span += 1;
    }
    let mut d: Vec<Vec3> = (0..=degree).map(|j| points[j + span - degree]).collect();
    for r in 1..=degree {
        for j in (r..=degree).rev() {
            let i = j + span - degree;
            let denom = knot(j + 1 + span - r) - knot(i);
            let alpha = if denom > 1e-12 {
                (t - knot(i)) / denom
            } else {
                0.0
            };
            d[j] = d[j - 1] * (1.0 - alpha) + d[j] * alpha;
        }
    }
    d[degree]
}

fn sample_spline_curve(
    points: &[Vec3],
    interpolation: u32,
    amount: f32,
    closed: bool,
    tension: f32,
) -> Vec3 {
    if !closed && points.len() == 2 {
        return sample_linear_spline(points, amount, false);
    }
    match interpolation {
        0 => sample_linear_spline(points, amount, closed),
        2 => sample_akima_spline(points, amount, closed),
        3 => sample_bspline_spline(points, amount, closed),
        _ => sample_catmull_rom_spline(points, amount, closed, tension),
    }
}

fn sample_thickness_stops(stops: &[(f32, f32)], position: f32) -> f32 {
    if stops.is_empty() {
        return 1.0;
    }
    let mut sorted: Vec<(f32, f32)> = stops
        .iter()
        .map(|&(p, v)| (p.clamp(0.0, 1.0), v.max(0.0)))
        .collect();
    sorted.sort_by(|a, b| a.0.total_cmp(&b.0));

    let u = position.clamp(0.0, 1.0);
    if u <= sorted[0].0 {
        return sorted[0].1;
    }
    let last = sorted[sorted.len() - 1];
    if u >= last.0 {
        return last.1;
    }
    for pair in sorted.windows(2) {
        let (p0, v0) = pair[0];
        let (p1, v1) = pair[1];
        if u <= p1 {
            let span = p1 - p0;
            let frac = if span > 1e-6 { (u - p0) / span } else { 0.0 };
            return v0 + (v1 - v0) * frac;
        }
    }
    last.1
}

/// Geometry+ `Spline` (`zoidium:geometry-plus/spline`): swept variable-radius tube along a 3D
/// curve supporting Linear (`0`), Cubic/Catmull-Rom (`1`), Akima (`2`), and B-Spline (`3`)
/// interpolation, parallel-transport Frenet frames, and Flat (`0`) or Round/Hemispherical (`1`)
/// end caps.
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn build_spline(
    points: &[[f32; 3]],
    interpolation: u32,
    closed: bool,
    start: f32,
    end: f32,
    thickness: f32,
    thickness_stops: &[(f32, f32)],
    tension: f32,
    path_segments: u32,
    radial_segments: u32,
    caps: bool,
    cap_style: u32,
    smooth: bool,
) -> MeshData {
    if points.len() < 2 {
        return MeshData::empty();
    }

    let mut s = start.clamp(0.0, 1.0);
    let mut e = end.clamp(0.0, 1.0);
    if e < s {
        std::mem::swap(&mut s, &mut e);
    }
    if e - s < 1e-4 {
        return MeshData::empty();
    }

    let pts: Vec<Vec3> = points.iter().map(|&p| Vec3::from_array(p)).collect();
    let tension = tension.clamp(0.0, 2.0);
    let base_radius = (thickness.max(0.0) * 0.5).max(1e-4);
    let p_segs = path_segments.max(1) as usize;
    let r_segs = radial_segments.max(3);

    let spans = if closed {
        pts.len()
    } else if interpolation == 3 && pts.len() >= 3 {
        pts.len() - 3.min(pts.len() - 1)
    } else {
        pts.len() - 1
    };
    let total_segments = (spans * p_segs).max(1);

    let mut centerline = Vec::with_capacity(total_segments + 1);
    let mut radii = Vec::with_capacity(total_segments + 1);
    for k in 0..=total_segments {
        let amount = s + (e - s) * (k as f32 / total_segments as f32);
        centerline.push(sample_spline_curve(
            &pts,
            interpolation,
            amount,
            closed,
            tension,
        ));
        let mult = sample_thickness_stops(thickness_stops, amount);
        radii.push((base_radius * mult).max(1e-4));
    }

    let full_closed_loop = closed && (e - s) >= 1.0 - 1e-4;
    sweep_tube_mesh(
        &centerline,
        &radii,
        r_segs,
        full_closed_loop,
        caps && !full_closed_loop,
        cap_style,
        smooth,
    )
}

/* ----------------------------------------------------- Material+ UV layout */

const UV_EPSILON: f32 = 1e-5;
const UV_QUANTIZE: f32 = 1e4;

fn uv_quantize_key(x: f32, y: f32) -> (i64, i64) {
    (
        (x * UV_QUANTIZE).round() as i64,
        (y * UV_QUANTIZE).round() as i64,
    )
}

struct PreparedContour {
    points: Vec<Vec2>,
    cumulative: Vec<f32>,
    total: f32,
}

fn reconstruct_xy_contours(mesh: &MeshData) -> Vec<PreparedContour> {
    let mut nodes: HashMap<(i64, i64), Vec2> = HashMap::new();
    let mut adjacency: HashMap<(i64, i64), HashSet<(i64, i64)>> = HashMap::new();
    let mut cap_tris = 0_usize;
    let mut wall_tris = 0_usize;
    let mut z_aligned_wall_tris = 0_usize;

    for tri in mesh.indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
            continue;
        }
        let ps = [
            Vec3::from_array(mesh.vertices[i0].position),
            Vec3::from_array(mesh.vertices[i1].position),
            Vec3::from_array(mesh.vertices[i2].position),
        ];
        let min_z = ps[0].z.min(ps[1].z).min(ps[2].z);
        let max_z = ps[0].z.max(ps[1].z).max(ps[2].z);
        if max_z - min_z <= UV_EPSILON {
            cap_tris += 1;
            continue; // cap plane
        }
        wall_tris += 1;

        let mut bottom = Vec::with_capacity(2);
        let mut top_count = 0_usize;
        for p in ps {
            if (p.z - min_z).abs() <= UV_EPSILON {
                bottom.push(Vec2::new(p.x, p.y));
            } else if (p.z - max_z).abs() <= UV_EPSILON {
                top_count += 1;
            }
        }
        if bottom.len() == 2 || top_count == 2 {
            z_aligned_wall_tris += 1;
        }
        if bottom.len() == 2 {
            let ka = uv_quantize_key(bottom[0].x, bottom[0].y);
            let kb = uv_quantize_key(bottom[1].x, bottom[1].y);
            if ka != kb {
                nodes.entry(ka).or_insert(bottom[0]);
                nodes.entry(kb).or_insert(bottom[1]);
                adjacency.entry(ka).or_default().insert(kb);
                adjacency.entry(kb).or_default().insert(ka);
            }
        }
    }

    // Only treat the mesh as XY-extruded when flat Z-caps exist and the vast majority of
    // non-cap triangles are Z-ring extrusion/bevel faces (excluding spheres, tubes, etc.).
    if cap_tris == 0 || wall_tris == 0 || z_aligned_wall_tris * 5 < wall_tris * 4 {
        return Vec::new();
    }

    let mut visited: HashSet<(i64, i64)> = HashSet::new();
    let mut prepared = Vec::new();
    let mut keys: Vec<(i64, i64)> = adjacency.keys().copied().collect();
    keys.sort_unstable();

    for start in keys {
        if visited.contains(&start) {
            continue;
        }
        let mut cycle = Vec::new();
        let mut prev: Option<(i64, i64)> = None;
        let mut curr = Some(start);
        while let Some(c) = curr {
            if !visited.insert(c) {
                break;
            }
            if let Some(&pt) = nodes.get(&c) {
                cycle.push(pt);
            }
            let next = adjacency.get(&c).and_then(|nbrs| {
                let mut sorted: Vec<(i64, i64)> = nbrs.iter().copied().collect();
                sorted.sort_unstable();
                sorted
                    .iter()
                    .copied()
                    .find(|&k| Some(k) != prev && !visited.contains(&k))
            });
            prev = Some(c);
            curr = next;
        }
        if cycle.len() >= 3 {
            let mut cumulative = Vec::with_capacity(cycle.len() + 1);
            cumulative.push(0.0_f32);
            let mut total = 0.0_f32;
            for i in 0..cycle.len() {
                let a = cycle[i];
                let b = cycle[(i + 1) % cycle.len()];
                total += a.distance(b);
                cumulative.push(total);
            }
            if total > 1e-6 {
                prepared.push(PreparedContour {
                    points: cycle,
                    cumulative,
                    total,
                });
            }
        }
    }
    prepared
}

fn contour_u_candidates(contours: &[PreparedContour], pt: Vec2, scale: f32) -> Vec<f32> {
    let mut best_d2 = f32::INFINITY;
    let mut out = Vec::new();
    let tie_eps = 1e-8_f32;

    for contour in contours {
        let len = contour.points.len();
        for i in 0..len {
            let a = contour.points[i];
            let b = contour.points[(i + 1) % len];
            let ab = b - a;
            let len2 = ab.length_squared();
            let t = if len2 > 0.0 {
                ((pt - a).dot(ab) / len2).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let proj = a + ab * t;
            let d2 = pt.distance_squared(proj);
            if d2 < best_d2 - tie_eps {
                best_d2 = d2;
                out.clear();
            } else if d2 > best_d2 + tie_eps {
                continue;
            }
            let arc = contour.cumulative[i] + len2.sqrt() * t;
            let u = if scale > 0.0 {
                arc / scale
            } else {
                arc / contour.total
            };
            out.push(u);
        }
    }
    out
}

/// Material+ normalized UV layout matching Zoidium `plugins/material-plus/uv-layout.js`.
///
/// - For extruded/3D-text geometry with reconstructable XY wall contours: unwraps caps into
///   uniform-density bounds `(x - min_x) / scale, (y - min_y) / scale` and unwraps side/bevel
///   walls along contour arc-length `u = arc / scale`, `v = (z - min_z) / scale`.
/// - For general 3D meshes without XY extrusion walls: applies uniform-density triplanar box
///   projection normalized by the dominant extent so texel density stays consistent across faces.
pub fn apply_normalized_uv_layout(mesh: &mut MeshData) {
    if mesh.vertices.is_empty() || mesh.indices.len() < 3 {
        return;
    }

    mesh.recompute_bounds();
    let b_min = mesh.bounds.min;
    let size = mesh.bounds.size();
    let scale = if size.y > 1e-6 {
        size.y
    } else if size.x > 1e-6 {
        size.x
    } else if size.z > 1e-6 {
        size.z
    } else {
        1.0
    };

    let contours = reconstruct_xy_contours(mesh);
    let has_colors = mesh.colors.len() == mesh.vertices.len() && !mesh.colors.is_empty();
    let tri_count = mesh.indices.len() / 3;
    let mut out_verts = Vec::with_capacity(tri_count * 3);
    let mut out_indices = Vec::with_capacity(tri_count * 3);
    let mut out_colors = if has_colors {
        Vec::with_capacity(tri_count * 3)
    } else {
        Vec::new()
    };

    if !contours.is_empty() {
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                continue;
            }
            let idxs = [i0, i1, i2];
            let ps = [
                Vec3::from_array(mesh.vertices[i0].position),
                Vec3::from_array(mesh.vertices[i1].position),
                Vec3::from_array(mesh.vertices[i2].position),
            ];
            let min_fz = ps[0].z.min(ps[1].z).min(ps[2].z);
            let max_fz = ps[0].z.max(ps[1].z).max(ps[2].z);
            let is_cap = (max_fz - min_fz) <= UV_EPSILON;

            let anchor_u = if is_cap {
                0.0
            } else {
                let centroid = Vec2::new(
                    (ps[0].x + ps[1].x + ps[2].x) / 3.0,
                    (ps[0].y + ps[1].y + ps[2].y) / 3.0,
                );
                contour_u_candidates(&contours, centroid, scale)
                    .first()
                    .copied()
                    .unwrap_or(0.0)
            };

            let base = out_verts.len() as u32;
            for k in 0..3 {
                let mut v = mesh.vertices[idxs[k]];
                let p = ps[k];
                if is_cap {
                    v.uv = [(p.x - b_min.x) / scale, (p.y - b_min.y) / scale];
                } else {
                    let cands = contour_u_candidates(&contours, Vec2::new(p.x, p.y), scale);
                    let best_u = cands
                        .into_iter()
                        .min_by(|a, b| (a - anchor_u).abs().total_cmp(&(b - anchor_u).abs()))
                        .unwrap_or(anchor_u);
                    v.uv = [best_u, (p.z - b_min.z) / scale];
                }
                out_verts.push(v);
                if has_colors {
                    out_colors.push(mesh.colors[idxs[k]]);
                }
            }
            out_indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    } else {
        // Uniform-density triplanar box projection fallback for general 3D meshes.
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= mesh.vertices.len() || i1 >= mesh.vertices.len() || i2 >= mesh.vertices.len() {
                continue;
            }
            let idxs = [i0, i1, i2];
            let ps = [
                Vec3::from_array(mesh.vertices[i0].position),
                Vec3::from_array(mesh.vertices[i1].position),
                Vec3::from_array(mesh.vertices[i2].position),
            ];
            let face_n = (ps[1] - ps[0]).cross(ps[2] - ps[0]);
            let ax = face_n.x.abs();
            let ay = face_n.y.abs();
            let az = face_n.z.abs();

            let base = out_verts.len() as u32;
            for k in 0..3 {
                let mut v = mesh.vertices[idxs[k]];
                let p = ps[k];
                v.uv = if ax >= ay && ax >= az {
                    [(p.z - b_min.z) / scale, (p.y - b_min.y) / scale]
                } else if ay >= ax && ay >= az {
                    [(p.x - b_min.x) / scale, (p.z - b_min.z) / scale]
                } else {
                    [(p.x - b_min.x) / scale, (p.y - b_min.y) / scale]
                };
                out_verts.push(v);
                if has_colors {
                    out_colors.push(mesh.colors[idxs[k]]);
                }
            }
            out_indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }

    mesh.vertices = out_verts;
    mesh.indices = out_indices;
    mesh.colors = out_colors;
    mesh.recompute_bounds();
}

/* ------------------------------------------------------ CM3 Preset Models */

fn set_slot(mesh: &mut MeshData, slot: u32) {
    for v in &mut mesh.vertices {
        v.material_slot = slot;
    }
}

fn add_box_part(dst: &mut MeshData, size: Vec3, translation: Vec3, rot_xyz: Vec3, slot: u32) {
    let mut part = build_box(size.x, size.y, size.z);
    set_slot(&mut part, slot);
    let m = Mat4::from_translation(translation)
        * Mat4::from_rotation_x(rot_xyz.x)
        * Mat4::from_rotation_y(rot_xyz.y)
        * Mat4::from_rotation_z(rot_xyz.z);
    part.apply_matrix(m);
    dst.merge(&part);
}

fn add_cyl_part(
    dst: &mut MeshData,
    radii: (f32, f32),
    height: f32,
    segs: u32,
    translation: Vec3,
    rot_xyz: Vec3,
    slot: u32,
) {
    let mut part = build_cylinder(radii.0, radii.1, height, segs, false);
    set_slot(&mut part, slot);
    let m = Mat4::from_translation(translation)
        * Mat4::from_rotation_x(rot_xyz.x)
        * Mat4::from_rotation_y(rot_xyz.y)
        * Mat4::from_rotation_z(rot_xyz.z);
    part.apply_matrix(m);
    dst.merge(&part);
}

/// Builds a detailed stylized procedural rifle silhouette (CM3 preset model 1).
fn build_rifle_preset() -> MeshData {
    let mut mesh = MeshData::empty();

    // 1. Main receiver body & upper Picatinny rail
    add_box_part(
        &mut mesh,
        Vec3::new(14.0, 3.2, 2.2),
        Vec3::new(0.0, 0.4, 0.0),
        Vec3::ZERO,
        0,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(13.0, 0.5, 1.3),
        Vec3::new(0.2, 2.2, 0.0),
        Vec3::ZERO,
        1,
    );
    // Lower receiver / magazine well
    add_box_part(
        &mut mesh,
        Vec3::new(10.0, 2.0, 2.0),
        Vec3::new(-0.5, -1.5, 0.0),
        Vec3::ZERO,
        0,
    );

    // 2. Forend / handguard (octagonal-ish upper & lower shrouds)
    add_box_part(
        &mut mesh,
        Vec3::new(13.0, 2.5, 2.0),
        Vec3::new(13.5, 0.5, 0.0),
        Vec3::ZERO,
        0,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(12.2, 0.45, 1.2),
        Vec3::new(13.5, 1.9, 0.0),
        Vec3::ZERO,
        1,
    );

    // 3. Long cylindrical barrel + gas block + muzzle brake (aligned along +X via -90 deg Z)
    add_cyl_part(
        &mut mesh,
        (0.52, 0.56),
        19.0,
        14,
        Vec3::new(22.0, 0.75, 0.0),
        Vec3::new(0.0, 0.0, -FRAC_PI_2),
        1,
    );
    // Gas tube above barrel
    add_cyl_part(
        &mut mesh,
        (0.22, 0.22),
        11.0,
        10,
        Vec3::new(18.0, 1.45, 0.0),
        Vec3::new(0.0, 0.0, -FRAC_PI_2),
        1,
    );
    // Muzzle brake / flash hider collar
    add_cyl_part(
        &mut mesh,
        (0.82, 0.72),
        2.6,
        12,
        Vec3::new(32.3, 0.75, 0.0),
        Vec3::new(0.0, 0.0, -FRAC_PI_2),
        1,
    );

    // 4. Shoulder stock + angled lower comb + buttplate pad
    add_box_part(
        &mut mesh,
        Vec3::new(13.5, 2.2, 1.8),
        Vec3::new(-13.2, -0.2, 0.0),
        Vec3::new(0.0, 0.0, 0.06),
        2,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(12.0, 2.8, 1.6),
        Vec3::new(-14.2, -2.1, 0.0),
        Vec3::new(0.0, 0.0, 0.18),
        2,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(1.2, 6.2, 1.9),
        Vec3::new(-20.2, -1.7, 0.0),
        Vec3::ZERO,
        1,
    );

    // 5. Angled pistol grip
    add_box_part(
        &mut mesh,
        Vec3::new(2.2, 6.4, 1.7),
        Vec3::new(-5.4, -5.0, 0.0),
        Vec3::new(0.0, 0.0, -0.32),
        2,
    );

    // 6. Curved two-segment box magazine
    add_box_part(
        &mut mesh,
        Vec3::new(3.2, 5.2, 1.6),
        Vec3::new(3.2, -4.6, 0.0),
        Vec3::new(0.0, 0.0, 0.14),
        1,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(3.0, 4.4, 1.55),
        Vec3::new(4.1, -8.8, 0.0),
        Vec3::new(0.0, 0.0, 0.28),
        1,
    );

    // 7. Trigger guard + trigger blade
    add_box_part(
        &mut mesh,
        Vec3::new(4.2, 0.4, 0.9),
        Vec3::new(-1.8, -4.2, 0.0),
        Vec3::ZERO,
        1,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(0.5, 1.5, 0.4),
        Vec3::new(-2.2, -3.1, 0.0),
        Vec3::new(0.0, 0.0, -0.22),
        1,
    );

    // 8. Sights & compact optic scope on rail
    add_box_part(
        &mut mesh,
        Vec3::new(1.2, 1.6, 1.0),
        Vec3::new(-5.0, 3.1, 0.0),
        Vec3::ZERO,
        1,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(0.9, 2.2, 0.8),
        Vec3::new(24.5, 2.1, 0.0),
        Vec3::ZERO,
        1,
    );
    // Scope mounts + optic tube
    add_box_part(
        &mut mesh,
        Vec3::new(1.0, 1.3, 1.1),
        Vec3::new(-1.0, 3.0, 0.0),
        Vec3::ZERO,
        1,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(1.0, 1.3, 1.1),
        Vec3::new(3.6, 3.0, 0.0),
        Vec3::ZERO,
        1,
    );
    add_cyl_part(
        &mut mesh,
        (1.1, 0.95),
        8.6,
        14,
        Vec3::new(1.4, 4.2, 0.0),
        Vec3::new(0.0, 0.0, -FRAC_PI_2),
        1,
    );

    mesh.center_at_origin();
    mesh
}

/// Builds a detailed Mystery Box chest (CM3 preset model 2) with base trim bands and an
/// articulated lid rotated by `lid_rotation_rad` around the back top hinge edge.
fn build_mystery_box_preset(lid_rotation_rad: f32) -> MeshData {
    let mut mesh = MeshData::empty();

    let w = 24.0_f32;
    let base_h = 10.0_f32;
    let d = 12.0_f32;
    let split_y = 2.0_f32; // Hinge Y height
    let base_center_y = split_y - base_h * 0.5;
    let hinge = Vec3::new(0.0, split_y, -d * 0.5);

    // 1. Chest base walls (hollow rim look via 4 walls + floor so interior is visible when open)
    let wall_t = 1.2_f32;
    // Floor
    add_box_part(
        &mut mesh,
        Vec3::new(w, wall_t, d),
        Vec3::new(0.0, split_y - base_h + wall_t * 0.5, 0.0),
        Vec3::ZERO,
        0,
    );
    // Front & back walls
    add_box_part(
        &mut mesh,
        Vec3::new(w, base_h, wall_t),
        Vec3::new(0.0, base_center_y, d * 0.5 - wall_t * 0.5),
        Vec3::ZERO,
        0,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(w, base_h, wall_t),
        Vec3::new(0.0, base_center_y, -d * 0.5 + wall_t * 0.5),
        Vec3::ZERO,
        0,
    );
    // Left & right walls
    add_box_part(
        &mut mesh,
        Vec3::new(wall_t, base_h, d - 2.0 * wall_t),
        Vec3::new(-w * 0.5 + wall_t * 0.5, base_center_y, 0.0),
        Vec3::ZERO,
        0,
    );
    add_box_part(
        &mut mesh,
        Vec3::new(wall_t, base_h, d - 2.0 * wall_t),
        Vec3::new(w * 0.5 - wall_t * 0.5, base_center_y, 0.0),
        Vec3::ZERO,
        0,
    );
    // Interior glowing bed pad (slot 2)
    add_box_part(
        &mut mesh,
        Vec3::new(w - 2.6, 0.8, d - 2.6),
        Vec3::new(0.0, split_y - base_h + wall_t + 0.4, 0.0),
        Vec3::ZERO,
        2,
    );

    // 2. Metal corner brackets, horizontal trim bands & front lock plate (slot 1)
    for sx in [-1.0_f32, 1.0_f32] {
        for sz in [-1.0_f32, 1.0_f32] {
            add_box_part(
                &mut mesh,
                Vec3::new(1.4, base_h + 0.3, 1.4),
                Vec3::new(sx * (w * 0.5 - 0.3), base_center_y, sz * (d * 0.5 - 0.3)),
                Vec3::ZERO,
                1,
            );
        }
    }
    // Lower & upper perimeter metal bands on front/back
    for band_y in [split_y - base_h + 1.1, split_y - 0.7] {
        add_box_part(
            &mut mesh,
            Vec3::new(w + 0.5, 1.0, 0.5),
            Vec3::new(0.0, band_y, d * 0.5 + 0.1),
            Vec3::ZERO,
            1,
        );
        add_box_part(
            &mut mesh,
            Vec3::new(w + 0.5, 1.0, 0.5),
            Vec3::new(0.0, band_y, -d * 0.5 - 0.1),
            Vec3::ZERO,
            1,
        );
    }
    // Front lock escutcheon plate
    add_box_part(
        &mut mesh,
        Vec3::new(3.4, 3.2, 0.7),
        Vec3::new(0.0, split_y - 1.8, d * 0.5 + 0.25),
        Vec3::ZERO,
        1,
    );
    // Rear hinge barrels
    for hx in [-7.0_f32, 7.0_f32] {
        add_cyl_part(
            &mut mesh,
            (0.55, 0.55),
            2.6,
            10,
            Vec3::new(hx, split_y, -d * 0.5 - 0.2),
            Vec3::new(0.0, 0.0, FRAC_PI_2),
            1,
        );
    }

    // 3. Articulated hinged lid (built closed, then rotated around back hinge edge)
    let mut lid = MeshData::empty();
    let lid_w = w + 0.4;
    let lid_d = d + 0.4;
    // Lower lid rim
    add_box_part(
        &mut lid,
        Vec3::new(lid_w, 2.2, lid_d),
        Vec3::new(0.0, split_y + 1.1, 0.0),
        Vec3::ZERO,
        0,
    );
    // Stepped arched upper crown
    add_box_part(
        &mut lid,
        Vec3::new(lid_w - 0.6, 1.8, lid_d - 2.2),
        Vec3::new(0.0, split_y + 3.0, 0.0),
        Vec3::ZERO,
        0,
    );
    add_box_part(
        &mut lid,
        Vec3::new(lid_w - 1.2, 1.2, lid_d - 4.8),
        Vec3::new(0.0, split_y + 4.4, 0.0),
        Vec3::ZERO,
        0,
    );
    // Metal straps over lid crown (left, center-left, center-right, right)
    for sx in [-w * 0.48, -w * 0.22, w * 0.22, w * 0.48] {
        add_box_part(
            &mut lid,
            Vec3::new(1.1, 2.4, lid_d + 0.35),
            Vec3::new(sx, split_y + 1.15, 0.0),
            Vec3::ZERO,
            1,
        );
        add_box_part(
            &mut lid,
            Vec3::new(1.1, 2.0, lid_d - 1.8),
            Vec3::new(sx, split_y + 3.05, 0.0),
            Vec3::ZERO,
            1,
        );
        add_box_part(
            &mut lid,
            Vec3::new(1.1, 1.35, lid_d - 4.4),
            Vec3::new(sx, split_y + 4.45, 0.0),
            Vec3::ZERO,
            1,
        );
    }
    // Front lid latch clasp
    add_box_part(
        &mut lid,
        Vec3::new(2.2, 2.0, 0.8),
        Vec3::new(0.0, split_y + 0.7, lid_d * 0.5 + 0.2),
        Vec3::ZERO,
        1,
    );

    if lid_rotation_rad.is_finite() && lid_rotation_rad.abs() > 1e-7 {
        // Negative X rotation lifts the +Z front edge UP around the -Z back hinge.
        let hinge_xf = Mat4::from_translation(hinge)
            * Mat4::from_rotation_x(-lid_rotation_rad)
            * Mat4::from_translation(-hinge);
        lid.apply_matrix(hinge_xf);
    }

    mesh.merge(&lid);
    mesh
}

/// Builds a CM3 preset 3D model mesh.
///
/// - `preset_type == 1`: Stylized multi-part rifle prop silhouette.
/// - `preset_type == 2`: Mystery Box chest with articulated lid rotated by `lid_rotation_rad`
///   around its back top hinge edge.
/// - Any other value: 10×10×10 fallback box.
#[must_use]
pub fn build_preset_model(preset_type: u32, lid_rotation_rad: f32) -> MeshData {
    match preset_type {
        1 => build_rifle_preset(),
        2 => build_mystery_box_preset(lid_rotation_rad),
        _ => build_box(10.0, 10.0, 10.0),
    }
}
