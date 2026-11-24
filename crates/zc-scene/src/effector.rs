use zc_geom::{Aabb3, Mat4, Vec2, Vec3};

use crate::mesh::{MeshData, Vertex3D};

/// Primary deformation axis for Twist and Warp effectors.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DeformAxis {
    X,
    #[default]
    Y,
    Z,
}

/// Effector `Repeater`: procedurally instances `mesh` `count` times with cumulative
/// translation `offset`, intrinsic XYZ Euler `rotation` (radians), and multiplicative `scale`.
#[must_use]
pub fn apply_repeater(
    mesh: &MeshData,
    count: u32,
    offset: Vec3,
    rotation: Vec3,
    scale: Vec3,
) -> MeshData {
    if count == 0 || mesh.vertices.is_empty() {
        return MeshData::empty();
    }

    let mut out = MeshData::empty();
    out.vertices.reserve(mesh.vertices.len() * count as usize);
    out.indices.reserve(mesh.indices.len() * count as usize);

    for i in 0..count {
        let mut instance = mesh.clone();
        if i > 0 {
            let fi = i as f32;
            let t = offset * fi;
            let r = rotation * fi;
            let s = Vec3::new(
                scale.x.powi(i as i32),
                scale.y.powi(i as i32),
                scale.z.powi(i as i32),
            );
            let rot_mat = Mat4::from_rotation_x(r.x)
                * Mat4::from_rotation_y(r.y)
                * Mat4::from_rotation_z(r.z);
            let mat = Mat4::from_translation(t) * rot_mat * Mat4::from_scale(s);
            instance.apply_matrix(mat);
        }
        out.merge(&instance);
    }

    out
}

/// Effector `Twist`: twists vertices linearly by `angle` radians around `axis` over `bounds`.
#[must_use]
pub fn apply_twist(mesh: &MeshData, angle: f32, axis: DeformAxis, bounds: Aabb3) -> MeshData {
    let mut out = mesh.clone();
    if out.vertices.is_empty() || angle.abs() <= 1e-8 {
        return out;
    }

    let (min_a, max_a) = match axis {
        DeformAxis::X => (bounds.min.x, bounds.max.x),
        DeformAxis::Y => (bounds.min.y, bounds.max.y),
        DeformAxis::Z => (bounds.min.z, bounds.max.z),
    };
    let span = max_a - min_a;
    if span.abs() <= 1e-6 {
        return out;
    }
    let center_a = 0.5 * (min_a + max_a);

    for v in &mut out.vertices {
        let p = Vec3::from_array(v.position);
        let coord = match axis {
            DeformAxis::X => p.x,
            DeformAxis::Y => p.y,
            DeformAxis::Z => p.z,
        };
        let t = (coord - center_a) / span;
        let theta = angle * t;
        let (sin_t, cos_t) = theta.sin_cos();

        let twisted = match axis {
            DeformAxis::X => Vec3::new(p.x, p.y * cos_t - p.z * sin_t, p.y * sin_t + p.z * cos_t),
            DeformAxis::Y => Vec3::new(p.x * cos_t + p.z * sin_t, p.y, -p.x * sin_t + p.z * cos_t),
            DeformAxis::Z => Vec3::new(p.x * cos_t - p.y * sin_t, p.x * sin_t + p.y * cos_t, p.z),
        };
        v.position = twisted.to_array();
    }

    out.compute_smooth_normals();
    out.recompute_bounds();
    out
}

/// Effector `Warp`: applies parabolic `bulge` and circular-arc `bend` (radians) along `axis`.
#[must_use]
pub fn apply_warp(mesh: &MeshData, bend: f32, bulge: f32, axis: DeformAxis) -> MeshData {
    let mut out = mesh.clone();
    if out.vertices.is_empty() || (bend.abs() <= 1e-8 && bulge.abs() <= 1e-8) {
        return out;
    }

    let bounds = out.bounds;
    let (min_a, max_a) = match axis {
        DeformAxis::X => (bounds.min.x, bounds.max.x),
        DeformAxis::Y => (bounds.min.y, bounds.max.y),
        DeformAxis::Z => (bounds.min.z, bounds.max.z),
    };
    let span = max_a - min_a;
    if span.abs() <= 1e-6 {
        return out;
    }
    let center_a = 0.5 * (min_a + max_a);

    for v in &mut out.vertices {
        let p = Vec3::from_array(v.position);
        let (mut along, mut perp1, mut perp2) = match axis {
            DeformAxis::X => (p.x, p.y, p.z),
            DeformAxis::Y => (p.y, p.x, p.z),
            DeformAxis::Z => (p.z, p.x, p.y),
        };

        let s = along - center_a;
        let t = s / span; // [-0.5, +0.5] across mesh bounds

        if bulge.abs() > 1e-8 {
            let bulge_factor = (1.0 + bulge * (1.0 - 4.0 * t * t)).max(0.01);
            perp1 *= bulge_factor;
            perp2 *= bulge_factor;
        }

        if bend.abs() > 1e-6 {
            let radius = span / bend;
            let theta = s / radius;
            let (sin_t, cos_t) = theta.sin_cos();
            let radial = radius - perp1;
            along = center_a + radial * sin_t;
            perp1 = radius - radial * cos_t;
        }

        let warped = match axis {
            DeformAxis::X => Vec3::new(along, perp1, perp2),
            DeformAxis::Y => Vec3::new(perp1, along, perp2),
            DeformAxis::Z => Vec3::new(perp1, perp2, along),
        };
        v.position = warped.to_array();
    }

    out.compute_smooth_normals();
    out.recompute_bounds();
    out
}

// ---------------------------------------------------------------------------
// Deterministic Voronoi Fracture (Spec 04 §5.3.3)
// ---------------------------------------------------------------------------

const MAX_VORONOI_OUTPUT_TRIS: usize = 1_600_000;
const MAX_CAP_CONTOUR_VERTS: usize = 4_096;

#[derive(Copy, Clone, Debug)]
struct ClipVert {
    pos: Vec3,
    normal: Vec3,
    uv: [f32; 2],
    material_slot: u32,
}

impl ClipVert {
    fn from_vertex(v: Vertex3D) -> Self {
        Self {
            pos: Vec3::from_array(v.position),
            normal: Vec3::from_array(v.normal),
            uv: v.uv,
            material_slot: v.material_slot,
        }
    }

    fn lerp(a: Self, b: Self, t: f32) -> Self {
        let n = a.normal.lerp(b.normal, t);
        let unit = if n.length_squared() > 1e-12 {
            n.normalize()
        } else {
            a.normal
        };
        Self {
            pos: a.pos.lerp(b.pos, t),
            normal: unit,
            uv: [
                a.uv[0] + (b.uv[0] - a.uv[0]) * t,
                a.uv[1] + (b.uv[1] - a.uv[1]) * t,
            ],
            material_slot: a.material_slot,
        }
    }

    fn to_vertex(self, offset: Vec3) -> Vertex3D {
        Vertex3D::new(
            (self.pos + offset).to_array(),
            self.normal.to_array(),
            self.uv,
            self.material_slot,
        )
    }
}

struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u32) -> Self {
        Self(u64::from(seed).wrapping_add(0x9E37_79B9_7F4A_7C15))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1_u32 << 24) as f32)
    }
}

/// Clips a convex polygon against half-space `(p - origin).dot(inward_normal) >= 0`.
/// If the plane cuts the polygon, pushes the boundary intersection segment `(start, end)`.
fn clip_polygon_halfspace(
    poly: &[ClipVert],
    plane_origin: Vec3,
    inward_normal: Vec3,
    cut_segments: &mut Vec<(Vec3, Vec3)>,
) -> Vec<ClipVert> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }

    let eps = 1e-5_f32;
    let dists: Vec<f32> = poly
        .iter()
        .map(|v| (v.pos - plane_origin).dot(inward_normal))
        .collect();

    let mut out = Vec::with_capacity(n + 1);
    let mut cross_pts = [Vec3::ZERO; 2];
    let mut cross_count = 0_usize;

    for i in 0..n {
        let j = (i + 1) % n;
        let vi = poly[i];
        let vj = poly[j];
        let di = dists[i];
        let dj = dists[j];
        let in_i = di >= -eps;
        let in_j = dj >= -eps;

        if in_i {
            out.push(vi);
        }
        if in_i != in_j {
            let denom = di - dj;
            let t = if denom.abs() > 1e-12 {
                (di / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let hit = ClipVert::lerp(vi, vj, t);
            out.push(hit);
            if cross_count < 2 {
                cross_pts[cross_count] = hit.pos;
                cross_count += 1;
            }
        }
    }

    if cross_count == 2 && cross_pts[0].distance_squared(cross_pts[1]) > 1e-10 {
        cut_segments.push((cross_pts[0], cross_pts[1]));
    }

    if out.len() >= 3 { out } else { Vec::new() }
}

/// Builds planar ear-clipped inner cap triangles on a Voronoi bisector plane (`material_slot = 1`).
fn build_planar_cap(
    segments: &[(Vec3, Vec3)],
    plane_origin: Vec3,
    inward_normal: Vec3,
    cell_offset: Vec3,
    out_verts: &mut Vec<Vertex3D>,
    out_indices: &mut Vec<u32>,
) {
    if segments.len() < 3 {
        return;
    }

    // Outward normal of the cell's cut face points opposite the inward half-space normal.
    let cap_normal = -inward_normal;
    let ref_axis = if cap_normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent_u = cap_normal.cross(ref_axis).normalize_or_zero();
    let tangent_v = cap_normal.cross(tangent_u).normalize_or_zero();

    // Project segment endpoints to 2D plane coordinates and chain into closed loops.
    let mut remaining: Vec<(Vec2, Vec2)> = segments
        .iter()
        .take(MAX_CAP_CONTOUR_VERTS)
        .map(|&(a, b)| {
            let da = a - plane_origin;
            let db = b - plane_origin;
            (
                Vec2::new(da.dot(tangent_u), da.dot(tangent_v)),
                Vec2::new(db.dot(tangent_u), db.dot(tangent_v)),
            )
        })
        .collect();

    let snap_eps2 = 1e-4_f32;
    let mut loops: Vec<Vec<Vec2>> = Vec::new();

    while let Some((start, next)) = remaining.pop() {
        let mut chain = vec![start, next];
        loop {
            let tail = *chain.last().unwrap_or(&start);
            if chain.len() >= 3 && tail.distance_squared(start) <= snap_eps2 {
                chain.pop();
                break;
            }
            let mut found = None;
            for (idx, &(a, b)) in remaining.iter().enumerate() {
                if a.distance_squared(tail) <= snap_eps2 {
                    found = Some((idx, b));
                    break;
                }
                if b.distance_squared(tail) <= snap_eps2 {
                    found = Some((idx, a));
                    break;
                }
            }
            if let Some((idx, pt)) = found {
                remaining.swap_remove(idx);
                if pt.distance_squared(tail) > 1e-10 {
                    chain.push(pt);
                }
                if chain.len() >= MAX_CAP_CONTOUR_VERTS {
                    break;
                }
            } else {
                break;
            }
        }
        if chain.len() >= 3 {
            loops.push(chain);
        }
    }

    for mut poly in loops {
        let area = polygon_area_2d(&poly);
        if area.abs() <= 1e-7 {
            continue;
        }
        if area < 0.0 {
            poly.reverse();
        }
        let tris = ear_clip_2d(&poly);
        if tris.is_empty() {
            continue;
        }

        let mut min_uv = poly[0];
        let mut max_uv = poly[0];
        for &p in &poly[1..] {
            min_uv = min_uv.min(p);
            max_uv = max_uv.max(p);
        }
        let span = (max_uv - min_uv).max(Vec2::splat(1e-5));

        let base = out_verts.len() as u32;
        for &p2 in &poly {
            let p3 = plane_origin + tangent_u * p2.x + tangent_v * p2.y + cell_offset;
            let uv = [(p2.x - min_uv.x) / span.x, (p2.y - min_uv.y) / span.y];
            out_verts.push(Vertex3D::new(p3.to_array(), cap_normal.to_array(), uv, 1));
        }
        for [i0, i1, i2] in tris {
            out_indices.extend_from_slice(&[base + i0 as u32, base + i1 as u32, base + i2 as u32]);
        }
    }
}

fn polygon_area_2d(poly: &[Vec2]) -> f32 {
    let n = poly.len();
    let mut sum = 0.0_f32;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        sum += a.x * b.y - b.x * a.y;
    }
    0.5 * sum
}

fn cross_2d(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x)
}

fn point_in_tri_2d(p: Vec2, a: Vec2, b: Vec2, c: Vec2) -> bool {
    let c1 = cross_2d(a, b, p);
    let c2 = cross_2d(b, c, p);
    let c3 = cross_2d(c, a, p);
    c1 >= -1e-6 && c2 >= -1e-6 && c3 >= -1e-6
}

fn ear_clip_2d(poly: &[Vec2]) -> Vec<[usize; 3]> {
    let n = poly.len();
    if n < 3 {
        return Vec::new();
    }
    let mut active: Vec<usize> = (0..n).collect();
    let mut tris = Vec::with_capacity(n - 2);
    let mut guard = n * n;

    while active.len() > 3 && guard > 0 {
        guard -= 1;
        let m = active.len();
        let mut clipped = false;

        for k in 0..m {
            let prev = active[(k + m - 1) % m];
            let curr = active[k];
            let next = active[(k + 1) % m];
            let a = poly[prev];
            let b = poly[curr];
            let c = poly[next];

            if cross_2d(a, b, c) <= 1e-8 {
                continue;
            }

            let mut contains_other = false;
            for &idx in &active {
                if idx == prev || idx == curr || idx == next {
                    continue;
                }
                if point_in_tri_2d(poly[idx], a, b, c) {
                    contains_other = true;
                    break;
                }
            }
            if !contains_other {
                tris.push([prev, curr, next]);
                active.remove(k);
                clipped = true;
                break;
            }
        }

        if !clipped {
            // Safe fallback on degenerate/open contours
            let root = active[0];
            for pair in active[1..].windows(2) {
                tris.push([root, pair[0], pair[1]]);
            }
            return tris;
        }
    }

    if active.len() == 3 {
        tris.push([active[0], active[1], active[2]]);
    }
    tris
}

/// Deterministic Effector `Voronoi Fracture`: clips `mesh` into up to `cell_count` convex
/// Voronoi fragments seeded by `seed`, caps interior cuts (`material_slot = 1`), and offsets
/// each fragment radially outward by `explode`.
#[must_use]
pub fn apply_voronoi_fracture(
    mesh: &MeshData,
    cell_count: u32,
    seed: u32,
    explode: f32,
) -> MeshData {
    let cells = cell_count.clamp(1, 1000) as usize;
    if mesh.indices.len() < 3 || mesh.vertices.is_empty() {
        return MeshData::empty();
    }
    if cells == 1 && explode.abs() <= 1e-8 {
        return mesh.clone();
    }

    let b_min = mesh.bounds.min;
    let b_max = mesh.bounds.max;
    let b_size = b_max - b_min;
    let mesh_center = (b_min + b_max) * 0.5;

    let mut rng = SplitMix64::new(seed);
    let mut sites = Vec::with_capacity(cells);
    for _ in 0..cells {
        let sx = b_min.x + (0.05 + 0.90 * rng.next_f32()) * b_size.x;
        let sy = b_min.y + (0.05 + 0.90 * rng.next_f32()) * b_size.y;
        let sz = b_min.z + (0.05 + 0.90 * rng.next_f32()) * b_size.z;
        sites.push(Vec3::new(sx, sy, sz));
    }

    type PlaneCut = (Vec3, Vec3, Vec<(Vec3, Vec3)>);

    // Collect source triangles once
    let mut source_tris: Vec<Vec<ClipVert>> = Vec::with_capacity(mesh.indices.len() / 3);
    for tri in mesh.indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 < mesh.vertices.len() && i1 < mesh.vertices.len() && i2 < mesh.vertices.len() {
            source_tris.push(vec![
                ClipVert::from_vertex(mesh.vertices[i0]),
                ClipVert::from_vertex(mesh.vertices[i1]),
                ClipVert::from_vertex(mesh.vertices[i2]),
            ]);
        }
    }

    let mut out_verts: Vec<Vertex3D> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::new();

    for (ci, &site_c) in sites.iter().enumerate() {
        let mut cell_polys = source_tris.clone();
        let mut plane_cuts: Vec<PlaneCut> = Vec::new();

        for (cj, &site_j) in sites.iter().enumerate() {
            if ci == cj || cell_polys.is_empty() {
                continue;
            }
            let diff = site_c - site_j;
            if diff.length_squared() <= 1e-10 {
                continue;
            }
            let inward_n = diff.normalize();
            let mid = (site_c + site_j) * 0.5;

            let mut cut_segs = Vec::new();
            let mut next_polys = Vec::with_capacity(cell_polys.len());
            for poly in &cell_polys {
                let clipped = clip_polygon_halfspace(poly, mid, inward_n, &mut cut_segs);
                if clipped.len() >= 3 {
                    next_polys.push(clipped);
                }
            }
            cell_polys = next_polys;

            if !cut_segs.is_empty() {
                // Clip earlier planes' cut segments against this new half-space too
                for (_, _, existing_segs) in &mut plane_cuts {
                    clip_segments_halfspace(existing_segs, mid, inward_n);
                }
                plane_cuts.push((mid, inward_n, cut_segs));
            }
        }

        if cell_polys.is_empty() {
            continue;
        }

        // Compute cell centroid from surviving vertices
        let mut centroid = Vec3::ZERO;
        let mut vert_total = 0_usize;
        for poly in &cell_polys {
            for v in poly {
                centroid += v.pos;
                vert_total += 1;
            }
        }
        let cell_center = if vert_total > 0 {
            centroid / vert_total as f32
        } else {
            site_c
        };

        let mut dir = (cell_center - mesh_center).normalize_or_zero();
        if dir.length_squared() <= 1e-10 {
            dir = (site_c - mesh_center).normalize_or_zero();
        }
        if dir.length_squared() <= 1e-10 {
            dir = Vec3::Y;
        }
        let cell_offset = dir * explode;

        // Emit outer shell triangles
        for poly in &cell_polys {
            if out_indices.len() / 3 >= MAX_VORONOI_OUTPUT_TRIS {
                break;
            }
            let base = out_verts.len() as u32;
            for v in poly {
                out_verts.push(v.to_vertex(cell_offset));
            }
            for k in 1..(poly.len() - 1) {
                out_indices.push(base);
                out_indices.push(base + k as u32);
                out_indices.push(base + k as u32 + 1);
            }
        }

        // Emit planar inner cut caps (material_slot = 1)
        for (origin, inward_n, segs) in &plane_cuts {
            if out_indices.len() / 3 >= MAX_VORONOI_OUTPUT_TRIS {
                break;
            }
            build_planar_cap(
                segs,
                *origin,
                *inward_n,
                cell_offset,
                &mut out_verts,
                &mut out_indices,
            );
        }
    }

    MeshData::new(out_verts, out_indices)
}

fn clip_segments_halfspace(
    segments: &mut Vec<(Vec3, Vec3)>,
    plane_origin: Vec3,
    inward_normal: Vec3,
) {
    let eps = 1e-5_f32;
    let mut kept = Vec::with_capacity(segments.len());
    for &(a, b) in segments.iter() {
        let da = (a - plane_origin).dot(inward_normal);
        let db = (b - plane_origin).dot(inward_normal);
        let in_a = da >= -eps;
        let in_b = db >= -eps;
        match (in_a, in_b) {
            (true, true) => kept.push((a, b)),
            (true, false) => {
                let t = (da / (da - db)).clamp(0.0, 1.0);
                let hit = a.lerp(b, t);
                if a.distance_squared(hit) > 1e-10 {
                    kept.push((a, hit));
                }
            }
            (false, true) => {
                let t = (da / (da - db)).clamp(0.0, 1.0);
                let hit = a.lerp(b, t);
                if hit.distance_squared(b) > 1e-10 {
                    kept.push((hit, b));
                }
            }
            (false, false) => {}
        }
    }
    *segments = kept;
}
