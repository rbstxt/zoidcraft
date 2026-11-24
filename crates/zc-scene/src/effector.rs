use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

use lyon_tessellation::math::point;
use lyon_tessellation::path::Path;
use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, VertexBuffers,
};
use zc_geom::{Aabb3, EulerOrder, Mat4, Transform3D, Vec2, Vec3};

use crate::mesh::{MeshData, Vertex3D, subdivide_mesh};

const TINY: f64 = 1e-9;
const SERIES_LIMIT: f64 = 1e-4;
const TWIST_PLANES: [(usize, usize); 3] = [(1, 2), (2, 0), (0, 1)];

/// Maximum repeater instance count matching Zoidium `scene-values.js:MAX_REPEATS`.
pub const MAX_REPEATS: u32 = 128;

/// Default Delay Spring damping (`effector-core.js:SPRING_DAMPING`).
pub const SPRING_DAMPING: f32 = 3.0;
/// Default Delay Spring cycles (`effector-core.js:SPRING_CYCLES`).
pub const SPRING_CYCLES: f32 = 1.0;
/// Minimum net Spring weight ratio (`effector-core.js:SPRING_MIN_TOTAL`).
pub const SPRING_MIN_TOTAL: f64 = 0.02;

/// Primary deformation axis for Twist and Warp effectors.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DeformAxis {
    X,
    #[default]
    Y,
    Z,
}

/// Bend destination axis for Zoidium `Warp` (`0 = bend along X into +Y`, `1 = bend along X into +Z`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum WarpAxis {
    #[default]
    IntoY = 0,
    IntoZ = 1,
}

impl WarpAxis {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::IntoZ,
            _ => Self::IntoY,
        }
    }
}

// ---------------------------------------------------------------------------
// Shared Zoidium Effector & Repeater PRNG (`zoidium/shared/random.js`)
// ---------------------------------------------------------------------------

/// Exact Zoidium `effectorHash01(seed, index, channel)` / `sceneSeededRandom` (`random.js:30-37, 72-79`).
///
/// Deterministic integer xorshift-mul hash returning a value in `[0, 1)`.
#[must_use]
pub fn effector_hash01(seed: i32, index: i32, channel: i32) -> f64 {
    let mut value: u32 = (seed as u32) ^ (index.wrapping_add(1) as u32).wrapping_mul(0x045d_9f3b);
    value ^= (channel.wrapping_add(1) as u32).wrapping_mul(0x27d4_eb2d);
    value = (value ^ (value >> 16)).wrapping_mul(0x045d_9f3b);
    value = (value ^ (value >> 16)).wrapping_mul(0x045d_9f3b);
    value ^= value >> 16;
    f64::from(value) / 4_294_967_296.0
}

/// Exact Zoidium `effectorHashUnitVector(seed, index, channel)` (`random.js:81-88`).
#[must_use]
pub fn effector_hash_unit_vector(seed: i32, index: i32, channel: i32) -> [f64; 3] {
    let x = 2.0 * effector_hash01(seed, index, channel) - 1.0;
    let y = 2.0 * effector_hash01(seed, index, channel + 1) - 1.0;
    let z = 2.0 * effector_hash01(seed, index, channel + 2) - 1.0;
    let len = (x * x + y * y + z * z).sqrt();
    if !len.is_finite() || len <= 1e-9 {
        [0.0, 0.0, 1.0]
    } else {
        [x / len, y / len, z / len]
    }
}

/// Exact Zoidium `rotateXYZ(point, degrees, inverse)` (`effector-core.js:111-120`).
#[must_use]
pub fn rotate_xyz_deg(point: [f64; 3], degrees: [f64; 3], inverse: bool) -> [f64; 3] {
    let mut out = point;
    let axes: [usize; 3] = if inverse { [2, 1, 0] } else { [0, 1, 2] };
    let sign = if inverse { -1.0 } else { 1.0 };
    for axis in axes {
        let (a, b) = TWIST_PLANES[axis];
        let deg = if degrees[axis].is_finite() {
            degrees[axis]
        } else {
            0.0
        };
        let angle = deg * (PI / 180.0) * sign;
        let c = angle.cos();
        let s = angle.sin();
        let x = out[a];
        let y = out[b];
        out[a] = c * x - s * y;
        out[b] = s * x + c * y;
    }
    out
}

fn clamp_unit(v: f64) -> f64 {
    if !v.is_finite() {
        0.0
    } else {
        v.clamp(0.0, 1.0)
    }
}

fn finite_or(v: f64, fallback: f64) -> f64 {
    if v.is_finite() { v } else { fallback }
}

// ---------------------------------------------------------------------------
// 6-Type Effector Field Evaluator (`effector-core.js:86-212`, `scene-fields.js`)
// ---------------------------------------------------------------------------

/// Zoidium 6-type spatial field (`scene-fields.js:227`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum EffectorFieldType {
    #[default]
    Infinite = 0,
    Linear = 1,
    Box = 2,
    Sphere = 3,
    Cylinder = 4,
    Noise = 5,
}

impl EffectorFieldType {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Linear,
            2 => Self::Box,
            3 => Self::Sphere,
            4 => Self::Cylinder,
            5 => Self::Noise,
            _ => Self::Infinite,
        }
    }
}

/// Zoidium field falloff easing curve (`scene-fields.js:232`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum FieldFalloffCurve {
    #[default]
    Linear = 0,
    Smooth = 1,
    EaseIn = 2,
    EaseOut = 3,
}

impl FieldFalloffCurve {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Smooth,
            2 => Self::EaseIn,
            3 => Self::EaseOut,
            _ => Self::Linear,
        }
    }
}

/// Zoidium Effector+ spatial field parameters (`scene-fields.js:205-238`).
#[derive(Clone, Debug, PartialEq)]
pub struct EffectorField {
    pub field_type: EffectorFieldType,
    pub position: Vec3,
    /// Rotation in degrees (`XYZ`).
    pub rotation_deg: Vec3,
    /// Scale in percent of object extent (default `[100, 100, 100]`, or raw world units in `legacy` mode).
    pub scale: Vec3,
    /// Falloff percentage (`0..100`, default `100`).
    pub falloff: f32,
    pub curve: FieldFalloffCurve,
    pub invert: bool,
    /// Linear field sweep percentage (`0..100`, default `100`).
    pub sweep: f32,
    /// Noise field scale percentage (`default 100`).
    pub noise_scale: f32,
    /// Noise field evolution offset (`default 0`).
    pub evolution: f32,
    pub seed: i32,
    pub legacy: bool,
}

impl Default for EffectorField {
    fn default() -> Self {
        Self {
            field_type: EffectorFieldType::Infinite,
            position: Vec3::ZERO,
            rotation_deg: Vec3::ZERO,
            scale: Vec3::splat(100.0),
            falloff: 100.0,
            curve: FieldFalloffCurve::Linear,
            invert: false,
            sweep: 100.0,
            noise_scale: 100.0,
            evolution: 0.0,
            seed: 1,
            legacy: false,
        }
    }
}

/// Precomputed field bounds and world-space scale for evaluating vertex weights.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedEffectorField {
    pub field: EffectorField,
    pub world_scale: [f64; 3],
    pub low: f64,
    pub high: f64,
}

fn value_noise_3d(x: f64, y: f64, z: f64, seed: i32) -> f64 {
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let iz = z.floor() as i32;
    let smooth = |t: f64| t * t * (3.0 - 2.0 * t);
    let tx = smooth(x - x.floor());
    let ty = smooth(y - y.floor());
    let tz = smooth(z - z.floor());
    let mut value = 0.0;
    for a in 0..2_i32 {
        for b in 0..2_i32 {
            for c in 0..2_i32 {
                let index = ix.wrapping_add(a).wrapping_mul(73_856_093)
                    ^ iy.wrapping_add(b).wrapping_mul(19_349_663)
                    ^ iz.wrapping_add(c).wrapping_mul(83_492_791);
                let wa = if a != 0 { tx } else { 1.0 - tx };
                let wb = if b != 0 { ty } else { 1.0 - ty };
                let wc = if c != 0 { tz } else { 1.0 - tz };
                value += effector_hash01(seed, index, 91) * wa * wb * wc;
            }
        }
    }
    value
}

/// Prepares an [`EffectorField`] from raw vertex positions (matching `effector-core.js:prepareField`).
#[must_use]
pub fn prepare_field_from_positions(
    field: &EffectorField,
    positions: &[[f32; 3]],
    matrix: Option<Mat4>,
) -> PreparedEffectorField {
    if field.field_type == EffectorFieldType::Infinite {
        return PreparedEffectorField {
            field: field.clone(),
            world_scale: [
                f64::from(field.scale.x).abs().max(TINY),
                f64::from(field.scale.y).abs().max(TINY),
                f64::from(field.scale.z).abs().max(TINY),
            ],
            low: -0.5,
            high: 0.5,
        };
    }

    let rot_deg = [
        f64::from(field.rotation_deg.x),
        f64::from(field.rotation_deg.y),
        f64::from(field.rotation_deg.z),
    ];
    let direction = rotate_xyz_deg([1.0, 0.0, 0.0], rot_deg, false);

    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    let mut low = f64::INFINITY;
    let mut high = f64::NEG_INFINITY;

    for &raw_p in positions {
        let p = match matrix {
            Some(m) => m.transform_point3(Vec3::from_array(raw_p)).to_array(),
            None => raw_p,
        };
        let (x, y, z) = (f64::from(p[0]), f64::from(p[1]), f64::from(p[2]));
        min[0] = min[0].min(x);
        max[0] = max[0].max(x);
        min[1] = min[1].min(y);
        max[1] = max[1].max(y);
        min[2] = min[2].min(z);
        max[2] = max[2].max(z);
        let projected = x * direction[0] + y * direction[1] + z * direction[2];
        low = low.min(projected);
        high = high.max(projected);
    }

    if !min[0].is_finite() {
        min = [0.0; 3];
        max = [0.0; 3];
        low = 0.0;
        high = 0.0;
    }

    let raw_scale = [
        f64::from(field.scale.x),
        f64::from(field.scale.y),
        f64::from(field.scale.z),
    ];
    let world_scale = [
        (raw_scale[0].abs() * (max[0] - min[0]).max(TINY) / 100.0).max(TINY),
        (raw_scale[1].abs() * (max[1] - min[1]).max(TINY) / 100.0).max(TINY),
        (raw_scale[2].abs() * (max[2] - min[2]).max(TINY) / 100.0).max(TINY),
    ];

    PreparedEffectorField {
        field: field.clone(),
        world_scale,
        low,
        high,
    }
}

/// Prepares an [`EffectorField`] from shared AABB bounds (matching `effector-core.js:fieldForBounds`).
#[must_use]
pub fn prepare_field_for_bounds(field: &EffectorField, bounds: Aabb3) -> PreparedEffectorField {
    let mut corners = [[0.0_f32; 3]; 8];
    let mut idx = 0;
    for x in 0..2 {
        for y in 0..2 {
            for z in 0..2 {
                corners[idx] = [
                    if x != 0 { bounds.max.x } else { bounds.min.x },
                    if y != 0 { bounds.max.y } else { bounds.min.y },
                    if z != 0 { bounds.max.z } else { bounds.min.z },
                ];
                idx += 1;
            }
        }
    }
    prepare_field_from_positions(field, &corners, None)
}

fn legacy_field_weight(x: f64, y: f64, z: f64, field: &EffectorField) -> f64 {
    let sx = f64::from(field.scale.x).abs().max(TINY);
    let sy = f64::from(field.scale.y).abs().max(TINY);
    let sz = f64::from(field.scale.z).abs().max(TINY);
    let px = f64::from(field.position.x);
    let py = f64::from(field.position.y);
    let pz = f64::from(field.position.z);
    let weight = match field.field_type {
        EffectorFieldType::Infinite => 1.0,
        EffectorFieldType::Linear => 1.0 - (x - px).max(0.0) / sx,
        EffectorFieldType::Box => {
            let dx = (x - px).abs() / (0.5 * sx);
            let dy = (y - py).abs() / (0.5 * sy);
            let dz = (z - pz).abs() / (0.5 * sz);
            let d = dx.max(dy).max(dz);
            if d <= 1.0 { 1.0 } else { 1.0 - (d - 1.0) }
        }
        EffectorFieldType::Sphere => {
            let dx = x - px;
            let dy = y - py;
            let dz = z - pz;
            1.0 - (dx * dx + dy * dy + dz * dz).sqrt() / sx
        }
        EffectorFieldType::Cylinder | EffectorFieldType::Noise => 1.0,
    };
    clamp_unit(weight)
}

/// Evaluates field weight in `[0, 1]` at `(x, y, z)` (`effector-core.js:164-200`).
#[must_use]
pub fn evaluate_field(point: Vec3, prepared: &PreparedEffectorField) -> f64 {
    let field = &prepared.field;
    if field.field_type == EffectorFieldType::Infinite {
        return if field.invert { 0.0 } else { 1.0 };
    }

    let (x, y, z) = (f64::from(point.x), f64::from(point.y), f64::from(point.z));
    let rot_deg = [
        f64::from(field.rotation_deg.x),
        f64::from(field.rotation_deg.y),
        f64::from(field.rotation_deg.z),
    ];
    let is_legacy = field.legacy
        && rot_deg.iter().all(|&v| v == 0.0)
        && (f64::from(field.sweep) - 100.0).abs() <= 1e-6
        && (f64::from(field.falloff) - 100.0).abs() <= 1e-6
        && field.curve == FieldFalloffCurve::Linear
        && !field.invert;
    if is_legacy
        && matches!(
            field.field_type,
            EffectorFieldType::Linear | EffectorFieldType::Box | EffectorFieldType::Sphere
        )
    {
        return legacy_field_weight(x, y, z, field);
    }

    let q = rotate_xyz_deg(
        [
            x - f64::from(field.position.x),
            y - f64::from(field.position.y),
            z - f64::from(field.position.z),
        ],
        rot_deg,
        true,
    );
    let sx = prepared.world_scale[0].abs().max(TINY);
    let sy = prepared.world_scale[1].abs().max(TINY);
    let sz = prepared.world_scale[2].abs().max(TINY);
    let falloff = clamp_unit(finite_or(f64::from(field.falloff), 100.0) / 100.0);

    let mut weight = match field.field_type {
        EffectorFieldType::Infinite => 1.0,
        EffectorFieldType::Linear => {
            let sweep = clamp_unit(finite_or(f64::from(field.sweep), 100.0) / 100.0);
            let low = finite_or(prepared.low, -sx * 0.5);
            let high = finite_or(prepared.high, sx * 0.5);
            let width = ((high - low) * falloff).max(TINY);
            let plane = low - width + (high - low + width) * sweep;
            if sweep == 0.0 {
                0.0
            } else if sweep == 1.0 {
                1.0
            } else if falloff > 0.0 {
                clamp_unit((plane + width - q[0]) / width)
            } else if q[0] <= plane {
                1.0
            } else {
                0.0
            }
        }
        EffectorFieldType::Box | EffectorFieldType::Sphere | EffectorFieldType::Cylinder => {
            let dx = q[0] / (sx * 0.5);
            let dy = q[1] / (sy * 0.5);
            let dz = q[2] / (sz * 0.5);
            let dist = match field.field_type {
                EffectorFieldType::Box => dx.abs().max(dy.abs()).max(dz.abs()),
                EffectorFieldType::Sphere => (dx * dx + dy * dy + dz * dz).sqrt(),
                EffectorFieldType::Cylinder => dx.hypot(dz).max(dy.abs()),
                _ => 0.0,
            };
            if falloff > 0.0 {
                clamp_unit((1.0 - dist) / falloff)
            } else if dist <= 1.0 {
                1.0
            } else {
                0.0
            }
        }
        EffectorFieldType::Noise => {
            let ns = (finite_or(f64::from(field.noise_scale), 100.0).abs() / 100.0).max(TINY);
            let evolution = finite_or(f64::from(field.evolution), 0.0);
            let seed = if field.seed == 0 { 1 } else { field.seed };
            let n = value_noise_3d(
                q[0] / sx / ns + evolution,
                q[1] / sy / ns + evolution * 0.73,
                q[2] / sz / ns + evolution * 0.37,
                seed,
            );
            if falloff > 0.0 {
                clamp_unit((n - 0.5) / falloff + 0.5)
            } else if n >= 0.5 {
                1.0
            } else {
                0.0
            }
        }
    };

    weight = match field.curve {
        FieldFalloffCurve::Linear => weight,
        FieldFalloffCurve::Smooth => weight * weight * (3.0 - 2.0 * weight),
        FieldFalloffCurve::EaseIn => weight * weight,
        FieldFalloffCurve::EaseOut => 1.0 - (1.0 - weight) * (1.0 - weight),
    };
    if field.invert { 1.0 - weight } else { weight }
}

// ---------------------------------------------------------------------------
// Deformer-Frame Relative Transform (`effector-core.js:324-360`)
// ---------------------------------------------------------------------------

/// Maps mesh-local vertices into a parent Effector container's local frame and back.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct DeformerFrame {
    pub forward: Mat4,
    pub inverse: Mat4,
    pub identity: bool,
}

impl DeformerFrame {
    pub const IDENTITY: Self = Self {
        forward: Mat4::IDENTITY,
        inverse: Mat4::IDENTITY,
        identity: true,
    };
}

fn is_approx_identity(m: Mat4) -> bool {
    let a = m.to_cols_array();
    let id = Mat4::IDENTITY.to_cols_array();
    a.iter().zip(id.iter()).all(|(x, y)| (x - y).abs() <= 1e-6)
}

/// Computes `forward = deformer_world^-1 * mesh_world` and its inverse (`effector-core.js:327-336`).
/// Returns `None` if `deformer_world` or `forward` is singular.
#[must_use]
pub fn compute_deformer_frame(deformer_world: Mat4, mesh_world: Mat4) -> Option<DeformerFrame> {
    if deformer_world.determinant().abs() < 1e-9 {
        return None;
    }
    let forward = deformer_world.inverse() * mesh_world;
    if is_approx_identity(forward) {
        return Some(DeformerFrame::IDENTITY);
    }
    if forward.determinant().abs() < 1e-9 {
        return None;
    }
    Some(DeformerFrame {
        forward,
        inverse: forward.inverse(),
        identity: false,
    })
}

/// Computes shared bounding box of child meshes expressed in a parent deformer's local frame.
#[must_use]
pub fn compute_shared_deformer_bounds(
    meshes_and_worlds: &[(&MeshData, Mat4)],
    deformer_world: Mat4,
) -> Option<Aabb3> {
    if deformer_world.determinant().abs() < 1e-9 {
        return None;
    }
    let inv_deformer = deformer_world.inverse();
    let mut min = Vec3::splat(f32::INFINITY);
    let mut max = Vec3::splat(f32::NEG_INFINITY);
    let mut any = false;
    for &(mesh, mesh_world) in meshes_and_worlds {
        let forward = inv_deformer * mesh_world;
        for v in &mesh.vertices {
            let p = forward.transform_point3(Vec3::from_array(v.position));
            if p.is_finite() {
                min = min.min(p);
                max = max.max(p);
                any = true;
            }
        }
    }
    any.then_some(Aabb3 { min, max })
}

// ---------------------------------------------------------------------------
// Connected Pieces & Welded Normals (`effector-core.js:511-530`, `effector-mesh.js`)
// ---------------------------------------------------------------------------

/// Welds coincident vertex positions (`extent * 1e-7`) and triangle connectivity via union-find
/// to assign each disconnected mesh island / character a deterministic piece ID (`0..piece_count`).
#[must_use]
pub fn connected_pieces(mesh: &MeshData) -> (Vec<u32>, usize) {
    let count = mesh.vertices.len();
    if count == 0 {
        return (Vec::new(), 0);
    }
    let mut parent: Vec<usize> = (0..count).collect();
    fn find(parent: &mut [usize], mut i: usize) -> usize {
        while parent[i] != i {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        i
    }

    let size = mesh.bounds.max - mesh.bounds.min;
    let extent = size.x.max(size.y).max(size.z).max(1.0);
    let eps = extent * 1e-7;

    let mut weld: HashMap<(i64, i64, i64), usize> = HashMap::with_capacity(count);
    for (i, v) in mesh.vertices.iter().enumerate() {
        let key = (
            (v.position[0] / eps).round() as i64,
            (v.position[1] / eps).round() as i64,
            (v.position[2] / eps).round() as i64,
        );
        if let Some(&prev) = weld.get(&key) {
            let ra = find(&mut parent, i);
            let rb = find(&mut parent, prev);
            parent[ra] = rb;
        } else {
            weld.insert(key, i);
        }
    }

    for tri in mesh.indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 < count && i1 < count && i2 < count {
            let r0 = find(&mut parent, i0);
            let r1 = find(&mut parent, i1);
            let r2 = find(&mut parent, i2);
            parent[r1] = r0;
            parent[r2] = r0;
        }
    }

    let mut ids: HashMap<usize, u32> = HashMap::new();
    let mut piece_ids = vec![0_u32; count];
    for (i, slot) in piece_ids.iter_mut().enumerate() {
        let r = find(&mut parent, i);
        let next_id = ids.len() as u32;
        *slot = *ids.entry(r).or_insert(next_id);
    }
    (piece_ids, ids.len())
}

/// Recomputes smooth normals welded by rest position, source normal, and material slot
/// (matching `effector-mesh.js:computeSmoothVertexNormals`), preserving hard edges.
fn recompute_deformed_normals(mesh: &mut MeshData, rest_vertices: &[Vertex3D]) {
    let count = mesh.vertices.len();
    if count == 0 || rest_vertices.len() != count {
        mesh.compute_smooth_normals();
        return;
    }

    type NormalWeldKey = (i64, i64, i64, i32, i32, i32, u32);
    let mut lookup: HashMap<NormalWeldKey, usize> = HashMap::with_capacity(count);
    let mut slots = vec![0_usize; count];
    for (i, v) in rest_vertices.iter().enumerate() {
        let key = (
            (f64::from(v.position[0]) * 100_000.0).round() as i64,
            (f64::from(v.position[1]) * 100_000.0).round() as i64,
            (f64::from(v.position[2]) * 100_000.0).round() as i64,
            (v.normal[0] * 10_000.0).round() as i32,
            (v.normal[1] * 10_000.0).round() as i32,
            (v.normal[2] * 10_000.0).round() as i32,
            v.material_slot,
        );
        let next_slot = lookup.len();
        slots[i] = *lookup.entry(key).or_insert(next_slot);
    }

    let mut sums = vec![[0.0_f64; 3]; lookup.len()];
    for tri in mesh.indices.as_chunks::<3>().0 {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        if i0 >= count || i1 >= count || i2 >= count {
            continue;
        }
        let p0 = mesh.vertices[i0].position;
        let p1 = mesh.vertices[i1].position;
        let p2 = mesh.vertices[i2].position;
        let ab = [
            f64::from(p1[0] - p0[0]),
            f64::from(p1[1] - p0[1]),
            f64::from(p1[2] - p0[2]),
        ];
        let ac = [
            f64::from(p2[0] - p0[0]),
            f64::from(p2[1] - p0[1]),
            f64::from(p2[2] - p0[2]),
        ];
        let fx = ab[1] * ac[2] - ab[2] * ac[1];
        let fy = ab[2] * ac[0] - ab[0] * ac[2];
        let fz = ab[0] * ac[1] - ab[1] * ac[0];
        for idx in [i0, i1, i2] {
            let s = slots[idx];
            sums[s][0] += fx;
            sums[s][1] += fy;
            sums[s][2] += fz;
        }
    }

    for sum in &mut sums {
        let len = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
        if len > 1e-12 {
            sum[0] /= len;
            sum[1] /= len;
            sum[2] /= len;
        } else {
            *sum = [0.0, 0.0, 1.0];
        }
    }

    for (v, &slot) in mesh.vertices.iter_mut().zip(&slots) {
        let n = sums[slot];
        v.normal = [n[0] as f32, n[1] as f32, n[2] as f32];
    }
}

// ---------------------------------------------------------------------------
// Repeater: Step, Linear, Random, Echo (`scene-values.js`, `scene-repeaters.js`)
// ---------------------------------------------------------------------------

/// Zoidium Repeater mode (`zoidium:repeater/{step,linear,random,echo}`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum RepeaterMode {
    #[default]
    Step,
    Linear,
    Random,
    Echo,
}

/// Per-instance TRS output for a Repeater copy.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct RepeaterTransform {
    pub position: Vec3,
    pub rotation_rad: Vec3,
    pub scale: Vec3,
}

/// Full Zoidium Repeater parameters (`scene-fields.js:51-184`, `scene-values.js:189-229`).
#[derive(Clone, Debug, PartialEq)]
pub struct RepeaterParams {
    pub mode: RepeaterMode,
    /// Instance count clamped to `1..=128` (`MAX_REPEATS`).
    pub count: u32,
    pub euler_order: EulerOrder,
    // Step controls
    pub position_step: Vec3,
    pub rotation_step_rad: Vec3,
    pub scale_step: Vec3,
    // Linear controls
    pub position_end: Vec3,
    pub rotation_end_rad: Vec3,
    pub scale_end: Vec3,
    // Random controls
    pub position_min: Vec3,
    pub position_max: Vec3,
    pub rotation_min_rad: Vec3,
    pub rotation_max_rad: Vec3,
    pub scale_min: Vec3,
    pub scale_max: Vec3,
    pub seed: i32,
    // Echo controls
    pub delay_frames: f64,
}

impl Default for RepeaterParams {
    fn default() -> Self {
        Self {
            mode: RepeaterMode::Step,
            count: 5,
            euler_order: EulerOrder::XYZ,
            position_step: Vec3::new(10.0, 0.0, 0.0),
            rotation_step_rad: Vec3::ZERO,
            scale_step: Vec3::ONE,
            position_end: Vec3::new(40.0, 0.0, 0.0),
            rotation_end_rad: Vec3::new(0.0, 0.0, std::f32::consts::FRAC_PI_2),
            scale_end: Vec3::ONE,
            position_min: Vec3::splat(-10.0),
            position_max: Vec3::splat(10.0),
            rotation_min_rad: Vec3::ZERO,
            rotation_max_rad: Vec3::ZERO,
            scale_min: Vec3::ONE,
            scale_max: Vec3::ONE,
            seed: 1,
            delay_frames: 5.0,
        }
    }
}

/// Computes the exact frame list for an `Echo` repeater (`scene-values.js:78-92`).
#[must_use]
pub fn repeater_echo_frames(frame: f64, count: u32, delay: f64) -> Vec<f64> {
    let copies = count.clamp(1, MAX_REPEATS);
    let current = finite_or(frame, 0.0).max(0.0);
    let frame_delay = finite_or(delay, 0.0).max(0.0);
    (0..copies)
        .map(|index| (current - f64::from(index) * frame_delay).max(0.0))
        .collect()
}

/// Evaluates the exact Zoidium transform for repeater copy `index` (`scene-values.js:189-229`).
#[must_use]
pub fn repeater_instance_transform(params: &RepeaterParams, index: u32) -> RepeaterTransform {
    let count = params.count.clamp(1, MAX_REPEATS);
    match params.mode {
        RepeaterMode::Linear => {
            let amount = if count > 1 {
                index as f32 / (count - 1) as f32
            } else {
                0.0
            };
            RepeaterTransform {
                position: params.position_end * amount,
                rotation_rad: params.rotation_end_rad * amount,
                scale: Vec3::ONE.lerp(params.scale_end, amount),
            }
        }
        RepeaterMode::Random => {
            if index == 0 {
                return RepeaterTransform {
                    position: Vec3::ZERO,
                    rotation_rad: Vec3::ZERO,
                    scale: Vec3::ONE,
                };
            }
            let rand_vec = |a: Vec3, b: Vec3, ch: i32| -> Vec3 {
                let lo = a.min(b);
                let hi = a.max(b);
                let idx = index as i32;
                Vec3::new(
                    lo.x + (hi.x - lo.x) * effector_hash01(params.seed, idx, ch) as f32,
                    lo.y + (hi.y - lo.y) * effector_hash01(params.seed, idx, ch + 1) as f32,
                    lo.z + (hi.z - lo.z) * effector_hash01(params.seed, idx, ch + 2) as f32,
                )
            };
            RepeaterTransform {
                position: rand_vec(params.position_min, params.position_max, 0),
                rotation_rad: rand_vec(params.rotation_min_rad, params.rotation_max_rad, 3),
                scale: rand_vec(params.scale_min, params.scale_max, 6).max(Vec3::splat(0.001)),
            }
        }
        RepeaterMode::Step => {
            let fi = index as f32;
            let sx = params.scale_step.x.max(0.001);
            let sy = params.scale_step.y.max(0.001);
            let sz = params.scale_step.z.max(0.001);
            RepeaterTransform {
                position: params.position_step * fi,
                rotation_rad: params.rotation_step_rad * fi,
                scale: Vec3::new(
                    sx.powi(index as i32),
                    sy.powi(index as i32),
                    sz.powi(index as i32),
                ),
            }
        }
        RepeaterMode::Echo => RepeaterTransform {
            position: Vec3::ZERO,
            rotation_rad: Vec3::ZERO,
            scale: Vec3::ONE,
        },
    }
}

/// Applies a Zoidium Repeater (`Step`, `Linear`, `Random`) with `euler_order`, `count <= 128`,
/// and clamped `scale_step >= 0.001`.
#[must_use]
pub fn apply_repeater_with_params(mesh: &MeshData, params: &RepeaterParams) -> MeshData {
    if params.count == 0 || mesh.vertices.is_empty() {
        return MeshData::empty();
    }
    let count = params.count.clamp(1, MAX_REPEATS);
    let mut out = MeshData::empty();
    out.vertices.reserve(mesh.vertices.len() * count as usize);
    out.indices.reserve(mesh.indices.len() * count as usize);

    for i in 0..count {
        let mut instance = mesh.clone();
        let tr = repeater_instance_transform(params, i);
        let tf = Transform3D {
            position: tr.position,
            rotation: tr.rotation_rad,
            scale: tr.scale,
            euler_order: params.euler_order,
        };
        let mat = tf.to_matrix();
        if !is_approx_identity(mat) {
            instance.apply_matrix(mat);
        }
        out.merge(&instance);
    }

    out
}

/// Effector `Repeater` (Step mode, backwards-compatible signature): instances `mesh` `count` times
/// (capped at `MAX_REPEATS = 128`) with cumulative `offset`, intrinsic XYZ Euler `rotation` (radians),
/// and multiplicative `scale` (clamped `>= 0.001` per axis matching Zoidium).
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
    let params = RepeaterParams {
        mode: RepeaterMode::Step,
        count: count.clamp(1, MAX_REPEATS),
        euler_order: EulerOrder::XYZ,
        position_step: offset,
        rotation_step_rad: rotation,
        scale_step: scale,
        ..RepeaterParams::default()
    };
    apply_repeater_with_params(mesh, &params)
}

// ---------------------------------------------------------------------------
// Twist Effector (`effector-core.js:214-234`)
// ---------------------------------------------------------------------------

/// Full Zoidium `Twist` effector parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct TwistParams {
    /// Total twist angle across bounds in **radians**.
    pub angle: f32,
    pub axis: DeformAxis,
    /// Pivot offset along `axis` (in deformer local units).
    pub offset: f32,
    /// Subdivision triangles per source triangle (`1, 4, 16, 64, 256`).
    pub subdivision: u32,
    pub field: Option<EffectorField>,
}

impl Default for TwistParams {
    fn default() -> Self {
        Self {
            angle: 0.0,
            axis: DeformAxis::Y,
            offset: 0.0,
            subdivision: 1,
            field: None,
        }
    }
}

/// Evaluates Zoidium `Twist` with optional subdivision, deformer frame, shared bounds, offset, and 6-type Field.
#[must_use]
pub fn apply_twist_with_params(
    mesh: &MeshData,
    params: &TwistParams,
    shared_bounds: Option<Aabb3>,
    deformer_frame: Option<&DeformerFrame>,
) -> MeshData {
    if mesh.vertices.is_empty() || !params.angle.is_finite() || params.angle.abs() <= 1e-8 {
        return mesh.clone();
    }

    let mut out = if params.subdivision > 1 {
        subdivide_mesh(mesh, params.subdivision)
    } else {
        mesh.clone()
    };
    let rest_vertices = out.vertices.clone();

    let use_frame = deformer_frame.is_some_and(|f| !f.identity);
    if let Some(frame) = deformer_frame
        && use_frame
    {
        for v in &mut out.vertices {
            v.position = frame
                .forward
                .transform_point3(Vec3::from_array(v.position))
                .to_array();
        }
        out.recompute_bounds();
    }

    let bounds = shared_bounds.unwrap_or(out.bounds);
    let along = match params.axis {
        DeformAxis::X => 0_usize,
        DeformAxis::Y => 1_usize,
        DeformAxis::Z => 2_usize,
    };
    let min_a = f64::from(bounds.min.to_array()[along]);
    let max_a = f64::from(bounds.max.to_array()[along]);
    let size = max_a - min_a;
    if !size.is_finite() || size <= 1e-6 {
        return mesh.clone();
    }
    let pivot = 0.5 * (min_a + max_a) + finite_or(f64::from(params.offset), 0.0);

    let prepared_field = params.field.as_ref().map(|f| {
        if let Some(sb) = shared_bounds {
            prepare_field_for_bounds(f, sb)
        } else {
            let pos_list: Vec<[f32; 3]> = out.vertices.iter().map(|v| v.position).collect();
            prepare_field_from_positions(f, &pos_list, None)
        }
    });

    let (first, second) = TWIST_PLANES[along];
    let angle = f64::from(params.angle);

    for v in &mut out.vertices {
        let p = v.position;
        let w = prepared_field
            .as_ref()
            .map_or(1.0, |pf| evaluate_field(Vec3::from_array(p), pf));
        let theta = angle * ((f64::from(p[along]) - pivot) / size) * w;
        let (sin_t, cos_t) = theta.sin_cos();
        let fv = f64::from(p[first]);
        let sv = f64::from(p[second]);
        let mut next = p;
        next[first] = (fv * cos_t - sv * sin_t) as f32;
        next[second] = (fv * sin_t + sv * cos_t) as f32;
        v.position = next;
    }

    if let Some(frame) = deformer_frame
        && use_frame
    {
        for v in &mut out.vertices {
            v.position = frame
                .inverse
                .transform_point3(Vec3::from_array(v.position))
                .to_array();
        }
    }

    recompute_deformed_normals(&mut out, &rest_vertices);
    out.recompute_bounds();
    out
}

/// Effector `Twist` (backwards-compatible signature): twists vertices linearly by `angle` radians
/// around `axis` over `bounds`.
#[must_use]
pub fn apply_twist(mesh: &MeshData, angle: f32, axis: DeformAxis, bounds: Aabb3) -> MeshData {
    let params = TwistParams {
        angle,
        axis,
        offset: 0.0,
        subdivision: 1,
        field: None,
    };
    apply_twist_with_params(mesh, &params, Some(bounds), None)
}

// ---------------------------------------------------------------------------
// Warp Effector (`effector-core.js:238-274`)
// ---------------------------------------------------------------------------

fn sinc_series(theta: f64) -> f64 {
    if theta.abs() < SERIES_LIMIT {
        1.0 - (theta * theta) / 6.0
    } else {
        theta.sin() / theta
    }
}

fn one_minus_cos_over_series(theta: f64) -> f64 {
    if theta.abs() < SERIES_LIMIT {
        theta * 0.5
    } else {
        (1.0 - theta.cos()) / theta
    }
}

/// Full Zoidium `Warp` effector parameters (`effector-core.js:253-274`).
#[derive(Clone, Debug, PartialEq)]
pub struct WarpParams {
    /// Bend strength across X extent in **radians**.
    pub strength: f32,
    /// Bend destination axis (`IntoY` for `axis = 0`, `IntoZ` for `axis = 1`).
    pub axis: WarpAxis,
    /// Pivot offset along X.
    pub offset: f32,
    /// Subdivision triangles per source triangle (`1, 4, 16, 64, 256`).
    pub subdivision: u32,
    pub field: Option<EffectorField>,
}

impl Default for WarpParams {
    fn default() -> Self {
        Self {
            strength: 0.0,
            axis: WarpAxis::IntoY,
            offset: 0.0,
            subdivision: 1,
            field: None,
        }
    }
}

/// Exact Zoidium `warpPositions` (`effector-core.js:253-274`):
/// always bends along X (`alongAxis = 0`) into `+Y` (`WarpAxis::IntoY`) or `+Z` (`WarpAxis::IntoZ`)
/// using `x' = pivot + s * sinc(theta)` and `bend' = bend + s * (1 - cos(theta)) / theta`.
#[must_use]
pub fn apply_warp_with_params(
    mesh: &MeshData,
    params: &WarpParams,
    shared_bounds: Option<Aabb3>,
    deformer_frame: Option<&DeformerFrame>,
) -> MeshData {
    if mesh.vertices.is_empty() || !params.strength.is_finite() || params.strength.abs() <= 1e-8 {
        return mesh.clone();
    }

    let mut out = if params.subdivision > 1 {
        subdivide_mesh(mesh, params.subdivision)
    } else {
        mesh.clone()
    };
    let rest_vertices = out.vertices.clone();

    let use_frame = deformer_frame.is_some_and(|f| !f.identity);
    if let Some(frame) = deformer_frame
        && use_frame
    {
        for v in &mut out.vertices {
            v.position = frame
                .forward
                .transform_point3(Vec3::from_array(v.position))
                .to_array();
        }
        out.recompute_bounds();
    }

    let bounds = shared_bounds.unwrap_or(out.bounds);
    let along_axis = 0_usize;
    let bend_axis = match params.axis {
        WarpAxis::IntoY => 1_usize,
        WarpAxis::IntoZ => 2_usize,
    };

    let min_x = f64::from(bounds.min.x);
    let max_x = f64::from(bounds.max.x);
    let size = max_x - min_x;
    if !size.is_finite() || size <= 1e-6 {
        return mesh.clone();
    }
    let pivot = 0.5 * (min_x + max_x) + finite_or(f64::from(params.offset), 0.0);
    let curvature = f64::from(params.strength) / size;

    let prepared_field = params.field.as_ref().map(|f| {
        if let Some(sb) = shared_bounds {
            prepare_field_for_bounds(f, sb)
        } else {
            let pos_list: Vec<[f32; 3]> = out.vertices.iter().map(|v| v.position).collect();
            prepare_field_from_positions(f, &pos_list, None)
        }
    });

    for v in &mut out.vertices {
        let p = v.position;
        let weight = prepared_field
            .as_ref()
            .map_or(1.0, |pf| evaluate_field(Vec3::from_array(p), pf));
        let effective = curvature * weight;
        if effective.abs() <= 1e-12 {
            continue;
        }
        let along = f64::from(p[along_axis]) - pivot;
        let theta = effective * along;
        let mut next = p;
        next[along_axis] = (pivot + along * sinc_series(theta)) as f32;
        next[bend_axis] =
            (f64::from(p[bend_axis]) + along * one_minus_cos_over_series(theta)) as f32;
        v.position = next;
    }

    if let Some(frame) = deformer_frame
        && use_frame
    {
        for v in &mut out.vertices {
            v.position = frame
                .inverse
                .transform_point3(Vec3::from_array(v.position))
                .to_array();
        }
    }

    recompute_deformed_normals(&mut out, &rest_vertices);
    out.recompute_bounds();
    out
}

/// Effector `Warp` (backwards-compatible signature): uses Zoidium's exact `warpPositions` formula
/// (bends along X into `+Y` for `DeformAxis::X | DeformAxis::Y` and into `+Z` for `DeformAxis::Z`;
/// `_bulge` is retained solely for caller signature compatibility).
#[must_use]
pub fn apply_warp(mesh: &MeshData, bend: f32, _bulge: f32, axis: DeformAxis) -> MeshData {
    let params = WarpParams {
        strength: bend,
        axis: match axis {
            DeformAxis::Z => WarpAxis::IntoZ,
            DeformAxis::X | DeformAxis::Y => WarpAxis::IntoY,
        },
        offset: 0.0,
        subdivision: 1,
        field: None,
    };
    apply_warp_with_params(mesh, &params, None, None)
}

// ---------------------------------------------------------------------------
// Plain Effector (`effector-core.js:532-567`)
// ---------------------------------------------------------------------------

/// Zoidium `Plain` effector parameters (`scene-effectors.js:307-336`).
#[derive(Clone, Debug, PartialEq)]
pub struct PlainParams {
    pub position_offset: Vec3,
    /// Rotation offset in **degrees** (`XYZ`).
    pub rotation_offset_deg: Vec3,
    /// Additive per-axis scale offset (final target scale = `1 + scale_offset + uniform_scale`).
    pub scale_offset: Vec3,
    /// Additive uniform scale offset (default `0.0`).
    pub uniform_scale: f32,
    pub field: Option<EffectorField>,
}

impl Default for PlainParams {
    fn default() -> Self {
        Self {
            position_offset: Vec3::ZERO,
            rotation_offset_deg: Vec3::ZERO,
            scale_offset: Vec3::ZERO,
            uniform_scale: 0.0,
            field: None,
        }
    }
}

/// Exact Zoidium `plainPositions` (`effector-core.js:532-567`): per-connected-piece TRS weighted by field.
#[must_use]
pub fn apply_plain(
    mesh: &MeshData,
    params: &PlainParams,
    shared_bounds: Option<Aabb3>,
    deformer_frame: Option<&DeformerFrame>,
) -> MeshData {
    if mesh.vertices.is_empty() {
        return mesh.clone();
    }

    let mut out = mesh.clone();
    let rest_vertices = out.vertices.clone();
    let (piece_ids, piece_count) = connected_pieces(&out);
    let count = piece_count.max(1);

    let use_frame = deformer_frame.is_some_and(|f| !f.identity);
    if let Some(frame) = deformer_frame
        && use_frame
    {
        for v in &mut out.vertices {
            v.position = frame
                .forward
                .transform_point3(Vec3::from_array(v.position))
                .to_array();
        }
        out.recompute_bounds();
    }

    let mut sums = vec![[0.0_f64; 3]; count];
    let mut sizes = vec![0_u32; count];
    for (i, v) in out.vertices.iter().enumerate() {
        let id = piece_ids[i] as usize;
        sizes[id] += 1;
        sums[id][0] += f64::from(v.position[0]);
        sums[id][1] += f64::from(v.position[1]);
        sums[id][2] += f64::from(v.position[2]);
    }

    let prepared_field = params.field.as_ref().map(|f| {
        if let Some(sb) = shared_bounds {
            prepare_field_for_bounds(f, sb)
        } else {
            let pos_list: Vec<[f32; 3]> = out.vertices.iter().map(|v| v.position).collect();
            prepare_field_from_positions(f, &pos_list, None)
        }
    });

    let target_scale = [
        1.0 + f64::from(params.scale_offset.x) + f64::from(params.uniform_scale),
        1.0 + f64::from(params.scale_offset.y) + f64::from(params.uniform_scale),
        1.0 + f64::from(params.scale_offset.z) + f64::from(params.uniform_scale),
    ];
    let pos_off = [
        f64::from(params.position_offset.x),
        f64::from(params.position_offset.y),
        f64::from(params.position_offset.z),
    ];
    let rot_deg = [
        f64::from(params.rotation_offset_deg.x),
        f64::from(params.rotation_offset_deg.y),
        f64::from(params.rotation_offset_deg.z),
    ];

    let mut centers = Vec::with_capacity(count);
    let mut weights = Vec::with_capacity(count);
    let mut scales = Vec::with_capacity(count);
    let mut rots = Vec::with_capacity(count);

    for id in 0..count {
        let denom = f64::from(sizes[id].max(1));
        let c = [
            sums[id][0] / denom,
            sums[id][1] / denom,
            sums[id][2] / denom,
        ];
        let w = prepared_field.as_ref().map_or(1.0, |pf| {
            evaluate_field(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32), pf)
        });
        centers.push(c);
        weights.push(w);
        scales.push([
            1.0 + (target_scale[0] - 1.0) * w,
            1.0 + (target_scale[1] - 1.0) * w,
            1.0 + (target_scale[2] - 1.0) * w,
        ]);
        rots.push([rot_deg[0] * w, rot_deg[1] * w, rot_deg[2] * w]);
    }

    for (i, v) in out.vertices.iter_mut().enumerate() {
        let id = piece_ids[i] as usize;
        let c = centers[id];
        let s = scales[id];
        let w = weights[id];
        let q = [
            (f64::from(v.position[0]) - c[0]) * s[0],
            (f64::from(v.position[1]) - c[1]) * s[1],
            (f64::from(v.position[2]) - c[2]) * s[2],
        ];
        let r = rotate_xyz_deg(q, rots[id], false);
        v.position = [
            (c[0] + r[0] + pos_off[0] * w) as f32,
            (c[1] + r[1] + pos_off[1] * w) as f32,
            (c[2] + r[2] + pos_off[2] * w) as f32,
        ];
    }

    if let Some(frame) = deformer_frame
        && use_frame
    {
        for v in &mut out.vertices {
            v.position = frame
                .inverse
                .transform_point3(Vec3::from_array(v.position))
                .to_array();
        }
    }

    recompute_deformed_normals(&mut out, &rest_vertices);
    out.recompute_bounds();
    out
}

// ---------------------------------------------------------------------------
// Delay Effector (`effector-core.js:572-653`, `effector-evaluate.js:205-252`)
// ---------------------------------------------------------------------------

/// Zoidium `Delay` filter mode (`scene-effectors.js:344`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum DelayMode {
    /// Exponential blend (`exp(-4 * age)`).
    #[default]
    Blend = 0,
    /// Uniform moving average (`1.0`).
    Average = 1,
    /// Damped sinusoid (`exp(-damping * age) * sin(2 * pi * cycles * age)`).
    Spring = 2,
}

impl DelayMode {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Average,
            2 => Self::Spring,
            _ => Self::Blend,
        }
    }
}

/// Zoidium `Delay` effector parameters (`scene-effectors.js:338-368`).
#[derive(Clone, Debug, PartialEq)]
pub struct DelayParams {
    pub mode: DelayMode,
    /// Blend strength in `[0.0, 1.0]` (UI `0..100` divided by `100`, default `0.75`).
    pub strength: f32,
    /// History window in frames (`1..=120`, default `12`).
    pub window: u32,
    /// Spring damping (`0.25..=12.0`, default `3.0`).
    pub spring_damping: f32,
    /// Spring cycles (`0.25..=8.0`, default `1.0`).
    pub spring_cycles: f32,
}

impl Default for DelayParams {
    fn default() -> Self {
        Self {
            mode: DelayMode::Blend,
            strength: 0.75,
            window: 12,
            spring_damping: SPRING_DAMPING,
            spring_cycles: SPRING_CYCLES,
        }
    }
}

/// Evaluates the exact Zoidium Delay history kernel weight for normalized `age = (i + 1) / history_len`
/// (`effector-evaluate.js:217`).
#[must_use]
pub fn delay_weight(mode: DelayMode, age: f64, damping: f64, cycles: f64) -> f64 {
    match mode {
        DelayMode::Blend => (-4.0 * age).exp(),
        DelayMode::Average => 1.0,
        DelayMode::Spring => (-damping * age).exp() * (age * TAU * cycles).sin(),
    }
}

#[derive(Clone, Debug)]
struct FittedPieceTransform {
    center: [f64; 3],
    scale: [f64; 3],
    rotation: [f64; 4], // [x, y, z, w]
}

struct PieceTransformBasis {
    count: usize,
    piece_ids: Vec<u32>,
    centers: Vec<[f64; 3]>,
    sizes: Vec<u32>,
    inverses: Vec<Option<[f64; 9]>>,
}

fn invert_3x3(g: [f64; 9]) -> Option<[f64; 9]> {
    let largest = g.iter().fold(TINY, |acc, &v| acc.max(v.abs()));
    let a = g[0] / largest;
    let b = g[1] / largest;
    let c = g[2] / largest;
    let d = g[3] / largest;
    let e = g[4] / largest;
    let f = g[5] / largest;
    let h = g[6] / largest;
    let i = g[7] / largest;
    let j = g[8] / largest;

    let c00 = e * j - f * i;
    let c01 = -(d * j - f * h);
    let c02 = d * i - e * h;
    let c10 = -(b * j - c * i);
    let c11 = a * j - c * h;
    let c12 = -(a * i - b * h);
    let c20 = b * f - c * e;
    let c21 = -(a * f - c * d);
    let c22 = a * e - b * d;

    let det = a * c00 + b * c01 + c * c02;
    if !det.is_finite() || det.abs() < TINY {
        return None;
    }
    let inv_scale = 1.0 / (det * largest);
    Some([
        c00 * inv_scale,
        c10 * inv_scale,
        c20 * inv_scale,
        c01 * inv_scale,
        c11 * inv_scale,
        c21 * inv_scale,
        c02 * inv_scale,
        c12 * inv_scale,
        c22 * inv_scale,
    ])
}

fn quaternion_of_mat3(m: [f64; 9]) -> [f64; 4] {
    let trace = m[0] + m[4] + m[8];
    if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        return [
            (m[7] - m[5]) / s,
            (m[2] - m[6]) / s,
            (m[3] - m[1]) / s,
            s * 0.25,
        ];
    }
    let i = if m[0] > m[4] && m[0] > m[8] {
        0
    } else if m[4] > m[8] {
        1
    } else {
        2
    };
    let j = (i + 1) % 3;
    let k = (i + 2) % 3;
    let mut q = [0.0_f64; 4];
    let raw_s = (1.0 + m[i * 3 + i] - m[j * 3 + j] - m[k * 3 + k])
        .max(0.0)
        .sqrt()
        * 2.0;
    let s = if raw_s > 0.0 { raw_s } else { 1.0 };
    q[i] = s * 0.25;
    q[j] = (m[j * 3 + i] + m[i * 3 + j]) / s;
    q[k] = (m[k * 3 + i] + m[i * 3 + k]) / s;
    q[3] = (m[k * 3 + j] - m[j * 3 + k]) / s;
    q
}

fn build_piece_transform_basis(rest: &MeshData) -> PieceTransformBasis {
    let (piece_ids, piece_count) = connected_pieces(rest);
    let count = piece_count.max(1);
    let mut centers = vec![[0.0_f64; 3]; count];
    let mut sizes = vec![0_u32; count];
    for (v, vert) in rest.vertices.iter().enumerate() {
        let id = piece_ids[v] as usize;
        sizes[id] += 1;
        centers[id][0] += f64::from(vert.position[0]);
        centers[id][1] += f64::from(vert.position[1]);
        centers[id][2] += f64::from(vert.position[2]);
    }
    for id in 0..count {
        let denom = f64::from(sizes[id].max(1));
        centers[id][0] /= denom;
        centers[id][1] /= denom;
        centers[id][2] /= denom;
    }

    let mut covariance = vec![[0.0_f64; 9]; count];
    for (v, vert) in rest.vertices.iter().enumerate() {
        let id = piece_ids[v] as usize;
        let d = [
            f64::from(vert.position[0]) - centers[id][0],
            f64::from(vert.position[1]) - centers[id][1],
            f64::from(vert.position[2]) - centers[id][2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                covariance[id][a * 3 + b] += d[a] * d[b];
            }
        }
    }

    let inverses = covariance.into_iter().map(invert_3x3).collect();
    PieceTransformBasis {
        count,
        piece_ids,
        centers,
        sizes,
        inverses,
    }
}

fn fit_piece_transforms(
    rest: &MeshData,
    deformed: &MeshData,
    basis: &PieceTransformBasis,
) -> Vec<Option<FittedPieceTransform>> {
    let mut centers = vec![[0.0_f64; 3]; basis.count];
    for (v, vert) in deformed.vertices.iter().enumerate() {
        let id = basis.piece_ids[v] as usize;
        centers[id][0] += f64::from(vert.position[0]);
        centers[id][1] += f64::from(vert.position[1]);
        centers[id][2] += f64::from(vert.position[2]);
    }
    for (id, center) in centers.iter_mut().enumerate().take(basis.count) {
        let denom = f64::from(basis.sizes[id].max(1));
        center[0] /= denom;
        center[1] /= denom;
        center[2] /= denom;
    }

    let mut cross = vec![[0.0_f64; 9]; basis.count];
    for (v, (rv, dv)) in rest.vertices.iter().zip(&deformed.vertices).enumerate() {
        let id = basis.piece_ids[v] as usize;
        let dr = [
            f64::from(rv.position[0]) - basis.centers[id][0],
            f64::from(rv.position[1]) - basis.centers[id][1],
            f64::from(rv.position[2]) - basis.centers[id][2],
        ];
        let dd = [
            f64::from(dv.position[0]) - centers[id][0],
            f64::from(dv.position[1]) - centers[id][1],
            f64::from(dv.position[2]) - centers[id][2],
        ];
        for a in 0..3 {
            for b in 0..3 {
                cross[id][a * 3 + b] += dd[a] * dr[b];
            }
        }
    }

    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let norm = |a: [f64; 3]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();

    basis
        .inverses
        .iter()
        .enumerate()
        .map(|(id, inv_opt)| {
            let inv = (*inv_opt)?;
            let h = cross[id];
            let mut m = [0.0_f64; 9];
            for a in 0..3 {
                for b in 0..3 {
                    for k in 0..3 {
                        m[a * 3 + b] += h[a * 3 + k] * inv[k * 3 + b];
                    }
                }
            }
            let cols = [[m[0], m[3], m[6]], [m[1], m[4], m[7]], [m[2], m[5], m[8]]];
            let mut scale = [norm(cols[0]), norm(cols[1]), norm(cols[2])];
            if scale.iter().any(|&v| v < TINY) {
                return None;
            }
            // Reject sheared transforms so vertex filtering preserves them (`effector-core.js:632-633`).
            for a in 0..3 {
                for b in (a + 1)..3 {
                    if dot(cols[a], cols[b]).abs() > scale[a] * scale[b] * 1e-4 {
                        return None;
                    }
                }
            }
            let x = [
                cols[0][0] / scale[0],
                cols[0][1] / scale[0],
                cols[0][2] / scale[0],
            ];
            let proj = dot(cols[1], x);
            let mut y = [
                cols[1][0] - proj * x[0],
                cols[1][1] - proj * x[1],
                cols[1][2] - proj * x[2],
            ];
            let ylen = norm(y);
            if ylen < TINY {
                return None;
            }
            y[0] /= ylen;
            y[1] /= ylen;
            y[2] /= ylen;
            let z = [
                x[1] * y[2] - x[2] * y[1],
                x[2] * y[0] - x[0] * y[2],
                x[0] * y[1] - x[1] * y[0],
            ];
            if dot(z, cols[2]) < 0.0 {
                scale[2] = -scale[2];
            }
            let rot = quaternion_of_mat3([x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]]);
            Some(FittedPieceTransform {
                center: centers[id],
                scale,
                rotation: rot,
            })
        })
        .collect()
}

/// Applies Zoidium `Delay` (`effector-evaluate.js:205-252`) over `history_meshes` ordered from
/// newest history frame (`t - 1`) to oldest (`t - window`).
///
/// When `curved == false` (no upstream Twist/Warp), uses least-squares rigid piece transform
/// filtering (`effector-core.js:572-653`) so piece rotations stay rigid without shrinking.
#[must_use]
pub fn apply_delay(
    rest_mesh: &MeshData,
    current_mesh: &MeshData,
    history_meshes: &[&MeshData],
    params: &DelayParams,
    curved: bool,
) -> MeshData {
    let strength = clamp_unit(f64::from(params.strength));
    if history_meshes.is_empty()
        || strength <= 0.0
        || current_mesh.vertices.is_empty()
        || rest_mesh.vertices.len() != current_mesh.vertices.len()
    {
        return current_mesh.clone();
    }

    let window = (params.window.clamp(1, 120) as usize).min(history_meshes.len());
    let damping = f64::from(params.spring_damping.clamp(0.25, 12.0));
    let cycles = f64::from(params.spring_cycles.clamp(0.25, 8.0));

    let vcount = current_mesh.vertices.len();
    let mut sum = vec![[0.0_f64; 3]; vcount];
    let basis = (!curved).then(|| build_piece_transform_basis(rest_mesh));
    let current_transforms = basis
        .as_ref()
        .map(|b| fit_piece_transforms(rest_mesh, current_mesh, b));
    let mut filtered: Option<Vec<Option<FittedPieceTransform>>> =
        current_transforms.as_ref().map(|ct| {
            ct.iter()
                .map(|opt| {
                    opt.as_ref().map(|_| FittedPieceTransform {
                        center: [0.0; 3],
                        scale: [0.0; 3],
                        rotation: [0.0; 4],
                    })
                })
                .collect()
        });

    let mut total = 0.0_f64;
    let mut peak = 0.0_f64;
    for (i, sample) in history_meshes.iter().take(window).enumerate() {
        if sample.vertices.len() != vcount {
            continue;
        }
        let age = (i + 1) as f64 / window as f64;
        let weight = delay_weight(params.mode, age, damping, cycles);
        total += weight;
        peak = peak.max(weight.abs());

        if let (Some(b), Some(cur), Some(filt)) = (
            basis.as_ref(),
            current_transforms.as_ref(),
            filtered.as_mut(),
        ) {
            let sample_t = fit_piece_transforms(rest_mesh, sample, b);
            for (id, t_opt) in sample_t.into_iter().enumerate() {
                match (t_opt, &cur[id], &mut filt[id]) {
                    (Some(t), Some(c), Some(acc)) => {
                        let qdot = t.rotation[0] * c.rotation[0]
                            + t.rotation[1] * c.rotation[1]
                            + t.rotation[2] * c.rotation[2]
                            + t.rotation[3] * c.rotation[3];
                        let sign = if qdot < 0.0 { -1.0 } else { 1.0 };
                        for a in 0..3 {
                            acc.center[a] += t.center[a] * weight;
                            acc.scale[a] += t.scale[a] * weight;
                        }
                        for a in 0..4 {
                            acc.rotation[a] += t.rotation[a] * weight * sign;
                        }
                    }
                    _ => {
                        filt[id] = None;
                    }
                }
            }
        }

        for (v, vert) in sample.vertices.iter().enumerate() {
            sum[v][0] += f64::from(vert.position[0]) * weight;
            sum[v][1] += f64::from(vert.position[1]) * weight;
            sum[v][2] += f64::from(vert.position[2]) * weight;
        }
    }

    let tuned = params.mode == DelayMode::Spring
        && ((params.spring_damping - SPRING_DAMPING).abs() > 1e-6
            || (params.spring_cycles - SPRING_CYCLES).abs() > 1e-6);
    let min_total = if tuned {
        (peak * SPRING_MIN_TOTAL).max(1e-6)
    } else {
        1e-6
    };
    if total.abs() <= min_total {
        return current_mesh.clone();
    }

    let mut out = current_mesh.clone();
    for (vert, avg) in out.vertices.iter_mut().zip(&sum) {
        for (coord, &acc) in vert.position.iter_mut().zip(avg.iter()) {
            let cur = f64::from(*coord);
            *coord = (cur + (acc / total - cur) * strength) as f32;
        }
    }

    if let (Some(b), Some(cur), Some(mut filt)) = (basis, current_transforms, filtered) {
        for id in 0..b.count {
            if let (Some(c), Some(t)) = (&cur[id], &mut filt[id]) {
                for a in 0..3 {
                    t.center[a] = c.center[a] * (1.0 - strength) + (t.center[a] / total) * strength;
                    t.scale[a] = c.scale[a] * (1.0 - strength) + (t.scale[a] / total) * strength;
                }
                for a in 0..4 {
                    t.rotation[a] =
                        c.rotation[a] * (1.0 - strength) + (t.rotation[a] / total) * strength;
                }
                let qlen = (t.rotation[0] * t.rotation[0]
                    + t.rotation[1] * t.rotation[1]
                    + t.rotation[2] * t.rotation[2]
                    + t.rotation[3] * t.rotation[3])
                    .sqrt();
                if qlen < 1e-8 {
                    t.rotation = c.rotation;
                } else {
                    for r in &mut t.rotation {
                        *r /= qlen;
                    }
                }
            }
        }

        for (v, vert) in out.vertices.iter_mut().enumerate() {
            let id = b.piece_ids[v] as usize;
            let Some(t) = &filt[id] else { continue };
            let [qx, qy, qz, qw] = t.rotation;
            let p = [
                (f64::from(rest_mesh.vertices[v].position[0]) - b.centers[id][0]) * t.scale[0],
                (f64::from(rest_mesh.vertices[v].position[1]) - b.centers[id][1]) * t.scale[1],
                (f64::from(rest_mesh.vertices[v].position[2]) - b.centers[id][2]) * t.scale[2],
            ];
            let uv = [
                qy * p[2] - qz * p[1],
                qz * p[0] - qx * p[2],
                qx * p[1] - qy * p[0],
            ];
            let uuv = [
                qy * uv[2] - qz * uv[1],
                qz * uv[0] - qx * uv[2],
                qx * uv[1] - qy * uv[0],
            ];
            vert.position = [
                (t.center[0] + p[0] + 2.0 * (qw * uv[0] + uuv[0])) as f32,
                (t.center[1] + p[1] + 2.0 * (qw * uv[1] + uuv[1])) as f32,
                (t.center[2] + p[2] + 2.0 * (qw * uv[2] + uuv[2])) as f32,
            ];
        }
    }

    recompute_deformed_normals(&mut out, &rest_mesh.vertices);
    out.recompute_bounds();
    out
}

// ---------------------------------------------------------------------------
// Deterministic Voronoi Fracture (`effector-fracture.js:521-921`)
// ---------------------------------------------------------------------------

const MAX_VORONOI_OUTPUT_TRIS: usize = 1_600_000;
const MAX_CAP_CONTOUR_VERTS: usize = 4_096;

/// Zoidium Voronoi Fracture site distribution (`scene-effectors.js:384`, `effector-fracture.js:588-613`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum FractureDistribution {
    #[default]
    Uniform = 0,
    Surface = 1,
    Center = 2,
    Edges = 3,
}

impl FractureDistribution {
    #[must_use]
    pub const fn from_u32(value: u32) -> Self {
        match value {
            1 => Self::Surface,
            2 => Self::Center,
            3 => Self::Edges,
            _ => Self::Uniform,
        }
    }
}

/// Full Zoidium Voronoi Fracture parameters (`scene-effectors.js:381-461`, `effector-fracture.js:557-921`).
#[derive(Clone, Debug, PartialEq)]
pub struct VoronoiFractureParams {
    pub cells: u32,
    pub seed: i32,
    /// When `true` (default), caps interior Voronoi cuts (`material_slot = 1`).
    pub closed: bool,
    pub distribution: FractureDistribution,
    /// Anisotropic Voronoi metric scale in percent (default `[100, 100, 100]`).
    pub cell_scale: Vec3,
    /// Radial outward explosion distance.
    pub distance: f32,
    /// Seeded random scatter magnitude.
    pub scatter: f32,
    /// Fragment shrink percentage (`0..100`, default `0`).
    pub offset: f32,
    /// Seeded Rodrigues spin angle in **radians**.
    pub spin_rad: f32,
    /// Explicit directional translation offset.
    pub direction: Vec3,
    /// Explicit XYZ Euler fragment rotation in **degrees**.
    pub rotation_deg: Vec3,
    /// Fragment scale percentage (`0..1000`, default `100`).
    pub fragment_scale: f32,
    /// Quadratic gravity drop along `-Y`.
    pub gravity: f32,
    /// Per-fragment motion variation randomness (`0..100`, default `0`).
    pub randomness: f32,
    pub field: Option<EffectorField>,
}

impl Default for VoronoiFractureParams {
    fn default() -> Self {
        Self {
            cells: 24,
            seed: 1,
            closed: true,
            distribution: FractureDistribution::Uniform,
            cell_scale: Vec3::splat(100.0),
            distance: 0.0,
            scatter: 0.0,
            offset: 0.0,
            spin_rad: 0.0,
            direction: Vec3::ZERO,
            rotation_deg: Vec3::ZERO,
            fragment_scale: 100.0,
            gravity: 0.0,
            randomness: 0.0,
            field: None,
        }
    }
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> [f32; 3] {
    if s <= 0.0 {
        let v = l as f32;
        return [v, v, v];
    }
    let hue2rgb = |p: f64, q: f64, mut t: f64| -> f64 {
        if t < 0.0 {
            t += 1.0;
        }
        if t > 1.0 {
            t -= 1.0;
        }
        if t < 1.0 / 6.0 {
            return p + (q - p) * 6.0 * t;
        }
        if t < 0.5 {
            return q;
        }
        if t < 2.0 / 3.0 {
            return p + (q - p) * (2.0 / 3.0 - t) * 6.0;
        }
        p
    };
    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - l * s
    };
    let p = 2.0 * l - q;
    [
        hue2rgb(p, q, h + 1.0 / 3.0) as f32,
        hue2rgb(p, q, h) as f32,
        hue2rgb(p, q, h - 1.0 / 3.0) as f32,
    ]
}

/// Exact Zoidium `pieceColor(seed, piece)` HSL fragment color (`effector-fracture.js:521-526`).
#[must_use]
pub fn voronoi_piece_color(seed: i32, piece: i32) -> [f32; 3] {
    let hue = effector_hash01(seed, piece, 40);
    let saturation = 0.45 + 0.2 * effector_hash01(seed, piece, 41);
    let lightness = 0.55 + 0.15 * effector_hash01(seed, piece, 42);
    hsl_to_rgb(hue, saturation, lightness)
}

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

    let mut out: Vec<ClipVert> = Vec::with_capacity(n + 1);
    let mut cross_pts = [Vec3::ZERO; 2];
    let mut cross_count = 0_usize;

    let push_dedup = |dst: &mut Vec<ClipVert>, v: ClipVert| {
        if dst
            .last()
            .is_none_or(|prev| prev.pos.distance_squared(v.pos) > 1e-10)
        {
            dst.push(v);
        }
    };

    for i in 0..n {
        let j = (i + 1) % n;
        let vi = poly[i];
        let vj = poly[j];
        let di = dists[i];
        let dj = dists[j];
        let in_i = di >= -eps;
        let in_j = dj >= -eps;

        if in_i {
            push_dedup(&mut out, vi);
        }
        if in_i != in_j {
            let denom = di - dj;
            let t = if denom.abs() > 1e-12 {
                (di / denom).clamp(0.0, 1.0)
            } else {
                0.5
            };
            let hit = ClipVert::lerp(vi, vj, t);
            push_dedup(&mut out, hit);
            if cross_count < 2 {
                cross_pts[cross_count] = hit.pos;
                cross_count += 1;
            }
        }
    }

    while out.len() >= 2 && out[0].pos.distance_squared(out[out.len() - 1].pos) <= 1e-10 {
        out.pop();
    }

    if cross_count == 2 && cross_pts[0].distance_squared(cross_pts[1]) > 1e-10 {
        cut_segments.push((cross_pts[0], cross_pts[1]));
    }

    if out.len() >= 3 { out } else { Vec::new() }
}

fn extend_cap_chain(
    chain: &mut Vec<Vec2>,
    remaining: &mut Vec<(Vec2, Vec2)>,
    snap_eps2: f32,
) -> bool {
    loop {
        let head = chain[0];
        let tail = *chain.last().unwrap_or(&head);
        if chain.len() >= 3 && tail.distance_squared(head) <= snap_eps2 {
            chain.pop();
            return true;
        }
        let mut best: Option<(usize, Vec2, f32)> = None;
        for (idx, &(a, b)) in remaining.iter().enumerate() {
            let da = a.distance_squared(tail);
            if da <= snap_eps2 && best.is_none_or(|(_, _, bd)| da < bd) {
                best = Some((idx, b, da));
            }
            let db = b.distance_squared(tail);
            if db <= snap_eps2 && best.is_none_or(|(_, _, bd)| db < bd) {
                best = Some((idx, a, db));
            }
        }
        if let Some((idx, pt, _)) = best {
            remaining.swap_remove(idx);
            if pt.distance_squared(tail) > 1e-10 {
                chain.push(pt);
            }
            if chain.len() >= MAX_CAP_CONTOUR_VERTS {
                return false;
            }
        } else {
            return false;
        }
    }
}

/// Builds planar inner cap triangles on a Voronoi bisector plane (`material_slot = 1`)
/// using `lyon_tessellation::FillRule::EvenOdd` so hollow/non-convex cuts preserve interior holes,
/// and appends them directly to `cell_polys` so subsequent bisector planes clip them cleanly.
fn build_planar_cap_polys(
    segments: &[(Vec3, Vec3)],
    plane_origin: Vec3,
    inward_normal: Vec3,
    cell_polys: &mut Vec<Vec<ClipVert>>,
) {
    if segments.len() < 3 {
        return;
    }

    let cap_normal = -inward_normal;
    let ref_axis = if cap_normal.x.abs() < 0.9 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let tangent_u = cap_normal.cross(ref_axis).normalize_or_zero();
    let tangent_v = cap_normal.cross(tangent_u).normalize_or_zero();

    let mut remaining: Vec<(Vec2, Vec2)> = segments
        .iter()
        .take(MAX_CAP_CONTOUR_VERTS)
        .filter_map(|&(a, b)| {
            let da = a - plane_origin;
            let db = b - plane_origin;
            let p0 = Vec2::new(da.dot(tangent_u), da.dot(tangent_v));
            let p1 = Vec2::new(db.dot(tangent_u), db.dot(tangent_v));
            (p0.distance_squared(p1) > 1e-10).then_some((p0, p1))
        })
        .collect();

    let snap_eps2 = 1e-4_f32;
    let mut loops: Vec<Vec<Vec2>> = Vec::new();

    while let Some((start, next)) = remaining.pop() {
        let mut chain = vec![start, next];
        if !extend_cap_chain(&mut chain, &mut remaining, snap_eps2) {
            chain.reverse();
            extend_cap_chain(&mut chain, &mut remaining, snap_eps2);
        }
        if chain.len() >= 3 && polygon_area_2d(&chain).abs() > 1e-7 {
            loops.push(chain);
        }
    }

    if loops.is_empty() {
        return;
    }

    let mut path_builder = Path::builder();
    for contour in &loops {
        if let Some((&first, rest)) = contour.split_first() {
            path_builder.begin(point(first.x, first.y));
            for &p in rest {
                path_builder.line_to(point(p.x, p.y));
            }
            path_builder.end(true);
        }
    }
    let path = path_builder.build();

    let mut cap_geo: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    let mut tessellator = FillTessellator::new();
    let options = FillOptions::default()
        .with_fill_rule(FillRule::EvenOdd)
        .with_tolerance(0.01);
    if tessellator
        .tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut cap_geo, |v: FillVertex| {
                let p = v.position();
                [p.x, p.y]
            }),
        )
        .is_err()
        || cap_geo.vertices.is_empty()
        || cap_geo.indices.len() < 3
    {
        return;
    }

    let mut min_uv = Vec2::splat(f32::INFINITY);
    let mut max_uv = Vec2::splat(f32::NEG_INFINITY);
    for &[x, y] in &cap_geo.vertices {
        let p = Vec2::new(x, y);
        min_uv = min_uv.min(p);
        max_uv = max_uv.max(p);
    }
    let span = (max_uv - min_uv).max(Vec2::splat(1e-5));

    let make_vert = |p2: [f32; 2]| -> ClipVert {
        let pos = plane_origin + tangent_u * p2[0] + tangent_v * p2[1];
        let uv = [(p2[0] - min_uv.x) / span.x, (p2[1] - min_uv.y) / span.y];
        ClipVert {
            pos,
            normal: cap_normal,
            uv,
            material_slot: 1,
        }
    };

    for tri in cap_geo.indices.as_chunks::<3>().0 {
        let p0 = cap_geo.vertices[tri[0] as usize];
        let p1 = cap_geo.vertices[tri[1] as usize];
        let p2 = cap_geo.vertices[tri[2] as usize];
        let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if area2.abs() <= 1e-10 {
            continue;
        }
        let ordered = if area2 >= 0.0 {
            [make_vert(p0), make_vert(p1), make_vert(p2)]
        } else {
            [make_vert(p0), make_vert(p2), make_vert(p1)]
        };
        cell_polys.push(ordered.to_vec());
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

/// Generates Voronoi sites matching Zoidium `effector-fracture.js:588-613`.
#[must_use]
pub fn generate_voronoi_sites(mesh: &MeshData, params: &VoronoiFractureParams) -> Vec<Vec3> {
    let cells = params.cells.clamp(1, 1000) as usize;
    let b_min = mesh.bounds.min;
    let b_max = mesh.bounds.max;
    let b_size = b_max - b_min;
    let seed = params.seed;

    let mut areas: Vec<f64> = Vec::new();
    let mut total_area = 0.0_f64;
    if params.distribution == FractureDistribution::Surface {
        let tri_count = mesh.indices.len() / 3;
        areas.reserve(tri_count);
        for tri in mesh.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 < mesh.vertices.len() && i1 < mesh.vertices.len() && i2 < mesh.vertices.len() {
                let p0 = Vec3::from_array(mesh.vertices[i0].position);
                let p1 = Vec3::from_array(mesh.vertices[i1].position);
                let p2 = Vec3::from_array(mesh.vertices[i2].position);
                total_area += f64::from((p1 - p0).cross(p2 - p0).length()) * 0.5;
            }
            areas.push(total_area);
        }
    }

    let mut sites = Vec::with_capacity(cells);
    for cell in 0..cells {
        let ci = cell as i32;
        let mut unit = [
            effector_hash01(seed, ci, 0),
            effector_hash01(seed, ci, 1),
            effector_hash01(seed, ci, 2),
        ];

        if params.distribution == FractureDistribution::Surface
            && total_area > 0.0
            && !areas.is_empty()
        {
            let target = effector_hash01(seed, ci, 3) * total_area;
            let mut lo = 0_usize;
            let mut hi = areas.len() - 1;
            while lo < hi {
                let mid = (lo + hi) >> 1;
                if areas[mid] < target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let base = lo * 3;
            let i0 = mesh.indices[base] as usize;
            let i1 = mesh.indices[base + 1] as usize;
            let i2 = mesh.indices[base + 2] as usize;
            let u = unit[0].sqrt();
            let v = unit[1];
            let w0 = (1.0 - u) as f32;
            let w1 = (u * (1.0 - v)) as f32;
            let w2 = (u * v) as f32;
            let p0 = Vec3::from_array(mesh.vertices[i0].position);
            let p1 = Vec3::from_array(mesh.vertices[i1].position);
            let p2 = Vec3::from_array(mesh.vertices[i2].position);
            sites.push(p0 * w0 + p1 * w1 + p2 * w2);
            continue;
        }

        match params.distribution {
            FractureDistribution::Center => {
                for v in &mut unit {
                    let d = *v - 0.5;
                    let sign = if d > 0.0 {
                        1.0
                    } else if d < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    *v = 0.5 + sign * 0.5 * (2.0 * d.abs()).powi(2);
                }
            }
            FractureDistribution::Edges => {
                for v in &mut unit {
                    let d = *v - 0.5;
                    let sign = if d > 0.0 {
                        1.0
                    } else if d < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                    *v = 0.5 + sign * 0.5 * (2.0 * d.abs()).sqrt();
                }
            }
            FractureDistribution::Uniform | FractureDistribution::Surface => {}
        }

        sites.push(Vec3::new(
            (f64::from(b_min.x) + f64::from(b_size.x) * unit[0]) as f32,
            (f64::from(b_min.y) + f64::from(b_size.y) * unit[1]) as f32,
            (f64::from(b_min.z) + f64::from(b_size.z) * unit[2]) as f32,
        ));
    }
    sites
}

/// Evaluates Zoidium Voronoi Fracture (`effector-fracture.js:557-921`) with `effectorHash01`,
/// 4 site distributions, anisotropic `cell_scale` metric, `closed` cap toggle, and full
/// `applyFragmentMotion` (distance, scatter, shrink offset, Rodrigues spin, explicit direction &
/// XYZ rotation, fragment scale, quadratic gravity, randomness, and 6-type Field).
#[must_use]
pub fn apply_voronoi_fracture_with_params(
    mesh: &MeshData,
    params: &VoronoiFractureParams,
    shared_bounds: Option<Aabb3>,
    deformer_frame: Option<&DeformerFrame>,
) -> MeshData {
    let cells = params.cells.clamp(1, 1000) as usize;
    if mesh.indices.len() < 3 || mesh.vertices.is_empty() {
        return MeshData::empty();
    }

    let has_motion = params.distance.abs() > 1e-8
        || params.scatter.abs() > 1e-8
        || params.offset.abs() > 1e-8
        || params.spin_rad.abs() > 1e-8
        || params.direction.length_squared() > 1e-12
        || params.rotation_deg.length_squared() > 1e-12
        || (params.fragment_scale - 100.0).abs() > 1e-6
        || params.gravity.abs() > 1e-8;
    if cells == 1 && !has_motion {
        return mesh.clone();
    }

    let sites = generate_voronoi_sites(mesh, params);
    let metric = [
        1.0 / (f64::from(params.cell_scale.x).abs() / 100.0)
            .max(0.0001)
            .powi(2),
        1.0 / (f64::from(params.cell_scale.y).abs() / 100.0)
            .max(0.0001)
            .powi(2),
        1.0 / (f64::from(params.cell_scale.z).abs() / 100.0)
            .max(0.0001)
            .powi(2),
    ];

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

    let use_frame = deformer_frame.is_some_and(|f| !f.identity);
    let forward = deformer_frame.map(|f| f.forward);
    let rest_center = 0.5 * (mesh.bounds.min + mesh.bounds.max);
    let center = match (forward, use_frame) {
        (Some(m), true) => m.transform_point3(rest_center),
        _ => rest_center,
    };

    let prepared_field = params.field.as_ref().map(|f| {
        if let Some(sb) = shared_bounds {
            prepare_field_for_bounds(f, sb)
        } else {
            let pos_list: Vec<[f32; 3]> = mesh.vertices.iter().map(|v| v.position).collect();
            prepare_field_from_positions(f, &pos_list, if use_frame { forward } else { None })
        }
    });

    let seed = params.seed;
    let distance = finite_or(f64::from(params.distance), 0.0);
    let scatter = finite_or(f64::from(params.scatter), 0.0);
    let shrink_offset = finite_or(f64::from(params.offset), 0.0);
    let spin_rad = finite_or(f64::from(params.spin_rad), 0.0);
    let explicit_dir = [
        finite_or(f64::from(params.direction.x), 0.0),
        finite_or(f64::from(params.direction.y), 0.0),
        finite_or(f64::from(params.direction.z), 0.0),
    ];
    let explicit_rot = [
        finite_or(f64::from(params.rotation_deg.x), 0.0),
        finite_or(f64::from(params.rotation_deg.y), 0.0),
        finite_or(f64::from(params.rotation_deg.z), 0.0),
    ];
    let fragment_scale = finite_or(f64::from(params.fragment_scale), 100.0);
    let gravity = finite_or(f64::from(params.gravity), 0.0);
    let randomness = clamp_unit(finite_or(f64::from(params.randomness), 0.0) / 100.0);

    let mut out_verts: Vec<Vertex3D> = Vec::new();
    let mut out_indices: Vec<u32> = Vec::new();

    for (ci, &site_c) in sites.iter().enumerate() {
        if out_indices.len() / 3 >= MAX_VORONOI_OUTPUT_TRIS {
            break;
        }
        let mut cell_polys = source_tris.clone();

        for (cj, &site_j) in sites.iter().enumerate() {
            if ci == cj || cell_polys.is_empty() {
                continue;
            }
            // Anisotropic bisector plane matching `effector-fracture.js:173-181`.
            let diff_m = [
                f64::from(site_c.x - site_j.x) * metric[0],
                f64::from(site_c.y - site_j.y) * metric[1],
                f64::from(site_c.z - site_j.z) * metric[2],
            ];
            let len =
                (diff_m[0] * diff_m[0] + diff_m[1] * diff_m[1] + diff_m[2] * diff_m[2]).sqrt();
            if len <= 1e-12 {
                continue;
            }
            let inward_n = Vec3::new(
                (diff_m[0] / len) as f32,
                (diff_m[1] / len) as f32,
                (diff_m[2] / len) as f32,
            );
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

            if params.closed && !cell_polys.is_empty() && !cut_segs.is_empty() {
                build_planar_cap_polys(&cut_segs, mid, inward_n, &mut cell_polys);
            }
        }

        if cell_polys.is_empty() {
            continue;
        }

        // Compute outer-surface fragment centroid (matching `effector-fracture.js:770-791`:
        // outer fragment vertices determine centroid; falls back to all cell vertices / site_c).
        let mut outer_sum = [0.0_f64; 3];
        let mut outer_count = 0_usize;
        let mut all_sum = [0.0_f64; 3];
        let mut all_count = 0_usize;
        for poly in &cell_polys {
            for v in poly {
                all_sum[0] += f64::from(v.pos.x);
                all_sum[1] += f64::from(v.pos.y);
                all_sum[2] += f64::from(v.pos.z);
                all_count += 1;
                if v.material_slot != 1 {
                    outer_sum[0] += f64::from(v.pos.x);
                    outer_sum[1] += f64::from(v.pos.y);
                    outer_sum[2] += f64::from(v.pos.z);
                    outer_count += 1;
                }
            }
        }
        let rest_centroid = if outer_count > 0 {
            Vec3::new(
                (outer_sum[0] / outer_count as f64) as f32,
                (outer_sum[1] / outer_count as f64) as f32,
                (outer_sum[2] / outer_count as f64) as f32,
            )
        } else if all_count > 0 {
            Vec3::new(
                (all_sum[0] / all_count as f64) as f32,
                (all_sum[1] / all_count as f64) as f32,
                (all_sum[2] / all_count as f64) as f32,
            )
        } else {
            site_c
        };

        let local_centroid = match (forward, use_frame) {
            (Some(m), true) => m.transform_point3(rest_centroid),
            _ => rest_centroid,
        };

        // Exact Zoidium `applyFragmentMotion` (`effector-fracture.js:871-904`).
        let piece = ci as i32;
        let field_weight = prepared_field
            .as_ref()
            .map_or(1.0, |pf| evaluate_field(local_centroid, pf));
        let variation = 1.0 - randomness * effector_hash01(seed, piece, 40);
        let weight = field_weight * variation;

        let raw_out = [
            f64::from(local_centroid.x - center.x),
            f64::from(local_centroid.y - center.y),
            f64::from(local_centroid.z - center.z),
        ];
        let out_len =
            (raw_out[0] * raw_out[0] + raw_out[1] * raw_out[1] + raw_out[2] * raw_out[2]).sqrt();
        let outward = if out_len > 1e-12 {
            [
                raw_out[0] / out_len,
                raw_out[1] / out_len,
                raw_out[2] / out_len,
            ]
        } else {
            effector_hash_unit_vector(seed, piece, 10)
        };

        let scatter_dir = effector_hash_unit_vector(seed, piece, 21);
        let scatter_mag = 0.5 + effector_hash01(seed, piece, 24);
        let spin_sign = 2.0 * effector_hash01(seed, piece, 30) - 1.0;
        let axis = effector_hash_unit_vector(seed, piece, 31);
        let angle = spin_rad * spin_sign * weight;
        let (sine, cosine) = angle.sin_cos();
        let one_minus = 1.0 - cosine;
        let [ax, ay, az] = axis;

        let shift = [
            (outward[0] * distance + scatter_dir[0] * scatter_mag * scatter + explicit_dir[0])
                * weight,
            (outward[1] * distance + scatter_dir[1] * scatter_mag * scatter + explicit_dir[1])
                * weight
                - gravity * field_weight * field_weight * variation,
            (outward[2] * distance + scatter_dir[2] * scatter_mag * scatter + explicit_dir[2])
                * weight,
        ];
        let scale = (1.0 - (shrink_offset / 100.0) * weight).max(0.0)
            * (1.0 + (fragment_scale / 100.0 - 1.0) * field_weight).max(0.0);

        let mut rot = [
            cosine + ax * ax * one_minus,
            ax * ay * one_minus - az * sine,
            ax * az * one_minus + ay * sine,
            ay * ax * one_minus + az * sine,
            cosine + ay * ay * one_minus,
            ay * az * one_minus - ax * sine,
            az * ax * one_minus - ay * sine,
            az * ay * one_minus + ax * sine,
            cosine + az * az * one_minus,
        ];
        let deg = [
            explicit_rot[0] * weight,
            explicit_rot[1] * weight,
            explicit_rot[2] * weight,
        ];
        if deg[0].abs() > 1e-12 || deg[1].abs() > 1e-12 || deg[2].abs() > 1e-12 {
            for col in 0..3 {
                let rotated = rotate_xyz_deg([rot[col], rot[3 + col], rot[6 + col]], deg, false);
                rot[col] = rotated[0];
                rot[3 + col] = rotated[1];
                rot[6 + col] = rotated[2];
            }
        }

        let cx = f64::from(local_centroid.x);
        let cy = f64::from(local_centroid.y);
        let cz = f64::from(local_centroid.z);

        // Emit transformed fragment vertices (and transform back from deformer frame if needed).
        for poly in &cell_polys {
            if out_indices.len() / 3 >= MAX_VORONOI_OUTPUT_TRIS {
                break;
            }
            let base = out_verts.len() as u32;
            for v in poly {
                let p_def = match (forward, use_frame) {
                    (Some(m), true) => m.transform_point3(v.pos),
                    _ => v.pos,
                };
                let qx = (f64::from(p_def.x) - cx) * scale;
                let qy = (f64::from(p_def.y) - cy) * scale;
                let qz = (f64::from(p_def.z) - cz) * scale;
                let moved_def = Vec3::new(
                    (cx + rot[0] * qx + rot[1] * qy + rot[2] * qz + shift[0]) as f32,
                    (cy + rot[3] * qx + rot[4] * qy + rot[5] * qz + shift[1]) as f32,
                    (cz + rot[6] * qx + rot[7] * qy + rot[8] * qz + shift[2]) as f32,
                );
                let final_pos = match (deformer_frame, use_frame) {
                    (Some(frame), true) => frame.inverse.transform_point3(moved_def),
                    _ => moved_def,
                };
                let nx = f64::from(v.normal.x);
                let ny = f64::from(v.normal.y);
                let nz = f64::from(v.normal.z);
                let rn = Vec3::new(
                    (rot[0] * nx + rot[1] * ny + rot[2] * nz) as f32,
                    (rot[3] * nx + rot[4] * ny + rot[5] * nz) as f32,
                    (rot[6] * nx + rot[7] * ny + rot[8] * nz) as f32,
                )
                .normalize_or_zero();
                let final_n = if rn.length_squared() > 1e-12 {
                    rn
                } else {
                    v.normal
                };
                out_verts.push(Vertex3D::new(
                    final_pos.to_array(),
                    final_n.to_array(),
                    v.uv,
                    v.material_slot,
                ));
            }
            for k in 1..(poly.len() - 1) {
                out_indices.push(base);
                out_indices.push(base + k as u32);
                out_indices.push(base + k as u32 + 1);
            }
        }
    }

    MeshData::new(out_verts, out_indices)
}

/// Deterministic Effector `Voronoi Fracture` (backwards-compatible signature): clips `mesh` into
/// up to `cell_count` convex Voronoi fragments seeded by `seed` using Zoidium `effectorHash01`,
/// caps interior cuts (`material_slot = 1`), and offsets each fragment radially outward by `explode`.
#[must_use]
pub fn apply_voronoi_fracture(
    mesh: &MeshData,
    cell_count: u32,
    seed: u32,
    explode: f32,
) -> MeshData {
    let params = VoronoiFractureParams {
        cells: cell_count,
        seed: seed as i32,
        closed: true,
        distance: explode,
        ..VoronoiFractureParams::default()
    };
    apply_voronoi_fracture_with_params(mesh, &params, None, None)
}
