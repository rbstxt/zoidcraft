use std::f32::consts::{FRAC_PI_2, PI, TAU};

use bytemuck::{Pod, Zeroable};
use zc_geom::{Aabb3, Mat4, Vec3, Vec4};

/// GPU-ready 3D mesh vertex with material slot tagging (e.g. cap vs. wall/bevel).
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
pub struct Vertex3D {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub material_slot: u32,
}

impl Vertex3D {
    #[must_use]
    pub const fn new(
        position: [f32; 3],
        normal: [f32; 3],
        uv: [f32; 2],
        material_slot: u32,
    ) -> Self {
        Self {
            position,
            normal,
            uv,
            material_slot,
        }
    }
}

/// Computes the tight axis-aligned bounding box of a vertex slice.
#[must_use]
pub fn compute_bounds(vertices: &[Vertex3D]) -> Aabb3 {
    let Some(first) = vertices.first() else {
        return Aabb3 {
            min: Vec3::ZERO,
            max: Vec3::ZERO,
        };
    };

    let mut min = Vec3::from_array(first.position);
    let mut max = min;
    for v in &vertices[1..] {
        let p = Vec3::from_array(v.position);
        min = min.min(p);
        max = max.max(p);
    }
    Aabb3 { min, max }
}

/// Indexed triangle mesh with axis-aligned bounds.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshData {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
    pub bounds: Aabb3,
}

impl MeshData {
    /// Creates a mesh and computes its bounding box from `vertices`.
    #[must_use]
    pub fn new(vertices: Vec<Vertex3D>, indices: Vec<u32>) -> Self {
        let bounds = compute_bounds(&vertices);
        Self {
            vertices,
            indices,
            bounds,
        }
    }

    /// Creates an empty mesh with zero bounds.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            bounds: Aabb3 {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            },
        }
    }

    /// Returns the number of triangles in this mesh.
    #[must_use]
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Recomputes `self.bounds` from current vertex positions.
    pub fn recompute_bounds(&mut self) {
        self.bounds = compute_bounds(&self.vertices);
    }

    /// Translates every vertex by `offset` and updates `self.bounds`.
    pub fn translate(&mut self, offset: Vec3) {
        if self.vertices.is_empty() {
            return;
        }
        for v in &mut self.vertices {
            let p = Vec3::from_array(v.position) + offset;
            v.position = p.to_array();
        }
        self.bounds = Aabb3 {
            min: self.bounds.min + offset,
            max: self.bounds.max + offset,
        };
    }

    /// Centers the mesh around the origin using its current AABB midpoint.
    pub fn center_at_origin(&mut self) {
        if self.vertices.is_empty() {
            return;
        }
        self.recompute_bounds();
        let center = (self.bounds.min + self.bounds.max) * 0.5;
        self.translate(-center);
    }

    /// Splits shared vertices per triangle and assigns exact geometric flat normals.
    pub fn compute_flat_normals(&mut self) {
        let tri_count = self.indices.len() / 3;
        let mut out_verts = Vec::with_capacity(tri_count * 3);
        let mut out_indices = Vec::with_capacity(tri_count * 3);

        for tri in self.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let mut v0 = self.vertices[i0];
            let mut v1 = self.vertices[i1];
            let mut v2 = self.vertices[i2];

            let p0 = Vec3::from_array(v0.position);
            let p1 = Vec3::from_array(v1.position);
            let p2 = Vec3::from_array(v2.position);
            let cross = (p1 - p0).cross(p2 - p0);
            let n = if cross.length_squared() > 1e-12 {
                cross.normalize()
            } else {
                Vec3::Z
            };
            let n_arr = n.to_array();
            v0.normal = n_arr;
            v1.normal = n_arr;
            v2.normal = n_arr;

            let base = out_verts.len() as u32;
            out_verts.push(v0);
            out_verts.push(v1);
            out_verts.push(v2);
            out_indices.push(base);
            out_indices.push(base + 1);
            out_indices.push(base + 2);
        }

        self.vertices = out_verts;
        self.indices = out_indices;
        self.recompute_bounds();
    }

    /// Computes area-weighted smooth vertex normals from incident triangles.
    pub fn compute_smooth_normals(&mut self) {
        let mut accum = vec![Vec3::ZERO; self.vertices.len()];
        for tri in self.indices.as_chunks::<3>().0 {
            let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
            if i0 >= self.vertices.len() || i1 >= self.vertices.len() || i2 >= self.vertices.len() {
                continue;
            }
            let p0 = Vec3::from_array(self.vertices[i0].position);
            let p1 = Vec3::from_array(self.vertices[i1].position);
            let p2 = Vec3::from_array(self.vertices[i2].position);
            let weighted_n = (p1 - p0).cross(p2 - p0);
            accum[i0] += weighted_n;
            accum[i1] += weighted_n;
            accum[i2] += weighted_n;
        }

        for (v, n) in self.vertices.iter_mut().zip(accum) {
            let unit = if n.length_squared() > 1e-12 {
                n.normalize()
            } else {
                Vec3::Z
            };
            v.normal = unit.to_array();
        }
    }

    /// Transforms vertex positions and normals by a 4x4 affine matrix.
    ///
    /// Uses the inverse-transpose of the upper-left 3x3 for normals and flips triangle
    /// winding when the matrix has negative determinant (mirroring).
    pub fn apply_matrix(&mut self, matrix: Mat4) {
        let linear = Mat4::from_cols(
            matrix.x_axis.truncate().extend(0.0),
            matrix.y_axis.truncate().extend(0.0),
            matrix.z_axis.truncate().extend(0.0),
            Vec4::W,
        );
        let det = linear.determinant();
        let normal_mat = if det.abs() > 1e-12 {
            linear.inverse().transpose()
        } else {
            linear
        };

        for v in &mut self.vertices {
            let p = matrix.transform_point3(Vec3::from_array(v.position));
            v.position = p.to_array();

            let n = normal_mat.transform_vector3(Vec3::from_array(v.normal));
            let unit = if n.length_squared() > 1e-12 {
                n.normalize()
            } else {
                Vec3::Z
            };
            v.normal = unit.to_array();
        }

        if det < 0.0 {
            for tri in self.indices.as_chunks_mut::<3>().0 {
                tri.swap(1, 2);
            }
        }

        self.recompute_bounds();
    }

    /// Appends another mesh into `self`, offsetting indices and updating bounds.
    pub fn merge(&mut self, other: &MeshData) {
        if other.vertices.is_empty() {
            return;
        }
        let base = self.vertices.len() as u32;
        self.vertices.extend_from_slice(&other.vertices);
        self.indices.extend(other.indices.iter().map(|&i| i + base));
        self.recompute_bounds();
    }
}

/// CM3 `Box` (`objectType = 1`): axis-aligned box centered at the origin.
#[must_use]
pub fn build_box(width: f32, height: f32, depth: f32) -> MeshData {
    let hx = width.max(0.0) * 0.5;
    let hy = height.max(0.0) * 0.5;
    let hz = depth.max(0.0) * 0.5;

    let faces: [(Vec3, Vec3, Vec3, f32, f32, f32); 6] = [
        (Vec3::Z, Vec3::X, Vec3::Y, hx, hy, hz),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y, hx, hy, hz),
        (Vec3::X, Vec3::NEG_Z, Vec3::Y, hz, hy, hx),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y, hz, hy, hx),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z, hx, hz, hy),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z, hx, hz, hy),
    ];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (normal, right, up, half_u, half_v, half_n) in faces {
        let base = vertices.len() as u32;
        let center = normal * half_n;
        let r = right * half_u;
        let u = up * half_v;
        let n_arr = normal.to_array();

        vertices.push(Vertex3D::new(
            (center - r - u).to_array(),
            n_arr,
            [0.0, 0.0],
            0,
        ));
        vertices.push(Vertex3D::new(
            (center + r - u).to_array(),
            n_arr,
            [1.0, 0.0],
            0,
        ));
        vertices.push(Vertex3D::new(
            (center + r + u).to_array(),
            n_arr,
            [1.0, 1.0],
            0,
        ));
        vertices.push(Vertex3D::new(
            (center - r + u).to_array(),
            n_arr,
            [0.0, 1.0],
            0,
        ));

        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    MeshData::new(vertices, indices)
}

/// CM3 `Cylinder` (`objectType = 2`): cylinder, frustum, or cone along Y centered at the origin.
#[must_use]
pub fn build_cylinder(
    top_radius: f32,
    bottom_radius: f32,
    height: f32,
    radial_segments: u32,
    open_ended: bool,
) -> MeshData {
    let r_top = top_radius.max(0.0);
    let r_bot = bottom_radius.max(0.0);
    let h = height.max(0.0);
    let half_h = h * 0.5;
    let segs = radial_segments.max(3);

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    let slope = if h > 1e-6 { (r_bot - r_top) / h } else { 0.0 };

    // Side wall: two rows (bottom = 0, top = 1)
    let side_base = vertices.len() as u32;
    for row in 0..2_u32 {
        let v_coord = row as f32;
        let y = if row == 0 { -half_h } else { half_h };
        let r = if row == 0 { r_bot } else { r_top };
        for i in 0..=segs {
            let u_coord = i as f32 / segs as f32;
            let theta = u_coord * TAU;
            let sin_t = theta.sin();
            let cos_t = theta.cos();
            let pos = [r * sin_t, y, r * cos_t];
            let normal = Vec3::new(sin_t, slope, cos_t).normalize_or_zero();
            let n_arr = if normal.length_squared() > 1e-12 {
                normal.to_array()
            } else {
                [sin_t, 0.0, cos_t]
            };
            vertices.push(Vertex3D::new(pos, n_arr, [u_coord, v_coord], 0));
        }
    }

    let stride = segs + 1;
    for i in 0..segs {
        let b0 = side_base + i;
        let b1 = side_base + i + 1;
        let t0 = side_base + stride + i;
        let t1 = side_base + stride + i + 1;
        if r_bot > 1e-6 {
            indices.extend_from_slice(&[b0, b1, t1]);
        }
        if r_top > 1e-6 {
            indices.extend_from_slice(&[b0, t1, t0]);
        }
    }

    if !open_ended {
        if r_top > 1e-6 {
            add_cylinder_cap(&mut vertices, &mut indices, r_top, half_h, segs, true);
        }
        if r_bot > 1e-6 {
            add_cylinder_cap(&mut vertices, &mut indices, r_bot, -half_h, segs, false);
        }
    }

    MeshData::new(vertices, indices)
}

fn add_cylinder_cap(
    vertices: &mut Vec<Vertex3D>,
    indices: &mut Vec<u32>,
    radius: f32,
    y: f32,
    segs: u32,
    is_top: bool,
) {
    let normal = if is_top {
        [0.0, 1.0, 0.0]
    } else {
        [0.0, -1.0, 0.0]
    };
    let center_idx = vertices.len() as u32;
    vertices.push(Vertex3D::new([0.0, y, 0.0], normal, [0.5, 0.5], 0));

    let ring_start = vertices.len() as u32;
    for i in 0..=segs {
        let theta = (i as f32 / segs as f32) * TAU;
        let sin_t = theta.sin();
        let cos_t = theta.cos();
        let u = 0.5 + 0.5 * sin_t;
        let v = if is_top {
            0.5 + 0.5 * cos_t
        } else {
            0.5 - 0.5 * cos_t
        };
        vertices.push(Vertex3D::new(
            [radius * sin_t, y, radius * cos_t],
            normal,
            [u, v],
            0,
        ));
    }

    for i in 0..segs {
        let r0 = ring_start + i;
        let r1 = ring_start + i + 1;
        if is_top {
            indices.extend_from_slice(&[center_idx, r0, r1]);
        } else {
            indices.extend_from_slice(&[center_idx, r1, r0]);
        }
    }
}

/// CM3 `Rectangle` (`objectType = 3`): 1x1 quad in the XY plane facing `+Z`.
#[must_use]
pub fn build_rectangle(width: f32, height: f32) -> MeshData {
    let hx = width.max(0.0) * 0.5;
    let hy = height.max(0.0) * 0.5;
    let n = [0.0, 0.0, 1.0];

    let vertices = vec![
        Vertex3D::new([-hx, -hy, 0.0], n, [0.0, 0.0], 0),
        Vertex3D::new([hx, -hy, 0.0], n, [1.0, 0.0], 0),
        Vertex3D::new([hx, hy, 0.0], n, [1.0, 1.0], 0),
        Vertex3D::new([-hx, hy, 0.0], n, [0.0, 1.0], 0),
    ];
    let indices = vec![0, 1, 2, 0, 2, 3];
    MeshData::new(vertices, indices)
}

/// CM3 `Circle` (`objectType = 4`): disk in the XY plane facing `+Z`.
#[must_use]
pub fn build_circle(radius: f32, segments: u32) -> MeshData {
    let r = radius.max(0.0);
    let segs = segments.max(3);
    let n = [0.0, 0.0, 1.0];

    let mut vertices = Vec::with_capacity((segs + 2) as usize);
    let mut indices = Vec::with_capacity((segs * 3) as usize);

    vertices.push(Vertex3D::new([0.0, 0.0, 0.0], n, [0.5, 0.5], 0));
    for i in 0..=segs {
        let theta = (i as f32 / segs as f32) * TAU;
        let cos_t = theta.cos();
        let sin_t = theta.sin();
        vertices.push(Vertex3D::new(
            [r * cos_t, r * sin_t, 0.0],
            n,
            [0.5 + 0.5 * cos_t, 0.5 + 0.5 * sin_t],
            0,
        ));
    }

    for i in 0..segs {
        indices.extend_from_slice(&[0, 1 + i, 2 + i]);
    }

    MeshData::new(vertices, indices)
}

/// CM3 `Sphere` (`objectType = 5`): UV sphere centered at the origin.
#[must_use]
pub fn build_sphere(radius: f32, width_segments: u32, height_segments: u32) -> MeshData {
    let r = radius.max(0.0);
    let w_segs = width_segments.max(3);
    let h_segs = height_segments.max(2);

    let mut vertices = Vec::with_capacity(((w_segs + 1) * (h_segs + 1)) as usize);
    let mut indices = Vec::with_capacity((w_segs * h_segs * 6) as usize);

    for iy in 0..=h_segs {
        let v = iy as f32 / h_segs as f32;
        let theta = v * PI;
        let sin_t = theta.sin();
        let cos_t = theta.cos();

        for ix in 0..=w_segs {
            let u = ix as f32 / w_segs as f32;
            let phi = u * TAU;
            let nx = sin_t * phi.sin();
            let ny = cos_t;
            let nz = sin_t * phi.cos();
            let normal = Vec3::new(nx, ny, nz).normalize_or_zero();
            let n_arr = if normal.length_squared() > 1e-12 {
                normal.to_array()
            } else {
                [0.0, if iy == 0 { 1.0 } else { -1.0 }, 0.0]
            };
            vertices.push(Vertex3D::new(
                [r * nx, r * ny, r * nz],
                n_arr,
                [u, 1.0 - v],
                0,
            ));
        }
    }

    let stride = w_segs + 1;
    for iy in 0..h_segs {
        for ix in 0..w_segs {
            let a = iy * stride + ix;
            let b = (iy + 1) * stride + ix;
            let c = (iy + 1) * stride + ix + 1;
            let d = iy * stride + ix + 1;
            if iy != h_segs - 1 {
                indices.extend_from_slice(&[a, b, c]);
            }
            if iy != 0 {
                indices.extend_from_slice(&[a, c, d]);
            }
        }
    }

    MeshData::new(vertices, indices)
}

/// CM3 `Donut` (`objectType = 6`): torus in the XY plane centered at the origin.
#[must_use]
pub fn build_donut(
    major_radius: f32,
    tube_radius: f32,
    radial_segments: u32,
    tubular_segments: u32,
) -> MeshData {
    let big_r = major_radius.max(0.0);
    let small_r = tube_radius.max(0.0);
    let r_segs = radial_segments.max(3);
    let t_segs = tubular_segments.max(3);

    let mut vertices = Vec::with_capacity(((t_segs + 1) * (r_segs + 1)) as usize);
    let mut indices = Vec::with_capacity((t_segs * r_segs * 6) as usize);

    for iu in 0..=t_segs {
        let u_frac = iu as f32 / t_segs as f32;
        let u = u_frac * TAU;
        let cos_u = u.cos();
        let sin_u = u.sin();

        for iv in 0..=r_segs {
            let v_frac = iv as f32 / r_segs as f32;
            let v = v_frac * TAU;
            let cos_v = v.cos();
            let sin_v = v.sin();

            let radial = big_r + small_r * cos_v;
            let pos = [radial * cos_u, radial * sin_u, small_r * sin_v];
            let normal = [cos_v * cos_u, cos_v * sin_u, sin_v];
            vertices.push(Vertex3D::new(pos, normal, [u_frac, v_frac], 0));
        }
    }

    let stride = r_segs + 1;
    for iu in 0..t_segs {
        for iv in 0..r_segs {
            let a = iu * stride + iv;
            let b = (iu + 1) * stride + iv;
            let c = (iu + 1) * stride + iv + 1;
            let d = iu * stride + iv + 1;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    MeshData::new(vertices, indices)
}

/// Evaluates a uniform Catmull-Rom spline segment for `t in [0, 1]`.
#[must_use]
fn catmull_rom_vec3(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: f32) -> Vec3 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Evaluates an open Catmull-Rom spline over `control_points` at normalized parameter `u in [0, 1]`.
fn sample_catmull_rom_chain(control_points: &[Vec3], u: f32) -> Vec3 {
    let n = control_points.len();
    debug_assert!(n >= 2);
    let seg_count = (n - 1) as f32;
    let scaled = u.clamp(0.0, 1.0) * seg_count;
    let idx = (scaled.floor() as usize).min(n - 2);
    let local_t = (scaled - idx as f32).clamp(0.0, 1.0);

    let p1 = control_points[idx];
    let p2 = control_points[idx + 1];
    let p0 = if idx == 0 {
        p1 + (p1 - p2)
    } else {
        control_points[idx - 1]
    };
    let p3 = if idx + 2 >= n {
        p2 + (p2 - p1)
    } else {
        control_points[idx + 2]
    };
    catmull_rom_vec3(p0, p1, p2, p3, local_t)
}

/// CM3 `Wire` (`objectType = 7`): circular tube along a Catmull-Rom helix, auto-centered.
#[must_use]
pub fn build_wire(
    size: Vec3,
    thickness: f32,
    loops: u32,
    inner_segs: u32,
    outer_segs: u32,
) -> MeshData {
    let loops = loops.max(1);
    let inner = inner_segs.max(3);
    let outer = outer_segs.max(3);
    let radius = thickness.max(1e-4);

    let point_count = (loops * outer) as usize + 1;
    let mut control_points = Vec::with_capacity(point_count);
    for k in 0..point_count {
        let frac = k as f32 / outer as f32;
        let angle = TAU * frac;
        control_points.push(Vec3::new(
            size.x * angle.cos(),
            size.y * angle.sin(),
            size.z * frac,
        ));
    }

    let steps = (10 * outer).max(4) as usize;
    let mut spine = Vec::with_capacity(steps + 1);
    for s in 0..=steps {
        let u = s as f32 / steps as f32;
        spine.push(sample_catmull_rom_chain(&control_points, u));
    }

    // Parallel-transport frames along the spline so the cross-section never twists wildly.
    let mut tangents = Vec::with_capacity(spine.len());
    for i in 0..spine.len() {
        let raw = if i == 0 {
            spine[1] - spine[0]
        } else if i == spine.len() - 1 {
            spine[i] - spine[i - 1]
        } else {
            spine[i + 1] - spine[i - 1]
        };
        let t = if raw.length_squared() > 1e-12 {
            raw.normalize()
        } else {
            Vec3::Z
        };
        tangents.push(t);
    }

    let mut normals = Vec::with_capacity(spine.len());
    let mut binormals = Vec::with_capacity(spine.len());

    let t0 = tangents[0];
    let ref_axis = if t0.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
    let mut prev_n = t0.cross(ref_axis).normalize_or_zero();
    if prev_n.length_squared() < 1e-12 {
        prev_n = Vec3::Y;
    }
    normals.push(prev_n);
    binormals.push(t0.cross(prev_n).normalize_or_zero());

    for i in 1..spine.len() {
        let prev_t = tangents[i - 1];
        let curr_t = tangents[i];
        let rot_axis = prev_t.cross(curr_t);
        let n = if rot_axis.length_squared() > 1e-10 {
            let axis = rot_axis.normalize();
            let angle = prev_t.dot(curr_t).clamp(-1.0, 1.0).acos();
            let rot = Mat4::from_axis_angle(axis, angle);
            rot.transform_vector3(prev_n).normalize_or_zero()
        } else {
            prev_n
        };
        // Re-orthogonalize against curr_t
        let n = (n - curr_t * n.dot(curr_t)).normalize_or_zero();
        let n = if n.length_squared() > 1e-12 {
            n
        } else {
            prev_n
        };
        let b = curr_t.cross(n).normalize_or_zero();
        normals.push(n);
        binormals.push(b);
        prev_n = n;
    }

    let mut vertices = Vec::with_capacity(spine.len() * (inner as usize + 1));
    let mut indices = Vec::with_capacity(steps * inner as usize * 6);

    for i in 0..spine.len() {
        let u = i as f32 / steps as f32;
        let center = spine[i];
        let n_vec = normals[i];
        let b_vec = binormals[i];
        for j in 0..=inner {
            let v = j as f32 / inner as f32;
            let theta = v * TAU;
            let radial_dir = (theta.cos() * n_vec + theta.sin() * b_vec).normalize_or_zero();
            let pos = center + radial_dir * radius;
            vertices.push(Vertex3D::new(
                pos.to_array(),
                radial_dir.to_array(),
                [u, v],
                0,
            ));
        }
    }

    let stride = inner + 1;
    for i in 0..steps as u32 {
        for j in 0..inner {
            let a = i * stride + j;
            let b = i * stride + j + 1;
            let c = (i + 1) * stride + j + 1;
            let d = (i + 1) * stride + j;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }

    let mut mesh = MeshData::new(vertices, indices);
    mesh.center_at_origin();
    mesh
}

/// Geometry+ `RoundedBox`: axis-aligned box with smooth filleted edges and corners.
#[must_use]
pub fn build_rounded_box(
    width: f32,
    height: f32,
    depth: f32,
    radius: f32,
    segments: u32,
) -> MeshData {
    let w = width.max(0.0);
    let h = height.max(0.0);
    let d = depth.max(0.0);
    let max_r = 0.5 * w.min(h).min(d);
    let r = radius.clamp(0.0, max_r);

    if r <= 1e-5 || segments == 0 {
        return build_box(w, h, d);
    }

    let hx = w * 0.5;
    let hy = h * 0.5;
    let hz = d * 0.5;
    let inner = Vec3::new((hx - r).max(0.0), (hy - r).max(0.0), (hz - r).max(0.0));

    let axis_samples = |half_ext: f32, inner_ext: f32| -> Vec<f32> {
        let s = segments as usize;
        let mut coords = Vec::with_capacity(2 * s + 2);
        for i in 0..=s {
            let t = i as f32 / s as f32;
            coords.push(-half_ext + t * (half_ext - inner_ext));
        }
        for i in 0..=s {
            let t = i as f32 / s as f32;
            coords.push(inner_ext + t * (half_ext - inner_ext));
        }
        coords
    };

    let xs = axis_samples(hx, inner.x);
    let ys = axis_samples(hy, inner.y);
    let zs = axis_samples(hz, inner.z);

    type RoundedBoxFace<'a> = (Vec3, Vec3, Vec3, &'a [f32], &'a [f32], f32);
    let faces: [RoundedBoxFace<'_>; 6] = [
        (Vec3::Z, Vec3::X, Vec3::Y, &xs, &ys, hz),
        (Vec3::NEG_Z, Vec3::NEG_X, Vec3::Y, &xs, &ys, hz),
        (Vec3::X, Vec3::NEG_Z, Vec3::Y, &zs, &ys, hx),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y, &zs, &ys, hx),
        (Vec3::Y, Vec3::X, Vec3::NEG_Z, &xs, &zs, hy),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z, &xs, &zs, hy),
    ];

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for (face_n, right, up, u_coords, v_coords, half_n) in faces {
        let cols = u_coords.len();
        let rows = v_coords.len();
        let base = vertices.len() as u32;
        let u_min = u_coords[0];
        let u_span = (u_coords[cols - 1] - u_min).max(1e-6);
        let v_min = v_coords[0];
        let v_span = (v_coords[rows - 1] - v_min).max(1e-6);

        for &vc in v_coords {
            for &uc in u_coords {
                let q = face_n * half_n + right * uc + up * vc;
                let clamped = q.clamp(-inner, inner);
                let delta = q - clamped;
                let (pos, normal) = if delta.length_squared() > 1e-12 {
                    let dir = delta.normalize();
                    (clamped + dir * r, dir)
                } else {
                    (q, face_n)
                };
                let uv = [(uc - u_min) / u_span, (vc - v_min) / v_span];
                vertices.push(Vertex3D::new(pos.to_array(), normal.to_array(), uv, 0));
            }
        }

        let stride = cols as u32;
        for row in 0..(rows as u32 - 1) {
            for col in 0..(cols as u32 - 1) {
                let a = base + row * stride + col;
                let b = base + row * stride + col + 1;
                let c = base + (row + 1) * stride + col + 1;
                let d_idx = base + (row + 1) * stride + col;
                indices.extend_from_slice(&[a, b, c, a, c, d_idx]);
            }
        }
    }

    MeshData::new(vertices, indices)
}

/// Builds a spherical/geodesic polyhedron from base vertices and triangular face indices.
fn build_polyhedron(
    base_verts: &[Vec3],
    base_faces: &[[usize; 3]],
    radius: f32,
    detail: u32,
) -> MeshData {
    let r = radius.max(0.0);
    let levels = detail.min(5);

    let mut triangles: Vec<[Vec3; 3]> = base_faces
        .iter()
        .map(|&[i0, i1, i2]| {
            let a = base_verts[i0].normalize();
            let b = base_verts[i1].normalize();
            let c = base_verts[i2].normalize();
            let centroid = a + b + c;
            if (b - a).cross(c - a).dot(centroid) >= 0.0 {
                [a, b, c]
            } else {
                [a, c, b]
            }
        })
        .collect();

    for _ in 0..levels {
        let mut next = Vec::with_capacity(triangles.len() * 4);
        for [a, b, c] in triangles {
            let ab = (a + b).normalize();
            let bc = (b + c).normalize();
            let ca = (c + a).normalize();
            next.push([a, ab, ca]);
            next.push([ab, b, bc]);
            next.push([ca, bc, c]);
            next.push([ab, bc, ca]);
        }
        triangles = next;
    }

    let mut vertices = Vec::with_capacity(triangles.len() * 3);
    let mut indices = Vec::with_capacity(triangles.len() * 3);

    for [a, b, c] in triangles {
        let face_n = (b - a).cross(c - a).normalize_or_zero();
        let base = vertices.len() as u32;
        for dir in [a, b, c] {
            let pos = dir * r;
            let normal = if levels == 0 { face_n } else { dir };
            let u = 0.5 + dir.x.atan2(dir.z) / TAU;
            let v = 0.5 + dir.y.clamp(-1.0, 1.0).asin() / PI;
            vertices.push(Vertex3D::new(pos.to_array(), normal.to_array(), [u, v], 0));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    MeshData::new(vertices, indices)
}

/// Geometry+ `Tetrahedron`: regular tetrahedron with optional geodesic subdivision.
#[must_use]
pub fn build_tetrahedron(radius: f32, detail: u32) -> MeshData {
    let verts = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, -1.0, -1.0),
    ];
    let faces = [[2, 1, 0], [0, 3, 2], [1, 3, 0], [2, 3, 1]];
    build_polyhedron(&verts, &faces, radius, detail)
}

/// Geometry+ `Octahedron`: regular octahedron with optional geodesic subdivision.
#[must_use]
pub fn build_octahedron(radius: f32, detail: u32) -> MeshData {
    let verts = [
        Vec3::X,
        Vec3::NEG_X,
        Vec3::Y,
        Vec3::NEG_Y,
        Vec3::Z,
        Vec3::NEG_Z,
    ];
    let faces = [
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 2, 5],
        [1, 5, 3],
        [1, 3, 4],
        [1, 4, 2],
    ];
    build_polyhedron(&verts, &faces, radius, detail)
}

/// Geometry+ `Icosahedron`: regular icosahedron with optional geodesic subdivision.
#[must_use]
pub fn build_icosahedron(radius: f32, detail: u32) -> MeshData {
    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let verts = [
        Vec3::new(-1.0, t, 0.0),
        Vec3::new(1.0, t, 0.0),
        Vec3::new(-1.0, -t, 0.0),
        Vec3::new(1.0, -t, 0.0),
        Vec3::new(0.0, -1.0, t),
        Vec3::new(0.0, 1.0, t),
        Vec3::new(0.0, -1.0, -t),
        Vec3::new(0.0, 1.0, -t),
        Vec3::new(t, 0.0, -1.0),
        Vec3::new(t, 0.0, 1.0),
        Vec3::new(-t, 0.0, -1.0),
        Vec3::new(-t, 0.0, 1.0),
    ];
    let faces = [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    build_polyhedron(&verts, &faces, radius, detail)
}

/// Geometry+ `Dodecahedron`: regular dodecahedron with optional geodesic subdivision.
#[must_use]
pub fn build_dodecahedron(radius: f32, detail: u32) -> MeshData {
    let t = (1.0 + 5.0_f32.sqrt()) * 0.5;
    let r = 1.0 / t;
    let verts = [
        // Cube vertices (0..8)
        Vec3::new(-1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
        // (0, ±1/t, ±t) (8..12)
        Vec3::new(0.0, -r, -t),
        Vec3::new(0.0, -r, t),
        Vec3::new(0.0, r, -t),
        Vec3::new(0.0, r, t),
        // (±1/t, ±t, 0) (12..16)
        Vec3::new(-r, -t, 0.0),
        Vec3::new(-r, t, 0.0),
        Vec3::new(r, -t, 0.0),
        Vec3::new(r, t, 0.0),
        // (±t, 0, ±1/t) (16..20)
        Vec3::new(-t, 0.0, -r),
        Vec3::new(t, 0.0, -r),
        Vec3::new(-t, 0.0, r),
        Vec3::new(t, 0.0, r),
    ];

    // 12 pentagonal faces triangulated into 36 triangles
    let pentagons: [[usize; 5]; 12] = [
        [3, 11, 7, 15, 13],
        [7, 19, 17, 6, 15],
        [17, 4, 8, 10, 6],
        [8, 0, 16, 2, 10],
        [0, 12, 1, 18, 16],
        [6, 10, 2, 13, 15],
        [2, 16, 18, 3, 13],
        [18, 1, 9, 11, 3],
        [4, 14, 12, 0, 8],
        [11, 9, 5, 19, 7],
        [19, 5, 14, 4, 17],
        [1, 12, 14, 5, 9],
    ];

    let mut faces = Vec::with_capacity(36);
    for p in pentagons {
        faces.push([p[0], p[1], p[2]]);
        faces.push([p[0], p[2], p[3]]);
        faces.push([p[0], p[3], p[4]]);
    }

    build_polyhedron(&verts, &faces, radius, detail)
}

/// Geometry+ `Capsule`: cylindrical barrel of `length` capped by hemispheres of `radius` along Y.
#[must_use]
pub fn build_capsule(
    radius: f32,
    length: f32,
    cap_segments: u32,
    radial_segments: u32,
) -> MeshData {
    let r = radius.max(0.0);
    let barrel = length.max(0.0);
    let half_l = barrel * 0.5;
    let cap_segs = cap_segments.max(2);
    let rad_segs = radial_segments.max(3);

    // Build latitude rings: north pole -> top equator -> bottom equator -> south pole
    let mut rings: Vec<(f32, f32, f32, f32)> = Vec::with_capacity((cap_segs * 2 + 2) as usize);
    let total_h = (barrel + 2.0 * r).max(1e-6);

    for i in 0..=cap_segs {
        let theta = (i as f32 / cap_segs as f32) * FRAC_PI_2;
        let ring_r = r * theta.sin();
        let y = half_l + r * theta.cos();
        let v = (y + half_l + r) / total_h;
        rings.push((ring_r, y, theta.sin(), theta.cos().max(0.0) + 0.0 * v));
    }
    for i in 0..=cap_segs {
        let theta = FRAC_PI_2 + (i as f32 / cap_segs as f32) * FRAC_PI_2;
        let ring_r = r * theta.sin();
        let y = -half_l + r * theta.cos();
        rings.push((ring_r, y, theta.sin(), theta.cos()));
    }

    let ring_count = rings.len() as u32;
    let mut vertices = Vec::with_capacity((ring_count * (rad_segs + 1)) as usize);
    let mut indices = Vec::with_capacity(((ring_count - 1) * rad_segs * 6) as usize);

    for &(ring_r, y, sin_t, cos_t) in &rings {
        let v = ((y + half_l + r) / total_h).clamp(0.0, 1.0);
        for ix in 0..=rad_segs {
            let u = ix as f32 / rad_segs as f32;
            let phi = u * TAU;
            let nx = sin_t * phi.sin();
            let ny = cos_t;
            let nz = sin_t * phi.cos();
            let normal = Vec3::new(nx, ny, nz).normalize_or_zero();
            let n_arr = if normal.length_squared() > 1e-12 {
                normal.to_array()
            } else {
                [0.0, if y >= 0.0 { 1.0 } else { -1.0 }, 0.0]
            };
            vertices.push(Vertex3D::new(
                [ring_r * phi.sin(), y, ring_r * phi.cos()],
                n_arr,
                [u, v],
                0,
            ));
        }
    }

    let stride = rad_segs + 1;
    for iy in 0..(ring_count - 1) {
        for ix in 0..rad_segs {
            let a = iy * stride + ix;
            let b = (iy + 1) * stride + ix;
            let c = (iy + 1) * stride + ix + 1;
            let d = iy * stride + ix + 1;
            if iy != ring_count - 2 {
                indices.extend_from_slice(&[a, b, c]);
            }
            if iy != 0 {
                indices.extend_from_slice(&[a, c, d]);
            }
        }
    }

    MeshData::new(vertices, indices)
}
