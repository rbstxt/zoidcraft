use super::*;

mod core_unit;
mod gpu_2d_effects;
mod gpu_3d_scene;
mod gpu_regressions;

#[allow(clippy::too_many_arguments)]
fn test_eval_env<'a>(
    project: &'a Project,
    plugins: &'a PluginRegistry,
    controls: &'a RefCell<BTreeMap<String, ExprValue>>,
    parent_props: &'a BTreeMap<String, ExprValue>,
    expr_stack: &'a RefCell<HashSet<String>>,
    frame: f64,
    time: f64,
    fps: f64,
) -> EvalEnv<'a> {
    EvalEnv {
        frame,
        abs_frame: frame,
        time,
        fps,
        sequence_resolution: project.sequence.resolution,
        project,
        sequence: &project.sequence,
        plugins,
        controls,
        parent_props,
        active_clip_id: project
            .sequence
            .video_tracks
            .first()
            .and_then(|t| t.clips.first())
            .map(|c| c.id),
        active_layer_id: project
            .sequence
            .video_tracks
            .first()
            .and_then(|t| t.clips.first())
            .map(|c| c.layer.id),
        active_object_id: None,
        active_effect_id: None,
        active_property_id: None,
        expr_stack,
    }
}

fn encode_test_png(width: u32, height: u32, rgba: Vec<u8>) -> Vec<u8> {
    let img = image::RgbaImage::from_raw(width, height, rgba).expect("valid rgba dims");
    let mut out = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
        .expect("encode test png");
    out
}

fn make_test_scene_project(objects: Vec<Object3D>, resolution: [u32; 2]) -> Project {
    let mut proj = Project::new_cm3_default();
    proj.sequence.resolution = resolution;
    proj.sequence.background_color = [0.0, 0.0, 0.0, 1.0];
    proj.sequence.video_tracks[0].clips[0].layer.resolution = resolution;
    proj.sequence.video_tracks[0].clips[0].layer.kind = LayerKind::Scene3D {
        objects,
        shadow_mode: ProjectCompatMode::Current,
    };
    proj
}
