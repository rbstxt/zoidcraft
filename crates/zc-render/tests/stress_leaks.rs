#![forbid(unsafe_code)]

use zc_plugin::PluginRegistry;
use zc_project::Project;
use zc_render::{
    FrameCache, MAX_CACHED_MESHES, MAX_CACHED_TEXTURES, ProjectRenderEngine, RenderError,
};

fn require_gpu() -> bool {
    std::env::var("ZOIDCRAFT_REQUIRE_GPU")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

#[test]
fn gpu_render_300_frames_across_resizes_stays_within_budgets() {
    let mut engine = match ProjectRenderEngine::new_headless() {
        Ok(engine) => engine,
        Err(RenderError::NoAdapter(_)) if !require_gpu() => {
            eprintln!("skipping GPU stress test: no wgpu adapter available");
            return;
        }
        Err(err) => panic!("failed to initialize ProjectRenderEngine: {err}"),
    };

    let target_budget = 16 * 1024 * 1024; // 16 MiB
    let frame_cache_budget = 4 * 1024 * 1024; // 4 MiB
    engine.targets.set_max_bytes(target_budget);
    engine.cache.set_max_bytes(frame_cache_budget);

    let project = Project::new_cm3_default();
    let plugins = PluginRegistry::new();
    let resolutions: &[(u32, u32)] = &[(320, 180), (640, 360), (480, 270), (256, 144)];

    for i in 0..300_u32 {
        let (w, h) = resolutions[(i as usize / 15) % resolutions.len()];
        let frame = f64::from(i % 120) * 0.5;

        if i % 10 == 0 {
            let cached = engine
                .render_frame_rgba8_cached(&project, &plugins, frame, w, h, u64::from(i / 30))
                .expect("cached frame render should succeed");
            assert_eq!(cached.len(), (w as usize) * (h as usize) * 4);
        } else {
            let _view = engine
                .render_frame_to_texture(&project, &plugins, frame, w, h)
                .expect("zero-copy GPU frame render should succeed");
        }

        assert!(
            engine.targets.pooled_bytes() <= target_budget,
            "TargetPool exceeded budget at frame {i}: {} > {target_budget}",
            engine.targets.pooled_bytes()
        );
        assert!(
            engine.cache.current_bytes() <= frame_cache_budget,
            "FrameCache exceeded byte budget at frame {i}: {} > {frame_cache_budget}",
            engine.cache.current_bytes()
        );
        assert!(
            engine.cache.len() <= FrameCache::MAX_ENTRIES,
            "FrameCache exceeded entry limit at frame {i}"
        );
        assert!(
            engine.cached_mesh_count() <= MAX_CACHED_MESHES,
            "mesh cache exceeded cap at frame {i}: {}",
            engine.cached_mesh_count()
        );
        assert!(
            engine.cached_texture_count() <= MAX_CACHED_TEXTURES,
            "texture cache exceeded cap at frame {i}: {}",
            engine.cached_texture_count()
        );
    }

    // Resize down to 160x90 and render two frames so `trim_unused` releases larger stale targets.
    for f in 0..2 {
        let _ = engine
            .render_frame_to_texture(&project, &plugins, f64::from(f), 160, 90)
            .expect("small viewport render should succeed");
    }
    assert!(
        engine.targets.pooled_bytes() <= 4 * 1024 * 1024,
        "stale high-resolution render targets were not trimmed after downsize: {} bytes",
        engine.targets.pooled_bytes()
    );

    engine.invalidate_caches();
    assert!(engine.cache.is_empty());
    assert_eq!(engine.cached_mesh_count(), 0);
    assert_eq!(engine.cached_texture_count(), 0);
}
