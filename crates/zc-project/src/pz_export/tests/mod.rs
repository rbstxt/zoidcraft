mod features;
mod invariants;
mod roundtrip;

use std::fs;
use std::io::Read as _;
use std::path::PathBuf;

use flate2::read::GzDecoder;

use super::*;
use crate::effects::{
    create_custom_shader, create_effect_group, create_standard_effect, create_zoidium_effect,
};
use crate::model::{
    AssetSource, AudioClipData, CameraDofParams, CameraMotionBlurParams, CameraVibrateParams,
    CustomPbrMaterial, EulerOrder, FollowConstraint, Layer3DProps, MatcapMaterial,
    MotionBlurSettings, PbrPlusMaterial, SceneBackdrop, TextureMaterial, TrackToConstraint,
};
use crate::pz_import::import_pz_from_bytes;

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!("zc-pz-export-test-{}", Uuid::new_v4()));
        fs::create_dir_all(&path).expect("create test dir");
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn inspect_tar_entries(pz_bytes: &[u8]) -> BTreeMap<String, Vec<u8>> {
    let decoder = GzDecoder::new(pz_bytes);
    let mut archive = tar::Archive::new(decoder);
    let mut out = BTreeMap::new();
    for entry in archive.entries().expect("entries") {
        let mut e = entry.expect("entry");
        let name = e.path().expect("path").to_string_lossy().into_owned();
        let mut buf = Vec::new();
        e.read_to_end(&mut buf).expect("read entry");
        out.insert(name, buf);
    }
    out
}
