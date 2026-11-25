//! Shape+ document, geometry, rasterizer, SVG and mask regression tests.

use super::*;

fn count_opaque_pixels(rgba: &[u8]) -> usize {
    rgba.as_chunks::<4>().0.iter().filter(|p| p[3] > 0).count()
}

fn pixel_at(rgba: &[u8], w: u32, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * w + x) * 4) as usize;
    [rgba[i], rgba[i + 1], rgba[i + 2], rgba[i + 3]]
}

fn make_doc(contents: Value) -> Value {
    json!({
        "schemaVersion": 1,
        "renderVersion": 1,
        "nextId": 100,
        "contents": contents
    })
}

#[test]
fn canonical_starter_and_crud_helpers() {
    let mut doc = default_shape_plus_payload();
    assert_eq!(doc["schemaVersion"], 1);
    assert_eq!(doc["renderVersion"], 1);
    assert_eq!(doc["nextId"], 4);
    assert_eq!(doc["contents"][0]["id"], "s1");
    assert_eq!(doc["contents"][0]["contents"][0]["id"], "s2");
    assert_eq!(doc["contents"][0]["contents"][1]["id"], "s3");
    validate_shape_plus_payload(&doc).expect("starter payload must validate");

    let parsed = parse_shape_plus_document(&doc);
    assert_eq!(parsed.next_id, 4);
    assert_eq!(parsed.contents.len(), 1);
    assert_eq!(parsed.contents[0].contents.len(), 2);

    let stroke_id = add_shape_plus_item(&mut doc, Some("s1"), "stroke").expect("add stroke");
    assert_eq!(stroke_id, "s4");
    assert_eq!(doc["contents"][0]["contents"].as_array().unwrap().len(), 3);

    assert!(reorder_shape_plus_item(&mut doc, &stroke_id, 0).expect("reorder"));
    assert_eq!(doc["contents"][0]["contents"][0]["id"], stroke_id);

    let dup_id = duplicate_shape_plus_item(&mut doc, "s1")
        .expect("dup")
        .expect("dup id");
    assert_ne!(dup_id, "s1");
    validate_shape_plus_payload(&doc).expect("duplicated payload must keep unique IDs");

    assert!(remove_shape_plus_item(&mut doc, &dup_id).expect("remove"));
    assert_eq!(doc["contents"].as_array().unwrap().len(), 1);
}

#[test]
fn primitives_gradient_fill_and_gradient_stroke() {
    // Gradient fill across a 160x160 rectangle: left red -> right blue.
    let grad_doc = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            { "id": "r1", "kind": "rectangle", "properties": { "size": [160.0, 100.0], "roundness": 12.0 } },
            {
                "id": "gf1", "kind": "gradientFill",
                "properties": {
                    "gradientType": "linear",
                    "startPoint": [-80.0, 0.0],
                    "endPoint": [80.0, 0.0],
                    "stops": [
                        { "position": 0.0, "color": [1.0, 0.0, 0.0, 1.0] },
                        { "position": 1.0, "color": [0.0, 0.0, 1.0, 1.0] }
                    ]
                }
            }
        ]
    }]));
    let rgba = rasterize_shape_plus(&grad_doc, 200, 200, [200, 200], 0.0, 30.0, None);
    let left_px = pixel_at(&rgba, 200, 35, 100);
    let right_px = pixel_at(&rgba, 200, 165, 100);
    assert!(
        left_px[0] > 200 && left_px[2] < 60,
        "expected red left, got {left_px:?}"
    );
    assert!(
        right_px[2] > 200 && right_px[0] < 60,
        "expected blue right, got {right_px:?}"
    );

    // Ellipse, polystar (with roundness), and cubic path with gradientStroke.
    let multi_doc = make_doc(json!([
        {
            "id": "g_e", "kind": "group", "properties": { "position": [-50.0, 0.0] },
            "contents": [
                { "id": "e1", "kind": "ellipse", "properties": { "size": [60.0, 40.0] } },
                { "id": "f1", "kind": "fill", "properties": { "color": [0.0, 1.0, 0.0] } }
            ]
        },
        {
            "id": "g_p", "kind": "group", "properties": { "position": [50.0, 0.0] },
            "contents": [
                {
                    "id": "ps1", "kind": "polystar",
                    "properties": { "mode": "star", "points": 5, "outerRadius": 40.0, "innerRadius": 18.0, "outerRoundness": 30.0, "innerRoundness": 20.0 }
                },
                {
                    "id": "gs1", "kind": "gradientStroke",
                    "properties": {
                        "gradientType": "radial",
                        "startPoint": [0.0, 0.0],
                        "endPoint": [40.0, 0.0],
                        "highlightLength": 35.0,
                        "width": 6.0,
                        "stops": [
                            { "position": 0.0, "color": "#ffff00" },
                            { "position": 1.0, "color": "#ff00ff" }
                        ]
                    }
                }
            ]
        }
    ]));
    let rgba2 = rasterize_shape_plus(&multi_doc, 200, 200, [200, 200], 0.0, 30.0, None);
    assert!(count_opaque_pixels(&rgba2) > 2000);
    // Check green ellipse center at (-50, 0) -> pixel (50, 100).
    let e_center = pixel_at(&rgba2, 200, 50, 100);
    assert!(e_center[1] > 240 && e_center[0] == 0);
}

#[test]
fn trim_degrees_offset_90_equals_quarter_turn() {
    // Square perimeter 400: start=0, end=25 draws exactly 1 side.
    // offset=0 draws bottom edge (y=-50 in Y-up -> pixel y=150).
    // offset=90 degrees (= 0.25) rotates to right edge (x=+50 -> pixel x=150)!
    let make_trim = |offset_deg: f64| {
        make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                { "id": "t1", "kind": "trim", "properties": { "start": 0.0, "end": 25.0, "offset": offset_deg } },
                { "id": "s1", "kind": "stroke", "properties": { "color": [1.0, 1.0, 1.0], "width": 8.0 } }
            ]
        }]))
    };
    let r0 = rasterize_shape_plus(&make_trim(0.0), 200, 200, [200, 200], 0.0, 30.0, None);
    let r90 = rasterize_shape_plus(&make_trim(90.0), 200, 200, [200, 200], 0.0, 30.0, None);
    // At offset=0: bottom edge (x=100, y=150) is lit, right edge (x=150, y=100) is dark.
    assert!(pixel_at(&r0, 200, 100, 150)[3] > 200);
    assert_eq!(pixel_at(&r0, 200, 150, 100)[3], 0);
    // At offset=90 deg: right edge (x=150, y=100) is lit, bottom edge (x=100, y=150) is dark.
    assert!(pixel_at(&r90, 200, 150, 100)[3] > 200);
    assert_eq!(pixel_at(&r90, 200, 100, 150)[3], 0);
}

#[test]
fn stroke_caps_dashes_taper_and_wave() {
    // Horizontal line from x=-40 to x=+40 at y=0 (pixels x=60..140, y=100) with width=20.
    let line_stroke = |cap: &str, extra: Value| {
        let mut props = json!({
            "color": [1.0, 1.0, 1.0],
            "width": 20.0,
            "cap": cap
        });
        if let (Some(dst), Some(src)) = (props.as_object_mut(), extra.as_object()) {
            for (k, v) in src {
                dst.insert(k.clone(), v.clone());
            }
        }
        make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                {
                    "id": "p1", "kind": "path", "vertexIds": ["v1", "v2"],
                    "properties": {
                        "closed": 0,
                        "path": [-40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    }
                },
                { "id": "s1", "kind": "stroke", "properties": props }
            ]
        }]))
    };

    let butt = rasterize_shape_plus(
        &line_stroke("butt", json!({})),
        200,
        200,
        [200, 200],
        0.0,
        30.0,
        None,
    );
    let square = rasterize_shape_plus(
        &line_stroke("square", json!({})),
        200,
        200,
        [200, 200],
        0.0,
        30.0,
        None,
    );
    let round = rasterize_shape_plus(
        &line_stroke("round", json!({})),
        200,
        200,
        [200, 200],
        0.0,
        30.0,
        None,
    );

    // Pixel (145, 100) is 5px past endpoint x=140: butt=0, square>200, round>200.
    assert_eq!(pixel_at(&butt, 200, 145, 100)[3], 0);
    assert!(pixel_at(&square, 200, 145, 100)[3] > 200);
    assert!(pixel_at(&round, 200, 145, 100)[3] > 200);
    // Corner pixel (148, 92) is inside square cap, outside round cap (dist from (140,100) = hypot(8,8)=11.3 > 10).
    assert!(pixel_at(&square, 200, 148, 92)[3] > 200);
    assert_eq!(pixel_at(&round, 200, 148, 92)[3], 0);

    // Dashes: dash1=20, gap1=20 -> gap at x=80..100 -> pixel (90, 100) must be empty!
    let dashed = rasterize_shape_plus(
        &line_stroke(
            "butt",
            json!({ "dashPairs": 1, "dash1": 20.0, "gap1": 20.0 }),
        ),
        200,
        200,
        [200, 200],
        0.0,
        30.0,
        None,
    );
    assert!(pixel_at(&dashed, 200, 70, 100)[3] > 200);
    assert_eq!(pixel_at(&dashed, 200, 90, 100)[3], 0);

    // Taper: startLength=50, startWidth=0 -> thinner near left end than center.
    let tapered = rasterize_shape_plus(
        &line_stroke(
            "butt",
            json!({ "startLength": 50.0, "startWidth": 0.0, "waveAmount": 25.0, "waveCycles": 2.0 }),
        ),
        200,
        200,
        [200, 200],
        0.0,
        30.0,
        None,
    );
    assert_eq!(pixel_at(&tapered, 200, 62, 93)[3], 0);
    assert!(pixel_at(&tapered, 200, 120, 96)[3] > 150);
}

#[test]
fn repeater_wiggle_transform_and_deformers_all_produce_distinct_outputs() {
    // Repeater: 3 copies shifted by +50 on X with opacity fade.
    let rep = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            { "id": "r1", "kind": "rectangle", "properties": { "size": [24.0, 24.0], "position": [-50.0, 0.0] } },
            { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } },
            {
                "id": "rep1", "kind": "repeater",
                "properties": { "copies": 3.0, "position": [50.0, 0.0], "startOpacity": 1.0, "endOpacity": 0.4 }
            }
        ]
    }]));
    let rep_px = rasterize_shape_plus(&rep, 200, 200, [200, 200], 0.0, 30.0, None);
    assert!(pixel_at(&rep_px, 200, 50, 100)[3] > 240);
    assert!(pixel_at(&rep_px, 200, 100, 100)[3] > 150);
    assert!(pixel_at(&rep_px, 200, 150, 100)[3] > 80 && pixel_at(&rep_px, 200, 150, 100)[3] < 130);

    // Baseline square vs roundCorners, twist, zigZag, puckerBloat, wigglePaths, wiggleTransform, offsetPaths.
    let base_doc = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
            { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
        ]
    }]));
    let base_px = rasterize_shape_plus(&base_doc, 200, 200, [200, 200], 10.0, 30.0, None);

    let modifiers = [
        json!({ "id": "m1", "kind": "roundCorners", "properties": { "radius": 30.0 } }),
        json!({ "id": "m2", "kind": "twist", "properties": { "angle": 1.2, "center": [0.0, 0.0] } }),
        json!({ "id": "m3", "kind": "zigZag", "properties": { "size": 12.0, "ridges": 4.0, "points": "smooth" } }),
        json!({ "id": "m4", "kind": "puckerBloat", "properties": { "amount": 45.0 } }),
        json!({ "id": "m5", "kind": "wigglePaths", "properties": { "size": 15.0, "detail": 8.0, "points": "smooth", "seed": 7.0 } }),
        json!({ "id": "m6", "kind": "wiggleTransform", "properties": { "position": [35.0, 25.0], "rotation": 0.4, "seed": 3.0 } }),
        json!({ "id": "m7", "kind": "offsetPaths", "properties": { "amount": 18.0, "join": "round" } }),
        json!({ "id": "m8", "kind": "offsetPaths", "properties": { "amount": -20.0, "join": "miter" } }),
    ];

    for m in modifiers {
        let kind = m["kind"].as_str().unwrap();
        let doc = make_doc(json!([{
            "id": "g1", "kind": "group", "properties": {},
            "contents": [
                { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                m,
                { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
            ]
        }]));
        let px = rasterize_shape_plus(&doc, 200, 200, [200, 200], 10.0, 30.0, None);
        assert!(
            count_opaque_pixels(&px) > 1000,
            "{kind} produced blank render"
        );
        assert_ne!(px, base_px, "{kind} had no visible effect vs baseline");
    }
}

#[test]
fn merge_paths_subtract_cuts_real_hole() {
    let doc = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            { "id": "r_outer", "kind": "rectangle", "properties": { "size": [140.0, 140.0] } },
            { "id": "r_inner", "kind": "rectangle", "properties": { "size": [60.0, 60.0] } },
            { "id": "mp1", "kind": "mergePaths", "properties": { "mode": "subtract" } },
            { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 0.5, 0.0], "fillRule": "nonzero" } }
        ]
    }]));
    let rgba = rasterize_shape_plus(&doc, 200, 200, [200, 200], 0.0, 30.0, None);
    // Center (100, 100) must be a transparent hole; ring at (50, 100) must be filled!
    assert_eq!(pixel_at(&rgba, 200, 100, 100)[3], 0);
    assert!(pixel_at(&rgba, 200, 50, 100)[3] > 240);
}

#[test]
fn isolated_group_opacity_paint_order_disabled_and_bezier_hold_eval() {
    // Isolated group opacity: two overlapping opaque rects inside group with opacity=0.5 ->
    // overlap pixel must have alpha ~128 (NOT ~191 from double blending!).
    let iso = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": { "opacity": 0.5 },
        "contents": [
            {
                "id": "g_a", "kind": "group", "properties": {},
                "contents": [
                    { "id": "ra", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                    { "id": "fa", "kind": "fill", "properties": { "color": [1.0, 0.0, 0.0] } }
                ]
            },
            {
                "id": "g_b", "kind": "group", "properties": {},
                "contents": [
                    { "id": "rb", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
                    { "id": "fb", "kind": "fill", "properties": { "color": [0.0, 0.0, 1.0] } }
                ]
            }
        ]
    }]));
    let iso_px = rasterize_shape_plus(&iso, 200, 200, [200, 200], 0.0, 30.0, None);
    let c = pixel_at(&iso_px, 200, 100, 100);
    assert!(
        (i32::from(c[3]) - 128).abs() <= 2,
        "expected isolated alpha ~128, got {c:?}"
    );

    // Paint order: below (default) vs above, plus disabled item skip.
    let order_below = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            { "id": "r1", "kind": "rectangle", "properties": { "size": [100.0, 100.0] } },
            { "id": "f_red", "kind": "fill", "properties": { "color": [1.0, 0.0, 0.0], "composite": "below" } },
            { "id": "f_blue", "kind": "fill", "properties": { "color": [0.0, 0.0, 1.0], "composite": "below" } },
            { "id": "f_off", "kind": "fill", "properties": { "enabled": 0, "color": [0.0, 1.0, 0.0], "composite": "above" } }
        ]
    }]));
    // In bottom-to-top default ("below"): f_red is painted AFTER f_blue -> red wins!
    let ob_px = rasterize_shape_plus(&order_below, 200, 200, [200, 200], 0.0, 30.0, None);
    assert!(pixel_at(&ob_px, 200, 100, 100)[0] > 240);

    let mut order_above = order_below.clone();
    order_above["contents"][0]["contents"][2]["properties"]["composite"] = json!("above");
    let oa_px = rasterize_shape_plus(&order_above, 200, 200, [200, 200], 0.0, 30.0, None);
    // f_blue has composite="above" -> blue draws on top of red!
    assert!(pixel_at(&oa_px, 200, 100, 100)[2] > 240);

    // Hold & Bezier keyframe + expression evaluation
    let hold_val = json!({
        "keyframes": [
            { "frame": 0, "value": 10.0, "tween": 0 },
            { "frame": 10, "value": 100.0, "tween": 1 }
        ]
    });
    assert!((eval_prop_f64(Some(&hold_val), 5.0, 0.0, None) - 10.0).abs() < 1e-6);

    let bez_val = json!({
        "keyframes": [
            { "frame": 0, "value": 0.0, "tween": 257, "bezier": [0.0, 1.0, 0.0, 1.0] },
            { "frame": 10, "value": 100.0, "tween": 1 }
        ]
    });
    assert!(eval_prop_f64(Some(&bez_val), 5.0, 0.0, None) > 75.0);

    let expr_val = json!({ "expression": "frame * 3 + 1" });
    let cb = |e: &str, f: f64| {
        if e == "frame * 3 + 1" {
            Some(f * 3.0 + 1.0)
        } else {
            None
        }
    };
    assert!((eval_prop_f64(Some(&expr_val), 4.0, 0.0, Some(&cb)) - 13.0).abs() < 1e-6);
}

#[test]
fn vector_text_and_text_animator_render() {
    let base = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            {
                "id": "t1", "kind": "text",
                "properties": { "sourceText": "HI\nOK", "fontSize": 60.0, "alignment": "center", "position": [0.0, 20.0] }
            },
            { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
        ]
    }]));
    let base_px = rasterize_shape_plus(&base, 256, 256, [256, 256], 0.0, 30.0, None);
    assert!(
        count_opaque_pixels(&base_px) > 500,
        "text must produce visible pixels"
    );

    let anim = make_doc(json!([{
        "id": "g1", "kind": "group", "properties": {},
        "contents": [
            {
                "id": "t1", "kind": "text",
                "properties": { "sourceText": "HI\nOK", "fontSize": 60.0, "alignment": "center", "position": [0.0, 20.0] }
            },
            {
                "id": "ta1", "kind": "textAnimator",
                "properties": {
                    "enabled_position": 1,
                    "position": [0.0, 30.0],
                    "enabled_fillColor": 1,
                    "fillColor": [1.0, 0.0, 0.0],
                    "enabled_opacity": 1,
                    "opacity": 0.5,
                    "selectorStart": 0.0,
                    "selectorEnd": 50.0,
                    "wigglyOn": 1,
                    "wigglySeed": 4.0
                }
            },
            { "id": "f1", "kind": "fill", "properties": { "color": [1.0, 1.0, 1.0] } }
        ]
    }]));
    let anim_px = rasterize_shape_plus(&anim, 256, 256, [256, 256], 15.0, 30.0, None);
    assert!(count_opaque_pixels(&anim_px) > 500);
    assert_ne!(anim_px, base_px);
}

#[test]
fn svg_import_and_export_roundtrip_renders() {
    let svg = r##"
        <svg viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
            <g transform="translate(10, 5)">
                <rect x="20" y="20" width="60" height="60" rx="8" fill="#ff3300" />
                <circle cx="140" cy="60" r="30" fill="rgb(0, 180, 255)" stroke="#ffffff" stroke-width="4" />
                <path d="M 30 140 A 40 30 0 0 1 150 140 L 150 170 Z" fill="green" />
            </g>
        </svg>
    "##;
    let payload = import_svg_to_shape_plus(svg).expect("SVG import must succeed");
    validate_shape_plus_payload(&payload).expect("imported SVG payload must be valid");
    let rgba = rasterize_shape_plus(&payload, 200, 200, [200, 200], 0.0, 30.0, None);
    assert!(
        count_opaque_pixels(&rgba) > 4000,
        "imported SVG must rasterize richly"
    );

    let d = export_shape_plus_path_to_svg_d(
        &[
            [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
            [50.0, 50.0, 0.0, -10.0, 0.0, 0.0],
            [0.0, 50.0, 0.0, 0.0, 0.0, 0.0],
        ],
        true,
    );
    assert!(d.starts_with("M0 0 C10 0 50 40 50 50 L0 50"));
    assert!(d.ends_with(" Z"));
}

#[test]
fn mask_rasterizer_aa_sdf_expansion_feather_invert_and_combine() {
    // Diagonal diamond to verify soft subpixel AA edges even at feather=0, expansion=0.
    let diamond = EvaluatedMask {
        vertices: vec![
            [100.0, 30.0, 0.0, 0.0, 0.0, 0.0],
            [170.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 170.0, 0.0, 0.0, 0.0, 0.0],
            [30.0, 100.0, 0.0, 0.0, 0.0, 0.0],
        ],
        closed: true,
        mode: "add".to_owned(),
        feather: 0.0,
        expansion: 0.0,
        opacity: 1.0,
        invert: false,
        enabled: true,
    };
    let aa = rasterize_masks_aa(std::slice::from_ref(&diamond), 200, 200);
    assert_eq!(aa[100 * 200 + 100], 255);
    assert_eq!(aa[0], 0);
    let partial_aa_pixels = aa.iter().filter(|&&v| v > 0 && v < 255).count();
    assert!(
        partial_aa_pixels > 40,
        "expected smooth subpixel AA edge pixels at feather=0, got {partial_aa_pixels}"
    );

    // Expansion grows coverage; negative expansion shrinks it.
    let mut expanded = diamond.clone();
    expanded.expansion = 12.0;
    let exp_plane = rasterize_masks_aa(&[expanded], 200, 200);
    assert!(
        exp_plane.iter().filter(|&&v| v > 128).count() > aa.iter().filter(|&&v| v > 128).count()
    );

    let mut contracted = diamond.clone();
    contracted.expansion = -12.0;
    let con_plane = rasterize_masks_aa(&[contracted], 200, 200);
    assert!(
        con_plane.iter().filter(|&&v| v > 128).count() < aa.iter().filter(|&&v| v > 128).count()
    );

    // Subtract hole + Intersect + Feather + Invert
    let hole = EvaluatedMask {
        vertices: vec![
            [80.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            [120.0, 80.0, 0.0, 0.0, 0.0, 0.0],
            [120.0, 120.0, 0.0, 0.0, 0.0, 0.0],
            [80.0, 120.0, 0.0, 0.0, 0.0, 0.0],
        ],
        closed: true,
        mode: "subtract".to_owned(),
        feather: 4.0,
        expansion: 0.0,
        opacity: 1.0,
        invert: false,
        enabled: true,
    };
    let sub_plane = rasterize_masks_aa(&[diamond.clone(), hole], 200, 200);
    assert!(
        sub_plane[100 * 200 + 100] < 10,
        "subtracted center must be cut out"
    );
    assert!(
        sub_plane[100 * 200 + 60] > 240,
        "outer diamond ring must remain opaque"
    );

    // Intersect with left half
    let left_half = EvaluatedMask {
        vertices: vec![
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [100.0, 200.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 200.0, 0.0, 0.0, 0.0, 0.0],
        ],
        mode: "intersect".to_owned(),
        ..EvaluatedMask::default()
    };
    let inter_plane = rasterize_masks_aa(&[diamond.clone(), left_half], 200, 200);
    assert!(inter_plane[100 * 200 + 70] > 240);
    assert_eq!(inter_plane[100 * 200 + 130], 0);

    // Post-opacity invert: outside becomes 255, inside at opacity=0.5 becomes ~128!
    let mut inv_mask = diamond;
    inv_mask.opacity = 0.5;
    inv_mask.invert = true;
    let inv_plane = rasterize_masks_aa(&[inv_mask], 200, 200);
    assert_eq!(inv_plane[0], 255);
    assert!((i32::from(inv_plane[100 * 200 + 100]) - 128).abs() <= 2);
}

#[test]
fn offscreen_negative_y_mask_with_expansion_does_not_hang_and_is_zero() {
    let offscreen_expanded = EvaluatedMask {
        vertices: vec![
            [20.0, -250.0, 0.0, 0.0, 0.0, 0.0],
            [180.0, -250.0, 0.0, 0.0, 0.0, 0.0],
            [180.0, -110.0, 0.0, 0.0, 0.0, 0.0],
            [20.0, -110.0, 0.0, 0.0, 0.0, 0.0],
        ],
        expansion: 4.0,
        ..EvaluatedMask::default()
    };
    let cov_exp = rasterize_masks_aa(std::slice::from_ref(&offscreen_expanded), 200, 200);
    assert_eq!(cov_exp.len(), 200 * 200);
    assert!(
        cov_exp.iter().all(|&b| b == 0),
        "offscreen y=-250..-110 expanded mask must produce all-zero coverage"
    );

    let mut offscreen_aa = offscreen_expanded;
    offscreen_aa.expansion = 0.0;
    let cov_aa = rasterize_masks_aa(&[offscreen_aa], 200, 200);
    assert_eq!(cov_aa.len(), 200 * 200);
    assert!(
        cov_aa.iter().all(|&b| b == 0),
        "offscreen y=-250..-110 AA mask must produce all-zero coverage"
    );
}

#[test]
fn symmetric_bowtie_mask_renders_both_lobes_and_lt3_masks_pass_through() {
    // Symmetric bowtie: opposing-winding left & right triangles have signed_area == 0.
    let bowtie_pts = [
        [40.0_f64, 40.0],
        [160.0, 160.0],
        [160.0, 40.0],
        [40.0, 160.0],
    ];
    assert!(signed_area(&bowtie_pts).abs() < 1e-9);

    let bowtie = EvaluatedMask {
        vertices: bowtie_pts
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, 0.0, 0.0, 0.0, 0.0])
            .collect(),
        ..EvaluatedMask::default()
    };
    let cov = rasterize_masks_aa(std::slice::from_ref(&bowtie), 200, 200);
    assert_eq!(
        cov[100 * 200 + 60],
        255,
        "left lobe of symmetric bowtie must be opaque"
    );
    assert_eq!(
        cov[100 * 200 + 140],
        255,
        "right lobe of symmetric bowtie must be opaque"
    );
    assert_eq!(cov[20 * 200 + 100], 0, "exterior must stay transparent");

    // Expanded bowtie must also render both lobes cleanly.
    let mut bowtie_exp = bowtie.clone();
    bowtie_exp.expansion = 2.0;
    let cov_exp = rasterize_masks_aa(&[bowtie_exp], 200, 200);
    assert_eq!(cov_exp[100 * 200 + 60], 255);
    assert_eq!(cov_exp[100 * 200 + 140], 255);

    // Degenerate <3 vertex masks (0, 1, 2 vertices) pass through as full-opaque (255).
    for verts in [
        vec![],
        vec![[50.0, 50.0, 0.0, 0.0, 0.0, 0.0]],
        vec![
            [50.0, 50.0, 0.0, 0.0, 0.0, 0.0],
            [150.0, 150.0, 0.0, 0.0, 0.0, 0.0],
        ],
    ] {
        let deg = EvaluatedMask {
            vertices: verts,
            ..EvaluatedMask::default()
        };
        let plane = rasterize_masks_aa(&[deg], 64, 64);
        assert!(
            plane.iter().all(|&b| b == 255),
            "<3 vertex degenerate mask must pass through full opaque"
        );
    }

    // An unfinished 2-vertex mask alongside a valid mask must not disturb the valid mask.
    let unfinished = EvaluatedMask {
        vertices: vec![
            [10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [20.0, 20.0, 0.0, 0.0, 0.0, 0.0],
        ],
        mode: "intersect".to_owned(),
        ..EvaluatedMask::default()
    };
    let combined = rasterize_masks_aa(&[bowtie, unfinished], 200, 200);
    assert_eq!(combined[100 * 200 + 60], 255);
    assert_eq!(combined[100 * 200 + 140], 255);
}
