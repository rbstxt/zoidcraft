use super::*;

fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

#[test]
fn formula_micro_language_degree_trig_right_assoc_power_and_fail_closed() {
    let scope = FormulaScope {
        u: 0.25,
        t: 0.5,
        f: 1.0,
        id: 2.0,
        count: 5.0,
        falloff: 1.0,
        ..Default::default()
    };
    // Degree-based trig: sin(90) == 1, cos(180) == -1, atan2(1, 1) == 45.
    assert!(approx_eq(eval_formula("sin(90)", &scope), 1.0, 1e-12));
    assert!(approx_eq(eval_formula("cos(180)", &scope), -1.0, 1e-12));
    assert!(approx_eq(eval_formula("atan2(1, 1)", &scope), 45.0, 1e-12));
    assert!(approx_eq(eval_formula("asin(1)", &scope), 90.0, 1e-12));
    assert!(approx_eq(eval_formula("acos(0)", &scope), 90.0, 1e-12));
    // Right-associative power: 2^3^2 == 512.
    assert!(approx_eq(eval_formula("2^3^2", &scope), 512.0, 1e-12));
    // Positive Euclidean-like mod: mod(-1, 3) == 2.
    assert!(approx_eq(eval_formula("mod(-1, 3)", &scope), 2.0, 1e-12));
    // Formula from expresstext-30fps.json fixture: `60*t + 30*sin(u*360)`
    assert!(approx_eq(
        eval_formula("60*t + 30*sin(u*360)", &scope),
        60.0,
        1e-12
    ));
    // Ternary, comparisons, min/max, constants.
    assert!(approx_eq(
        eval_formula("id >= 2 ? max(1, 4, 2) : min(9, 3)", &scope),
        4.0,
        1e-12
    ));
    // Invalid syntax / unknown identifiers fail closed to 0.0.
    assert_eq!(eval_formula("window.alert(1)", &scope), 0.0);
    assert_eq!(eval_formula("unknownVar + 1", &scope), 0.0);
    assert_eq!(eval_formula("", &scope), 0.0);
}

#[test]
fn expresstext_30fps_golden_fixture_parity() {
    // Matches Zoidium `test/expresstext-plus-identity.test.js` & `test/fixtures/expresstext-30fps.json`:
    // Text root at [4, 3, 2], two letters with base positions [-8, 0, 0] and [6, 0, 0].
    let list = LetterList {
        total: 2,
        order: vec![0, 1],
    };
    let bases = [[-8.0, 0.0, 0.0], [6.0, 0.0, 0.0]];
    let root_pos = [4.0, 3.0, 2.0];
    let aux = EvalAux::default();

    // Case 2: Plain { plainPosition: [12, -4, 2], plainRotation: [0, 0, 0.3] }
    let mut plain_props = BTreeMap::new();
    plain_props.insert("plainPosition".into(), EtValue::Vec3([12.0, -4.0, 2.0]));
    plain_props.insert("plainRotation".into(), EtValue::Vec3([0.0, 0.0, 0.3]));
    let plain_layers = [EffectorLayer {
        kind: EffectorKind::Plain,
        props: &plain_props,
    }];
    let c0 = evaluate_stack(&plain_layers, 13.0, 0, &list, bases[0], &aux);
    assert!(approx_eq(
        root_pos[0] + bases[0][0] + c0.position[0],
        8.0,
        1e-12
    ));
    assert!(approx_eq(
        root_pos[1] + bases[0][1] + c0.position[1],
        -1.0,
        1e-12
    ));
    assert!(approx_eq(
        root_pos[2] + bases[0][2] + c0.position[2],
        4.0,
        1e-12
    ));
    assert!(approx_eq(c0.rotation[2], 0.3, 1e-12));

    // Case 4: Formula { formulaExpression: "60*t + 30*sin(u*360)", formulaPosition: [1, 0, 0] }
    // At frame 13 (30fps): t = 13/30, u=0 for letter 0, u=1 for letter 1 -> 60*(13/30) = 26.
    let mut formula_props = BTreeMap::new();
    formula_props.insert(
        "formulaExpression".into(),
        EtValue::Text("60*t + 30*sin(u*360)".into()),
    );
    formula_props.insert("formulaPosition".into(), EtValue::Vec3([1.0, 0.0, 0.0]));
    let formula_layers = [EffectorLayer {
        kind: EffectorKind::Formula,
        props: &formula_props,
    }];
    let cf0 = evaluate_stack(&formula_layers, 13.0, 0, &list, bases[0], &aux);
    let cf1 = evaluate_stack(&formula_layers, 13.0, 1, &list, bases[1], &aux);
    assert!(approx_eq(cf0.position[0], 26.0, 1e-12));
    assert!(approx_eq(cf1.position[0], 26.0, 1e-12));

    // Case 5: Shake { shakePosition: [2, 5, 1], shakeFrequency: 1.25 } at frame 13 (30fps).
    let mut shake_props = BTreeMap::new();
    shake_props.insert("shakePosition".into(), EtValue::Vec3([2.0, 5.0, 1.0]));
    shake_props.insert("shakeFrequency".into(), EtValue::Number(1.25));
    let shake_layers = [EffectorLayer {
        kind: EffectorKind::Shake,
        props: &shake_props,
    }];
    let cs = evaluate_stack(&shake_layers, 13.0, 0, &list, bases[0], &aux);
    let expected_s = (13.0 * (1.25 / 30.0) * 2.0 * PI).sin();
    assert!(approx_eq(cs.position[0], 2.0 * expected_s, 1e-12));
    assert!(approx_eq(cs.position[1], 5.0 * expected_s, 1e-12));
    assert!(approx_eq(cs.position[2], expected_s, 1e-12));

    // Case 3: Random { randomMode: 2 (Noise), randomAnimationSpeed: 50, randomSeed: 7 }
    // Verify exact determinism and range across frames 0, 13, 30.
    let mut rand_props = BTreeMap::new();
    rand_props.insert("randomMode".into(), EtValue::Number(2.0));
    rand_props.insert("randomAnimationSpeed".into(), EtValue::Number(50.0));
    rand_props.insert("randomSeed".into(), EtValue::Number(7.0));
    let rand_layers = [EffectorLayer {
        kind: EffectorKind::Random,
        props: &rand_props,
    }];
    let r0 = evaluate_stack(&rand_layers, 0.0, 0, &list, bases[0], &aux);
    let r13 = evaluate_stack(&rand_layers, 13.0, 0, &list, bases[0], &aux);
    assert_ne!(r0.position, r13.position);
    assert_eq!(
        r13,
        evaluate_stack(&rand_layers, 13.0, 0, &list, bases[0], &aux)
    );
}

#[test]
fn delay_even_blend_and_spring_chase_modes() {
    struct StepProps;
    impl EtPropReader for StepProps {
        fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
            match key {
                "plainPosition" => Some(EtValue::Vec3([
                    if frame >= 5.0 { 100.0 } else { 0.0 },
                    0.0,
                    0.0,
                ])),
                _ => None,
            }
        }
    }
    let step = StepProps;
    let list = LetterList {
        total: 1,
        order: vec![0],
    };
    let aux = EvalAux::default();

    // Blend mode (mode 1, strength 100%): at frame 5 catches up DELAY_BLEND_CATCH (0.25) * 100 = 25.
    let mut blend = BTreeMap::new();
    blend.insert("delayMode".into(), EtValue::Number(1.0));
    blend.insert("delayStrength".into(), EtValue::Number(100.0));
    let blend_layers = [
        EffectorLayer {
            kind: EffectorKind::Plain,
            props: &step,
        },
        EffectorLayer {
            kind: EffectorKind::Delay,
            props: &blend,
        },
    ];
    let b5 = evaluate_stack(&blend_layers, 5.0, 0, &list, [0.0, 0.0, 0.0], &aux);
    assert!(approx_eq(b5.position[0], 25.0, 1e-12));

    // Even mode (mode 0, strength 100%): pace = 100 / 2.5 = 40 per frame.
    let mut even = BTreeMap::new();
    even.insert("delayMode".into(), EtValue::Number(0.0));
    even.insert("delayStrength".into(), EtValue::Number(100.0));
    let even_layers = [
        EffectorLayer {
            kind: EffectorKind::Plain,
            props: &step,
        },
        EffectorLayer {
            kind: EffectorKind::Delay,
            props: &even,
        },
    ];
    let e5 = evaluate_stack(&even_layers, 5.0, 0, &list, [0.0, 0.0, 0.0], &aux);
    assert!(approx_eq(e5.position[0], 40.0, 1e-12));

    // Spring mode (mode 2, strength 100%): overshoots 100 on later frames.
    let mut spring = BTreeMap::new();
    spring.insert("delayMode".into(), EtValue::Number(2.0));
    spring.insert("delayStrength".into(), EtValue::Number(100.0));
    let spring_layers = [
        EffectorLayer {
            kind: EffectorKind::Plain,
            props: &step,
        },
        EffectorLayer {
            kind: EffectorKind::Delay,
            props: &spring,
        },
    ];
    let max_spring = (5..=25)
        .map(|f| {
            evaluate_stack(&spring_layers, f as f64, 0, &list, [0.0, 0.0, 0.0], &aux).position[0]
        })
        .fold(0.0_f64, f64::max);
    assert!(
        max_spring > 100.0,
        "spring mode should overshoot 100, got {max_spring}"
    );
}

#[test]
fn repeater_linear_random_echo_fade_and_step_offsets() {
    let list = LetterList {
        total: 3,
        order: vec![0, 1, 2],
    };
    let aux = EvalAux::default();

    // 1. Linear repeater with rotation/scale step & opacity fade.
    let mut lin = BTreeMap::new();
    lin.insert("repeaterEnable".into(), EtValue::Number(1.0));
    lin.insert("repeaterMode".into(), EtValue::Number(0.0));
    lin.insert("repeaterCopies".into(), EtValue::Number(4.0));
    lin.insert("repeaterOffset".into(), EtValue::Vec3([5.0, 2.0, -1.0]));
    lin.insert("repeaterRotation".into(), EtValue::Vec3([0.0, 0.0, 0.25]));
    lin.insert("repeaterScale".into(), EtValue::Vec3([0.9, 0.9, 1.0]));
    lin.insert("repeaterFade".into(), EtValue::Number(80.0));

    let layers = [EffectorLayer {
        kind: EffectorKind::Repeater,
        props: &lin,
    }];
    let instances =
        evaluate_expresstext_repeater_instances(&layers, 10.0, 1, &list, [0.0, 0.0, 0.0], &aux);
    assert_eq!(instances.len(), 4);
    assert_eq!(instances[0].copy_index, 1);
    assert!(approx_eq(instances[0].position_offset[0], 5.0, 1e-12));
    assert!(approx_eq(instances[3].position_offset[0], 20.0, 1e-12));
    assert!(approx_eq(instances[3].position_offset[1], 8.0, 1e-12));
    assert!(approx_eq(instances[3].rotation_offset[2], 1.0, 1e-12));
    assert!(approx_eq(instances[1].scale_multiplier[0], 0.81, 1e-12));
    // Slot 1/4 fade 80% -> 1 - 0.8*0.25 = 0.8; slot 4/4 -> 1 - 0.8 = 0.2.
    assert!(approx_eq(instances[0].opacity, 0.8, 1e-12));
    assert!(approx_eq(instances[3].opacity, 0.2, 1e-12));

    // 2. Explicit opacity start/end overrides default fade.
    lin.insert("repeaterOpacityStart".into(), EtValue::Number(0.9));
    lin.insert("repeaterOpacityEnd".into(), EtValue::Number(0.1));
    let layers_op = [EffectorLayer {
        kind: EffectorKind::Repeater,
        props: &lin,
    }];
    let op_inst =
        evaluate_expresstext_repeater_instances(&layers_op, 10.0, 1, &list, [0.0, 0.0, 0.0], &aux);
    assert!(approx_eq(op_inst[3].opacity, 0.1, 1e-12));
    assert!(approx_eq(op_inst[1].opacity, 0.5, 1e-12));

    // 3. Random mode determinism & range bounds.
    let mut rnd = BTreeMap::new();
    rnd.insert("repeaterMode".into(), EtValue::Number(1.0));
    rnd.insert("repeaterCopies".into(), EtValue::Number(5.0));
    rnd.insert(
        "repeaterRangeMin".into(),
        EtValue::Vec3([-10.0, -20.0, -5.0]),
    );
    rnd.insert("repeaterRangeMax".into(), EtValue::Vec3([10.0, 20.0, 5.0]));
    rnd.insert("repeaterSeed".into(), EtValue::Number(42.0));
    let rnd_layers = [EffectorLayer {
        kind: EffectorKind::Repeater,
        props: &rnd,
    }];
    let r_a =
        evaluate_expresstext_repeater_instances(&rnd_layers, 0.0, 0, &list, [0.0, 0.0, 0.0], &aux);
    let r_b =
        evaluate_expresstext_repeater_instances(&rnd_layers, 0.0, 0, &list, [0.0, 0.0, 0.0], &aux);
    assert_eq!(r_a, r_b);
    for inst in &r_a {
        assert!((-10.0..=10.0).contains(&inst.position_offset[0]));
        assert!((-20.0..=20.0).contains(&inst.position_offset[1]));
        assert!((-5.0..=5.0).contains(&inst.position_offset[2]));
    }

    // 4. Echo mode trails animated velocity `-v * lag * slot`.
    struct MovingPlain;
    impl EtPropReader for MovingPlain {
        fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
            if key == "plainPosition" {
                Some(EtValue::Vec3([frame * 4.0, 0.0, 0.0]))
            } else {
                None
            }
        }
    }
    let moving = MovingPlain;
    let mut echo = BTreeMap::new();
    echo.insert("repeaterMode".into(), EtValue::Number(2.0));
    echo.insert("repeaterCopies".into(), EtValue::Number(3.0));
    echo.insert("repeaterLag".into(), EtValue::Number(2.5));
    let echo_layers = [
        EffectorLayer {
            kind: EffectorKind::Plain,
            props: &moving,
        },
        EffectorLayer {
            kind: EffectorKind::Repeater,
            props: &echo,
        },
    ];
    let echo_inst = evaluate_expresstext_repeater_instances(
        &echo_layers,
        10.0,
        0,
        &list,
        [0.0, 0.0, 0.0],
        &aux,
    );
    assert_eq!(echo_inst.len(), 3);
    // Velocity = [4, 0, 0], lag = 2.5 -> slot 1 offset = -10, slot 3 offset = -30.
    assert!(approx_eq(echo_inst[0].position_offset[0], -10.0, 1e-12));
    assert!(approx_eq(echo_inst[2].position_offset[0], -30.0, 1e-12));
    assert!(approx_eq(echo_inst[2].controls.position[0], 10.0, 1e-12));
}

#[test]
fn voronoi_halton_shatter_is_deterministic_and_conserves_triangles() {
    // Halton sequence exact base-2 and base-3 values.
    assert!(approx_eq(halton_sequence(1.0, 2.0), 0.5, 1e-12));
    assert!(approx_eq(halton_sequence(2.0, 2.0), 0.25, 1e-12));
    assert!(approx_eq(halton_sequence(3.0, 2.0), 0.75, 1e-12));
    assert!(approx_eq(halton_sequence(1.0, 3.0), 1.0 / 3.0, 1e-12));
    assert!(approx_eq(halton_sequence(2.0, 3.0), 2.0 / 3.0, 1e-12));

    // Build a 4x4 grid of 32 triangles in [0..4, 0..4, 0..1].
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut uvs = Vec::new();
    for gy in 0..4 {
        for gx in 0..4 {
            let x0 = gx as f32;
            let x1 = (gx + 1) as f32;
            let y0 = gy as f32;
            let y1 = (gy + 1) as f32;
            let z = ((gx + gy) % 2) as f32 * 0.5;
            // Tri 1
            positions.extend_from_slice(&[x0, y0, z, x1, y0, z, x1, y1, z]);
            normals.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
            uvs.extend_from_slice(&[x0 / 4.0, y0 / 4.0, x1 / 4.0, y0 / 4.0, x1 / 4.0, y1 / 4.0]);
            // Tri 2
            positions.extend_from_slice(&[x0, y0, z, x1, y1, z, x0, y1, z]);
            normals.extend_from_slice(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
            uvs.extend_from_slice(&[x0 / 4.0, y0 / 4.0, x1 / 4.0, y1 / 4.0, x0 / 4.0, y1 / 4.0]);
        }
    }

    let params = VoronoiGlyphParams {
        cells: 8,
        seed: 3.25,
        scatter: 1.5,
        distance: 10.0,
        spin: [0.5, 1.0, 0.25],
        gravity: 2.0,
        randomness: 1.0,
        scale: 0.85,
        weight: 1.0,
        letter_index: 2,
    };

    let chunks_a = shatter_glyph_triangles(&positions, &normals, &uvs, None, &params);
    let chunks_b = shatter_glyph_triangles(&positions, &normals, &uvs, None, &params);
    assert_eq!(chunks_a, chunks_b);
    assert!(chunks_a.len() >= 4 && chunks_a.len() <= 8);

    // Every input triangle must appear in exactly one Voronoi chunk.
    let total_tris: usize = chunks_a.iter().map(|c| c.positions.len() / 9).sum();
    assert_eq!(total_tris, 32);
    for chunk in &chunks_a {
        assert_eq!(chunk.normals.len(), chunk.positions.len());
        assert_eq!(chunk.uvs.len(), (chunk.positions.len() / 9) * 6);
        assert!(approx_eq(chunk.scale[0], 0.85, 1e-12));
        let baked = apply_chunk_pose_to_positions(chunk);
        assert_eq!(baked.len(), chunk.positions.len());
        assert!(baked.iter().all(|v| v.is_finite()));
    }

    // Stack-level Voronoi evaluation with field falloff.
    let mut vor = BTreeMap::new();
    vor.insert("voronoiEnable".into(), EtValue::Number(1.0));
    vor.insert("voronoiChunks".into(), EtValue::Number(6.0));
    vor.insert("voronoiExplode".into(), EtValue::Number(12.0));
    vor.insert("voronoiSpin".into(), EtValue::Vec3([0.2, 0.4, 0.6]));
    vor.insert("voronoiSeed".into(), EtValue::Number(9.0));
    vor.insert("voronoiField".into(), EtValue::Number(1.0)); // Linear X ramp
    vor.insert(
        "voronoiFieldPosition".into(),
        EtValue::Vec3([0.0, 0.0, 0.0]),
    );
    vor.insert(
        "voronoiFieldScale".into(),
        EtValue::Vec3([100.0, 100.0, 100.0]),
    );
    let layers = [EffectorLayer {
        kind: EffectorKind::Voronoi,
        props: &vor,
    }];
    let list = LetterList {
        total: 2,
        order: vec![0, 1],
    };
    let evaluated = evaluate_expresstext_voronoi_params(&layers, 0.0, 0, &list, [0.0, 0.0, 0.0])
        .expect("voronoi should be active at linear midpoint");
    assert_eq!(evaluated.cells, 6);
    // Midpoint of linear field -> weight 0.5 -> explode 6.0.
    assert!(approx_eq(evaluated.distance, 6.0, 1e-12));
}

#[test]
fn parses_express_properties_json_and_letters_mode() {
    let json = r#"{
        "type": "zoidium:expresstext-plus/expresstext",
        "expressProperties": {
            "rotationOrder": "YXZ",
            "lettersMode": 0,
            "lettersVersion": 1
        },
        "customProperties": [
            {
                "type": {
                    "name": "Repeater",
                    "objects": [
                        { "effectorKey": "repeaterEnable" },
                        { "effectorKey": "repeaterCopies" },
                        { "effectorKey": "repeaterOffset" }
                    ]
                },
                "objects": [
                    { "value": 1 },
                    { "value": 5 },
                    { "value": [12, 0, -4] }
                ]
            },
            {
                "type": { "name": "Voronoi" },
                "objects": [
                    { "effectorKey": "voronoiEnable", "value": 1 },
                    { "effectorKey": "voronoiChunks", "value": 12 },
                    { "effectorKey": "voronoiExplode", "value": 25.0 }
                ]
            }
        ]
    }"#;
    let cfg = parse_express_text_config_str(json).expect("config must parse");
    assert_eq!(cfg.letters_mode, LettersMode::PerText);
    assert_eq!(cfg.rotation_order, "YXZ");
    assert_eq!(cfg.effectors.len(), 2);
    assert_eq!(cfg.effectors[0].kind, EffectorKind::Repeater);
    assert_eq!(
        cfg.effectors[0].properties.get("repeaterCopies"),
        Some(&EtValue::Number(5.0))
    );
    assert_eq!(
        cfg.effectors[0].properties.get("repeaterOffset"),
        Some(&EtValue::Vec3([12.0, 0.0, -4.0]))
    );
    assert_eq!(cfg.effectors[1].kind, EffectorKind::Voronoi);
    assert_eq!(
        cfg.effectors[1].properties.get("voronoiChunks"),
        Some(&EtValue::Number(12.0))
    );

    // Pre-scope legacy container with content defaults to AcrossTexts.
    let legacy = serde_json::json!({
        "expressProperties": {
            "plainEnable": 1,
            "plainPosition": [4, 5, 6]
        },
        "objects": [{ "type": "text" }]
    });
    let legacy_cfg = parse_express_text_config(&legacy).expect("legacy must parse");
    assert_eq!(legacy_cfg.letters_mode, LettersMode::AcrossTexts);
    assert_eq!(legacy_cfg.effectors.len(), 1);
    assert_eq!(legacy_cfg.effectors[0].kind, EffectorKind::Plain);

    // Verify PerText vs AcrossTexts letter run helper.
    let (per_list, per_start) = build_letter_run_for_text(LettersMode::PerText, &[3, 5, 2], 1);
    assert_eq!(per_list.total, 5);
    assert_eq!(per_start, 0);
    let (acr_list, acr_start) = build_letter_run_for_text(LettersMode::AcrossTexts, &[3, 5, 2], 1);
    assert_eq!(acr_list.total, 10);
    assert_eq!(acr_start, 3);
}

#[test]
fn formula_cache_static_keys_chunk_matrix_and_hoisted_delay_sections() {
    // 1. FormulaCache cap & stack-evaluated 1/2/variadic math functions.
    let mut cache = FormulaCache::default();
    let compiled = Rc::new(CompiledFormula::compile("sin(90)").unwrap());
    for i in 0..(FORMULA_CACHE_CAP + 16) {
        cache.insert(&format!("f{i}"), Rc::clone(&compiled));
    }
    assert_eq!(cache.map.len(), FORMULA_CACHE_CAP);
    assert!(cache.get("f0").is_none());

    let scope = FormulaScope {
        u: 0.5,
        t: 1.0,
        f: 2.0,
        ..FormulaScope::default()
    };
    assert!(
        (eval_formula("sin(90) + pow(2, 3) + min(5, 2, 9) + max(1, 4, 3)", &scope) - 15.0).abs()
            < 1e-12
    );
    assert!((eval_formula("mod(-1, 4)", &scope) - 3.0).abs() < 1e-12);

    // 2. All 11 EffectorKind static prop keys match dynamic format! strings.
    for kind in EffectorKind::ALL {
        let p = kind.as_str();
        let k = kind.prop_keys();
        assert_eq!(k.letters, format!("{p}Letters"));
        assert_eq!(k.delay_order, format!("{p}DelayOrder"));
        assert_eq!(k.seed, format!("{p}Seed"));
        assert_eq!(k.delay, format!("{p}Delay"));
        assert_eq!(k.strength, format!("{p}Strength"));
        assert_eq!(k.field, format!("{p}Field"));
        assert_eq!(k.field_position, format!("{p}FieldPosition"));
        assert_eq!(k.field_scale, format!("{p}FieldScale"));
    }

    // 3. apply_chunk_pose_to_positions precomputed 3x3 matrix matches rotate_euler_xyz per vertex.
    let chunk = ShatteredVoronoiChunk {
        cell_index: 0,
        positions: vec![1.0, 2.0, 3.0, -4.5, 0.25, 7.5, 2.0, -1.5, 0.0],
        normals: Vec::new(),
        uvs: Vec::new(),
        center: [0.5, -0.25, 1.25],
        offset: [3.0, -2.0, 4.5],
        rotation: [0.43, -1.17, 2.05],
        scale: [1.5, 0.75, 2.25],
        signs: [1.0, -1.0, 1.0],
    };
    let fast_baked = apply_chunk_pose_to_positions(&chunk);
    for (i, v) in chunk.positions.as_chunks::<3>().0.iter().enumerate() {
        let rel = [
            (f64::from(v[0]) - chunk.center[0]) * chunk.scale[0],
            (f64::from(v[1]) - chunk.center[1]) * chunk.scale[1],
            (f64::from(v[2]) - chunk.center[2]) * chunk.scale[2],
        ];
        let spun = rotate_euler_xyz(rel, chunk.rotation);
        for axis in 0..3 {
            let expected = (chunk.center[axis] + spun[axis] + chunk.offset[axis]) as f32;
            let actual = fast_baked[i * 3 + axis];
            assert!(
                (actual - expected).abs() <= 1e-5,
                "vertex {i} axis {axis}: actual={actual}, expected={expected}"
            );
        }
    }

    // 4. Hoisted sections_at for static reader matches dynamic reader in evaluate_stack.
    struct DynamicWrap(BTreeMap<String, EtValue>);
    impl EtPropReader for DynamicWrap {
        fn read(&self, key: &str, frame: f64) -> Option<EtValue> {
            self.0.read(key, frame)
        }
    }
    let mut plain_map = BTreeMap::new();
    plain_map.insert(
        "plainPosition".to_string(),
        EtValue::Vec3([10.0, 5.0, -2.0]),
    );
    plain_map.insert("plainLetters".to_string(), EtValue::Text("1-3".to_string()));
    let mut delay_map = BTreeMap::new();
    delay_map.insert("delayMode".to_string(), EtValue::Number(2.0));
    delay_map.insert("delayStrength".to_string(), EtValue::Number(60.0));

    let dyn_plain = DynamicWrap(plain_map.clone());
    let dyn_delay = DynamicWrap(delay_map.clone());

    let static_layers = [
        EffectorLayer {
            kind: EffectorKind::Plain,
            props: &plain_map,
        },
        EffectorLayer {
            kind: EffectorKind::Delay,
            props: &delay_map,
        },
    ];
    let dynamic_layers = [
        EffectorLayer {
            kind: EffectorKind::Plain,
            props: &dyn_plain,
        },
        EffectorLayer {
            kind: EffectorKind::Delay,
            props: &dyn_delay,
        },
    ];
    let list = LetterList {
        total: 4,
        order: vec![0, 1, 2, 3],
    };
    let aux = EvalAux::default();
    let res_static = evaluate_stack(&static_layers, 12.5, 1, &list, [0.0; 3], &aux);
    let res_dyn = evaluate_stack(&dynamic_layers, 12.5, 1, &list, [0.0; 3], &aux);
    assert_eq!(res_static, res_dyn);
}
