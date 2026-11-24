#![forbid(unsafe_code)]

use zc_session::{Command, Session};

#[test]
fn rapid_600_edits_undo_redo_and_transactions_respect_memory_and_entry_bounds() {
    let mut session = Session::new();
    let max_entries = 48_usize;
    let max_bytes = 2 * 1024 * 1024; // 2 MiB
    session.history.set_limits(max_entries, max_bytes);

    let default_track_id = session.project.sequence.video_tracks[0].id.to_string();
    let default_clip_id = session.project.sequence.video_tracks[0].clips[0]
        .id
        .to_string();

    let assert_bounds = |s: &Session, step: usize| {
        assert!(
            s.history.undo_len() <= max_entries,
            "step {step}: undo_len {} > {max_entries}",
            s.history.undo_len()
        );
        assert!(
            s.history.redo_len() <= max_entries,
            "step {step}: redo_len {} > {max_entries}",
            s.history.redo_len()
        );
        assert!(
            s.history.total_bytes() <= max_bytes,
            "step {step}: total_bytes {} > {max_bytes}",
            s.history.total_bytes()
        );
    };

    for step in 0..600_usize {
        match step % 10 {
            0 => {
                session
                    .execute(Command::SetProjectName {
                        name: format!("Stress Project #{step}"),
                    })
                    .expect("set project name");
            }
            1 => {
                let start = f64::from((step % 20) as u32) * 20.0 + 350.0;
                let res = session.execute(Command::AddClip {
                    track_id: default_track_id.clone(),
                    kind: "scene".to_owned(),
                    name: Some(format!("Clip {step}")),
                    start,
                    length: 16.0,
                });
                assert!(res.is_ok(), "step {step}: AddClip failed: {res:?}");
            }
            2 => {
                // Split the last clip on the primary video track if long enough.
                if let Some(clip) = session.project.sequence.video_tracks[0]
                    .clips
                    .last()
                    .filter(|c| c.length >= 6.0)
                {
                    let clip_id = clip.id.to_string();
                    let mid = clip.start + (clip.length * 0.5).floor();
                    let _ = session.execute(Command::SplitClip {
                        clip_id,
                        at_frame: mid,
                    });
                }
            }
            3 => {
                if let Some(clip) = session.project.sequence.video_tracks[0].clips.last() {
                    let clip_id = clip.id.to_string();
                    let _ = session.execute(Command::DuplicateClip { clip_id });
                }
            }
            4 => {
                // Keep track clip count bounded while exercising delete + undo history.
                while session.project.sequence.video_tracks[0].clips.len() > 6 {
                    let clip_id = session.project.sequence.video_tracks[0]
                        .clips
                        .last()
                        .expect("non-empty clips")
                        .id
                        .to_string();
                    session
                        .execute(Command::DeleteClip { clip_id })
                        .expect("delete clip");
                }
            }
            5 => {
                let _ = session.execute(Command::AddObject3D {
                    clip_id: default_clip_id.clone(),
                    parent_id: None,
                    kind: "box".to_owned(),
                    name: Some(format!("Cube {step}")),
                });
            }
            6 => {
                // Live scrub transaction that commits as a single history entry.
                session
                    .begin_transaction(format!("Scrub #{step}"))
                    .expect("begin tx");
                for sub in 0..5_u32 {
                    session
                        .execute(Command::SetProjectName {
                            name: format!("Scrub {step}.{sub}"),
                        })
                        .expect("tx edit");
                }
                let committed = session.commit_transaction(true).expect("commit tx");
                assert!(committed);
            }
            7 => {
                // Live scrub transaction that cancels cleanly with zero history leak.
                let name_before = session.project.name.clone();
                let undo_before = session.history.undo_len();
                session
                    .begin_transaction(format!("Cancelled Scrub #{step}"))
                    .expect("begin cancel tx");
                session
                    .execute(Command::SetProjectName {
                        name: "Temporary Cancelled Name".to_owned(),
                    })
                    .expect("cancel tx edit");
                session.cancel_transaction().expect("cancel tx");
                assert_eq!(session.project.name, name_before);
                assert_eq!(session.history.undo_len(), undo_before);
            }
            8 => {
                if session.history.can_undo() {
                    session.execute(Command::Undo).expect("undo");
                }
            }
            9 => {
                if session.history.can_redo() {
                    session.execute(Command::Redo).expect("redo");
                }
            }
            _ => unreachable!(),
        }

        assert_bounds(&session, step);
    }

    // Drain undo stack completely, then redo completely, verifying invariants throughout.
    let mut undone = 0_usize;
    while session.history.can_undo() {
        session.execute(Command::Undo).expect("drain undo");
        undone += 1;
        assert_bounds(&session, 1000 + undone);
    }
    assert!(undone > 0 && undone <= max_entries);

    let mut redone = 0_usize;
    while session.history.can_redo() {
        session.execute(Command::Redo).expect("drain redo");
        redone += 1;
        assert_bounds(&session, 2000 + redone);
    }
    assert_eq!(redone, undone);
}
