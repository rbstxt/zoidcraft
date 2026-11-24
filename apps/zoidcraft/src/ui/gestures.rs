//! Shared trackpad, multi-touch, and mouse-wheel gesture classification helpers.
//!
//! `egui 0.37` folds both true trackpad pinch (`Event::Zoom`) and `Ctrl/Cmd + MouseWheel`
//! into `InputState::zoom_delta()` while zeroing `smooth_scroll_delta`, and remaps
//! `Shift + Wheel` onto X and `Alt + Wheel` onto Y. [`GestureClassifier`] separates these
//! signals cleanly so panels never double-zoom or drop modifier-wheel gestures.

use egui::{
    CursorIcon, Event, InputState, Modifiers, MouseWheelSource, MouseWheelUnit, Pos2, Rect,
    TouchPhase, Ui, Vec2,
};

/// Sticky inertial window (in seconds) over which scroll device classification persists.
const INERTIAL_TAIL_SECS: f64 = 0.22;

/// Logical points per discrete mouse-wheel line notch (matches `egui`'s default `line_scroll_speed`).
const LINE_SCROLL_POINTS: f32 = 40.0;

/// Coarse classification of the pointing device driving a scroll gesture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScrollDevice {
    #[default]
    MouseWheel,
    Trackpad,
}

/// Per-widget sticky classifier distinguishing smooth two-finger trackpad panning
/// from discrete notched mouse-wheel ticks across multi-frame smoothing and OS momentum.
#[derive(Debug, Clone, Copy, Default)]
pub struct GestureClassifier {
    last_device: ScrollDevice,
    last_time: f64,
    has_sample: bool,
}

impl GestureClassifier {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_device: ScrollDevice::MouseWheel,
            last_time: 0.0,
            has_sample: false,
        }
    }

    /// Samples all non-overlapping gesture signals for the current `egui` frame.
    #[must_use]
    pub fn sample(&mut self, i: &InputState) -> FrameGestures {
        // 1. Extract true pinch ONLY (`Event::Zoom` or touch-screen multi-touch).
        let pinch_zoom = if let Some(mt) = i.multi_touch() {
            if mt.zoom_delta.is_finite() && mt.zoom_delta > 0.0 {
                mt.zoom_delta
            } else {
                1.0
            }
        } else {
            i.events.iter().fold(1.0_f32, |acc, ev| match ev {
                Event::Zoom(f) if f.is_finite() && *f > 0.0 => acc * *f,
                _ => acc,
            })
        };

        // 2. Extract two-finger rotation radians (`Event::Rotate` or touch-screen multi-touch).
        let rotate_radians = if let Some(mt) = i.multi_touch() {
            if mt.rotation_delta.is_finite() {
                mt.rotation_delta
            } else {
                0.0
            }
        } else {
            i.events.iter().fold(0.0_f32, |acc, ev| match ev {
                Event::Rotate(rad) if rad.is_finite() => acc + *rad,
                _ => acc,
            })
        };

        // 3. Separate egui's Ctrl/Cmd+wheel zoom factor from true pinch (no double-counting).
        let total_zoom = i.zoom_delta();
        let mod_wheel_zoom = if total_zoom.is_finite() && total_zoom > 0.0 && pinch_zoom > 1e-4 {
            let ratio = total_zoom / pinch_zoom;
            if ratio.is_finite() && (ratio - 1.0).abs() > 1e-4 {
                ratio
            } else {
                1.0
            }
        } else {
            1.0
        };

        // 4. Collect un-remapped wheel points, effective modifiers, and classify MouseWheel vs Trackpad.
        let mut raw_wheel_points = Vec2::ZERO;
        let mut saw_wheel_event = false;
        let mut classified_this_frame: Option<ScrollDevice> = None;
        let mut modifiers = i.modifiers;

        for ev in &i.events {
            if let Event::MouseWheel {
                unit,
                delta,
                phase,
                source,
                modifiers: ev_mods,
            } = *ev
            {
                saw_wheel_event = true;
                modifiers |= ev_mods;
                let pts = match unit {
                    MouseWheelUnit::Point => delta,
                    MouseWheelUnit::Line => delta * LINE_SCROLL_POINTS,
                    MouseWheelUnit::Page => delta * i.viewport_rect().height(),
                };
                raw_wheel_points += pts;

                // macOS reports reliable `source`; Windows/Linux fall back to unit + delta heuristics.
                classified_this_frame = Some(match source {
                    MouseWheelSource::Trackpad | MouseWheelSource::Momentum => {
                        ScrollDevice::Trackpad
                    }
                    MouseWheelSource::Wheel => ScrollDevice::MouseWheel,
                    MouseWheelSource::Unknown => match unit {
                        MouseWheelUnit::Line | MouseWheelUnit::Page => ScrollDevice::MouseWheel,
                        MouseWheelUnit::Point => {
                            let is_touch_phase =
                                matches!(phase, TouchPhase::Start | TouchPhase::End);
                            let is_discrete_step = delta.x == 0.0
                                && delta.y.fract() == 0.0
                                && delta.y.abs() >= LINE_SCROLL_POINTS;
                            if is_touch_phase || !is_discrete_step {
                                ScrollDevice::Trackpad
                            } else {
                                ScrollDevice::MouseWheel
                            }
                        }
                    },
                });
            }
        }

        // Also respect egui's live `scroll_source()` during multi-frame wheel smoothing.
        if classified_this_frame.is_none() {
            classified_this_frame = match i.scroll_source() {
                Some(MouseWheelSource::Trackpad | MouseWheelSource::Momentum) => {
                    Some(ScrollDevice::Trackpad)
                }
                Some(MouseWheelSource::Wheel) => Some(ScrollDevice::MouseWheel),
                _ => None,
            };
        }

        let dt_since_last = i.time - self.last_time;
        let within_tail = self.has_sample && (0.0..INERTIAL_TAIL_SECS).contains(&dt_since_last);

        let device = if let Some(d) = classified_this_frame {
            if within_tail && !saw_wheel_event {
                self.last_device
            } else {
                self.last_device = d;
                self.last_time = i.time;
                self.has_sample = true;
                d
            }
        } else if within_tail {
            self.last_device
        } else {
            ScrollDevice::MouseWheel
        };

        // For direct unmodified trackpad events (`MouseWheelUnit::Point`), use the exact
        // unsmoothed `raw_wheel_points` so fast swipes stay 1:1 with zero latency and never
        // inherit leftover `unprocessed_wheel_delta` from a preceding discrete `Cmd+Wheel` notch.
        // On non-wheel frames during the trackpad inertial tail, avoid replaying any leftover
        // `egui` smoothing buffer that was already applied 1:1 from `raw_wheel_points`.
        let ctrl_or_cmd = modifiers.ctrl || modifiers.command;
        let scroll_delta = if device == ScrollDevice::Trackpad {
            if saw_wheel_event && !ctrl_or_cmd && !modifiers.shift && !modifiers.alt {
                raw_wheel_points
            } else if saw_wheel_event {
                i.smooth_scroll_delta
            } else {
                Vec2::ZERO
            }
        } else {
            i.smooth_scroll_delta
        };

        FrameGestures {
            pinch_zoom,
            rotate_radians,
            scroll_delta,
            raw_wheel_points,
            mod_wheel_zoom,
            ctrl_or_cmd,
            modifiers,
            device,
        }
    }
}

/// Decoded, non-overlapping per-frame gesture signals returned by [`GestureClassifier::sample`].
#[derive(Debug, Clone, Copy)]
pub struct FrameGestures {
    /// Multiplicative factor from true pinch gestures ONLY (`Event::Zoom` / multi-touch).
    /// `1.0` = no pinch; `> 1.0` = spread (zoom in); `< 1.0` = pinch (zoom out).
    pub pinch_zoom: f32,
    /// Two-finger rotation delta in radians (`Event::Rotate` / multi-touch).
    pub rotate_radians: f32,
    /// Smoothed scroll delta from `InputState::smooth_scroll_delta` when NO `Ctrl`/`Cmd` is held.
    /// Note: `egui` already remapped `Shift` -> X and `Alt` -> Y in this vector.
    pub scroll_delta: Vec2,
    /// Un-remapped raw wheel delta in logical points for this frame (`Line` * 40, `Point` as-is).
    /// Non-zero even when `Ctrl`/`Cmd`/`Shift`/`Alt` are held.
    pub raw_wheel_points: Vec2,
    /// Zoom factor synthesized by `egui` from `Ctrl`/`Cmd` + wheel ONLY (excludes `pinch_zoom`).
    /// `1.0` when not zooming via `Ctrl/Cmd` + wheel.
    pub mod_wheel_zoom: f32,
    /// `true` if `Ctrl` or `Cmd` is active on the current frame or wheel event.
    pub ctrl_or_cmd: bool,
    /// Effective keyboard modifiers for the gesture.
    pub modifiers: Modifiers,
    /// Classified scroll device (`Trackpad` vs discrete `MouseWheel`).
    pub device: ScrollDevice,
}

impl FrameGestures {
    #[must_use]
    pub fn has_pinch(&self) -> bool {
        (self.pinch_zoom - 1.0).abs() > 1e-4
    }

    #[must_use]
    pub fn has_mod_wheel_zoom(&self) -> bool {
        (self.mod_wheel_zoom - 1.0).abs() > 1e-4
    }

    #[must_use]
    pub fn has_scroll(&self) -> bool {
        self.scroll_delta.x.abs() > 0.01 || self.scroll_delta.y.abs() > 0.01
    }

    #[must_use]
    pub fn has_raw_wheel(&self) -> bool {
        self.raw_wheel_points.x.abs() > 0.01 || self.raw_wheel_points.y.abs() > 0.01
    }

    #[must_use]
    pub fn has_rotate(&self) -> bool {
        self.rotate_radians.abs() > 1e-4
    }
}

/// Returns `true` only when `pos` lands directly on `ui`'s top-most egui layer
/// (not occluded by a floating window, popup, dropdown, or modal) and no eyedropper
/// crosshair capture is active.
#[must_use]
pub fn pointer_on_layer(ui: &Ui, pos: Pos2) -> bool {
    ui.ctx().layer_id_at(pos) == Some(ui.layer_id())
        && !ui.ctx().output(|o| o.cursor_icon == CursorIcon::Crosshair)
}

/// Occlusion-safe hit test verifying `pos` is inside `rect` AND directly on `ui`'s layer.
#[must_use]
pub fn rect_contains_unoccluded_pointer(ui: &Ui, rect: Rect, pos: Pos2) -> bool {
    rect.contains(pos) && pointer_on_layer(ui, pos)
}

/// Returns the current pointer hover position iff it lies inside `rect` on `ui`'s unoccluded layer.
#[must_use]
pub fn unoccluded_hover_pos(ui: &Ui, rect: Rect) -> Option<Pos2> {
    let pos = ui.input(|i| i.pointer.hover_pos())?;
    rect_contains_unoccluded_pointer(ui, rect, pos).then_some(pos)
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::{Context, RawInput, pos2, vec2};

    fn sample_frame(
        ctx: &Context,
        classifier: &mut GestureClassifier,
        time: f64,
        modifiers: Modifiers,
        mut events: Vec<Event>,
    ) -> FrameGestures {
        events.insert(0, Event::ModifiersChanged(modifiers));
        let raw = RawInput {
            screen_rect: Some(Rect::from_min_size(pos2(0.0, 0.0), vec2(800.0, 600.0))),
            time: Some(time),
            events,
            focused: true,
            ..Default::default()
        };
        let mut sampled = None;
        let mut full = ctx.run_ui(raw, |ui| {
            sampled = Some(ui.input(|i| classifier.sample(i)));
        });
        full.textures_delta.clear();
        sampled.expect("sampled gestures")
    }

    #[test]
    fn pinch_and_cmd_wheel_are_separated_without_double_counting() {
        let ctx = Context::default();
        let mut cls = GestureClassifier::new();

        // Prime time so dt is non-zero.
        let _ = sample_frame(&ctx, &mut cls, 0.0, Modifiers::NONE, vec![]);

        // Pure pinch Event::Zoom(1.25).
        let g_pinch = sample_frame(
            &ctx,
            &mut cls,
            0.016,
            Modifiers::NONE,
            vec![Event::Zoom(1.25)],
        );
        assert!((g_pinch.pinch_zoom - 1.25).abs() < 1e-4);
        assert!(!g_pinch.has_mod_wheel_zoom());

        // Pure Cmd+Scroll (Line): egui zeroes smooth_scroll_delta and moves zoom into zoom_delta().
        let g_cmd = sample_frame(
            &ctx,
            &mut cls,
            0.032,
            Modifiers::COMMAND,
            vec![Event::MouseWheel {
                unit: MouseWheelUnit::Line,
                delta: vec2(0.0, 1.0),
                phase: TouchPhase::Move,
                source: MouseWheelSource::Wheel,
                modifiers: Modifiers::COMMAND,
            }],
        );
        assert!(!g_cmd.has_pinch());
        assert!(g_cmd.has_mod_wheel_zoom());
        assert!(g_cmd.mod_wheel_zoom > 1.0);
        assert_eq!(g_cmd.scroll_delta, Vec2::ZERO);
        assert!((g_cmd.raw_wheel_points.y - 40.0).abs() < 1e-3);
        assert_eq!(g_cmd.device, ScrollDevice::MouseWheel);

        // Simultaneous pinch + Cmd+Scroll: each component isolated, product equals total zoom_delta.
        let g_both = sample_frame(
            &ctx,
            &mut cls,
            0.048,
            Modifiers::COMMAND,
            vec![
                Event::Zoom(1.2),
                Event::MouseWheel {
                    unit: MouseWheelUnit::Point,
                    delta: vec2(0.0, 20.0),
                    phase: TouchPhase::Move,
                    source: MouseWheelSource::Trackpad,
                    modifiers: Modifiers::COMMAND,
                },
            ],
        );
        assert!((g_both.pinch_zoom - 1.2).abs() < 1e-4);
        assert!(g_both.has_mod_wheel_zoom());
        assert!(g_both.mod_wheel_zoom > 1.0);
    }

    #[test]
    fn trackpad_classification_persists_through_inertial_tail_and_decays() {
        let ctx = Context::default();
        let mut cls = GestureClassifier::new();

        let _ = sample_frame(&ctx, &mut cls, 1.0, Modifiers::NONE, vec![]);

        let g_tp = sample_frame(
            &ctx,
            &mut cls,
            1.02,
            Modifiers::NONE,
            vec![Event::MouseWheel {
                unit: MouseWheelUnit::Point,
                delta: vec2(6.0, -4.0),
                phase: TouchPhase::Move,
                source: MouseWheelSource::Trackpad,
                modifiers: Modifiers::NONE,
            }],
        );
        assert_eq!(g_tp.device, ScrollDevice::Trackpad);

        // Within 220 ms inertial tail with no new wheel event -> remains Trackpad.
        let g_tail = sample_frame(&ctx, &mut cls, 1.12, Modifiers::NONE, vec![]);
        assert_eq!(g_tail.device, ScrollDevice::Trackpad);

        // Past 220 ms -> decays back to MouseWheel default.
        let g_after = sample_frame(&ctx, &mut cls, 1.40, Modifiers::NONE, vec![]);
        assert_eq!(g_after.device, ScrollDevice::MouseWheel);
    }

    #[test]
    fn rotate_event_is_captured() {
        let ctx = Context::default();
        let mut cls = GestureClassifier::new();
        let g = sample_frame(
            &ctx,
            &mut cls,
            0.016,
            Modifiers::NONE,
            vec![Event::Rotate(0.15)],
        );
        assert!(g.has_rotate());
        assert!((g.rotate_radians - 0.15).abs() < 1e-5);
    }
}
