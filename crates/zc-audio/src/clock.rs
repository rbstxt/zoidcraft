//! Sample-accurate master playback clock with lock-free atomic epoch tracking
//! and optional `cpal` live audio output stream (Spec 02 §5.4).
//!
//! Eliminates CM3's HTMLMediaElement seek-desync defects by anchoring playhead
//! advancement directly to monotonic hardware audio sample counts (or a wall-clock
//! fallback in headless environments) and invalidating epochs atomically on every
//! scrub, loop wrap, speed change, or clip schedule edit.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence};
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

#[cfg(not(target_arch = "wasm32"))]
use cpal::SampleFormat;
#[cfg(not(target_arch = "wasm32"))]
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::AudioError;
use crate::decode::STANDARD_SAMPLE_RATE;
#[cfg(not(target_arch = "wasm32"))]
use crate::decode::f32_to_pcm16;

#[cfg(target_arch = "wasm32")]
mod web;

/// Maximum time (500 ms) a live CPAL stream callback may go silent before `exact_frame()`
/// falls back to monotonic wall-clock advancement so playback never freezes on device sleep.
const CALLBACK_STALL_TIMEOUT_NANOS: u64 = 500_000_000;

/// Shared pre-mixed stereo audio buffer (`[L, R]` at 48 kHz).
type SharedStereoMix = Arc<Vec<[f32; 2]>>;
/// Shared swappable slot for a loop-head stereo buffer.
type LoopHeadSlot = Arc<Mutex<Option<SharedStereoMix>>>;

/// Bounded pre-mixed 48 kHz stereo audio window anchored at timeline frame `origin_frame`.
#[derive(Debug, Clone)]
pub struct AudioMixWindow {
    /// Timeline frame corresponding to sample index `0` of `samples`.
    pub origin_frame: f64,
    /// Interleaved stereo `[L, R]` samples at `STANDARD_SAMPLE_RATE` (48 kHz).
    pub samples: Arc<Vec<[f32; 2]>>,
}

/// Atomic, tear-free snapshot of the master playback clock state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaybackSnapshot {
    /// Monotonically increasing epoch incremented on every seek/play/pause/speed/loop event.
    pub epoch_id: u64,
    /// Timeline frame at which the current epoch started.
    pub anchor_exact_frame: f64,
    /// Hardware sample counter at which the current epoch started.
    pub anchor_sample_count: u64,
    /// Playback speed multiplier (`0.0` = paused, `1.0` = normal forward, `-1.0` = reverse).
    pub speed: f64,
    /// Sequence frame rate in frames per second.
    pub fps: f64,
    /// Sequence length in frames.
    pub total_frames: f64,
    /// Whether looping is enabled when crossing sequence boundaries.
    pub loop_enabled: bool,
    /// Active audio clock sample rate in Hz.
    pub sample_rate: u32,
    /// Whether a live hardware audio stream is driving the sample counter.
    pub hardware_clock_active: bool,
}

struct CommitContext {
    hw_now: u64,
    now_nanos: u64,
    current_frame: f64,
    hit_stop: bool,
}

struct ClockShared {
    /// Seqlock counter (odd while a control-thread mutation is in flight, even when stable).
    seq: AtomicU64,
    /// Serializes control-thread writers (never locked inside the real-time audio callback).
    writer_lock: Mutex<()>,
    /// Monotonic origin timestamp set once at construction for lock-free wall-clock offsets.
    clock_origin: Instant,
    /// Nanoseconds elapsed since `clock_origin` when the current epoch was anchored.
    wall_anchor_nanos: AtomicU64,
    /// Monotonically increasing identifier of the currently active live output stream (`0` = none).
    active_stream_id: AtomicU64,

    epoch_id: AtomicU64,
    anchor_exact_frame_bits: AtomicU64,
    anchor_sample_count: AtomicU64,
    hardware_sample_count: AtomicU64,
    /// Queued hardware output latency in samples derived from CPAL's `OutputCallbackInfo`
    /// playback timestamp (so the visual playhead tracks audible output rather than leading it).
    output_latency_samples: AtomicU64,
    /// Epoch id most recently latched by the real-time audio callback.
    callback_latched_epoch: AtomicU64,
    /// Exact submission sample index where `callback_latched_epoch` first began rendering,
    /// eliminating any 1-buffer skip when seek/resume races an in-flight callback.
    callback_epoch_start_sample: AtomicU64,
    /// Monotonic timestamp (`now_nanos`) when the live audio callback last executed (`0` = never).
    last_callback_nanos: AtomicU64,
    speed_bits: AtomicU64,
    fps_bits: AtomicU64,
    total_frames_bits: AtomicU64,
    loop_enabled: AtomicBool,
    sample_rate: AtomicU32,
    hardware_clock_active: AtomicBool,
}

impl ClockShared {
    fn new(fps: f64, total_frames: f64, sample_rate: u32) -> Self {
        let clean_fps = if fps.is_finite() && fps > 0.0 {
            fps
        } else {
            30.0
        };
        let clean_total = if total_frames.is_finite() && total_frames >= 0.0 {
            total_frames
        } else {
            0.0
        };
        let clean_sr = sample_rate.max(1);

        Self {
            seq: AtomicU64::new(0),
            writer_lock: Mutex::new(()),
            clock_origin: Instant::now(),
            wall_anchor_nanos: AtomicU64::new(0),
            active_stream_id: AtomicU64::new(0),
            epoch_id: AtomicU64::new(1),
            anchor_exact_frame_bits: AtomicU64::new(0.0_f64.to_bits()),
            anchor_sample_count: AtomicU64::new(0),
            hardware_sample_count: AtomicU64::new(0),
            output_latency_samples: AtomicU64::new(0),
            callback_latched_epoch: AtomicU64::new(0),
            callback_epoch_start_sample: AtomicU64::new(0),
            last_callback_nanos: AtomicU64::new(0),
            speed_bits: AtomicU64::new(0.0_f64.to_bits()),
            fps_bits: AtomicU64::new(clean_fps.to_bits()),
            total_frames_bits: AtomicU64::new(clean_total.to_bits()),
            loop_enabled: AtomicBool::new(false),
            sample_rate: AtomicU32::new(clean_sr),
            hardware_clock_active: AtomicBool::new(false),
        }
    }

    #[inline]
    fn now_nanos(&self) -> u64 {
        u64::try_from(self.clock_origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Lock-free, tear-free read of the current clock parameters and wall-clock anchor
    /// (safe in real-time audio callbacks).
    fn read_snapshot_with_wall(&self) -> (PlaybackSnapshot, u64) {
        for _ in 0..64 {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let (snap, wall_nanos) = self.read_snapshot_relaxed_with_wall();
            fence(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Relaxed);
            if s1 == s2 {
                return (snap, wall_nanos);
            }
            std::hint::spin_loop();
        }
        // Fallback after bounded spin: return a paused snapshot to avoid ever blocking audio.
        let (mut snap, wall_nanos) = self.read_snapshot_relaxed_with_wall();
        snap.speed = 0.0;
        (snap, wall_nanos)
    }

    /// Lock-free, tear-free read of the current clock parameters (safe in real-time audio callbacks).
    fn read_snapshot(&self) -> PlaybackSnapshot {
        self.read_snapshot_with_wall().0
    }

    #[inline]
    fn read_snapshot_relaxed_with_wall(&self) -> (PlaybackSnapshot, u64) {
        let snap = PlaybackSnapshot {
            epoch_id: self.epoch_id.load(Ordering::Relaxed),
            anchor_exact_frame: f64::from_bits(
                self.anchor_exact_frame_bits.load(Ordering::Relaxed),
            ),
            anchor_sample_count: self.anchor_sample_count.load(Ordering::Relaxed),
            speed: f64::from_bits(self.speed_bits.load(Ordering::Relaxed)),
            fps: f64::from_bits(self.fps_bits.load(Ordering::Relaxed)),
            total_frames: f64::from_bits(self.total_frames_bits.load(Ordering::Relaxed)),
            loop_enabled: self.loop_enabled.load(Ordering::Relaxed),
            sample_rate: self.sample_rate.load(Ordering::Relaxed).max(1),
            hardware_clock_active: self.hardware_clock_active.load(Ordering::Relaxed),
        };
        let wall_nanos = self.wall_anchor_nanos.load(Ordering::Relaxed);
        (snap, wall_nanos)
    }

    /// Computes exact frame at a given hardware sample count relative to `anchor_sample`.
    #[inline]
    fn evaluate_at_sample_from(
        snap: &PlaybackSnapshot,
        hw_sample: u64,
        anchor_sample: u64,
    ) -> (f64, bool) {
        if snap.speed == 0.0 {
            return (snap.anchor_exact_frame, false);
        }
        let elapsed_samples = hw_sample.saturating_sub(anchor_sample) as f64;
        let elapsed_sec = elapsed_samples / f64::from(snap.sample_rate);
        let raw_frame = snap.anchor_exact_frame + elapsed_sec * snap.fps * snap.speed;
        apply_bounds(raw_frame, snap.total_frames, snap.loop_enabled)
    }

    /// Returns the estimated currently audible hardware sample count (submitted samples minus
    /// the device's queued playback latency reported by CPAL).
    #[inline]
    fn audible_sample_count(&self) -> u64 {
        let written = self.hardware_sample_count.load(Ordering::Acquire);
        let latency = self.output_latency_samples.load(Ordering::Acquire);
        written.saturating_sub(latency)
    }

    /// Evaluates current exact frame and stop flag from an already-consistent `(snap, wall_anchor_nanos)`.
    fn evaluate_snapshot_at(
        &self,
        snap: &PlaybackSnapshot,
        wall_anchor_nanos: u64,
        now_nanos: u64,
    ) -> (f64, bool) {
        if snap.speed == 0.0 {
            return (snap.anchor_exact_frame, false);
        }

        let raw_frame = if snap.hardware_clock_active {
            let latched_epoch = self.callback_latched_epoch.load(Ordering::Acquire);
            let live_stream = self.active_stream_id.load(Ordering::Acquire) != 0;
            if latched_epoch != snap.epoch_id {
                // Audio callback has not yet started rendering this epoch; hold at anchor frame
                // unless a live CPAL stream callback has stalled past the heartbeat timeout.
                if live_stream
                    && now_nanos.saturating_sub(wall_anchor_nanos) >= CALLBACK_STALL_TIMEOUT_NANOS
                {
                    let elapsed_sec = (now_nanos.saturating_sub(wall_anchor_nanos) as f64) * 1e-9;
                    snap.anchor_exact_frame + elapsed_sec * snap.fps * snap.speed
                } else {
                    snap.anchor_exact_frame
                }
            } else {
                let anchor = self.callback_epoch_start_sample.load(Ordering::Acquire);
                let hw = self.audible_sample_count();
                let elapsed_samples = hw.saturating_sub(anchor) as f64;
                let elapsed_sec = elapsed_samples / f64::from(snap.sample_rate);
                let last_cb = self.last_callback_nanos.load(Ordering::Acquire);
                let stall_extra_sec = if live_stream
                    && last_cb != 0
                    && now_nanos.saturating_sub(last_cb) >= CALLBACK_STALL_TIMEOUT_NANOS
                {
                    (now_nanos.saturating_sub(last_cb) as f64) * 1e-9
                } else {
                    0.0
                };
                snap.anchor_exact_frame + (elapsed_sec + stall_extra_sec) * snap.fps * snap.speed
            }
        } else {
            let elapsed_sec = (now_nanos.saturating_sub(wall_anchor_nanos) as f64) * 1e-9;
            snap.anchor_exact_frame + elapsed_sec * snap.fps * snap.speed
        };

        apply_bounds(raw_frame, snap.total_frames, snap.loop_enabled)
    }

    /// Computes current exact frame (using hardware counter if active, else monotonic wall clock).
    fn current_exact_frame_and_stop_flag(&self) -> (f64, bool, PlaybackSnapshot) {
        let (snap, wall_anchor_nanos) = self.read_snapshot_with_wall();
        let now_nanos = self.now_nanos();
        let (bounded, hit_stop) = self.evaluate_snapshot_at(&snap, wall_anchor_nanos, now_nanos);
        (bounded, hit_stop, snap)
    }

    /// Conditionally mutates the clock atomically under the control-thread writer lock and seqlock.
    ///
    /// Evaluates the current exact frame while holding `writer_lock` so concurrent control-thread
    /// mutations never race on a stale frame reading. If `mutate` returns `false`, no seqlock
    /// write or epoch bump is performed and `None` is returned.
    fn try_commit_update<F>(&self, bump_epoch: bool, mutate: F) -> Option<u64>
    where
        F: FnOnce(&mut PlaybackSnapshot, &mut u64, CommitContext) -> bool,
    {
        let _guard = self
            .writer_lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let (mut snap, mut wall_anchor_nanos) = self.read_snapshot_relaxed_with_wall();
        let now_nanos = self.now_nanos();
        let (current_frame, hit_stop) =
            self.evaluate_snapshot_at(&snap, wall_anchor_nanos, now_nanos);
        let hw_now = self.hardware_sample_count.load(Ordering::Acquire);
        let ctx = CommitContext {
            hw_now,
            now_nanos,
            current_frame,
            hit_stop,
        };

        if !mutate(&mut snap, &mut wall_anchor_nanos, ctx) {
            return None;
        }

        if bump_epoch {
            snap.epoch_id = snap.epoch_id.wrapping_add(1);
        }

        // Begin seqlock write (odd) — strictly lock-free atomic stores only.
        let prev_seq = self.seq.load(Ordering::Relaxed);
        self.seq.store(prev_seq.wrapping_add(1), Ordering::Release);
        fence(Ordering::Release);

        self.epoch_id.store(snap.epoch_id, Ordering::Relaxed);
        self.anchor_exact_frame_bits
            .store(snap.anchor_exact_frame.to_bits(), Ordering::Relaxed);
        self.anchor_sample_count
            .store(snap.anchor_sample_count, Ordering::Relaxed);
        self.speed_bits
            .store(snap.speed.to_bits(), Ordering::Relaxed);
        self.fps_bits.store(snap.fps.to_bits(), Ordering::Relaxed);
        self.total_frames_bits
            .store(snap.total_frames.to_bits(), Ordering::Relaxed);
        self.loop_enabled
            .store(snap.loop_enabled, Ordering::Relaxed);
        self.sample_rate
            .store(snap.sample_rate.max(1), Ordering::Relaxed);
        self.hardware_clock_active
            .store(snap.hardware_clock_active, Ordering::Relaxed);
        self.wall_anchor_nanos
            .store(wall_anchor_nanos, Ordering::Relaxed);

        // End seqlock write (even)
        self.seq.store(prev_seq.wrapping_add(2), Ordering::Release);

        if !snap.hardware_clock_active {
            self.output_latency_samples.store(0, Ordering::Release);
        }

        Some(snap.epoch_id)
    }

    /// Mutates the clock atomically under the control-thread seqlock and optionally bumps `epoch_id`.
    ///
    /// Anchors `hw_now` to the **submission cursor** (`hardware_sample_count`) so the very next
    /// audio callback synthesizes sample 0 at the exact requested frame without skipping the
    /// queued latency interval, while visual playhead queries subtract `output_latency_samples`.
    fn commit_update<F>(&self, bump_epoch: bool, mutate: F) -> u64
    where
        F: FnOnce(&mut PlaybackSnapshot, &mut u64, CommitContext),
    {
        self.try_commit_update(bump_epoch, |snap, wall_anchor_nanos, ctx| {
            mutate(snap, wall_anchor_nanos, ctx);
            true
        })
        .unwrap_or_else(|| self.epoch_id.load(Ordering::Acquire))
    }

    fn attach_hardware_clock(&self, sample_rate: u32) -> u64 {
        // Anchor clock to hardware sample counter, bump epoch, and claim a unique stream ID
        // before starting playback so re-attaching never double-counts prior samples.
        let mut stream_id = 0_u64;
        self.commit_update(true, |snap, wall_anchor_nanos, ctx| {
            let mut next_id = self
                .active_stream_id
                .fetch_add(1, Ordering::AcqRel)
                .wrapping_add(1);
            if next_id == 0 {
                next_id = self
                    .active_stream_id
                    .fetch_add(1, Ordering::AcqRel)
                    .wrapping_add(1);
            }
            stream_id = next_id;
            snap.anchor_exact_frame = ctx.current_frame;
            snap.anchor_sample_count = ctx.hw_now;
            snap.sample_rate = sample_rate;
            snap.hardware_clock_active = true;
            *wall_anchor_nanos = ctx.now_nanos;
            self.last_callback_nanos
                .store(ctx.now_nanos, Ordering::Release);
        });

        stream_id
    }

    fn detach_hardware_clock(&self, expected_stream_id: u64) {
        self.try_commit_update(false, |snap, wall_anchor_nanos, ctx| {
            if self.active_stream_id.load(Ordering::Acquire) != expected_stream_id {
                return false;
            }
            snap.anchor_exact_frame = ctx.current_frame;
            snap.anchor_sample_count = ctx.hw_now;
            snap.hardware_clock_active = false;
            *wall_anchor_nanos = ctx.now_nanos;
            if ctx.hit_stop {
                snap.speed = 0.0;
            }
            true
        });
    }
}

/// Applies Spec 02 §5.1 boundary and loop wrap rules.
/// Returns `(bounded_frame, should_stop_playback)`.
#[inline]
fn apply_bounds(raw_frame: f64, total_frames: f64, loop_enabled: bool) -> (f64, bool) {
    if !raw_frame.is_finite() || !total_frames.is_finite() || total_frames <= 0.0 {
        return (0.0, !loop_enabled);
    }
    if raw_frame >= 0.0 && raw_frame < total_frames {
        return (raw_frame, false);
    }
    if loop_enabled {
        let wrapped = ((raw_frame % total_frames) + total_frames) % total_frames;
        (wrapped, false)
    } else {
        let max_frame = (total_frames - 1.0).max(0.0);
        (raw_frame.clamp(0.0, max_frame), true)
    }
}

/// Sample-accurate, lock-free master playback clock shared by audio mixing,
/// video decoding, and timeline UI rendering.
#[derive(Clone)]
pub struct MasterPlaybackClock {
    shared: Arc<ClockShared>,
}

impl std::fmt::Debug for MasterPlaybackClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let snap = self.snapshot();
        f.debug_struct("MasterPlaybackClock")
            .field("epoch_id", &snap.epoch_id)
            .field("exact_frame", &self.exact_frame())
            .field("speed", &snap.speed)
            .field("fps", &snap.fps)
            .field("total_frames", &snap.total_frames)
            .field("loop_enabled", &snap.loop_enabled)
            .field("sample_rate", &snap.sample_rate)
            .field("hardware_clock_active", &snap.hardware_clock_active)
            .finish()
    }
}

impl MasterPlaybackClock {
    /// Creates a new paused playback clock anchored at frame `0.0` and `48,000 Hz`.
    #[must_use]
    pub fn new(fps: f64, total_frames: f64) -> Self {
        Self::with_sample_rate(fps, total_frames, STANDARD_SAMPLE_RATE)
    }

    /// Creates a new paused playback clock with an explicit initial sample rate.
    #[must_use]
    pub fn with_sample_rate(fps: f64, total_frames: f64, sample_rate: u32) -> Self {
        Self {
            shared: Arc::new(ClockShared::new(fps, total_frames, sample_rate)),
        }
    }

    /// Returns a consistent lock-free snapshot of clock state.
    #[must_use]
    pub fn snapshot(&self) -> PlaybackSnapshot {
        self.shared.read_snapshot()
    }

    /// Current monotonically increasing epoch identifier.
    #[must_use]
    pub fn epoch(&self) -> u64 {
        self.shared.epoch_id.load(Ordering::Acquire)
    }

    /// Current high-precision timeline frame position.
    ///
    /// Automatically stops playback if non-looping playback has reached either sequence boundary.
    #[must_use]
    pub fn exact_frame(&self) -> f64 {
        let (frame, hit_stop, observed_snap) = self.shared.current_exact_frame_and_stop_flag();
        if hit_stop {
            let mut result_frame = frame;
            self.shared
                .try_commit_update(true, |snap, wall_anchor_nanos, ctx| {
                    if snap.epoch_id != observed_snap.epoch_id || !ctx.hit_stop {
                        result_frame = ctx.current_frame;
                        return false;
                    }
                    snap.anchor_exact_frame = ctx.current_frame;
                    snap.anchor_sample_count = ctx.hw_now;
                    snap.speed = 0.0;
                    *wall_anchor_nanos = ctx.now_nanos;
                    result_frame = ctx.current_frame;
                    true
                });
            return result_frame;
        }
        frame
    }

    /// Integer frame passed to sequence evaluation and compositing (`round(exact_frame)`).
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.exact_frame().round().max(0.0) as u64
    }

    /// Current timeline time in seconds (`exact_frame / fps`), preserving fractional sample precision.
    #[must_use]
    pub fn current_time_secs(&self) -> f64 {
        let snap = self.snapshot();
        if snap.fps <= 0.0 {
            0.0
        } else {
            self.exact_frame() / snap.fps
        }
    }

    /// Evaluates the exact timeline frame corresponding to a specific hardware sample counter.
    #[must_use]
    pub fn exact_frame_at_sample(&self, sample_count: u64) -> f64 {
        let snap = self.snapshot();
        let anchor = if self.shared.callback_latched_epoch.load(Ordering::Acquire) == snap.epoch_id
        {
            self.shared
                .callback_epoch_start_sample
                .load(Ordering::Acquire)
        } else {
            snap.anchor_sample_count
        };
        ClockShared::evaluate_at_sample_from(&snap, sample_count, anchor).0
    }

    /// Returns `true` when playback speed is non-zero.
    #[must_use]
    pub fn is_playing(&self) -> bool {
        self.snapshot().speed != 0.0
    }

    /// Returns the current playback speed multiplier.
    #[must_use]
    pub fn speed(&self) -> f64 {
        self.snapshot().speed
    }

    /// Returns the current hardware sample counter.
    #[must_use]
    pub fn hardware_sample_count(&self) -> u64 {
        self.shared.hardware_sample_count.load(Ordering::Acquire)
    }

    /// Starts normal forward playback (`speed = 1.0`) from the current frame.
    pub fn play(&self) -> u64 {
        self.set_speed(1.0)
    }

    /// Pauses playback (`speed = 0.0`) at the exact current frame.
    pub fn pause(&self) -> u64 {
        self.set_speed(0.0)
    }

    /// Sets playback speed multiplier, re-anchoring smoothly at the current frame and bumping epoch.
    pub fn set_speed(&self, speed: f64) -> u64 {
        let clean_speed = if speed.is_finite() { speed } else { 0.0 };
        self.shared
            .commit_update(true, |snap, wall_anchor_nanos, ctx| {
                snap.anchor_exact_frame = ctx.current_frame;
                snap.anchor_sample_count = ctx.hw_now;
                snap.speed = clean_speed;
                *wall_anchor_nanos = ctx.now_nanos;
            })
    }

    /// Scrubs immediately to `frame`, pausing playback (`speed = 0.0`), clamping to
    /// `[0, max(total_frames - 1, 0)]`, and incrementing `epoch_id` (Spec 02 §5.1).
    pub fn set_frame(&self, frame: f64) -> u64 {
        self.shared
            .commit_update(true, |snap, wall_anchor_nanos, ctx| {
                let max_frame = (snap.total_frames - 1.0).max(0.0);
                let clamped = if frame.is_finite() {
                    frame.clamp(0.0, max_frame)
                } else {
                    0.0
                };
                snap.anchor_exact_frame = clamped;
                snap.anchor_sample_count = ctx.hw_now;
                snap.speed = 0.0;
                *wall_anchor_nanos = ctx.now_nanos;
            })
    }

    /// Seeks immediately to `frame` while preserving the active playback speed,
    /// re-anchoring `anchor_sample_count` to the current hardware sample counter and
    /// bumping `epoch_id` so audio and video reposition with zero desync.
    pub fn seek_preserving_playback(&self, frame: f64) -> u64 {
        self.shared
            .commit_update(true, |snap, wall_anchor_nanos, ctx| {
                let max_frame = (snap.total_frames - 1.0).max(0.0);
                let clamped = if frame.is_finite() {
                    frame.clamp(0.0, max_frame)
                } else {
                    0.0
                };
                snap.anchor_exact_frame = clamped;
                snap.anchor_sample_count = ctx.hw_now;
                *wall_anchor_nanos = ctx.now_nanos;
            })
    }

    /// Enables or disables loop wrapping at sequence boundaries.
    pub fn set_loop(&self, enabled: bool) {
        self.shared
            .commit_update(false, |snap, _wall_anchor_nanos, _ctx| {
                snap.loop_enabled = enabled;
            });
    }

    /// Updates sequence `fps` and `total_frames`, clamping the current playhead and bumping epoch.
    pub fn set_timeline_config(&self, fps: f64, total_frames: f64) -> u64 {
        self.shared
            .commit_update(true, |snap, wall_anchor_nanos, ctx| {
                if fps.is_finite() && fps > 0.0 {
                    snap.fps = fps;
                }
                if total_frames.is_finite() && total_frames >= 0.0 {
                    snap.total_frames = total_frames;
                }
                let max_frame = (snap.total_frames - 1.0).max(0.0);
                snap.anchor_exact_frame = ctx.current_frame.clamp(0.0, max_frame);
                snap.anchor_sample_count = ctx.hw_now;
                *wall_anchor_nanos = ctx.now_nanos;
            })
    }

    /// Invalidates downstream clip/decoder schedules after a timeline edit while preserving playhead.
    pub fn invalidate_schedule(&self) -> u64 {
        self.shared
            .commit_update(true, |snap, wall_anchor_nanos, ctx| {
                snap.anchor_exact_frame = ctx.current_frame;
                snap.anchor_sample_count = ctx.hw_now;
                *wall_anchor_nanos = ctx.now_nanos;
            })
    }

    /// Enables or disables hardware sample-counter clock mode manually (useful for deterministic tests
    /// and headless drivers without an OS audio device).
    pub fn set_hardware_clock_mode(&self, active: bool, sample_rate: u32) {
        self.shared
            .commit_update(true, |snap, wall_anchor_nanos, ctx| {
                snap.anchor_exact_frame = ctx.current_frame;
                snap.anchor_sample_count = ctx.hw_now;
                snap.hardware_clock_active = active;
                if sample_rate > 0 {
                    snap.sample_rate = sample_rate;
                }
                if !active && ctx.hit_stop {
                    snap.speed = 0.0;
                }
                *wall_anchor_nanos = ctx.now_nanos;
            });
    }

    /// Advances the hardware sample counter by `sample_frames` (called by headless drivers or tests).
    pub fn advance_hardware_samples(&self, sample_frames: u64) -> u64 {
        let start_hw = self.shared.hardware_sample_count.load(Ordering::Acquire);
        let snap = self.shared.read_snapshot();
        if self.shared.callback_latched_epoch.load(Ordering::Relaxed) != snap.epoch_id {
            self.shared
                .callback_epoch_start_sample
                .store(start_hw, Ordering::Release);
            self.shared
                .callback_latched_epoch
                .store(snap.epoch_id, Ordering::Release);
        }
        self.shared
            .hardware_sample_count
            .fetch_add(sample_frames, Ordering::AcqRel)
            .saturating_add(sample_frames)
    }

    /// Attempts to open a live audio output stream synchronized to this clock.
    ///
    /// Gracefully returns `None` (keeping silent wall-clock fallback active) if no audio output
    /// device is available (e.g., headless servers or CI environments).
    #[must_use]
    pub fn try_start_audio_output(
        &self,
        stereo_mix_48k: Option<Arc<Vec<[f32; 2]>>>,
    ) -> Option<AudioOutputHandle> {
        self.try_start_audio_output_windowed(stereo_mix_48k.map(|samples| AudioMixWindow {
            origin_frame: 0.0,
            samples,
        }))
    }

    /// Attempts to open a live audio output stream backed by a bounded timeline window.
    #[must_use]
    pub fn try_start_audio_output_windowed(
        &self,
        window: Option<AudioMixWindow>,
    ) -> Option<AudioOutputHandle> {
        match self.start_audio_output_windowed(window) {
            Ok(handle) => Some(handle),
            Err(err) => {
                tracing::warn!(
                    "live audio output unavailable ({err}); falling back to silent wall-clock mode"
                );
                None
            }
        }
    }

    /// Opens a live audio output stream synchronized to this clock.
    ///
    /// Dropping the returned [`AudioOutputHandle`] stops the stream, releases OS audio resources,
    /// and seamlessly transitions the clock back to wall-clock mode.
    pub fn start_audio_output(
        &self,
        stereo_mix_48k: Option<Arc<Vec<[f32; 2]>>>,
    ) -> Result<AudioOutputHandle, AudioError> {
        self.start_audio_output_windowed(stereo_mix_48k.map(|samples| AudioMixWindow {
            origin_frame: 0.0,
            samples,
        }))
    }

    /// Opens a live audio output stream backed by a bounded [`AudioMixWindow`] anchored
    /// at `window.origin_frame`, allowing arbitrarily long timelines to stream audio without
    /// exceeding the mixer sample cap.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_audio_output_windowed(
        &self,
        window: Option<AudioMixWindow>,
    ) -> Result<AudioOutputHandle, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;

        let supported = device
            .default_output_config()
            .map_err(|err| AudioError::OutputDevice(err.to_string()))?;
        let sample_format = supported.sample_format();
        let config = supported.config();
        let device_sr = config.sample_rate;
        let device_channels = config.channels;

        if device_sr == 0 || device_channels == 0 {
            return Err(AudioError::InvalidFormat(
                "audio output device reported zero sample rate or zero channels".to_string(),
            ));
        }

        let stream_id = self.shared.attach_hardware_clock(device_sr);

        let shared_cb = Arc::clone(&self.shared);
        let err_fn = |err: cpal::Error| {
            log_cpal_stream_error(&err);
        };

        let ch_usize = usize::from(device_channels);
        let mix_sr = f64::from(STANDARD_SAMPLE_RATE);

        let window_slot = Arc::new(Mutex::new(window.clone()));
        let window_gen = Arc::new(AtomicU64::new(0));
        let loop_head_slot: LoopHeadSlot = Arc::new(Mutex::new(None));
        let loop_head_gen = Arc::new(AtomicU64::new(0));

        let mut cb_state = CallbackMixState {
            active_window: window,
            seen_window_gen: 0,
            window_slot: Arc::clone(&window_slot),
            window_gen: Arc::clone(&window_gen),
            active_loop_head: None,
            seen_loop_head_gen: 0,
            loop_head_slot: Arc::clone(&loop_head_slot),
            loop_head_gen: Arc::clone(&loop_head_gen),
        };

        let stream_result = match sample_format {
            SampleFormat::F32 => device.build_output_stream(
                config,
                move |data: &mut [f32], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        |s| s,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::F64 => device.build_output_stream(
                config,
                move |data: &mut [f64], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        f32_to_f64_sample,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::I16 => device.build_output_stream(
                config,
                move |data: &mut [i16], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        f32_to_pcm16,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::U16 => device.build_output_stream(
                config,
                move |data: &mut [u16], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        |s| ((s.clamp(-1.0, 1.0) * 0.5 + 0.5) * f32::from(u16::MAX)).round() as u16,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::I32 => device.build_output_stream(
                config,
                move |data: &mut [i32], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        f32_to_i32_sample,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::U32 => device.build_output_stream(
                config,
                move |data: &mut [u32], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        f32_to_u32_sample,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::I8 => device.build_output_stream(
                config,
                move |data: &mut [i8], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        f32_to_i8_sample,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::U8 => device.build_output_stream(
                config,
                move |data: &mut [u8], info: &cpal::OutputCallbackInfo| {
                    cb_state.poll_updates();
                    let (origin, mix) = cb_state.active_mix();
                    render_callback_frames_windowed_with_loop_head(
                        &shared_cb,
                        mix,
                        origin,
                        cb_state.active_loop_head.as_deref(),
                        mix_sr,
                        ch_usize,
                        output_delay_secs(info),
                        data,
                        f32_to_u8_sample,
                    );
                },
                err_fn,
                None,
            ),
            other => {
                self.shared.detach_hardware_clock(stream_id);
                return Err(AudioError::OutputDevice(format!(
                    "unsupported output sample format: {other:?}"
                )));
            }
        };

        let stream = match stream_result {
            Ok(stream) => stream,
            Err(err) => {
                self.shared.detach_hardware_clock(stream_id);
                return Err(AudioError::OutputDevice(err.to_string()));
            }
        };

        if let Err(err) = stream.play() {
            drop(stream);
            self.shared.detach_hardware_clock(stream_id);
            return Err(AudioError::OutputDevice(err.to_string()));
        }

        Ok(AudioOutputHandle {
            stream: Some(stream),
            stream_id,
            shared: Arc::clone(&self.shared),
            sample_rate: device_sr,
            channels: device_channels,
            window_slot,
            window_gen,
            loop_head_slot,
            loop_head_gen,
        })
    }
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn is_transient_cpal_xrun(err: &cpal::Error) -> bool {
    if err.kind() == cpal::ErrorKind::Xrun {
        return true;
    }
    let msg = err.to_string().to_ascii_lowercase();
    msg.contains("underrun") || msg.contains("overrun")
}

#[cfg(not(target_arch = "wasm32"))]
fn log_cpal_stream_error(err: &cpal::Error) {
    if is_transient_cpal_xrun(err) {
        tracing::debug!("cpal output stream transient xrun: {err}");
    } else {
        tracing::error!("cpal output stream error: {err}");
    }
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn f32_to_f64_sample(s: f32) -> f64 {
    if s.is_finite() {
        f64::from(s.clamp(-1.0, 1.0))
    } else {
        0.0
    }
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn f32_to_i32_sample(s: f32) -> i32 {
    if !s.is_finite() {
        return 0;
    }
    let c = s.clamp(-1.0, 1.0);
    let scaled = if c < 0.0 {
        f64::from(c) * 2_147_483_648.0
    } else {
        f64::from(c) * 2_147_483_647.0
    };
    scaled
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn f32_to_u32_sample(s: f32) -> u32 {
    if !s.is_finite() {
        return u32::MAX / 2;
    }
    let c = f64::from(s.clamp(-1.0, 1.0));
    ((c * 0.5 + 0.5) * f64::from(u32::MAX))
        .round()
        .clamp(0.0, f64::from(u32::MAX)) as u32
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn f32_to_i8_sample(s: f32) -> i8 {
    if !s.is_finite() {
        return 0;
    }
    let c = s.clamp(-1.0, 1.0);
    let scaled = if c < 0.0 {
        f64::from(c) * 128.0
    } else {
        f64::from(c) * 127.0
    };
    scaled.round().clamp(f64::from(i8::MIN), f64::from(i8::MAX)) as i8
}

#[inline]
#[cfg(not(target_arch = "wasm32"))]
fn f32_to_u8_sample(s: f32) -> u8 {
    if !s.is_finite() {
        return 128;
    }
    let c = s.clamp(-1.0, 1.0);
    ((c * 0.5 + 0.5) * f32::from(u8::MAX))
        .round()
        .clamp(0.0, f32::from(u8::MAX)) as u8
}

struct CallbackMixState {
    active_window: Option<AudioMixWindow>,
    seen_window_gen: u64,
    window_slot: Arc<Mutex<Option<AudioMixWindow>>>,
    window_gen: Arc<AtomicU64>,
    active_loop_head: Option<SharedStereoMix>,
    seen_loop_head_gen: u64,
    loop_head_slot: LoopHeadSlot,
    loop_head_gen: Arc<AtomicU64>,
}

impl CallbackMixState {
    #[inline]
    fn poll_updates(&mut self) {
        let current_gen = self.window_gen.load(Ordering::Acquire);
        if current_gen != self.seen_window_gen
            && let Ok(guard) = self.window_slot.try_lock()
        {
            self.active_window = guard.clone();
            self.seen_window_gen = current_gen;
        }
        let loop_gen = self.loop_head_gen.load(Ordering::Acquire);
        if loop_gen != self.seen_loop_head_gen
            && let Ok(guard) = self.loop_head_slot.try_lock()
        {
            self.active_loop_head = guard.clone();
            self.seen_loop_head_gen = loop_gen;
        }
    }

    #[inline]
    fn active_mix(&self) -> (f64, Option<&Vec<[f32; 2]>>) {
        match self.active_window.as_ref() {
            Some(w) => (w.origin_frame, Some(w.samples.as_ref())),
            None => (0.0, None),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn output_delay_secs(info: &cpal::OutputCallbackInfo) -> f64 {
    let timestamp = info.timestamp();
    timestamp
        .playback
        .saturating_duration_since(timestamp.callback)
        .as_secs_f64()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
fn render_callback_frames<T, F>(
    shared: &ClockShared,
    mix: Option<&Vec<[f32; 2]>>,
    mix_sample_rate: f64,
    channels: usize,
    delay_secs: f64,
    output: &mut [T],
    convert: F,
) where
    F: Fn(f32) -> T,
{
    render_callback_frames_windowed(
        shared,
        mix,
        0.0,
        mix_sample_rate,
        channels,
        delay_secs,
        output,
        convert,
    );
}

/// Evaluates 4-point Catmull-Rom cubic Hermite interpolation for `t in [0.0, 1.0]`
/// between `y1` (at `t = 0`) and `y2` (at `t = 1`) using neighbor samples `y0` and `y3`.
#[inline]
fn catmull_rom_sample(y0: f32, y1: f32, y2: f32, y3: f32, t: f32) -> f32 {
    let a0 = -0.5 * y0 + 1.5 * y1 - 1.5 * y2 + 0.5 * y3;
    let a1 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3;
    let a2 = -0.5 * y0 + 0.5 * y2;
    let a3 = y1;
    ((a0 * t + a1) * t + a2) * t + a3
}

/// Reads a stereo frame from `slice` at fractional sample position `pos` using 4-point
/// Catmull-Rom cubic Hermite interpolation (real-time safe, zero allocations).
#[inline]
fn sample_stereo_slice_catmull_rom(slice: &[[f32; 2]], pos: f64) -> Option<[f32; 2]> {
    let len = slice.len();
    if len == 0 || !pos.is_finite() || pos < 0.0 {
        return None;
    }
    let idx1 = pos.floor() as usize;
    if idx1 >= len {
        return None;
    }
    let frac = (pos - idx1 as f64) as f32;
    if frac == 0.0 {
        return Some(slice[idx1]);
    }
    let last = len - 1;
    let idx0 = idx1.saturating_sub(1);
    let idx2 = (idx1 + 1).min(last);
    let idx3 = (idx1 + 2).min(last);
    let s0 = slice[idx0];
    let s1 = slice[idx1];
    let s2 = slice[idx2];
    let s3 = slice[idx3];
    Some([
        catmull_rom_sample(s0[0], s1[0], s2[0], s3[0], frac),
        catmull_rom_sample(s0[1], s1[1], s2[1], s3[1], frac),
    ])
}

#[allow(clippy::too_many_arguments)]
fn render_callback_frames_windowed_with_loop_head<T, F>(
    shared: &ClockShared,
    mix: Option<&Vec<[f32; 2]>>,
    mix_origin_frame: f64,
    loop_head: Option<&Vec<[f32; 2]>>,
    mix_sample_rate: f64,
    channels: usize,
    delay_secs: f64,
    output: &mut [T],
    convert: F,
) where
    F: Fn(f32) -> T,
{
    if channels == 0 || output.is_empty() {
        return;
    }
    shared
        .last_callback_nanos
        .store(shared.now_nanos(), Ordering::Release);
    let frame_count = (output.len() / channels) as u64;
    let snap = shared.read_snapshot();
    // Read the current submitted sample cursor BEFORE filling, then advance it AFTER
    // filling the buffer so the visual playhead never jumps ahead of an unfilled buffer.
    let start_hw = shared.hardware_sample_count.load(Ordering::Acquire);

    if snap.speed == 0.0 || snap.fps <= 0.0 {
        // Paused or seqlock-contention fallback (`speed == 0.0`): emit silence without
        // latching `callback_latched_epoch` so the first active callback for a newly
        // committed epoch still starts at its exact anchor frame with zero skipped samples.
        for slot in output.iter_mut() {
            *slot = convert(0.0);
        }
    } else {
        // Latch the exact submission cursor where this active epoch first starts rendering
        // so an in-flight callback racing a seek/resume can never skip a buffer.
        let epoch_anchor = if shared.callback_latched_epoch.load(Ordering::Relaxed) != snap.epoch_id
        {
            shared
                .callback_epoch_start_sample
                .store(start_hw, Ordering::Release);
            shared
                .callback_latched_epoch
                .store(snap.epoch_id, Ordering::Release);
            start_hw
        } else {
            shared.callback_epoch_start_sample.load(Ordering::Relaxed)
        };

        if mix.is_some() || loop_head.is_some() {
            let sample_slice =
                |slice: &[[f32; 2]], origin: f64, exact_frame: f64| -> Option<[f32; 2]> {
                    let timeline_sec = (exact_frame - origin) / snap.fps;
                    let pos = timeline_sec * mix_sample_rate;
                    sample_stereo_slice_catmull_rom(slice, pos)
                };

            for (i, frame_chunk) in output.chunks_exact_mut(channels).enumerate() {
                let hw = start_hw.saturating_add(i as u64);
                let (exact_frame, hit_stop) =
                    ClockShared::evaluate_at_sample_from(&snap, hw, epoch_anchor);
                let sample_pair = if hit_stop {
                    [0.0_f32, 0.0_f32]
                } else {
                    mix.and_then(|m| sample_slice(m.as_slice(), mix_origin_frame, exact_frame))
                        .or_else(|| {
                            loop_head.and_then(|lh| sample_slice(lh.as_slice(), 0.0, exact_frame))
                        })
                        .unwrap_or([0.0_f32, 0.0_f32])
                };

                for (ch_idx, slot) in frame_chunk.iter_mut().enumerate() {
                    let val = match (channels, ch_idx) {
                        (1, _) => 0.5 * (sample_pair[0] + sample_pair[1]),
                        (_, 0) => sample_pair[0],
                        (_, 1) => sample_pair[1],
                        _ => 0.0,
                    };
                    *slot = convert(val);
                }
            }
        } else {
            for slot in output.iter_mut() {
                *slot = convert(0.0);
            }
        }
    }

    // Account for samples queued for playback and bound the estimate to two seconds.
    let reported_delay_samples = (delay_secs * f64::from(snap.sample_rate)).round().max(0.0) as u64;
    let max_reasonable_latency = u64::from(snap.sample_rate).saturating_mul(2);
    let min_latency = frame_count.min(max_reasonable_latency);
    let latency_samples = reported_delay_samples.clamp(min_latency, max_reasonable_latency);
    shared
        .output_latency_samples
        .store(latency_samples, Ordering::Release);
    shared
        .hardware_sample_count
        .fetch_add(frame_count, Ordering::Release);
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[allow(clippy::too_many_arguments)]
fn render_callback_frames_windowed<T, F>(
    shared: &ClockShared,
    mix: Option<&Vec<[f32; 2]>>,
    mix_origin_frame: f64,
    mix_sample_rate: f64,
    channels: usize,
    delay_secs: f64,
    output: &mut [T],
    convert: F,
) where
    F: Fn(f32) -> T,
{
    render_callback_frames_windowed_with_loop_head(
        shared,
        mix,
        mix_origin_frame,
        None,
        mix_sample_rate,
        channels,
        delay_secs,
        output,
        convert,
    );
}

/// RAII handle owning a live audio output stream attached to a [`MasterPlaybackClock`].
///
/// Dropping this handle cleanly detaches the clock back to wall-clock fallback mode and
/// stops/releases the underlying OS audio stream with zero leaked resources.
pub struct AudioOutputHandle {
    #[cfg(not(target_arch = "wasm32"))]
    stream: Option<cpal::Stream>,
    #[cfg(target_arch = "wasm32")]
    stream: Option<web::WebAudioStream>,
    stream_id: u64,
    shared: Arc<ClockShared>,
    sample_rate: u32,
    channels: u16,
    window_slot: Arc<Mutex<Option<AudioMixWindow>>>,
    window_gen: Arc<AtomicU64>,
    loop_head_slot: LoopHeadSlot,
    loop_head_gen: Arc<AtomicU64>,
}

impl AudioOutputHandle {
    /// Opens a live output stream synchronized to `clock`.
    pub fn start_with_window(
        clock: &MasterPlaybackClock,
        window: Option<AudioMixWindow>,
    ) -> Result<Self, AudioError> {
        clock.start_audio_output_windowed(window)
    }

    /// Returns whether this handle still owns an output stream.
    #[must_use]
    pub const fn is_active(&self) -> bool {
        self.stream.is_some()
    }

    /// Stops the stream and returns its clock to wall-clock playback.
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.take() {
            drop(stream);
            self.shared.detach_hardware_clock(self.stream_id);
        }
    }

    /// Device output sample rate in Hz.
    #[must_use]
    pub const fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Device output channel count.
    #[must_use]
    pub const fn channels(&self) -> u16 {
        self.channels
    }

    /// Seamlessly swaps the active [`AudioMixWindow`] used by the live audio callback
    /// without dropping or recreating the underlying OS audio stream.
    pub fn update_window(&self, window: Option<AudioMixWindow>) {
        let mut guard = self
            .window_slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *guard = window;
        drop(guard);
        self.window_gen.fetch_add(1, Ordering::Release);
    }

    /// Supplies a pre-mixed loop-head audio buffer anchored at frame `0.0` so looping
    /// timelines longer than one window never emit silence at the loop wrap seam.
    pub fn set_loop_head(&self, loop_head: Option<Arc<Vec<[f32; 2]>>>) {
        let mut guard = self
            .loop_head_slot
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        *guard = loop_head;
        drop(guard);
        self.loop_head_gen.fetch_add(1, Ordering::Release);
    }
}

impl Drop for AudioOutputHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn master_clock_seek_reanchors_without_desync() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);

        let initial_epoch = clock.epoch();
        clock.play();
        assert_eq!(clock.epoch(), initial_epoch + 1);
        assert!(clock.is_playing());

        // Advance 1.0 second of hardware audio samples -> exact_frame must be 30.0
        clock.advance_hardware_samples(48_000);
        assert!((clock.exact_frame() - 30.0).abs() < 1e-6);

        // Scrub inside an active clip to frame 120.0 -> pauses and bumps epoch
        let epoch_after_scrub = clock.set_frame(120.0);
        assert!(epoch_after_scrub > initial_epoch + 1);
        assert!(!clock.is_playing());
        assert!((clock.exact_frame() - 120.0).abs() < 1e-6);

        // Even if hardware audio callbacks keep firing silence while paused, frame stays at 120.0
        clock.advance_hardware_samples(24_000);
        assert!((clock.exact_frame() - 120.0).abs() < 1e-6);

        // Resume playback: must start from EXACTLY 120.0 with zero jump/desync!
        clock.play();
        assert!((clock.exact_frame() - 120.0).abs() < 1e-6);

        // Advance 0.5s of audio -> must be exactly 135.0
        clock.advance_hardware_samples(24_000);
        assert!((clock.exact_frame() - 135.0).abs() < 1e-6);

        // Seek while playing (preserving playback) to frame 50.0
        let epoch_before_seek = clock.epoch();
        clock.seek_preserving_playback(50.0);
        assert_eq!(clock.epoch(), epoch_before_seek + 1);
        assert!(clock.is_playing());
        assert!((clock.exact_frame() - 50.0).abs() < 1e-6);

        // Advance 0.25s -> exactly 57.5
        clock.advance_hardware_samples(12_000);
        assert!((clock.exact_frame() - 57.5).abs() < 1e-6);
    }

    #[test]
    fn master_clock_loop_wrap_and_non_loop_stop() {
        let clock = MasterPlaybackClock::new(60.0, 60.0); // 1 second sequence
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);

        // With loop enabled, advancing 1.25s wraps to frame 15.0
        clock.set_loop(true);
        clock.play();
        clock.advance_hardware_samples(60_000);
        assert!((clock.exact_frame() - 15.0).abs() < 1e-6);
        assert!(clock.is_playing());

        // Disable loop and seek to frame 50; advancing 0.5s hits end and pauses at frame 59
        clock.set_loop(false);
        clock.seek_preserving_playback(50.0);
        clock.advance_hardware_samples(24_000);
        assert!((clock.exact_frame() - 59.0).abs() < 1e-6);
        assert!(!clock.is_playing());
    }

    #[test]
    fn seek_racing_inflight_callback_does_not_skip_first_buffer() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock.play();

        // Build a ramp mix where sample value encodes exact sample index / 100000.0.
        let mix: Vec<[f32; 2]> = (0..96_000)
            .map(|i| {
                let v = (i as f32) / 100_000.0;
                [v, v]
            })
            .collect();

        // Simulate callback #1 capturing old epoch before seek, then seek happens mid-callback,
        // then callback #1 finishes and advances `hardware_sample_count` by 512 frames.
        clock.seek_preserving_playback(30.0); // frame 30 = 1.0s = sample 48_000
        clock
            .shared
            .hardware_sample_count
            .fetch_add(512, Ordering::Release);

        // Now callback #2 starts with the new epoch and fills 512 stereo frames.
        let mut out = vec![0.0_f32; 512 * 2];
        render_callback_frames(
            &clock.shared,
            Some(&mix),
            f64::from(STANDARD_SAMPLE_RATE),
            2,
            0.0,
            &mut out,
            |s| s,
        );

        // Sample 0 of callback #2 must be at EXACTLY sample 48_000 (0.48), NOT 48_512!
        assert!(
            (out[0] - 0.48).abs() < 1e-5,
            "expected first sample at exact seek target 0.48, got {}",
            out[0]
        );
        // Visual playhead stays at 30.0 while that 512-sample buffer is queued at the DAC.
        assert!((clock.exact_frame() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn seqlock_contention_fallback_does_not_latch_new_epoch() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock.play();

        let mix: Vec<[f32; 2]> = (0..96_000)
            .map(|i| {
                let v = (i as f32) / 100_000.0;
                [v, v]
            })
            .collect();

        // Render one normal callback on the initial playing epoch.
        let mut out = vec![0.0_f32; 512 * 2];
        render_callback_frames(
            &clock.shared,
            Some(&mix),
            f64::from(STANDARD_SAMPLE_RATE),
            2,
            0.0,
            &mut out,
            |s| s,
        );
        let prev_epoch = clock.epoch();

        // Seek to frame 30 (1.0s = sample 48_000), then simulate the seqlock still being
        // held odd by the writer when the audio callback runs and hits the fallback path.
        let new_epoch = clock.seek_preserving_playback(30.0);
        assert_ne!(new_epoch, prev_epoch);
        let prev_seq = clock.shared.seq.load(Ordering::Acquire);
        clock
            .shared
            .seq
            .store(prev_seq.wrapping_add(1), Ordering::Release);

        let mut fallback_out = vec![1.0_f32; 512 * 2];
        render_callback_frames(
            &clock.shared,
            Some(&mix),
            f64::from(STANDARD_SAMPLE_RATE),
            2,
            0.0,
            &mut fallback_out,
            |s| s,
        );
        // Fallback callback must emit silence and MUST NOT latch `new_epoch`.
        assert!(fallback_out.iter().all(|&s| s == 0.0));
        assert_eq!(
            clock.shared.callback_latched_epoch.load(Ordering::Acquire),
            prev_epoch
        );

        // Writer completes the seqlock update (even).
        clock
            .shared
            .seq
            .store(prev_seq.wrapping_add(2), Ordering::Release);

        // Next callback renders `new_epoch` starting at exact sample 48_000 (0.48), NOT 48_512.
        let mut active_out = vec![0.0_f32; 512 * 2];
        render_callback_frames(
            &clock.shared,
            Some(&mix),
            f64::from(STANDARD_SAMPLE_RATE),
            2,
            0.0,
            &mut active_out,
            |s| s,
        );
        assert_eq!(
            clock.shared.callback_latched_epoch.load(Ordering::Acquire),
            new_epoch
        );
        assert!(
            (active_out[0] - 0.48).abs() < 1e-5,
            "expected first active sample at exact seek target 0.48, got {}",
            active_out[0]
        );
        assert!((clock.exact_frame() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn hardware_reattach_bumps_epoch_and_clears_latency_without_double_count() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock.play();

        // Advance 1.0s of hardware samples with non-zero output latency recorded.
        clock
            .shared
            .output_latency_samples
            .store(512, Ordering::Release);
        clock.advance_hardware_samples(48_000);
        let frame_before_detach = clock.exact_frame();
        assert!((frame_before_detach - ((48_000.0 - 512.0) / 48_000.0) * 30.0).abs() < 1e-4);

        // Detach back to wall-clock mode: latency must reset to 0 and epoch must bump.
        let epoch_before_detach = clock.epoch();
        clock.set_hardware_clock_mode(false, STANDARD_SAMPLE_RATE);
        assert!(clock.epoch() > epoch_before_detach);
        assert_eq!(
            clock.shared.output_latency_samples.load(Ordering::Acquire),
            0
        );

        // Re-attach hardware mode: epoch must bump again so stale callback latch is ignored
        // and prior hardware samples are never double-counted.
        let epoch_before_reattach = clock.epoch();
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        assert!(clock.epoch() > epoch_before_reattach);
        let frame_after_reattach = clock.exact_frame();
        assert!(
            (frame_after_reattach - frame_before_detach).abs() < 5.0,
            "expected no double-count jump on re-attach: before={frame_before_detach}, after={frame_after_reattach}"
        );
    }

    #[test]
    fn detach_ignores_stale_stream_id() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock
            .shared
            .output_latency_samples
            .store(256, Ordering::Release);
        clock.shared.active_stream_id.store(2, Ordering::Release);

        // Stale stream handle (id 1) must NOT clobber active stream (id 2).
        clock.shared.detach_hardware_clock(1);
        assert!(clock.snapshot().hardware_clock_active);
        assert_eq!(
            clock.shared.output_latency_samples.load(Ordering::Acquire),
            256
        );

        // Matching stream handle (id 2) cleanly detaches and clears output latency.
        clock.shared.detach_hardware_clock(2);
        assert!(!clock.snapshot().hardware_clock_active);
        assert_eq!(
            clock.shared.output_latency_samples.load(Ordering::Acquire),
            0
        );
    }

    #[test]
    fn current_time_secs_preserves_fractional_sample_precision() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock.play();

        // Advance 0.25s = 12,000 samples => exact_frame = 7.5 => current_time_secs = 0.25,
        // NOT round(7.5) / 30 = 8 / 30 = 0.266667.
        clock.advance_hardware_samples(12_000);
        assert!((clock.exact_frame() - 7.5).abs() < 1e-6);
        assert!(
            (clock.current_time_secs() - 0.25).abs() < 1e-6,
            "expected exact 0.25s, got {}",
            clock.current_time_secs()
        );
    }

    #[test]
    fn hardware_clock_falls_back_to_wall_clock_when_live_callback_stalls() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock.shared.active_stream_id.store(1, Ordering::Release);
        clock.play();

        // Simulate wall_anchor_nanos = 0 and now_nanos = 1_000_000_000 (1.0s elapsed without callback latching epoch).
        let snap = clock.snapshot();
        let (frame, hit_stop) = clock.shared.evaluate_snapshot_at(&snap, 0, 1_000_000_000);
        assert!(!hit_stop);
        assert!(
            (frame - 30.0).abs() < 1e-6,
            "stalled unlatched live stream must fall back to wall clock (30.0), got {frame}"
        );
    }

    #[test]
    fn loop_head_eliminates_loop_seam_silence_on_windowed_timeline() {
        let clock = MasterPlaybackClock::new(30.0, 300.0);
        clock.set_hardware_clock_mode(true, STANDARD_SAMPLE_RATE);
        clock.set_loop(true);
        // Playhead at last frame (299.0), active window starts at frame 240.0.
        clock.seek_preserving_playback(299.0);
        clock.play();

        let tail_window = vec![[0.25_f32, 0.25]; 96_000];
        let loop_head = vec![[0.75_f32, 0.75]; 48_000];

        // 1 frame at 30 fps = 1600 samples at 48 kHz. Rendering 2400 samples spans 299.0..300.5 -> wraps to 0.0..0.5!
        let mut out = vec![0.0_f32; 2_400 * 2];
        render_callback_frames_windowed_with_loop_head(
            &clock.shared,
            Some(&tail_window),
            240.0,
            Some(&loop_head),
            f64::from(STANDARD_SAMPLE_RATE),
            2,
            0.0,
            &mut out,
            |s| s,
        );
        // Before wrap (sample 0, frame 299.0): comes from tail_window (0.25)
        assert!((out[0] - 0.25).abs() < 1e-5);
        // After wrap (sample 2000, frame ~0.25 < 240.0): seamlessly comes from loop_head (0.75), NOT silence!
        assert!(
            (out[2_000 * 2] - 0.75).abs() < 1e-5,
            "expected loop_head sample 0.75 after wrap, got {}",
            out[2_000 * 2]
        );
    }

    #[test]
    fn cpal_xrun_classified_as_transient_while_device_errors_remain_errors() {
        assert!(is_transient_cpal_xrun(&cpal::Error::new(
            cpal::ErrorKind::Xrun
        )));
        assert!(is_transient_cpal_xrun(&cpal::Error::with_message(
            cpal::ErrorKind::BackendError,
            "A buffer underrun or overrun occurred."
        )));
        assert!(is_transient_cpal_xrun(&cpal::Error::with_message(
            cpal::ErrorKind::Other,
            "CoreAudio buffer OVERRUN on startup"
        )));
        assert!(!is_transient_cpal_xrun(&cpal::Error::new(
            cpal::ErrorKind::DeviceNotAvailable
        )));
        assert!(!is_transient_cpal_xrun(&cpal::Error::new(
            cpal::ErrorKind::HostUnavailable
        )));
        assert!(!is_transient_cpal_xrun(&cpal::Error::new(
            cpal::ErrorKind::StreamInvalidated
        )));
        assert!(!is_transient_cpal_xrun(&cpal::Error::new(
            cpal::ErrorKind::BackendError
        )));
    }

    #[test]
    fn live_catmull_rom_resampling_preserves_ramps_and_beats_linear_on_sine() {
        // Out-of-bounds and empty guards.
        assert_eq!(sample_stereo_slice_catmull_rom(&[], 0.0), None);
        let ramp: Vec<[f32; 2]> = (0..64)
            .map(|i| {
                let v = i as f32 * 0.01;
                [v, -v]
            })
            .collect();
        assert_eq!(sample_stereo_slice_catmull_rom(&ramp, -0.1), None);
        assert_eq!(sample_stereo_slice_catmull_rom(&ramp, 64.0), None);
        assert_eq!(sample_stereo_slice_catmull_rom(&ramp, f64::NAN), None);

        // Exact reproduction on linear ramp at integer and fractional positions.
        for &pos in &[0.0, 10.0, 10.25, 10.5, 31.75] {
            let s = sample_stereo_slice_catmull_rom(&ramp, pos).expect("in range");
            let expected = (pos * 0.01) as f32;
            assert!(
                (s[0] - expected).abs() < 1e-5 && (s[1] + expected).abs() < 1e-5,
                "ramp mismatch at pos={pos}: got={s:?}, expected=[{expected}, -{}]",
                expected
            );
        }

        // High-frequency sine at 6 kHz / 48 kHz resampled at fractional step 0.73:
        // 4-point Catmull-Rom must be dramatically cleaner than 2-point linear interpolation.
        let sr = 48_000.0_f64;
        let freq = 6_000.0_f64;
        let len = 512_usize;
        let sine: Vec<[f32; 2]> = (0..len)
            .map(|i| {
                let v = (std::f64::consts::TAU * freq * (i as f64) / sr).sin() as f32;
                [v, v]
            })
            .collect();

        let mut cr_err_sq = 0.0_f64;
        let mut lin_err_sq = 0.0_f64;
        let mut count = 0_usize;
        for step in 16..480 {
            let pos = step as f64 + 0.37;
            let expected = (std::f64::consts::TAU * freq * pos / sr).sin();
            let cr = sample_stereo_slice_catmull_rom(&sine, pos).unwrap()[0];
            let err_cr = f64::from(cr) - expected;
            cr_err_sq += err_cr * err_cr;

            let i0 = pos.floor() as usize;
            let frac = pos - i0 as f64;
            let lin = f64::from(sine[i0][0])
                + (f64::from(sine[i0 + 1][0]) - f64::from(sine[i0][0])) * frac;
            let err_lin = lin - expected;
            lin_err_sq += err_lin * err_lin;
            count += 1;
        }
        let cr_rms = (cr_err_sq / count as f64).sqrt();
        let lin_rms = (lin_err_sq / count as f64).sqrt();
        assert!(
            cr_rms < 0.015,
            "Catmull-Rom fractional sine RMS error too high: {cr_rms}"
        );
        assert!(
            lin_rms > cr_rms * 3.0,
            "expected Catmull-Rom ({cr_rms}) to significantly beat linear ({lin_rms})"
        );
    }
}
