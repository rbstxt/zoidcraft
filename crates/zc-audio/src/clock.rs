//! Sample-accurate master playback clock with lock-free atomic epoch tracking
//! and optional `cpal` live audio output stream (Spec 02 §5.4).
//!
//! Eliminates CM3's HTMLMediaElement seek-desync defects by anchoring playhead
//! advancement directly to monotonic hardware audio sample counts (or a wall-clock
//! fallback in headless environments) and invalidating epochs atomically on every
//! scrub, loop wrap, speed change, or clip schedule edit.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering, fence};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::AudioError;
use crate::decode::STANDARD_SAMPLE_RATE;

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

struct ClockShared {
    /// Seqlock counter (odd while a control-thread mutation is in flight, even when stable).
    seq: AtomicU64,
    /// Serializes control-thread writers (never locked inside the real-time audio callback).
    writer_lock: Mutex<()>,
    /// Wall-clock anchor used when no hardware audio stream is active.
    wall_anchor: Mutex<Instant>,

    epoch_id: AtomicU64,
    anchor_exact_frame_bits: AtomicU64,
    anchor_sample_count: AtomicU64,
    hardware_sample_count: AtomicU64,
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
            wall_anchor: Mutex::new(Instant::now()),
            epoch_id: AtomicU64::new(1),
            anchor_exact_frame_bits: AtomicU64::new(0.0_f64.to_bits()),
            anchor_sample_count: AtomicU64::new(0),
            hardware_sample_count: AtomicU64::new(0),
            speed_bits: AtomicU64::new(0.0_f64.to_bits()),
            fps_bits: AtomicU64::new(clean_fps.to_bits()),
            total_frames_bits: AtomicU64::new(clean_total.to_bits()),
            loop_enabled: AtomicBool::new(false),
            sample_rate: AtomicU32::new(clean_sr),
            hardware_clock_active: AtomicBool::new(false),
        }
    }

    /// Lock-free, tear-free read of the current clock parameters (safe in real-time audio callbacks).
    fn read_snapshot(&self) -> PlaybackSnapshot {
        for _ in 0..64 {
            let s1 = self.seq.load(Ordering::Acquire);
            if s1 & 1 != 0 {
                std::hint::spin_loop();
                continue;
            }
            let snap = self.read_snapshot_relaxed();
            fence(Ordering::Acquire);
            let s2 = self.seq.load(Ordering::Relaxed);
            if s1 == s2 {
                return snap;
            }
            std::hint::spin_loop();
        }
        // Fallback after bounded spin: return a paused snapshot to avoid ever blocking audio.
        let mut snap = self.read_snapshot_relaxed();
        snap.speed = 0.0;
        snap
    }

    #[inline]
    fn read_snapshot_relaxed(&self) -> PlaybackSnapshot {
        PlaybackSnapshot {
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
        }
    }

    /// Computes exact frame at a given hardware sample count without mutating state.
    #[inline]
    fn evaluate_at_sample(snap: &PlaybackSnapshot, hw_sample: u64) -> (f64, bool) {
        if snap.speed == 0.0 {
            return (snap.anchor_exact_frame, false);
        }
        let elapsed_samples = hw_sample.saturating_sub(snap.anchor_sample_count) as f64;
        let elapsed_sec = elapsed_samples / f64::from(snap.sample_rate);
        let raw_frame = snap.anchor_exact_frame + elapsed_sec * snap.fps * snap.speed;
        apply_bounds(raw_frame, snap.total_frames, snap.loop_enabled)
    }

    /// Computes current exact frame (using hardware counter if active, else monotonic wall clock).
    fn current_exact_frame_and_stop_flag(&self) -> (f64, bool, PlaybackSnapshot) {
        let snap = self.read_snapshot();
        if snap.speed == 0.0 {
            return (snap.anchor_exact_frame, false, snap);
        }

        let raw_frame = if snap.hardware_clock_active {
            let hw = self.hardware_sample_count.load(Ordering::Acquire);
            let elapsed_samples = hw.saturating_sub(snap.anchor_sample_count) as f64;
            let elapsed_sec = elapsed_samples / f64::from(snap.sample_rate);
            snap.anchor_exact_frame + elapsed_sec * snap.fps * snap.speed
        } else {
            let elapsed_sec = self
                .wall_anchor
                .lock()
                .map(|inst| inst.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            snap.anchor_exact_frame + elapsed_sec * snap.fps * snap.speed
        };

        let (bounded, hit_stop) = apply_bounds(raw_frame, snap.total_frames, snap.loop_enabled);
        (bounded, hit_stop, snap)
    }

    /// Mutates the clock atomically under the control-thread seqlock and bumps `epoch_id`.
    fn commit_update<F>(&self, bump_epoch: bool, mutate: F) -> u64
    where
        F: FnOnce(&mut PlaybackSnapshot, u64),
    {
        let _guard = self
            .writer_lock
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());

        let hw_now = self.hardware_sample_count.load(Ordering::Acquire);
        let mut snap = self.read_snapshot_relaxed();
        mutate(&mut snap, hw_now);

        if bump_epoch {
            snap.epoch_id = snap.epoch_id.wrapping_add(1);
        }

        // Begin seqlock write (odd)
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

        if let Ok(mut wall) = self.wall_anchor.lock() {
            *wall = Instant::now();
        }

        // End seqlock write (even)
        self.seq.store(prev_seq.wrapping_add(2), Ordering::Release);
        snap.epoch_id
    }

    fn detach_hardware_clock(&self) {
        let (current_frame, hit_stop, _) = self.current_exact_frame_and_stop_flag();
        self.commit_update(false, |snap, hw_now| {
            snap.anchor_exact_frame = current_frame;
            snap.anchor_sample_count = hw_now;
            snap.hardware_clock_active = false;
            if hit_stop {
                snap.speed = 0.0;
            }
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
        let (frame, hit_stop, _) = self.shared.current_exact_frame_and_stop_flag();
        if hit_stop {
            self.shared.commit_update(true, |snap, hw_now| {
                snap.anchor_exact_frame = frame;
                snap.anchor_sample_count = hw_now;
                snap.speed = 0.0;
            });
        }
        frame
    }

    /// Integer frame passed to sequence evaluation and compositing (`round(exact_frame)`).
    #[must_use]
    pub fn current_frame(&self) -> u64 {
        self.exact_frame().round().max(0.0) as u64
    }

    /// Current timeline time in seconds (`current_frame / fps`).
    #[must_use]
    pub fn current_time_secs(&self) -> f64 {
        let snap = self.snapshot();
        if snap.fps <= 0.0 {
            0.0
        } else {
            self.current_frame() as f64 / snap.fps
        }
    }

    /// Evaluates the exact timeline frame corresponding to a specific hardware sample counter.
    #[must_use]
    pub fn exact_frame_at_sample(&self, sample_count: u64) -> f64 {
        let snap = self.snapshot();
        ClockShared::evaluate_at_sample(&snap, sample_count).0
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
        let (frame, _, _) = self.shared.current_exact_frame_and_stop_flag();
        self.shared.commit_update(true, |snap, hw_now| {
            snap.anchor_exact_frame = frame;
            snap.anchor_sample_count = hw_now;
            snap.speed = clean_speed;
        })
    }

    /// Scrubs immediately to `frame`, pausing playback (`speed = 0.0`), clamping to
    /// `[0, max(total_frames - 1, 0)]`, and incrementing `epoch_id` (Spec 02 §5.1).
    pub fn set_frame(&self, frame: f64) -> u64 {
        self.shared.commit_update(true, |snap, hw_now| {
            let max_frame = (snap.total_frames - 1.0).max(0.0);
            let clamped = if frame.is_finite() {
                frame.clamp(0.0, max_frame)
            } else {
                0.0
            };
            snap.anchor_exact_frame = clamped;
            snap.anchor_sample_count = hw_now;
            snap.speed = 0.0;
        })
    }

    /// Seeks immediately to `frame` while preserving the active playback speed,
    /// re-anchoring `anchor_sample_count` to the current hardware sample counter and
    /// bumping `epoch_id` so audio and video reposition with zero desync.
    pub fn seek_preserving_playback(&self, frame: f64) -> u64 {
        self.shared.commit_update(true, |snap, hw_now| {
            let max_frame = (snap.total_frames - 1.0).max(0.0);
            let clamped = if frame.is_finite() {
                frame.clamp(0.0, max_frame)
            } else {
                0.0
            };
            snap.anchor_exact_frame = clamped;
            snap.anchor_sample_count = hw_now;
        })
    }

    /// Enables or disables loop wrapping at sequence boundaries.
    pub fn set_loop(&self, enabled: bool) {
        self.shared.commit_update(false, |snap, _hw| {
            snap.loop_enabled = enabled;
        });
    }

    /// Updates sequence `fps` and `total_frames`, clamping the current playhead and bumping epoch.
    pub fn set_timeline_config(&self, fps: f64, total_frames: f64) -> u64 {
        let (frame, _, _) = self.shared.current_exact_frame_and_stop_flag();
        self.shared.commit_update(true, |snap, hw_now| {
            if fps.is_finite() && fps > 0.0 {
                snap.fps = fps;
            }
            if total_frames.is_finite() && total_frames >= 0.0 {
                snap.total_frames = total_frames;
            }
            let max_frame = (snap.total_frames - 1.0).max(0.0);
            snap.anchor_exact_frame = frame.clamp(0.0, max_frame);
            snap.anchor_sample_count = hw_now;
        })
    }

    /// Invalidates downstream clip/decoder schedules after a timeline edit while preserving playhead.
    pub fn invalidate_schedule(&self) -> u64 {
        let (frame, _, _) = self.shared.current_exact_frame_and_stop_flag();
        self.shared.commit_update(true, |snap, hw_now| {
            snap.anchor_exact_frame = frame;
            snap.anchor_sample_count = hw_now;
        })
    }

    /// Enables or disables hardware sample-counter clock mode manually (useful for deterministic tests
    /// and headless drivers without an OS audio device).
    pub fn set_hardware_clock_mode(&self, active: bool, sample_rate: u32) {
        let (frame, _, _) = self.shared.current_exact_frame_and_stop_flag();
        self.shared.commit_update(false, |snap, hw_now| {
            snap.anchor_exact_frame = frame;
            snap.anchor_sample_count = hw_now;
            snap.hardware_clock_active = active;
            if sample_rate > 0 {
                snap.sample_rate = sample_rate;
            }
        });
    }

    /// Advances the hardware sample counter by `sample_frames` (called by the `cpal` callback or tests).
    pub fn advance_hardware_samples(&self, sample_frames: u64) -> u64 {
        self.shared
            .hardware_sample_count
            .fetch_add(sample_frames, Ordering::AcqRel)
            .saturating_add(sample_frames)
    }

    /// Attempts to open a live `cpal` audio output stream synchronized to this clock.
    ///
    /// Gracefully returns `None` (keeping silent wall-clock fallback active) if no audio output
    /// device is available (e.g., headless servers or CI environments).
    #[must_use]
    pub fn try_start_audio_output(
        &self,
        stereo_mix_48k: Option<Arc<Vec<[f32; 2]>>>,
    ) -> Option<AudioOutputHandle> {
        match self.start_audio_output(stereo_mix_48k) {
            Ok(handle) => Some(handle),
            Err(err) => {
                tracing::warn!(
                    "live audio output unavailable ({err}); falling back to silent wall-clock mode"
                );
                None
            }
        }
    }

    /// Opens a live `cpal` audio output stream synchronized to this clock.
    ///
    /// Dropping the returned [`AudioOutputHandle`] stops the stream, releases OS audio resources,
    /// and seamlessly transitions the clock back to wall-clock mode.
    pub fn start_audio_output(
        &self,
        stereo_mix_48k: Option<Arc<Vec<[f32; 2]>>>,
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

        // Anchor clock to hardware sample counter before starting playback.
        let (frame, _, _) = self.shared.current_exact_frame_and_stop_flag();
        self.shared.commit_update(false, |snap, hw_now| {
            snap.anchor_exact_frame = frame;
            snap.anchor_sample_count = hw_now;
            snap.sample_rate = device_sr;
            snap.hardware_clock_active = true;
        });

        let shared_cb = Arc::clone(&self.shared);
        let err_fn = |err| {
            tracing::error!("cpal output stream error: {err}");
        };

        let ch_usize = usize::from(device_channels);
        let mix_sr = f64::from(STANDARD_SAMPLE_RATE);

        let stream_result = match sample_format {
            SampleFormat::F32 => device.build_output_stream(
                config,
                move |data: &mut [f32], _| {
                    render_callback_frames(
                        &shared_cb,
                        stereo_mix_48k.as_deref(),
                        mix_sr,
                        ch_usize,
                        data,
                        |s| s,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::I16 => device.build_output_stream(
                config,
                move |data: &mut [i16], _| {
                    render_callback_frames(
                        &shared_cb,
                        stereo_mix_48k.as_deref(),
                        mix_sr,
                        ch_usize,
                        data,
                        |s| (s.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16,
                    );
                },
                err_fn,
                None,
            ),
            SampleFormat::U16 => device.build_output_stream(
                config,
                move |data: &mut [u16], _| {
                    render_callback_frames(
                        &shared_cb,
                        stereo_mix_48k.as_deref(),
                        mix_sr,
                        ch_usize,
                        data,
                        |s| ((s.clamp(-1.0, 1.0) * 0.5 + 0.5) * f32::from(u16::MAX)).round() as u16,
                    );
                },
                err_fn,
                None,
            ),
            other => {
                self.shared.detach_hardware_clock();
                return Err(AudioError::OutputDevice(format!(
                    "unsupported output sample format: {other:?}"
                )));
            }
        };

        let stream = match stream_result {
            Ok(stream) => stream,
            Err(err) => {
                self.shared.detach_hardware_clock();
                return Err(AudioError::OutputDevice(err.to_string()));
            }
        };

        if let Err(err) = stream.play() {
            self.shared.detach_hardware_clock();
            drop(stream);
            return Err(AudioError::OutputDevice(err.to_string()));
        }

        Ok(AudioOutputHandle {
            _stream: stream,
            shared: Arc::clone(&self.shared),
            sample_rate: device_sr,
            channels: device_channels,
        })
    }
}

fn render_callback_frames<T, F>(
    shared: &ClockShared,
    mix: Option<&Vec<[f32; 2]>>,
    mix_sample_rate: f64,
    channels: usize,
    output: &mut [T],
    convert: F,
) where
    F: Fn(f32) -> T,
{
    if channels == 0 || output.is_empty() {
        return;
    }
    let frame_count = output.len() / channels;
    let start_hw = shared
        .hardware_sample_count
        .fetch_add(frame_count as u64, Ordering::AcqRel);
    let snap = shared.read_snapshot();

    if snap.speed == 0.0 || mix.is_none() || snap.fps <= 0.0 {
        for slot in output.iter_mut() {
            *slot = convert(0.0);
        }
        return;
    }

    let mix_slice = mix.map(Vec::as_slice).unwrap_or(&[]);
    let mix_len = mix_slice.len();

    for (i, frame_chunk) in output.chunks_exact_mut(channels).enumerate() {
        let hw = start_hw.saturating_add(i as u64);
        let (exact_frame, hit_stop) = ClockShared::evaluate_at_sample(&snap, hw);
        let sample_pair = if hit_stop || mix_len == 0 {
            [0.0_f32, 0.0_f32]
        } else {
            let timeline_sec = exact_frame / snap.fps;
            let pos = timeline_sec * mix_sample_rate;
            if pos >= 0.0 && (pos.floor() as usize) < mix_len {
                let idx0 = pos.floor() as usize;
                let idx1 = (idx0 + 1).min(mix_len - 1);
                let frac = (pos - idx0 as f64) as f32;
                let a = mix_slice[idx0];
                let b = mix_slice[idx1];
                [a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]
            } else {
                [0.0, 0.0]
            }
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
}

/// RAII handle owning a live `cpal` output stream attached to a [`MasterPlaybackClock`].
///
/// Dropping this handle cleanly detaches the clock back to wall-clock fallback mode and
/// stops/releases the underlying OS audio stream with zero leaked resources.
pub struct AudioOutputHandle {
    _stream: cpal::Stream,
    shared: Arc<ClockShared>,
    sample_rate: u32,
    channels: u16,
}

impl AudioOutputHandle {
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
}

impl Drop for AudioOutputHandle {
    fn drop(&mut self) {
        self.shared.detach_hardware_clock();
    }
}

#[cfg(test)]
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
}
