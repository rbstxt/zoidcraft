//! Audio clip waveform peak cache and per-pixel column sampler (Spec 02 §1.7).

use crate::decode::AudioBuffer;

/// Default peak cache resolution in buckets per second (Spec 02 §1.7).
pub const DEFAULT_POINTS_PER_SEC: u32 = 200;

/// One 1px horizontal column of a rendered timeline audio waveform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveformColumn {
    /// Minimum signed 8-bit peak (`<= 0`) in this pixel column.
    pub min_i8: i8,
    /// Maximum signed 8-bit peak (`>= 0`) in this pixel column.
    pub max_i8: i8,
    /// Peak-to-peak bar height `|max_i8 - min_i8|` in `0..=255` (Spec 02 §1.7).
    pub amplitude_u8: u8,
}

/// Precomputed signed 8-bit min/max peak buckets for crisp timeline waveform rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WaveformPeaks {
    points_per_sec: u32,
    min_peaks: Vec<i8>,
    max_peaks: Vec<i8>,
}

impl WaveformPeaks {
    /// Builds a peak cache at [`DEFAULT_POINTS_PER_SEC`] (200 buckets/sec).
    #[must_use]
    pub fn from_audio_buffer(buffer: &AudioBuffer) -> Self {
        Self::from_audio_buffer_with_rate(buffer, DEFAULT_POINTS_PER_SEC)
    }

    /// Builds a peak cache at a custom `points_per_sec` resolution.
    #[must_use]
    pub fn from_audio_buffer_with_rate(buffer: &AudioBuffer, points_per_sec: u32) -> Self {
        let total_frames = buffer.frame_count();
        let ch = usize::from(buffer.channels);
        if total_frames == 0 || ch == 0 || buffer.sample_rate == 0 || points_per_sec == 0 {
            return Self {
                points_per_sec,
                min_peaks: Vec::new(),
                max_peaks: Vec::new(),
            };
        }

        let sr = f64::from(buffer.sample_rate);
        let pps = f64::from(points_per_sec);
        let bucket_count = (((total_frames as f64) * pps) / sr).ceil().max(1.0) as usize;
        let mut min_peaks = Vec::with_capacity(bucket_count);
        let mut max_peaks = Vec::with_capacity(bucket_count);
        let inv_ch = 1.0_f32 / ch as f32;

        let samples = &buffer.samples_interleaved_f32;
        match ch {
            1 => {
                for bucket_idx in 0..bucket_count {
                    let start =
                        (((bucket_idx as f64) * sr / pps).floor() as usize).min(total_frames - 1);
                    let end = ((((bucket_idx + 1) as f64) * sr / pps)
                        .floor()
                        .max((start + 1) as f64) as usize)
                        .min(total_frames);
                    let mut min_sum = 0.0_f32;
                    let mut max_sum = 0.0_f32;

                    for &s in &samples[start..end] {
                        if s.is_finite() {
                            if s < min_sum {
                                min_sum = s;
                            }
                            if s > max_sum {
                                max_sum = s;
                            }
                        }
                    }

                    let min_avg = min_sum.clamp(-1.0, 1.0);
                    let max_avg = max_sum.clamp(-1.0, 1.0);
                    min_peaks.push((min_avg * 127.0).clamp(-127.0, 127.0) as i8);
                    max_peaks.push((max_avg * 127.0).clamp(-127.0, 127.0) as i8);
                }
            }
            2 => {
                for bucket_idx in 0..bucket_count {
                    let start =
                        (((bucket_idx as f64) * sr / pps).floor() as usize).min(total_frames - 1);
                    let end = ((((bucket_idx + 1) as f64) * sr / pps)
                        .floor()
                        .max((start + 1) as f64) as usize)
                        .min(total_frames);
                    let mut min_sum = 0.0_f32;
                    let mut max_sum = 0.0_f32;

                    for &[p0, p1] in samples[start * 2..end * 2].as_chunks::<2>().0 {
                        let s0 = if p0.is_finite() { p0 } else { 0.0 };
                        let s1 = if p1.is_finite() { p1 } else { 0.0 };
                        let sum = s0 + s1;
                        if sum < min_sum {
                            min_sum = sum;
                        }
                        if sum > max_sum {
                            max_sum = sum;
                        }
                    }

                    let min_avg = (min_sum * 0.5).clamp(-1.0, 1.0);
                    let max_avg = (max_sum * 0.5).clamp(-1.0, 1.0);
                    min_peaks.push((min_avg * 127.0).clamp(-127.0, 127.0) as i8);
                    max_peaks.push((max_avg * 127.0).clamp(-127.0, 127.0) as i8);
                }
            }
            _ => {
                for bucket_idx in 0..bucket_count {
                    let start =
                        (((bucket_idx as f64) * sr / pps).floor() as usize).min(total_frames - 1);
                    let end = ((((bucket_idx + 1) as f64) * sr / pps)
                        .floor()
                        .max((start + 1) as f64) as usize)
                        .min(total_frames);
                    let mut min_sum = 0.0_f32;
                    let mut max_sum = 0.0_f32;

                    for frame in samples[start * ch..end * ch].chunks_exact(ch) {
                        let mut sum = 0.0_f32;
                        for &s in frame {
                            if s.is_finite() {
                                sum += s;
                            }
                        }
                        if sum < min_sum {
                            min_sum = sum;
                        }
                        if sum > max_sum {
                            max_sum = sum;
                        }
                    }

                    let min_avg = (min_sum * inv_ch).clamp(-1.0, 1.0);
                    let max_avg = (max_sum * inv_ch).clamp(-1.0, 1.0);
                    min_peaks.push((min_avg * 127.0).clamp(-127.0, 127.0) as i8);
                    max_peaks.push((max_avg * 127.0).clamp(-127.0, 127.0) as i8);
                }
            }
        }

        Self {
            points_per_sec,
            min_peaks,
            max_peaks,
        }
    }

    /// Samples `width_px` pixel columns covering `[media_start_sec, media_end_sec]`.
    #[must_use]
    pub fn sample_columns(
        &self,
        media_start_sec: f64,
        media_end_sec: f64,
        width_px: usize,
    ) -> Vec<WaveformColumn> {
        if !media_start_sec.is_finite()
            || !media_end_sec.is_finite()
            || width_px == 0
            || self.is_empty()
            || self.points_per_sec == 0
        {
            return Vec::new();
        }

        let pps = f64::from(self.points_per_sec);
        let len = self.len();
        let len_f = len as f64;
        let width_f = width_px as f64;
        let span_sec = media_end_sec - media_start_sec;
        let col_time = |frac: f64| -> f64 {
            if span_sec.is_finite() {
                media_start_sec + span_sec * frac
            } else {
                media_start_sec * (1.0 - frac) + media_end_sec * frac
            }
        };

        let mut cols = Vec::with_capacity(width_px);
        for x in 0..width_px {
            let b0 = col_time(x as f64 / width_f) * pps;
            let b1 = col_time((x + 1) as f64 / width_f) * pps;
            let b_lo = b0.min(b1);
            let b_hi = b0.max(b1);

            let mut hi: i8 = 0;
            let mut lo: i8 = 0;

            if !b_lo.is_nan()
                && !b_hi.is_nan()
                && (b_lo < len_f || ((b_lo - b_hi).abs() < 1e-9 && (b_lo - len_f).abs() < 1e-6))
                && (b_hi > 0.0 || (b_hi == 0.0 && b_lo == 0.0))
            {
                let start = (b_lo.floor().max(0.0) as usize).min(len.saturating_sub(1));
                let mut end = (b_hi.ceil().clamp(0.0, len_f) as usize).min(len);
                if end <= start && start < len {
                    end = (start + 1).min(len);
                }
                for u in start..end {
                    if self.max_peaks[u] > hi {
                        hi = self.max_peaks[u];
                    }
                    if self.min_peaks[u] < lo {
                        lo = self.min_peaks[u];
                    }
                }
            }

            let amp = (i16::from(hi) - i16::from(lo)).clamp(0, 255) as u8;
            cols.push(WaveformColumn {
                min_i8: lo,
                max_i8: hi,
                amplitude_u8: amp,
            });
        }

        cols
    }

    /// Bucket resolution in points per second.
    #[must_use]
    pub const fn points_per_sec(&self) -> u32 {
        self.points_per_sec
    }

    /// Number of stored peak buckets.
    #[must_use]
    pub fn len(&self) -> usize {
        self.min_peaks.len()
    }

    /// Returns `true` if no peak buckets are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.min_peaks.is_empty()
    }

    /// Raw signed 8-bit minimum peaks per bucket.
    #[must_use]
    pub fn min_peaks(&self) -> &[i8] {
        &self.min_peaks
    }

    /// Raw signed 8-bit maximum peaks per bucket.
    #[must_use]
    pub fn max_peaks(&self) -> &[i8] {
        &self.max_peaks
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;
    use crate::decode::STANDARD_SAMPLE_RATE;

    #[test]
    fn waveform_peaks_capture_quiet_and_loud_sections() {
        let sr = STANDARD_SAMPLE_RATE;
        // 1 second total: first 0.5s silence, second 0.5s full-scale 200Hz sine.
        let mut samples = vec![0.0_f32; (sr * 2) as usize];
        for i in (sr / 2)..sr {
            let t = i as f64 / f64::from(sr);
            let s = (TAU * 200.0 * t).sin() as f32;
            let idx = (i * 2) as usize;
            samples[idx] = s;
            samples[idx + 1] = s;
        }

        let buf = AudioBuffer {
            sample_rate: sr,
            channels: 2,
            samples_interleaved_f32: samples,
        };

        let peaks = WaveformPeaks::from_audio_buffer(&buf);
        assert_eq!(peaks.points_per_sec(), DEFAULT_POINTS_PER_SEC);
        assert_eq!(peaks.len(), 200);

        let cols = peaks.sample_columns(0.0, 1.0, 100);
        assert_eq!(cols.len(), 100);

        // First half is silent
        for col in &cols[0..45] {
            assert_eq!(col.amplitude_u8, 0);
        }
        // Second half has near full-scale peak-to-peak amplitude (~254)
        for col in &cols[55..95] {
            assert!(
                col.amplitude_u8 >= 240,
                "expected loud column >= 240, got {}",
                col.amplitude_u8
            );
            assert!(col.max_i8 >= 120);
            assert!(col.min_i8 <= -120);
        }
    }

    #[test]
    fn waveform_peaks_44100_exact_bucket_count_no_drift() {
        let sr = 44_100_u32;
        let duration_secs = 180_usize;
        let total_frames = (sr as usize) * duration_secs;
        let buf = AudioBuffer {
            sample_rate: sr,
            channels: 1,
            samples_interleaved_f32: vec![0.25_f32; total_frames],
        };

        let peaks = WaveformPeaks::from_audio_buffer(&buf);
        // Exact 180s * 200 pts/sec = 36,000 buckets with zero drift.
        assert_eq!(peaks.len(), duration_secs * DEFAULT_POINTS_PER_SEC as usize);
    }

    #[test]
    fn sample_columns_handles_non_finite_and_extreme_bounds() {
        let sr = STANDARD_SAMPLE_RATE;
        let mut samples = vec![0.0_f32; sr as usize];
        for sample in &mut samples[(sr as usize / 2)..] {
            *sample = 0.75;
        }
        let buf = AudioBuffer {
            sample_rate: sr,
            channels: 1,
            samples_interleaved_f32: samples,
        };
        let peaks = WaveformPeaks::from_audio_buffer(&buf);

        assert!(peaks.sample_columns(f64::NAN, 1.0, 100).is_empty());
        assert!(peaks.sample_columns(0.0, f64::INFINITY, 100).is_empty());
        assert!(peaks.sample_columns(f64::NEG_INFINITY, 1.0, 100).is_empty());

        let far_out = peaks.sample_columns(1e20, 1e20 + 1.0, 16);
        assert_eq!(far_out.len(), 16);
        assert!(far_out.iter().all(|c| c.amplitude_u8 == 0));

        let huge_span = peaks.sample_columns(-1e20, 1e20, 16);
        assert_eq!(huge_span.len(), 16);

        // Reverse sampling mirrors quiet/loud halves cleanly.
        let rev = peaks.sample_columns(1.0, 0.0, 100);
        assert_eq!(rev.len(), 100);
        assert!(rev[10].max_i8 >= 90);
        assert_eq!(rev[90].amplitude_u8, 0);
    }

    #[test]
    fn waveform_peaks_dc_buckets_keep_zero_bounds_and_freeze_frame_samples_frozen_bucket() {
        let sr = STANDARD_SAMPLE_RATE;
        // First 0.5s: +0.5 DC; second 0.5s: -0.6 DC.
        let mut samples = vec![0.5_f32; sr as usize];
        for s in &mut samples[(sr as usize / 2)..] {
            *s = -0.6;
        }
        let buf = AudioBuffer {
            sample_rate: sr,
            channels: 1,
            samples_interleaved_f32: samples,
        };
        let peaks = WaveformPeaks::from_audio_buffer(&buf);

        // Every bucket must obey min_i8 <= 0 and max_i8 >= 0 (matching Zoidium waveform-samples.js).
        assert!(
            peaks.min_peaks().iter().all(|&m| m <= 0),
            "min_peaks must never exceed 0 on DC-positive audio"
        );
        assert!(
            peaks.max_peaks().iter().all(|&m| m >= 0),
            "max_peaks must never drop below 0 on DC-negative audio"
        );
        assert!(peaks.max_peaks()[10] >= 60);
        assert_eq!(peaks.min_peaks()[10], 0);
        assert!(peaks.min_peaks()[150] <= -70);
        assert_eq!(peaks.max_peaks()[150], 0);

        // Freeze-frame sampling (media_start_sec == media_end_sec) samples the frozen bucket consistently.
        let frozen_pos = peaks.sample_columns(0.25, 0.25, 16);
        assert_eq!(frozen_pos.len(), 16);
        assert!(frozen_pos.iter().all(|c| c.max_i8 >= 60 && c.min_i8 == 0));

        let frozen_end = peaks.sample_columns(1.0, 1.0, 8);
        assert_eq!(frozen_end.len(), 8);
        assert!(frozen_end.iter().all(|c| c.min_i8 <= -70 && c.max_i8 == 0));
    }

    #[test]
    fn waveform_peaks_channel_specializations_match_reference_and_handle_non_finite() {
        for ch in [1_u16, 2, 4] {
            let sr = 48_000_u32;
            let frames = 2_400_usize;
            let ch_u = usize::from(ch);
            let mut samples = Vec::with_capacity(frames * ch_u);
            for i in 0..frames {
                let t = i as f64 / f64::from(sr);
                for c in 0..ch_u {
                    let v = if i == 100 && c == 0 {
                        f32::NAN
                    } else if i == 250 && c == ch_u - 1 {
                        f32::INFINITY
                    } else if i == 500 {
                        1.5 // tests > 1.0 clamping
                    } else if i == 750 {
                        -1.5 // tests < -1.0 clamping
                    } else {
                        ((TAU * (150.0 + c as f64 * 90.0) * t).sin() as f32) * 0.85
                    };
                    samples.push(v);
                }
            }
            let buf = AudioBuffer {
                sample_rate: sr,
                channels: ch,
                samples_interleaved_f32: samples,
            };
            let peaks = WaveformPeaks::from_audio_buffer_with_rate(&buf, 200);

            // Reference calculation matching original per-sample implementation.
            let pps = 200.0_f64;
            let sr_f = f64::from(sr);
            let bucket_count = (((frames as f64) * pps) / sr_f).ceil().max(1.0) as usize;
            let inv_ch = 1.0_f32 / ch_u as f32;
            let mut ref_min = Vec::with_capacity(bucket_count);
            let mut ref_max = Vec::with_capacity(bucket_count);
            for b in 0..bucket_count {
                let start = (((b as f64) * sr_f / pps).floor() as usize).min(frames - 1);
                let end = ((((b + 1) as f64) * sr_f / pps)
                    .floor()
                    .max((start + 1) as f64) as usize)
                    .min(frames);
                let mut min_avg = 0.0_f32;
                let mut max_avg = 0.0_f32;
                for f in start..end {
                    let mut sum = 0.0_f32;
                    for c in 0..ch_u {
                        let s = buf.samples_interleaved_f32[f * ch_u + c];
                        if s.is_finite() {
                            sum += s;
                        }
                    }
                    let avg = (sum * inv_ch).clamp(-1.0, 1.0);
                    min_avg = min_avg.min(avg);
                    max_avg = max_avg.max(avg);
                }
                ref_min.push((min_avg * 127.0).clamp(-127.0, 127.0) as i8);
                ref_max.push((max_avg * 127.0).clamp(-127.0, 127.0) as i8);
            }

            assert_eq!(
                peaks.min_peaks(),
                ref_min.as_slice(),
                "min mismatch ch={ch}"
            );
            assert_eq!(
                peaks.max_peaks(),
                ref_max.as_slice(),
                "max mismatch ch={ch}"
            );
        }
    }
}
