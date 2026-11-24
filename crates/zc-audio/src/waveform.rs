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

        let samples_per_bucket = (buffer.sample_rate / points_per_sec).max(1) as usize;
        let bucket_count = total_frames.div_ceil(samples_per_bucket);
        let mut min_peaks = Vec::with_capacity(bucket_count);
        let mut max_peaks = Vec::with_capacity(bucket_count);
        let inv_ch = 1.0_f32 / ch as f32;

        for bucket_idx in 0..bucket_count {
            let start = bucket_idx * samples_per_bucket;
            let end = (start + samples_per_bucket).min(total_frames);
            let mut min_avg = 1.0_f32;
            let mut max_avg = -1.0_f32;

            for frame_idx in start..end {
                let base = frame_idx * ch;
                let mut sum = 0.0_f32;
                for c in 0..ch {
                    let s = buffer.samples_interleaved_f32[base + c];
                    if s.is_finite() {
                        sum += s;
                    }
                }
                let avg = (sum * inv_ch).clamp(-1.0, 1.0);
                if avg < min_avg {
                    min_avg = avg;
                }
                if avg > max_avg {
                    max_avg = avg;
                }
            }

            min_peaks.push((min_avg * 127.0).clamp(-127.0, 127.0) as i8);
            max_peaks.push((max_avg * 127.0).clamp(-127.0, 127.0) as i8);
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
        if width_px == 0 || self.is_empty() || self.points_per_sec == 0 {
            return Vec::new();
        }

        let pps = f64::from(self.points_per_sec);
        let base_idx = (media_start_sec * pps).floor();
        let span_buckets = ((media_end_sec - media_start_sec) * pps).floor();
        let step = span_buckets / width_px as f64;
        let scan_count = step.abs().ceil().max(1.0) as usize;
        let len = self.len() as isize;

        let mut cols = Vec::with_capacity(width_px);
        for x in 0..width_px {
            let start_idx = (step * x as f64).floor() as isize + base_idx as isize;
            let mut hi: i8 = 0;
            let mut lo: i8 = 0;

            for p in 0..scan_count {
                let idx = if step >= 0.0 {
                    start_idx + p as isize
                } else {
                    start_idx - p as isize
                };
                if idx >= 0 && idx < len {
                    let u = idx as usize;
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
}
