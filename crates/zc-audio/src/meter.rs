//! Audio level meter state implementing exact CM3 RMS and peak-hold decay math
//! (Spec 02 §5.2, 45px minimum meter widget).

/// Minimum width and height in logical pixels for the CM3 audio meter panel.
pub const METER_MIN_PX: u32 = 45;

/// Default time-domain analysis window size (matching CM3 `AnalyserNode.fftSize = 1024`).
pub const METER_ANALYSIS_WINDOW: usize = 1_024;

/// Tracks real-time RMS level and peak-hold indicator with CM3's exact exponential decay.
///
/// Peak-hold decay formula per animation frame:
/// `peak_{t+1} = max(frame_peak, peak_t - 0.01 / exp(peak_t))`
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AudioMeterState {
    rms: f32,
    peak_hold: f32,
    last_frame_peak: f32,
}

impl AudioMeterState {
    /// Creates a zeroed meter state.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            rms: 0.0,
            peak_hold: 0.0,
            last_frame_peak: 0.0,
        }
    }

    /// Resets RMS and peak-hold levels to `0.0`.
    pub fn reset(&mut self) {
        self.rms = 0.0;
        self.peak_hold = 0.0;
        self.last_frame_peak = 0.0;
    }

    /// Updates meter state from a mono slice of normalized `f32` samples in `[-1.0, 1.0]`.
    pub fn update_mono(&mut self, samples: &[f32]) {
        if samples.is_empty() {
            self.apply_frame_stats(0.0, 0.0);
            return;
        }

        let mut sum_sq = 0.0_f64;
        let mut frame_peak = 0.0_f32;
        for &s in samples {
            let abs = if s.is_finite() { s.abs().min(1.0) } else { 0.0 };
            sum_sq += f64::from(abs) * f64::from(abs);
            if abs > frame_peak {
                frame_peak = abs;
            }
        }

        let rms = (sum_sq / samples.len() as f64).sqrt().clamp(0.0, 1.0) as f32;
        self.apply_frame_stats(rms, frame_peak);
    }

    /// Updates meter state from interleaved stereo frames `[L, R]` in `[-1.0, 1.0]`.
    pub fn update_stereo(&mut self, frames: &[[f32; 2]]) {
        if frames.is_empty() {
            self.apply_frame_stats(0.0, 0.0);
            return;
        }

        let mut sum_sq = 0.0_f64;
        let mut frame_peak = 0.0_f32;
        for &[l, r] in frames {
            let la = if l.is_finite() { l.abs().min(1.0) } else { 0.0 };
            let ra = if r.is_finite() { r.abs().min(1.0) } else { 0.0 };
            sum_sq += 0.5 * (f64::from(la) * f64::from(la) + f64::from(ra) * f64::from(ra));
            let peak = la.max(ra);
            if peak > frame_peak {
                frame_peak = peak;
            }
        }

        let rms = (sum_sq / frames.len() as f64).sqrt().clamp(0.0, 1.0) as f32;
        self.apply_frame_stats(rms, frame_peak);
    }

    /// Updates meter state from unsigned 8-bit time-domain analyser bytes (`0..=255`)
    /// using CM3's exact normalization `2 * |sample / 256 - 0.5|`.
    pub fn update_u8_time_domain(&mut self, bytes: &[u8]) {
        if bytes.is_empty() {
            self.apply_frame_stats(0.0, 0.0);
            return;
        }

        let mut sum_sq = 0.0_f64;
        let mut frame_peak = 0.0_f32;
        for &b in bytes {
            let norm = (2.0 * (f32::from(b) / 256.0 - 0.5).abs()).clamp(0.0, 1.0);
            sum_sq += f64::from(norm) * f64::from(norm);
            if norm > frame_peak {
                frame_peak = norm;
            }
        }

        let rms = (sum_sq / bytes.len() as f64).sqrt().clamp(0.0, 1.0) as f32;
        self.apply_frame_stats(rms, frame_peak);
    }

    /// Advances one animation frame of silence (RMS drops to 0, peak-hold decays).
    pub fn decay_silence(&mut self) {
        self.apply_frame_stats(0.0, 0.0);
    }

    #[inline]
    fn apply_frame_stats(&mut self, rms: f32, frame_peak: f32) {
        let decayed = (self.peak_hold - 0.01_f32 / self.peak_hold.exp()).max(0.0);
        self.rms = rms.clamp(0.0, 1.0);
        self.last_frame_peak = frame_peak.clamp(0.0, 1.0);
        self.peak_hold = self.last_frame_peak.max(decayed).clamp(0.0, 1.0);
    }

    /// Current RMS level in `[0.0, 1.0]`.
    #[must_use]
    pub const fn rms(&self) -> f32 {
        self.rms
    }

    /// Current decaying peak-hold level in `[0.0, 1.0]`.
    #[must_use]
    pub const fn peak_hold(&self) -> f32 {
        self.peak_hold
    }

    /// Instantaneous maximum sample magnitude observed in the most recent frame.
    #[must_use]
    pub const fn last_frame_peak(&self) -> f32 {
        self.last_frame_peak
    }

    /// Vertical dark-mask scale factor `1.0 - rms` per Spec 02 §5.2.
    #[must_use]
    pub fn mask_scale(&self) -> f32 {
        (1.0 - self.rms).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meter_rms_and_peak_hold_decay_formula() {
        let mut meter = AudioMeterState::new();
        assert_eq!(meter.rms(), 0.0);
        assert_eq!(meter.peak_hold(), 0.0);

        // Full-scale DC pulse of 0.8
        meter.update_mono(&[0.8; 128]);
        assert!((meter.rms() - 0.8).abs() < 1e-5);
        assert!((meter.peak_hold() - 0.8).abs() < 1e-5);
        assert!((meter.mask_scale() - 0.2).abs() < 1e-5);

        // Next frame is silent: peak must decay by exact 0.01 / exp(0.8)
        let expected_after_1 = 0.8_f32 - 0.01_f32 / 0.8_f32.exp();
        meter.decay_silence();
        assert_eq!(meter.rms(), 0.0);
        assert!((meter.peak_hold() - expected_after_1).abs() < 1e-6);

        // A second silent frame decays from expected_after_1
        let expected_after_2 = expected_after_1 - 0.01_f32 / expected_after_1.exp();
        meter.decay_silence();
        assert!((meter.peak_hold() - expected_after_2).abs() < 1e-6);

        // Eventually decays smoothly to 0.0 without going negative
        for _ in 0..500 {
            meter.decay_silence();
        }
        assert_eq!(meter.peak_hold(), 0.0);
    }

    #[test]
    fn meter_u8_time_domain_and_stereo_rms() {
        let mut meter = AudioMeterState::new();
        // 128 in u8 maps to |128/256 - 0.5| * 2 = 0.0 (silence)
        meter.update_u8_time_domain(&[128; 64]);
        assert!(meter.rms() < 1e-6);
        assert!(meter.peak_hold() < 1e-6);

        // 0 in u8 maps to |0/256 - 0.5| * 2 = 1.0
        meter.update_u8_time_domain(&[0; 64]);
        assert!((meter.rms() - 1.0).abs() < 1e-6);
        assert!((meter.peak_hold() - 1.0).abs() < 1e-6);

        // Stereo hard-panned signal [1.0, 0.0] -> RMS = sqrt(0.5), peak = 1.0
        meter.reset();
        meter.update_stereo(&[[1.0, 0.0]; 64]);
        assert!((meter.rms() - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        assert!((meter.peak_hold() - 1.0).abs() < 1e-6);
    }
}
