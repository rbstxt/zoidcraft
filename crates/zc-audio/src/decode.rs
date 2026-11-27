//! Audio decoding via `symphonia` (WAV, MP3, OGG/Vorbis, FLAC, AAC, MP4),
//! windowed-sinc stereo resampling to 48 kHz, and lightweight WAV encoders.

use std::f32::consts::FRAC_1_SQRT_2;
use std::f64::consts::PI;
use std::io::{self, Cursor};

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::AudioError;

/// Standard internal mixing and export sample rate (48,000 Hz).
pub const STANDARD_SAMPLE_RATE: u32 = 48_000;
/// Minimum accepted audio stream sample rate in Hz.
pub const MIN_AUDIO_SAMPLE_RATE: u32 = 8_000;
/// Maximum accepted audio stream sample rate in Hz.
pub const MAX_AUDIO_SAMPLE_RATE: u32 = 384_000;
/// Maximum accepted channel count per decoded audio asset.
pub const MAX_AUDIO_CHANNELS: u16 = 32;
/// Maximum interleaved f32 samples decoded from a single asset (64M samples = 256 MiB).
pub const MAX_DECODED_AUDIO_SAMPLES: usize = 64 * 1024 * 1024;
/// Maximum stereo frames allocated by a single resampling call (32M stereo frames = 256 MiB).
pub const MAX_RESAMPLED_STEREO_FRAMES: usize = 32 * 1024 * 1024;

/// Decoded audio buffer storing interleaved 32-bit float PCM samples in `[-1.0, 1.0]`.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioBuffer {
    /// Sampling frequency in Hertz (e.g., 44100 or 48000).
    pub sample_rate: u32,
    /// Number of interleaved channels (1 = mono, 2 = stereo, etc.).
    pub channels: u16,
    /// Interleaved 32-bit floating-point samples.
    pub samples_interleaved_f32: Vec<f32>,
}

impl AudioBuffer {
    /// Returns the number of multi-channel sample frames in this buffer.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples_interleaved_f32.len() / usize::from(self.channels)
        }
    }

    /// Returns the duration of the audio buffer in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frame_count() as f64 / f64::from(self.sample_rate)
        }
    }

    /// Returns `true` if the buffer contains zero audio frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.frame_count() == 0
    }

    /// Reads frame `frame_idx` and downmixes it to stereo `[L, R]` using standard
    /// ITU-R BS.775 / Web Audio channel layout rules.
    #[must_use]
    #[inline]
    pub fn read_stereo_frame(&self, frame_idx: usize) -> [f32; 2] {
        downmix_frame_to_stereo(self, frame_idx)
    }
}

/// Decodes encoded audio bytes (WAV, MP3, OGG, FLAC, AAC, MP4) into an interleaved `f32` [`AudioBuffer`].
pub fn decode_audio_bytes(bytes: &[u8], hint_ext: Option<&str>) -> Result<AudioBuffer, AudioError> {
    if bytes.is_empty() {
        return Err(AudioError::EmptyInput);
    }

    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

    let mut hint = Hint::new();
    if let Some(ext) = hint_ext {
        let trimmed = ext.trim().trim_start_matches('.');
        if !trimmed.is_empty() {
            hint.with_extension(trimmed);
        }
    }

    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|err| AudioError::Probe(err.to_string()))?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or(AudioError::NoAudioTrack)?;
    let track_id = track.id;
    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or(AudioError::NoAudioTrack)?;

    let header_rate = audio_params.sample_rate;
    let header_channels = audio_params
        .channels
        .as_ref()
        .and_then(|ch| u16::try_from(ch.count()).ok());
    let estimated_samples = track
        .num_frames
        .and_then(|frames| header_channels.map(|ch| frames.saturating_mul(u64::from(ch))))
        .map_or(0, |n| {
            usize::try_from(n)
                .unwrap_or(MAX_DECODED_AUDIO_SAMPLES)
                .min(MAX_DECODED_AUDIO_SAMPLES)
        });

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|err| AudioError::UnsupportedCodec(err.to_string()))?;

    let mut sample_rate: Option<u32> = None;
    let mut channels: Option<u16> = None;
    let mut samples_interleaved_f32 = Vec::new();
    if estimated_samples > 0 {
        let _ = samples_interleaved_f32.try_reserve(estimated_samples);
    }
    let mut scratch: Vec<f32> = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::IoError(err)) if err.kind() == io::ErrorKind::UnexpectedEof => {
                break;
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
                continue;
            }
            Err(err) => return Err(AudioError::Decode(err.to_string())),
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                let spec = decoded.spec();
                let pkt_rate = spec.rate();
                let pkt_ch_count = spec.channels().count();
                let pkt_channels = u16::try_from(pkt_ch_count).map_err(|_| {
                    AudioError::InvalidFormat(format!("unsupported channel count: {pkt_ch_count}"))
                })?;

                if !(MIN_AUDIO_SAMPLE_RATE..=MAX_AUDIO_SAMPLE_RATE).contains(&pkt_rate)
                    || pkt_channels == 0
                    || pkt_channels > MAX_AUDIO_CHANNELS
                {
                    return Err(AudioError::InvalidFormat(format!(
                        "unsupported audio stream format: {pkt_rate} Hz / {pkt_channels} ch"
                    )));
                }

                match (sample_rate, channels) {
                    (None, None) => {
                        sample_rate = Some(pkt_rate);
                        channels = Some(pkt_channels);
                    }
                    (Some(expected_rate), Some(expected_channels)) => {
                        if pkt_rate != expected_rate || pkt_channels != expected_channels {
                            return Err(AudioError::InvalidFormat(format!(
                                "mid-stream format change ({expected_rate}Hz/{expected_channels}ch -> {pkt_rate}Hz/{pkt_channels}ch)"
                            )));
                        }
                    }
                    _ => unreachable!(),
                }

                let count = decoded.samples_interleaved();
                if count > 0 {
                    let next_len = samples_interleaved_f32
                        .len()
                        .checked_add(count)
                        .ok_or_else(|| {
                            AudioError::Decode("decoded audio sample count overflow".to_string())
                        })?;
                    if next_len > MAX_DECODED_AUDIO_SAMPLES {
                        return Err(AudioError::Decode(format!(
                            "decoded audio exceeds 256 MiB memory budget ({MAX_DECODED_AUDIO_SAMPLES} samples)"
                        )));
                    }
                    scratch.resize(count, 0.0);
                    decoded.copy_to_slice_interleaved(&mut scratch);
                    samples_interleaved_f32.extend_from_slice(&scratch);
                }
            }
            Err(SymphoniaError::DecodeError(msg)) => {
                tracing::warn!("skipping corrupt audio packet during decode: {msg}");
            }
            Err(SymphoniaError::ResetRequired) => {
                decoder.reset();
            }
            Err(err) => return Err(AudioError::Decode(err.to_string())),
        }
    }

    let final_rate = sample_rate
        .or(header_rate)
        .filter(|&r| r > 0)
        .ok_or_else(|| AudioError::InvalidFormat("unknown audio sample rate".to_string()))?;
    let final_channels = channels
        .or(header_channels)
        .filter(|&c| c > 0)
        .ok_or_else(|| AudioError::InvalidFormat("unknown audio channel count".to_string()))?;

    Ok(AudioBuffer {
        sample_rate: final_rate,
        channels: final_channels,
        samples_interleaved_f32,
    })
}

/// Converts and resamples an [`AudioBuffer`] to stereo `48,000 Hz` using windowed-sinc
/// anti-aliased interpolation.
#[must_use]
pub fn resample_stereo_48k(buffer: &AudioBuffer) -> Vec<[f32; 2]> {
    resample_stereo(buffer, STANDARD_SAMPLE_RATE)
}

/// Converts and resamples an [`AudioBuffer`] to stereo at `target_sample_rate` using a
/// Hann-windowed sinc anti-aliasing filter when downsampling and a 16-tap Blackman-windowed
/// sinc filter when upsampling.
///
/// Rejects out-of-range sample rates and caps output allocations to
/// [`MAX_RESAMPLED_STEREO_FRAMES`] (256 MiB) so malformed metadata cannot exhaust memory.
#[must_use]
pub fn resample_stereo(buffer: &AudioBuffer, target_sample_rate: u32) -> Vec<[f32; 2]> {
    let in_frames = buffer.frame_count();
    if in_frames == 0
        || !(MIN_AUDIO_SAMPLE_RATE..=MAX_AUDIO_SAMPLE_RATE).contains(&buffer.sample_rate)
        || !(MIN_AUDIO_SAMPLE_RATE..=MAX_AUDIO_SAMPLE_RATE).contains(&target_sample_rate)
        || buffer.channels == 0
        || buffer.channels > MAX_AUDIO_CHANNELS
    {
        return Vec::new();
    }

    if buffer.sample_rate == target_sample_rate {
        let capped = in_frames.min(MAX_RESAMPLED_STEREO_FRAMES);
        let mut out = Vec::with_capacity(capped);
        for frame_idx in 0..capped {
            out.push(downmix_frame_to_stereo(buffer, frame_idx));
        }
        return out;
    }

    let out_frames_f64 =
        (in_frames as f64) * (f64::from(target_sample_rate) / f64::from(buffer.sample_rate));
    if !out_frames_f64.is_finite()
        || out_frames_f64 <= 0.0
        || out_frames_f64 > MAX_RESAMPLED_STEREO_FRAMES as f64
    {
        return Vec::new();
    }
    let out_frames = (out_frames_f64.round().max(1.0) as usize).min(MAX_RESAMPLED_STEREO_FRAMES);
    let ratio = f64::from(buffer.sample_rate) / f64::from(target_sample_rate);
    let use_antialias = ratio > 1.0 + 1e-6;

    let mut out = Vec::with_capacity(out_frames);
    if use_antialias {
        let sinc_kernel = SincResampleKernel::new(ratio);
        for out_idx in 0..out_frames {
            let src_pos = (out_idx as f64) * ratio;
            out.push(sinc_kernel.sample_stereo(buffer, src_pos));
        }
    } else {
        let upsample_kernel = BlackmanSincUpsampleKernel::new();
        for out_idx in 0..out_frames {
            let src_pos = (out_idx as f64) * ratio;
            out.push(upsample_kernel.sample_stereo(buffer, src_pos));
        }
    }

    out
}

/// Precomputed 16-tap (half-width 8) Blackman-windowed sinc interpolation kernel for
/// anti-imaged audio upsampling (e.g. 44.1 kHz -> 48 kHz).
///
/// Uses a single window oscillator recurrence (`cos(2u) = 2 * cos^2(u) - 1`) and exact
/// sign-alternating `sin(pi * d)` evaluation so integer source positions are preserved
/// bit-accurately and high-frequency imaging artifacts above source Nyquist are suppressed
/// by >55 dB.
#[derive(Debug, Clone, Copy)]
struct BlackmanSincUpsampleKernel {
    radius: isize,
    inv_radius_pi: f64,
    w_sin_step: f64,
    w_cos_step: f64,
}

impl BlackmanSincUpsampleKernel {
    const RADIUS: isize = 8;

    #[must_use]
    fn new() -> Self {
        let inv_radius_pi = PI / (Self::RADIUS as f64);
        let (w_sin_step, w_cos_step) = (-inv_radius_pi).sin_cos();
        Self {
            radius: Self::RADIUS,
            inv_radius_pi,
            w_sin_step,
            w_cos_step,
        }
    }

    #[must_use]
    fn sample_stereo(&self, buffer: &AudioBuffer, src_pos: f64) -> [f32; 2] {
        let frames = buffer.frame_count();
        if !src_pos.is_finite() || src_pos < 0.0 || frames == 0 {
            return [0.0, 0.0];
        }
        let center = src_pos.floor() as isize;
        if center as usize >= frames {
            return [0.0, 0.0];
        }
        let frac = src_pos - center as f64;
        if frac < 1e-12 {
            return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
        }
        if frac > 1.0 - 1e-12 {
            return downmix_frame_to_stereo(buffer, ((center + 1) as usize).min(frames - 1));
        }

        let max_idx = (frames - 1) as isize;
        let start = (center - (self.radius - 1)).max(0);
        let end = (center + self.radius).min(max_idx);
        let start_u = start as usize;
        let end_u = end as usize;

        let d0 = src_pos - start as f64;
        let (mut w_sin, mut w_cos) = (self.inv_radius_pi * d0).sin_cos();
        let base_sin = (PI * frac).sin();
        let mut x_sin = if (center - start) % 2 == 0 {
            base_sin
        } else {
            -base_sin
        };
        let mut x = PI * d0;

        let w_sin_step = self.w_sin_step;
        let w_cos_step = self.w_cos_step;

        let mut acc_l = 0.0_f64;
        let mut acc_r = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        let samples = &buffer.samples_interleaved_f32;
        match buffer.channels {
            1 => {
                for &raw in &samples[start_u..=end_u] {
                    let w_cos2 = 2.0 * w_cos * w_cos - 1.0;
                    let window = 0.42 + 0.5 * w_cos + 0.08 * w_cos2;
                    let w = (x_sin / x) * window;
                    acc_l += f64::from(clean_sample(raw)) * w;
                    weight_sum += w;

                    let next_w_sin = w_sin * w_cos_step + w_cos * w_sin_step;
                    let next_w_cos = w_cos * w_cos_step - w_sin * w_sin_step;
                    w_sin = next_w_sin;
                    w_cos = next_w_cos;
                    x -= PI;
                    x_sin = -x_sin;
                }
                if weight_sum.abs() <= 1e-12 {
                    return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
                }
                let out = (acc_l / weight_sum) as f32;
                return [out, out];
            }
            2 => {
                for &[raw_l, raw_r] in samples[start_u * 2..(end_u + 1) * 2].as_chunks::<2>().0 {
                    let w_cos2 = 2.0 * w_cos * w_cos - 1.0;
                    let window = 0.42 + 0.5 * w_cos + 0.08 * w_cos2;
                    let w = (x_sin / x) * window;
                    acc_l += f64::from(clean_sample(raw_l)) * w;
                    acc_r += f64::from(clean_sample(raw_r)) * w;
                    weight_sum += w;

                    let next_w_sin = w_sin * w_cos_step + w_cos * w_sin_step;
                    let next_w_cos = w_cos * w_cos_step - w_sin * w_sin_step;
                    w_sin = next_w_sin;
                    w_cos = next_w_cos;
                    x -= PI;
                    x_sin = -x_sin;
                }
            }
            _ => {
                for idx in start_u..=end_u {
                    let w_cos2 = 2.0 * w_cos * w_cos - 1.0;
                    let window = 0.42 + 0.5 * w_cos + 0.08 * w_cos2;
                    let w = (x_sin / x) * window;
                    let s = downmix_frame_to_stereo(buffer, idx);
                    acc_l += f64::from(s[0]) * w;
                    acc_r += f64::from(s[1]) * w;
                    weight_sum += w;

                    let next_w_sin = w_sin * w_cos_step + w_cos * w_sin_step;
                    let next_w_cos = w_cos * w_cos_step - w_sin * w_sin_step;
                    w_sin = next_w_sin;
                    w_cos = next_w_cos;
                    x -= PI;
                    x_sin = -x_sin;
                }
            }
        }

        if weight_sum.abs() <= 1e-12 {
            return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
        }
        let inv = 1.0 / weight_sum;
        [(acc_l * inv) as f32, (acc_r * inv) as f32]
    }
}

/// Precomputed Hann-windowed sinc oscillator step constants for a fixed downsampling ratio.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SincResampleKernel {
    cutoff: f64,
    radius: isize,
    radius_f: f64,
    inv_radius_pi: f64,
    w_sin_step: f64,
    w_cos_step: f64,
    x_step: f64,
    x_sin_step: f64,
    x_cos_step: f64,
}

impl SincResampleKernel {
    #[must_use]
    pub(crate) fn new(ratio: f64) -> Self {
        let effective_ratio = ratio.max(1.0);
        // Cutoff slightly below target Nyquist (0.9 / ratio) to prevent fold-back aliasing.
        let cutoff = 0.9 / effective_ratio;
        let radius = ((8.0 * effective_ratio - 1e-9).ceil() as isize).clamp(8, 64);
        let radius_f = radius as f64;
        let inv_radius_pi = PI / radius_f;
        let (w_sin_step, w_cos_step) = (-inv_radius_pi).sin_cos();
        let x_step = -PI * cutoff;
        let (x_sin_step, x_cos_step) = x_step.sin_cos();
        Self {
            cutoff,
            radius,
            radius_f,
            inv_radius_pi,
            w_sin_step,
            w_cos_step,
            x_step,
            x_sin_step,
            x_cos_step,
        }
    }

    /// Samples a stereo frame from `buffer` at fractional source frame index `src_pos` using a
    /// 2-oscillator trigonometric recurrence (2 `sin_cos` calls total before the tap loop) and
    /// channel-specialized inner loops.
    #[must_use]
    pub(crate) fn sample_stereo(&self, buffer: &AudioBuffer, src_pos: f64) -> [f32; 2] {
        let frames = buffer.frame_count();
        if !src_pos.is_finite() || src_pos < 0.0 || frames == 0 {
            return [0.0, 0.0];
        }
        let center = src_pos.floor() as isize;
        if center as usize >= frames {
            return [0.0, 0.0];
        }
        let max_idx = (frames - 1) as isize;
        let mut start = (center - self.radius).max(0);
        let end = (center + self.radius).min(max_idx);
        if start <= end && (src_pos - start as f64) > self.radius_f {
            start += 1;
        }
        if start > end {
            return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
        }

        let start_u = start as usize;
        let end_u = end as usize;
        let d0 = src_pos - start as f64;
        let (mut w_sin, mut w_cos) = (self.inv_radius_pi * d0).sin_cos();
        let mut x = PI * self.cutoff * d0;
        let (mut x_sin, mut x_cos) = x.sin_cos();

        let w_sin_step = self.w_sin_step;
        let w_cos_step = self.w_cos_step;
        let x_step = self.x_step;
        let x_sin_step = self.x_sin_step;
        let x_cos_step = self.x_cos_step;

        let mut acc_l = 0.0_f64;
        let mut acc_r = 0.0_f64;
        let mut weight_sum = 0.0_f64;

        let samples = &buffer.samples_interleaved_f32;
        match buffer.channels {
            1 => {
                for &raw in &samples[start_u..=end_u] {
                    let window = 0.5 * (1.0 + w_cos);
                    let sinc = if x.abs() < 1e-4 {
                        1.0 - x * x * (1.0 / 6.0)
                    } else {
                        x_sin / x
                    };
                    let w = sinc * window;
                    let m = f64::from(clean_sample(raw));
                    acc_l += m * w;
                    weight_sum += w;

                    let next_w_sin = w_sin * w_cos_step + w_cos * w_sin_step;
                    let next_w_cos = w_cos * w_cos_step - w_sin * w_sin_step;
                    w_sin = next_w_sin;
                    w_cos = next_w_cos;

                    x += x_step;
                    let next_x_sin = x_sin * x_cos_step + x_cos * x_sin_step;
                    let next_x_cos = x_cos * x_cos_step - x_sin * x_sin_step;
                    x_sin = next_x_sin;
                    x_cos = next_x_cos;
                }
                if weight_sum.abs() <= 1e-12 {
                    return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
                }
                let out = (acc_l / weight_sum) as f32;
                return [out, out];
            }
            2 => {
                for &[raw_l, raw_r] in samples[start_u * 2..(end_u + 1) * 2].as_chunks::<2>().0 {
                    let window = 0.5 * (1.0 + w_cos);
                    let sinc = if x.abs() < 1e-4 {
                        1.0 - x * x * (1.0 / 6.0)
                    } else {
                        x_sin / x
                    };
                    let w = sinc * window;
                    acc_l += f64::from(clean_sample(raw_l)) * w;
                    acc_r += f64::from(clean_sample(raw_r)) * w;
                    weight_sum += w;

                    let next_w_sin = w_sin * w_cos_step + w_cos * w_sin_step;
                    let next_w_cos = w_cos * w_cos_step - w_sin * w_sin_step;
                    w_sin = next_w_sin;
                    w_cos = next_w_cos;

                    x += x_step;
                    let next_x_sin = x_sin * x_cos_step + x_cos * x_sin_step;
                    let next_x_cos = x_cos * x_cos_step - x_sin * x_sin_step;
                    x_sin = next_x_sin;
                    x_cos = next_x_cos;
                }
            }
            _ => {
                for idx in start_u..=end_u {
                    let window = 0.5 * (1.0 + w_cos);
                    let sinc = if x.abs() < 1e-4 {
                        1.0 - x * x * (1.0 / 6.0)
                    } else {
                        x_sin / x
                    };
                    let w = sinc * window;
                    let s = downmix_frame_to_stereo(buffer, idx);
                    acc_l += f64::from(s[0]) * w;
                    acc_r += f64::from(s[1]) * w;
                    weight_sum += w;

                    let next_w_sin = w_sin * w_cos_step + w_cos * w_sin_step;
                    let next_w_cos = w_cos * w_cos_step - w_sin * w_sin_step;
                    w_sin = next_w_sin;
                    w_cos = next_w_cos;

                    x += x_step;
                    let next_x_sin = x_sin * x_cos_step + x_cos * x_sin_step;
                    let next_x_cos = x_cos * x_cos_step - x_sin * x_sin_step;
                    x_sin = next_x_sin;
                    x_cos = next_x_cos;
                }
            }
        }

        if weight_sum.abs() <= 1e-12 {
            return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
        }
        let inv = 1.0 / weight_sum;
        [(acc_l * inv) as f32, (acc_r * inv) as f32]
    }
}

/// Samples a stereo frame from `buffer` at fractional source frame index `src_pos` using a
/// Hann-windowed sinc low-pass anti-aliasing filter tuned for downsampling ratio `ratio > 1.0`.
#[must_use]
pub(crate) fn sample_stereo_sinc(buffer: &AudioBuffer, src_pos: f64, ratio: f64) -> [f32; 2] {
    if !ratio.is_finite() {
        return [0.0, 0.0];
    }
    SincResampleKernel::new(ratio).sample_stereo(buffer, src_pos)
}

#[inline]
pub(crate) fn clean_sample(s: f32) -> f32 {
    if s.is_finite() { s } else { 0.0 }
}

/// Reads frame `frame_idx` from `buffer` and downmixes multi-channel layouts to stereo `[L, R]`
/// following ITU-R BS.775 / W3C Web Audio channel downmix rules:
///
/// - 1 ch (mono): `[M, M]`
/// - 2 ch (stereo): `[L, R]`
/// - 3 ch (3.0 FL/FR/FC): `[FL + 0.7071*FC, FR + 0.7071*FC]`
/// - 4 ch (quad FL/FR/BL/BR): `[0.5*(FL + BL), 0.5*(FR + BR)]`
/// - 6 ch (5.1 FL/FR/FC/LFE/BL/BR): `[FL + 0.7071*FC + 0.7071*BL, FR + 0.7071*FC + 0.7071*BR]` (LFE omitted)
/// - 8 ch (7.1 FL/FR/FC/LFE/BL/BR/SL/SR): `[FL + 0.7071*(FC + BL + SL), FR + 0.7071*(FC + BR + SR)]`
#[must_use]
#[inline]
pub fn downmix_frame_to_stereo(buffer: &AudioBuffer, frame_idx: usize) -> [f32; 2] {
    let ch = usize::from(buffer.channels);
    if ch == 0 || frame_idx >= buffer.frame_count() {
        return [0.0, 0.0];
    }
    let base = frame_idx * ch;
    let s = |offset: usize| clean_sample(buffer.samples_interleaved_f32[base + offset]);
    match ch {
        1 => {
            let m = s(0);
            [m, m]
        }
        2 => [s(0), s(1)],
        3 => {
            let fc = s(2) * FRAC_1_SQRT_2;
            [s(0) + fc, s(1) + fc]
        }
        4 => [0.5 * (s(0) + s(2)), 0.5 * (s(1) + s(3))],
        6 => {
            // Standard 5.1 channel order: FL=0, FR=1, FC=2, LFE=3, BL=4, BR=5.
            let fc = s(2) * FRAC_1_SQRT_2;
            let l = s(0) + fc + s(4) * FRAC_1_SQRT_2;
            let r = s(1) + fc + s(5) * FRAC_1_SQRT_2;
            [l, r]
        }
        8 => {
            // Standard 7.1 channel order: FL=0, FR=1, FC=2, LFE=3, BL=4, BR=5, SL=6, SR=7.
            let fc = s(2) * FRAC_1_SQRT_2;
            let l = s(0) + fc + (s(4) + s(6)) * FRAC_1_SQRT_2;
            let r = s(1) + fc + (s(5) + s(7)) * FRAC_1_SQRT_2;
            [l, r]
        }
        _ => {
            let mut left = 0.0_f32;
            let mut right = 0.0_f32;
            let mut left_count = 0_u32;
            let mut right_count = 0_u32;
            for c in 0..ch {
                let sample = s(c);
                if c % 2 == 0 {
                    left += sample;
                    left_count += 1;
                } else {
                    right += sample;
                    right_count += 1;
                }
            }
            let l = if left_count > 0 {
                left / left_count as f32
            } else {
                0.0
            };
            let r = if right_count > 0 {
                right / right_count as f32
            } else {
                l
            };
            [l, r]
        }
    }
}

/// Converts a normalized `f32` audio sample in `[-1.0, 1.0]` to signed 16-bit PCM using
/// asymmetric full-range scaling (`-1.0 -> -32768`, `+1.0 -> +32767`, matching Zoidium).
#[must_use]
#[inline]
pub fn f32_to_pcm16(sample: f32) -> i16 {
    if !sample.is_finite() {
        return 0;
    }
    let clamped = sample.clamp(-1.0, 1.0);
    let scaled = if clamped < 0.0 {
        clamped * 32_768.0
    } else {
        clamped * 32_767.0
    };
    scaled.round().clamp(-32_768.0, 32_767.0) as i16
}

pub(crate) fn checked_wav_chunk_sizes(
    frame_count: usize,
    block_align: u16,
    header_overhead: u32,
) -> Result<(u32, u32), AudioError> {
    let data_bytes_usize = frame_count
        .checked_mul(usize::from(block_align))
        .ok_or_else(|| {
            AudioError::InvalidFormat("WAV payload byte size overflow (> 4 GiB RIFF limit)".into())
        })?;
    let data_bytes = u32::try_from(data_bytes_usize).map_err(|_| {
        AudioError::InvalidFormat("WAV payload exceeds 4 GiB RIFF/WAVE limit".into())
    })?;
    let riff_size = header_overhead.checked_add(data_bytes).ok_or_else(|| {
        AudioError::InvalidFormat("WAV RIFF chunk size exceeds 4 GiB limit".into())
    })?;
    Ok((data_bytes, riff_size))
}

/// Fallible version of [`encode_wav_pcm16_stereo`] that rejects invalid sample rates and
/// payloads exceeding the 4 GiB RIFF/WAVE header limit.
pub fn try_encode_wav_pcm16_stereo(
    samples: &[[f32; 2]],
    sample_rate: u32,
) -> Result<Vec<u8>, AudioError> {
    if sample_rate == 0 {
        return Err(AudioError::InvalidFormat(
            "WAV sample_rate must be non-zero".into(),
        ));
    }
    let num_channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let block_align = num_channels * (bits_per_sample / 8);
    let byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| AudioError::InvalidFormat("WAV byte_rate overflow".into()))?;
    let (data_bytes, riff_size) = checked_wav_chunk_sizes(samples.len(), block_align, 36)?;

    let mut out = Vec::with_capacity(44 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&1_u16.to_le_bytes()); // WAVE_FORMAT_PCM
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &[l, r] in samples {
        let l16 = f32_to_pcm16(l);
        let r16 = f32_to_pcm16(r);
        out.extend_from_slice(&l16.to_le_bytes());
        out.extend_from_slice(&r16.to_le_bytes());
    }
    Ok(out)
}

/// Encodes stereo `f32` samples as a standard 16-bit signed PCM RIFF/WAVE byte vector.
///
/// Panics if `samples` exceeds the 4 GiB RIFF/WAVE header limit; use
/// [`try_encode_wav_pcm16_stereo`] for fallible encoding of untrusted buffer sizes.
#[must_use]
pub fn encode_wav_pcm16_stereo(samples: &[[f32; 2]], sample_rate: u32) -> Vec<u8> {
    try_encode_wav_pcm16_stereo(samples, sample_rate.max(1))
        .expect("WAV PCM16 payload exceeds 4 GiB RIFF/WAVE limit")
}

/// Fallible version of [`encode_wav_f32_stereo`] that rejects invalid sample rates and
/// payloads exceeding the 4 GiB RIFF/WAVE header limit.
pub fn try_encode_wav_f32_stereo(
    samples: &[[f32; 2]],
    sample_rate: u32,
) -> Result<Vec<u8>, AudioError> {
    if sample_rate == 0 {
        return Err(AudioError::InvalidFormat(
            "WAV sample_rate must be non-zero".into(),
        ));
    }
    let num_channels: u16 = 2;
    let bits_per_sample: u16 = 32;
    let block_align = num_channels * (bits_per_sample / 8);
    let byte_rate = sample_rate
        .checked_mul(u32::from(block_align))
        .ok_or_else(|| AudioError::InvalidFormat("WAV byte_rate overflow".into()))?;
    let (data_bytes, riff_size) = checked_wav_chunk_sizes(samples.len(), block_align, 48)?;
    let frame_count_u32 = u32::try_from(samples.len()).map_err(|_| {
        AudioError::InvalidFormat("WAV frame count exceeds 4 GiB RIFF/WAVE limit".into())
    })?;

    let mut out = Vec::with_capacity(56 + data_bytes as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16_u32.to_le_bytes());
    out.extend_from_slice(&3_u16.to_le_bytes()); // WAVE_FORMAT_IEEE_FLOAT
    out.extend_from_slice(&num_channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    out.extend_from_slice(b"fact");
    out.extend_from_slice(&4_u32.to_le_bytes());
    out.extend_from_slice(&frame_count_u32.to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &[l, r] in samples {
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    Ok(out)
}

/// Encodes stereo `f32` samples as a standard 32-bit IEEE float RIFF/WAVE byte vector.
///
/// Panics if `samples` exceeds the 4 GiB RIFF/WAVE header limit; use
/// [`try_encode_wav_f32_stereo`] for fallible encoding of untrusted buffer sizes.
#[must_use]
pub fn encode_wav_f32_stereo(samples: &[[f32; 2]], sample_rate: u32) -> Vec<u8> {
    try_encode_wav_f32_stereo(samples, sample_rate.max(1))
        .expect("WAV float32 payload exceeds 4 GiB RIFF/WAVE limit")
}

#[cfg(test)]
mod tests {
    use std::f64::consts::TAU;

    use super::*;

    #[test]
    fn wav_pcm16_symphonia_decode_round_trip() {
        let sample_rate = 44_100;
        let frames = 2_048;
        let source: Vec<[f32; 2]> = (0..frames)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                let l = (TAU * 440.0 * t).sin() as f32 * 0.75;
                let r = (TAU * 880.0 * t).cos() as f32 * 0.5;
                [l, r]
            })
            .collect();

        let wav_bytes = encode_wav_pcm16_stereo(&source, sample_rate);
        let decoded = decode_audio_bytes(&wav_bytes, Some("wav")).expect("WAV decode succeeds");

        assert_eq!(decoded.sample_rate, sample_rate);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.frame_count(), frames);

        for (i, &[orig_l, orig_r]) in source.iter().enumerate() {
            let dec_l = decoded.samples_interleaved_f32[i * 2];
            let dec_r = decoded.samples_interleaved_f32[i * 2 + 1];
            assert!(
                (dec_l - orig_l).abs() < 5e-4,
                "left mismatch at {i}: got {dec_l}, expected {orig_l}"
            );
            assert!(
                (dec_r - orig_r).abs() < 5e-4,
                "right mismatch at {i}: got {dec_r}, expected {orig_r}"
            );
        }
    }

    #[test]
    fn wav_f32_symphonia_decode_round_trip() {
        let sample_rate = STANDARD_SAMPLE_RATE;
        let source: Vec<[f32; 2]> = (0..512)
            .map(|i| {
                let phase = (i as f32) / 512.0;
                [phase * 2.0 - 1.0, 1.0 - phase * 2.0]
            })
            .collect();

        let wav_bytes = encode_wav_f32_stereo(&source, sample_rate);
        let decoded = decode_audio_bytes(&wav_bytes, None).expect("float32 WAV decode succeeds");

        assert_eq!(decoded.sample_rate, STANDARD_SAMPLE_RATE);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.frame_count(), source.len());

        for (i, &[orig_l, orig_r]) in source.iter().enumerate() {
            let dec_l = decoded.samples_interleaved_f32[i * 2];
            let dec_r = decoded.samples_interleaved_f32[i * 2 + 1];
            assert!((dec_l - orig_l).abs() < 1e-6);
            assert!((dec_r - orig_r).abs() < 1e-6);
        }
    }

    #[test]
    fn resample_mono_24k_to_stereo_48k_preserves_ramp() {
        let buffer = AudioBuffer {
            sample_rate: 24_000,
            channels: 1,
            samples_interleaved_f32: vec![0.0, 0.5, 1.0, 0.5, 0.0],
        };

        let stereo = resample_stereo_48k(&buffer);
        assert_eq!(stereo.len(), 10);
        // Exact integer source positions are preserved bit-accurately by windowed-sinc upsampling.
        assert!((stereo[0][0] - 0.0).abs() < 1e-6);
        assert!((stereo[0][1] - 0.0).abs() < 1e-6);
        assert!((stereo[2][0] - 0.5).abs() < 1e-6);
        assert!((stereo[4][0] - 1.0).abs() < 1e-6);
        assert!((stereo[6][0] - 0.5).abs() < 1e-6);
        assert!((stereo[8][0] - 0.0).abs() < 1e-6);
        // Fractional positions stay symmetric (L == R) and smoothly monotonic across the rise.
        assert!((stereo[1][0] - stereo[1][1]).abs() < 1e-6);
        assert!(stereo[1][0] > 0.0 && stereo[1][0] < 0.5);
        assert!(stereo[3][0] > 0.5 && stereo[3][0] < 1.0);
    }

    #[test]
    fn upsampling_44k1_to_48k_blackman_sinc_suppresses_imaging_vs_linear() {
        let in_sr = 44_100_u32;
        let out_sr = STANDARD_SAMPLE_RATE;
        let in_frames = 2_205_usize; // 50 ms at 44.1 kHz
        let freq_hz = 14_000.0_f64;

        // 1. Constant DC signal upsampled 44.1 kHz -> 48 kHz remains flat within 1e-6.
        let dc_buf = AudioBuffer {
            sample_rate: in_sr,
            channels: 2,
            samples_interleaved_f32: vec![0.6_f32; in_frames * 2],
        };
        let dc_out = resample_stereo(&dc_buf, out_sr);
        assert_eq!(dc_out.len(), 2_400);
        for &[l, r] in &dc_out {
            assert!((l - 0.6).abs() < 1e-5, "DC ripple on L: {l}");
            assert!((r - 0.6).abs() < 1e-5, "DC ripple on R: {r}");
        }

        // 2. 14 kHz sine wave upsampled 44.1 kHz -> 48 kHz: compare against exact analytic sine
        //    and verify Blackman-windowed sinc dramatically beats 2-point linear interpolation.
        let sine_samples: Vec<f32> = (0..in_frames)
            .map(|i| (TAU * freq_hz * (i as f64) / f64::from(in_sr)).sin() as f32)
            .collect();
        let sine_buf = AudioBuffer {
            sample_rate: in_sr,
            channels: 1,
            samples_interleaved_f32: sine_samples,
        };
        let sinc_out = resample_stereo(&sine_buf, out_sr);
        let ratio = f64::from(in_sr) / f64::from(out_sr);

        let mut sinc_err_sq = 0.0_f64;
        let mut linear_err_sq = 0.0_f64;
        let margin = 32_usize;
        let count = sinc_out.len() - 2 * margin;
        for (idx, &[got_l, _]) in sinc_out.iter().enumerate().skip(margin).take(count) {
            let expected = (TAU * freq_hz * (idx as f64) / f64::from(out_sr)).sin();
            let err_sinc = f64::from(got_l) - expected;
            sinc_err_sq += err_sinc * err_sinc;

            let src_pos = (idx as f64) * ratio;
            let i0 = (src_pos.floor() as usize).min(in_frames - 1);
            let i1 = (i0 + 1).min(in_frames - 1);
            let frac = src_pos - i0 as f64;
            let a = f64::from(sine_buf.samples_interleaved_f32[i0]);
            let b = f64::from(sine_buf.samples_interleaved_f32[i1]);
            let lin = a + (b - a) * frac;
            let err_lin = lin - expected;
            linear_err_sq += err_lin * err_lin;
        }

        let sinc_rms_err = (sinc_err_sq / count as f64).sqrt();
        let linear_rms_err = (linear_err_sq / count as f64).sqrt();
        assert!(
            sinc_rms_err < 0.005,
            "Blackman-windowed sinc 14 kHz upsampling RMS error too high: {sinc_rms_err}"
        );
        assert!(
            linear_rms_err > sinc_rms_err * 20.0,
            "expected windowed sinc ({sinc_rms_err}) to beat linear ({linear_rms_err}) by >20x"
        );
    }

    #[test]
    fn decode_empty_returns_error() {
        let err = decode_audio_bytes(&[], Some("wav")).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput));
    }

    #[test]
    fn pcm16_scaling_is_asymmetric_full_range() {
        assert_eq!(f32_to_pcm16(-1.0), -32_768);
        assert_eq!(f32_to_pcm16(1.0), 32_767);
        assert_eq!(f32_to_pcm16(0.0), 0);
        assert_eq!(f32_to_pcm16(-2.0), -32_768);
        assert_eq!(f32_to_pcm16(2.0), 32_767);
        assert_eq!(f32_to_pcm16(f32::NAN), 0);

        let wav = try_encode_wav_pcm16_stereo(&[[-1.0, 1.0]], 48_000).expect("encode wav");
        let l = i16::from_le_bytes([wav[44], wav[45]]);
        let r = i16::from_le_bytes([wav[46], wav[47]]);
        assert_eq!(l, -32_768);
        assert_eq!(r, 32_767);
    }

    #[test]
    fn wav_4gib_overflow_guard_rejects_oversized_payloads() {
        // Max valid PCM16 stereo frames: (u32::MAX - 36) / 4
        let max_pcm16_frames = ((u32::MAX - 36) / 4) as usize;
        assert!(checked_wav_chunk_sizes(max_pcm16_frames, 4, 36).is_ok());
        assert!(matches!(
            checked_wav_chunk_sizes(max_pcm16_frames + 1, 4, 36),
            Err(AudioError::InvalidFormat(_))
        ));

        // Max valid float32 stereo frames: (u32::MAX - 48) / 8
        let max_f32_frames = ((u32::MAX - 48) / 8) as usize;
        assert!(checked_wav_chunk_sizes(max_f32_frames, 8, 48).is_ok());
        assert!(matches!(
            checked_wav_chunk_sizes(max_f32_frames + 1, 8, 48),
            Err(AudioError::InvalidFormat(_))
        ));

        // Zero sample rate rejected by fallible encoders
        assert!(matches!(
            try_encode_wav_pcm16_stereo(&[[0.0, 0.0]], 0),
            Err(AudioError::InvalidFormat(_))
        ));
        assert!(matches!(
            try_encode_wav_f32_stereo(&[[0.0, 0.0]], 0),
            Err(AudioError::InvalidFormat(_))
        ));
    }

    #[test]
    fn surround_5_1_downmix_centers_dialogue_and_omits_lfe() {
        // 5.1 layout: FL=0, FR=1, FC=2, LFE=3, BL=4, BR=5
        // Dialogue only in FC=1.0, LFE=0.9 (should be ignored, never panned right)
        let buf = AudioBuffer {
            sample_rate: 48_000,
            channels: 6,
            samples_interleaved_f32: vec![0.0, 0.0, 1.0, 0.9, 0.0, 0.0],
        };
        let stereo = downmix_frame_to_stereo(&buf, 0);
        assert!((stereo[0] - FRAC_1_SQRT_2).abs() < 1e-6);
        assert!((stereo[1] - FRAC_1_SQRT_2).abs() < 1e-6);

        // Quad layout: FL=0.8, FR=0.2, BL=0.4, BR=0.6
        let quad = AudioBuffer {
            sample_rate: 48_000,
            channels: 4,
            samples_interleaved_f32: vec![0.8, 0.2, 0.4, 0.6],
        };
        let q = downmix_frame_to_stereo(&quad, 0);
        assert!((q[0] - 0.6).abs() < 1e-6);
        assert!((q[1] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn downsampling_96k_to_48k_attenuates_above_nyquist_aliasing() {
        let sr = 96_000_u32;
        let frames = 2_048_usize;
        // 36 kHz tone at 96 kHz is well above 48 kHz Nyquist (24 kHz) -> would alias to 12 kHz without LPF.
        let high_freq: Vec<f32> = (0..frames)
            .map(|i| (TAU * 36_000.0 * (i as f64) / f64::from(sr)).sin() as f32)
            .collect();
        let hi_buf = AudioBuffer {
            sample_rate: sr,
            channels: 1,
            samples_interleaved_f32: high_freq,
        };
        let resampled_hi = resample_stereo(&hi_buf, 48_000);
        let steady_hi_peak = resampled_hi[64..resampled_hi.len() - 64]
            .iter()
            .fold(0.0_f32, |acc, &[l, _]| acc.max(l.abs()));
        assert!(
            steady_hi_peak < 0.05,
            "expected >26 dB attenuation of 36 kHz aliasing tone, got peak {steady_hi_peak}"
        );

        // 1 kHz in-band tone passes cleanly near unity gain.
        let low_freq: Vec<f32> = (0..frames)
            .map(|i| (TAU * 1_000.0 * (i as f64) / f64::from(sr)).sin() as f32)
            .collect();
        let lo_buf = AudioBuffer {
            sample_rate: sr,
            channels: 1,
            samples_interleaved_f32: low_freq,
        };
        let resampled_lo = resample_stereo(&lo_buf, 48_000);
        let steady_lo_peak = resampled_lo[64..resampled_lo.len() - 64]
            .iter()
            .fold(0.0_f32, |acc, &[l, _]| acc.max(l.abs()));
        assert!(
            (steady_lo_peak - 1.0).abs() < 0.05,
            "expected 1 kHz passband tone near 1.0, got {steady_lo_peak}"
        );
    }

    #[test]
    fn trig_recurrence_sinc_matches_direct_sin_cos_reference() {
        fn reference_sample_stereo_sinc(
            buffer: &AudioBuffer,
            src_pos: f64,
            ratio: f64,
        ) -> [f32; 2] {
            let frames = buffer.frame_count();
            if !src_pos.is_finite() || !ratio.is_finite() || src_pos < 0.0 || frames == 0 {
                return [0.0, 0.0];
            }
            let center = src_pos.floor() as isize;
            if center as usize >= frames {
                return [0.0, 0.0];
            }
            let effective_ratio = ratio.max(1.0);
            let cutoff = 0.9 / effective_ratio;
            let radius = ((8.0 * effective_ratio - 1e-9).ceil() as isize).clamp(8, 64);
            let radius_f = radius as f64;
            let max_idx = (frames - 1) as isize;

            let mut acc_l = 0.0_f64;
            let mut acc_r = 0.0_f64;
            let mut weight_sum = 0.0_f64;

            let start = (center - radius).max(0);
            let end = (center + radius).min(max_idx);
            for idx in start..=end {
                let d = src_pos - idx as f64;
                if d.abs() > radius_f {
                    continue;
                }
                let window = 0.5 * (1.0 + (PI * d / radius_f).cos());
                let x = PI * cutoff * d;
                let sinc = if x.abs() < 1e-9 { 1.0 } else { x.sin() / x };
                let w = sinc * window;
                let s = downmix_frame_to_stereo(buffer, idx as usize);
                acc_l += f64::from(s[0]) * w;
                acc_r += f64::from(s[1]) * w;
                weight_sum += w;
            }
            if weight_sum.abs() <= 1e-12 {
                return downmix_frame_to_stereo(buffer, (center as usize).min(frames - 1));
            }
            let inv = 1.0 / weight_sum;
            [(acc_l * inv) as f32, (acc_r * inv) as f32]
        }

        let frames = 512_usize;
        for channels in [1_u16, 2, 6] {
            let mut samples = Vec::with_capacity(frames * usize::from(channels));
            for i in 0..frames {
                let t = i as f64 / 48_000.0;
                for c in 0..channels {
                    let freq = 440.0 + f64::from(c) * 310.0;
                    samples.push((TAU * freq * t + f64::from(c) * 0.37).sin() as f32 * 0.8);
                }
            }
            let buf = AudioBuffer {
                sample_rate: 96_000,
                channels,
                samples_interleaved_f32: samples,
            };
            for ratio in [1.1, 1.5, 2.0, 4.0, 10.0] {
                let kernel = SincResampleKernel::new(ratio);
                for &pos in &[
                    0.0,
                    0.35,
                    1.0 + 1e-10,
                    12.0 + 1e-6,
                    64.0,
                    128.5,
                    250.999999,
                    511.25,
                ] {
                    let expected = reference_sample_stereo_sinc(&buf, pos, ratio);
                    let got = kernel.sample_stereo(&buf, pos);
                    let diff_l = (got[0] - expected[0]).abs();
                    let diff_r = (got[1] - expected[1]).abs();
                    assert!(
                        diff_l < 1e-6 && diff_r < 1e-6,
                        "ch={channels} ratio={ratio} pos={pos}: got={got:?} expected={expected:?} diff=({diff_l}, {diff_r})"
                    );
                }
            }
        }
    }
}
