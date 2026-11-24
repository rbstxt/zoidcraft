//! Audio decoding via `symphonia` (WAV, MP3, OGG/Vorbis, FLAC, AAC, MP4),
//! stereo linear resampling to 48 kHz, and lightweight WAV encoders.

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

    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(audio_params, &AudioDecoderOptions::default())
        .map_err(|err| AudioError::UnsupportedCodec(err.to_string()))?;

    let mut sample_rate: Option<u32> = None;
    let mut channels: Option<u16> = None;
    let mut samples_interleaved_f32 = Vec::new();
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

/// Converts and linearly resamples an [`AudioBuffer`] to stereo `48,000 Hz`.
#[must_use]
pub fn resample_stereo_48k(buffer: &AudioBuffer) -> Vec<[f32; 2]> {
    resample_stereo(buffer, STANDARD_SAMPLE_RATE)
}

/// Converts and linearly resamples an [`AudioBuffer`] to stereo at `target_sample_rate`.
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
            out.push(read_stereo_frame(buffer, frame_idx));
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
    let last_frame = in_frames - 1;

    let mut out = Vec::with_capacity(out_frames);
    for out_idx in 0..out_frames {
        let src_pos = (out_idx as f64) * ratio;
        let idx0 = (src_pos.floor() as usize).min(last_frame);
        let idx1 = (idx0 + 1).min(last_frame);
        let frac = (src_pos - idx0 as f64).clamp(0.0, 1.0) as f32;

        let a = read_stereo_frame(buffer, idx0);
        let b = read_stereo_frame(buffer, idx1);
        out.push([a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac]);
    }

    out
}

#[inline]
fn read_stereo_frame(buffer: &AudioBuffer, frame_idx: usize) -> [f32; 2] {
    let ch = usize::from(buffer.channels);
    let base = frame_idx * ch;
    match ch {
        1 => {
            let s = buffer.samples_interleaved_f32[base];
            [s, s]
        }
        2 => [
            buffer.samples_interleaved_f32[base],
            buffer.samples_interleaved_f32[base + 1],
        ],
        _ => {
            let mut left = 0.0_f32;
            let mut right = 0.0_f32;
            let mut left_count = 0_u32;
            let mut right_count = 0_u32;
            for c in 0..ch {
                let sample = buffer.samples_interleaved_f32[base + c];
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

/// Encodes stereo `f32` samples as a standard 16-bit signed PCM RIFF/WAVE byte vector.
#[must_use]
pub fn encode_wav_pcm16_stereo(samples: &[[f32; 2]], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let block_align = num_channels * (bits_per_sample / 8);
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));
    let data_bytes = (samples.len() * usize::from(block_align)) as u32;
    let riff_size = 36_u32.saturating_add(data_bytes);

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
        let l16 = (l.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        let r16 = (r.clamp(-1.0, 1.0) * f32::from(i16::MAX)).round() as i16;
        out.extend_from_slice(&l16.to_le_bytes());
        out.extend_from_slice(&r16.to_le_bytes());
    }
    out
}

/// Encodes stereo `f32` samples as a standard 32-bit IEEE float RIFF/WAVE byte vector.
#[must_use]
pub fn encode_wav_f32_stereo(samples: &[[f32; 2]], sample_rate: u32) -> Vec<u8> {
    let num_channels: u16 = 2;
    let bits_per_sample: u16 = 32;
    let block_align = num_channels * (bits_per_sample / 8);
    let byte_rate = sample_rate.saturating_mul(u32::from(block_align));
    let data_bytes = (samples.len() * usize::from(block_align)) as u32;
    let riff_size = 48_u32.saturating_add(data_bytes);

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
    out.extend_from_slice(&(samples.len() as u32).to_le_bytes());

    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());
    for &[l, r] in samples {
        out.extend_from_slice(&l.to_le_bytes());
        out.extend_from_slice(&r.to_le_bytes());
    }
    out
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
        // Sample 0 is 0.0, sample 1 interpolates halfway to 0.5 => 0.25.
        assert!((stereo[0][0] - 0.0).abs() < 1e-6);
        assert!((stereo[0][1] - 0.0).abs() < 1e-6);
        assert!((stereo[1][0] - 0.25).abs() < 1e-6);
        assert!((stereo[1][1] - 0.25).abs() < 1e-6);
        assert!((stereo[2][0] - 0.5).abs() < 1e-6);
        assert!((stereo[4][0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decode_empty_returns_error() {
        let err = decode_audio_bytes(&[], Some("wav")).unwrap_err();
        assert!(matches!(err, AudioError::EmptyInput));
    }
}
