//! Clean-room audio decoding, resampling, offline stereo mixing, metering,
//! waveform peak caching, and sample-accurate playback clock for ZoidCraft.

#![forbid(unsafe_code)]

pub mod clock;
pub mod decode;
pub mod meter;
pub mod mixer;
pub mod waveform;

pub use clock::{AudioOutputHandle, MasterPlaybackClock, PlaybackSnapshot};
pub use decode::{
    AudioBuffer, STANDARD_SAMPLE_RATE, decode_audio_bytes, encode_wav_f32_stereo,
    encode_wav_pcm16_stereo, resample_stereo, resample_stereo_48k,
};
pub use meter::{AudioMeterState, METER_ANALYSIS_WINDOW, METER_MIN_PX};
pub use mixer::{
    AudioClipScheduleItem, AudioClipTrackItem, ClipAudioSampleParams, apply_gain_and_pan,
    effective_playback_rate, mix_offline_stereo, mix_offline_stereo_animated,
};
pub use waveform::{DEFAULT_POINTS_PER_SEC, WaveformColumn, WaveformPeaks};

use thiserror::Error;

/// Errors produced by audio decoding, resampling, or live playback output.
#[derive(Debug, Error)]
pub enum AudioError {
    /// Input byte slice was empty.
    #[error("audio input is empty")]
    EmptyInput,

    /// Container probing failed.
    #[error("failed to probe audio container: {0}")]
    Probe(String),

    /// No audio stream existed in the probed media container.
    #[error("media container does not contain an audio track")]
    NoAudioTrack,

    /// Codec instantiation failed or codec is unsupported.
    #[error("unsupported or invalid audio codec: {0}")]
    UnsupportedCodec(String),

    /// Packet demuxing or audio decoding failed.
    #[error("audio decode error: {0}")]
    Decode(String),

    /// Stream had invalid channel count or sample rate metadata.
    #[error("invalid audio stream format: {0}")]
    InvalidFormat(String),

    /// No audio device was available for live output.
    #[error("no audio output device available")]
    NoOutputDevice,

    /// Live `cpal` output stream initialization or control failed.
    #[error("audio output device error: {0}")]
    OutputDevice(String),
}
