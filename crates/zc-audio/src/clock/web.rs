//! `globalThis.__zcWebAudio.start(sampleRate, pull)` opens a stereo output stream
//! at the AudioContext's native sample rate (via `AudioWorkletNode` with automatic
//! scheduler fallback) and returns a function that stops it and releases `pull`.
//! `globalThis.__zcWebAudio.sampleRate()` reports the AudioContext's live sample rate.
//! `pull(frameCount, outputDelaySeconds)` returns interleaved stereo Float32Array samples.

use js_sys::{Float32Array, Function, Reflect};
use wasm_bindgen::{JsCast, JsValue, closure::Closure};

use super::*;

type PullAudio = dyn FnMut(u32, f64) -> Float32Array;

pub(super) struct WebAudioStream {
    stop: Function,
    _pull: Closure<PullAudio>,
}

impl Drop for WebAudioStream {
    fn drop(&mut self) {
        let _ = self.stop.call0(&JsValue::UNDEFINED);
    }
}

fn query_web_audio_sample_rate(bridge: &JsValue) -> u32 {
    Reflect::get(bridge, &"sampleRate".into())
        .ok()
        .and_then(|value| value.dyn_into::<Function>().ok())
        .and_then(|func| func.call0(bridge).ok())
        .and_then(|value| value.as_f64())
        .filter(|sr| sr.is_finite() && (8_000.0..=384_000.0).contains(sr))
        .map_or(STANDARD_SAMPLE_RATE, |sr| sr.round() as u32)
}

impl MasterPlaybackClock {
    /// Opens a browser output stream for the supplied timeline window.
    pub fn start_audio_output_windowed(
        &self,
        window: Option<AudioMixWindow>,
    ) -> Result<AudioOutputHandle, AudioError> {
        let bridge = Reflect::get(&js_sys::global(), &"__zcWebAudio".into())
            .map_err(|_| AudioError::NoOutputDevice)?;
        let start = Reflect::get(&bridge, &"start".into())
            .ok()
            .and_then(|value| value.dyn_into::<Function>().ok())
            .ok_or(AudioError::NoOutputDevice)?;
        let output_sample_rate = query_web_audio_sample_rate(&bridge);

        let window_slot = Arc::new(Mutex::new(window.clone()));
        let window_gen = Arc::new(AtomicU64::new(0));
        let loop_head_slot = Arc::new(Mutex::new(None));
        let loop_head_gen = Arc::new(AtomicU64::new(0));
        let mut state = CallbackMixState {
            active_window: window,
            seen_window_gen: 0,
            window_slot: Arc::clone(&window_slot),
            window_gen: Arc::clone(&window_gen),
            active_loop_head: None,
            seen_loop_head_gen: 0,
            loop_head_slot: Arc::clone(&loop_head_slot),
            loop_head_gen: Arc::clone(&loop_head_gen),
        };

        let shared = Arc::clone(&self.shared);
        let stream_id = shared.attach_hardware_clock(output_sample_rate);
        let pull_shared = Arc::clone(&shared);
        let mut output = Vec::new();
        let pull = Closure::<PullAudio>::new(move |frame_count: u32, delay_secs| {
            if pull_shared.active_stream_id.load(Ordering::Acquire) != stream_id {
                return Float32Array::new_with_length(0);
            }
            let Some(sample_count) = (frame_count as usize).checked_mul(2) else {
                return Float32Array::new_with_length(0);
            };
            output.resize(sample_count, 0.0);
            state.poll_updates();
            let (origin, mix) = state.active_mix();
            render_callback_frames_windowed_with_loop_head(
                &pull_shared,
                mix,
                origin,
                state.active_loop_head.as_deref(),
                f64::from(STANDARD_SAMPLE_RATE),
                2,
                delay_secs,
                &mut output,
                |sample| sample,
            );
            Float32Array::from(output.as_slice())
        });

        let stop = start
            .call2(&bridge, &output_sample_rate.into(), pull.as_ref())
            .and_then(JsCast::dyn_into::<Function>);
        let stop = match stop {
            Ok(stop) => stop,
            Err(error) => {
                shared.detach_hardware_clock(stream_id);
                return Err(AudioError::OutputDevice(format!(
                    "could not start Web Audio output: {error:?}"
                )));
            }
        };

        Ok(AudioOutputHandle {
            stream: Some(WebAudioStream { stop, _pull: pull }),
            stream_id,
            shared,
            sample_rate: output_sample_rate,
            channels: 2,
            window_slot,
            window_gen,
            loop_head_slot,
            loop_head_gen,
        })
    }
}
