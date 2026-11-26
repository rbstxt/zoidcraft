//! Live audio mixing, CPAL playback, waveform peak sync, and transport ticking.

use super::*;

/// Maximum live audio playback mix window duration in seconds (60s = 2.88M stereo frames at 48 kHz,
/// well inside `zc-audio`'s 32 Mi sample cap even for multi-hour timelines).
#[cfg(not(target_arch = "wasm32"))]
pub(crate) const LIVE_AUDIO_WINDOW_SECS: f64 = 60.0;
#[cfg(target_arch = "wasm32")]
pub(crate) const LIVE_AUDIO_WINDOW_SECS: f64 = 4.0;

/// Latches OS audio-output open failures so headless or muted machines neither retry CPAL
/// every 60 seconds during playback nor spam repeated "Audio output unavailable" toasts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AudioOutputFailureLatch {
    pub(crate) unavailable_for_run: bool,
    pub(crate) notified: bool,
}

impl AudioOutputFailureLatch {
    pub(crate) fn should_attempt_open(&self) -> bool {
        !self.unavailable_for_run
    }

    pub(crate) fn on_open_success(&mut self) {
        self.unavailable_for_run = false;
        self.notified = false;
    }

    /// Records an audio device open failure. Returns `true` only the first time a failure
    /// occurs so the caller shows a single status toast instead of spamming repeatedly.
    pub(crate) fn on_open_failure(&mut self) -> bool {
        self.unavailable_for_run = true;
        if !self.notified {
            self.notified = true;
            true
        } else {
            false
        }
    }

    /// Clears the per-playback-run block when playback stops so the next Play press can
    /// re-probe (e.g., if headphones were plugged in), while keeping `notified` latched so
    /// persistent failures never re-spam toasts.
    pub(crate) fn on_playback_stopped(&mut self) {
        self.unavailable_for_run = false;
    }
}

pub(crate) type AudioMixHashCacheKey = (u64, usize, u64, u64, bool);

pub(crate) struct AudioMixRawAsset {
    pub(crate) bytes: Vec<u8>,
    pub(crate) ext: Option<String>,
}

pub(crate) struct AudioMixJobResult {
    pub(crate) mix_hash: u64,
    pub(crate) win_start: u32,
    pub(crate) win_end: u32,
    pub(crate) window: Result<Option<AudioMixWindow>, String>,
    pub(crate) loop_head: Option<Arc<Vec<[f32; 2]>>>,
}

#[cfg(test)]
pub(crate) fn collect_in_range_audio_arcs(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    settings: &CoreExportSettings,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> HashMap<String, Arc<AudioBuffer>> {
    collect_in_range_audio_arcs_for_project(
        &session.project,
        session.plugins.is_enabled("precomp-plus"),
        cached_buffers,
        settings,
        start_seq_frame,
        end_seq_frame,
    )
}

pub(crate) fn collect_in_range_audio_arcs_for_project(
    project: &Project,
    precomp_enabled: bool,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    settings: &CoreExportSettings,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> HashMap<String, Arc<AudioBuffer>> {
    let mut selected = HashMap::new();
    if matches!(
        settings.container,
        CoreExportContainer::PngSequence | CoreExportContainer::Gif
    ) || (!settings.container.is_audio_only() && settings.audio_codec == AudioCodec::None)
        || cached_buffers.is_empty()
    {
        return selected;
    }
    let flattened = collect_flattened_audio_clips_with_precomp(
        project,
        f64::from(start_seq_frame),
        f64::from(end_seq_frame),
        precomp_enabled,
    );
    for clip in &flattened {
        if let Some(buf) = cached_buffers.get(&clip.asset_id) {
            selected
                .entry(clip.asset_id.clone())
                .or_insert_with(|| Arc::clone(buf));
        }
    }
    selected
}

/// Returns `true` when the live CPAL audio output window must be rebuilt and re-opened.
pub(crate) fn live_audio_needs_rebuild(
    audio_mix_hash: u64,
    audio_window_start: u32,
    audio_window_end: u32,
    audio_stream_attempted_hash: u64,
    has_audio_output: bool,
    mix_hash: u64,
    cur_f: f64,
) -> bool {
    let outside_window = audio_window_end <= audio_window_start
        || cur_f < f64::from(audio_window_start)
        || cur_f >= f64::from(audio_window_end);
    let needs_mix = audio_mix_hash != mix_hash || outside_window;
    needs_mix && (has_audio_output || audio_stream_attempted_hash != mix_hash || outside_window)
}

/// Returns `true` when the playhead has crossed 75% of the currently loaded live audio window
/// and more timeline exists in the active playback direction, triggering an off-thread prefetch
/// so the next window can be hot-swapped seamlessly before the boundary is reached.
pub(crate) fn live_audio_should_prefetch(
    audio_mix_hash: u64,
    audio_window_start: u32,
    audio_window_end: u32,
    mix_hash: u64,
    cur_f: f64,
    speed: f64,
    total_frames: u32,
) -> bool {
    if audio_mix_hash != mix_hash || audio_window_end <= audio_window_start {
        return false;
    }
    let span = f64::from(audio_window_end - audio_window_start);
    if span <= 1.0 {
        return false;
    }
    if speed > 0.0 && audio_window_end < total_frames {
        let progress = (cur_f - f64::from(audio_window_start)) / span;
        progress >= 0.75
    } else if speed < 0.0 && audio_window_start > 0 {
        let progress = (f64::from(audio_window_end) - cur_f) / span;
        progress >= 0.75
    } else {
        false
    }
}

pub(crate) type AudioMixInputs = (
    Vec<FlattenedAudioClip>,
    HashMap<String, Arc<AudioBuffer>>,
    HashMap<String, AudioMixRawAsset>,
);

pub(crate) fn collect_audio_mix_inputs(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    range_start: f64,
    range_end: f64,
) -> AudioMixInputs {
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        range_start,
        range_end,
        session.plugins.is_enabled("precomp-plus"),
    );
    let mut buffers: HashMap<String, Arc<AudioBuffer>> = HashMap::new();
    let mut raw_assets: HashMap<String, AudioMixRawAsset> = HashMap::new();
    let mut active_clips = Vec::with_capacity(flattened.len());

    for clip in flattened {
        if !buffers.contains_key(&clip.asset_id) && !raw_assets.contains_key(&clip.asset_id) {
            if let Some(cached) = cached_buffers.get(&clip.asset_id) {
                if cached.is_empty() {
                    continue;
                }
                buffers.insert(clip.asset_id.clone(), Arc::clone(cached));
            } else if let Some(rec) = session.project.assets.get(&clip.asset_id)
                && let Some(bytes) = rec.bytes.as_deref()
            {
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_owned);
                raw_assets.insert(
                    clip.asset_id.clone(),
                    AudioMixRawAsset {
                        bytes: bytes.to_vec(),
                        ext,
                    },
                );
            } else {
                continue;
            }
        }
        active_clips.push(clip);
    }
    (active_clips, buffers, raw_assets)
}

pub(crate) fn execute_audio_mix_work(
    clips: Vec<FlattenedAudioClip>,
    mut buffers: HashMap<String, Arc<AudioBuffer>>,
    raw_assets: HashMap<String, AudioMixRawAsset>,
    win_start: u32,
    win_end: u32,
    fps: f64,
) -> Result<Option<AudioMixWindow>, String> {
    if clips.is_empty() {
        return Ok(None);
    }
    let mut silent_assets: HashSet<String> = HashSet::new();
    for (asset_id, raw) in raw_assets {
        match decode_audio_bytes(&raw.bytes, raw.ext.as_deref()) {
            Ok(decoded) => {
                if decoded.is_empty() {
                    silent_assets.insert(asset_id);
                } else {
                    buffers.insert(asset_id, Arc::new(decoded));
                }
            }
            Err(AudioError::NoAudioTrack) => {
                silent_assets.insert(asset_id);
            }
            Err(err) => {
                return Err(format!("failed to decode audio asset `{asset_id}`: {err}"));
            }
        }
    }
    let active_clips: Vec<FlattenedAudioClip> = clips
        .into_iter()
        .filter(|c| !silent_assets.contains(&c.asset_id) && buffers.contains_key(&c.asset_id))
        .collect();
    if active_clips.is_empty() {
        return Ok(None);
    }
    let range_start = f64::from(win_start);
    let span_frames = f64::from(win_end.saturating_sub(win_start).max(1));
    let samples = mix_flattened_audio_clips(
        &active_clips,
        range_start,
        span_frames,
        fps.max(1.0),
        STANDARD_SAMPLE_RATE,
        &buffers,
    )
    .map_err(|err| format!("failed to mix sequence audio: {err}"))?;
    Ok(Some(AudioMixWindow {
        origin_frame: range_start,
        samples: Arc::new(samples),
    }))
}

/// Computes a bounded `[start_frame, end_frame)` live audio mix window covering `current_frame`
/// without exceeding `window_frames`, supporting both forward and reverse playback on long timelines.
pub(crate) fn compute_live_audio_window(
    current_frame: f64,
    speed: f64,
    total_frames: u32,
    window_frames: u32,
) -> (u32, u32) {
    let total = total_frames.max(1);
    let span = window_frames.max(1);
    if total <= span {
        return (0, total);
    }
    let cur = (current_frame.floor().max(0.0) as u32).min(total.saturating_sub(1));
    if speed < 0.0 {
        let end = cur.saturating_add(1).min(total);
        let start = end.saturating_sub(span);
        (start, end.max(start + 1))
    } else {
        let start = cur;
        let end = start.saturating_add(span).min(total);
        (start, end.max(start + 1))
    }
}

pub(crate) fn compute_audio_mix_hash(
    session: &Session,
    decoded_audio: &HashMap<String, Arc<AudioBuffer>>,
    fps: f64,
    seq_len: f64,
) -> u64 {
    let precomp_enabled = session.plugins.is_enabled("precomp-plus");
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        0.0,
        seq_len.max(1.0),
        precomp_enabled,
    );
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut active_clips = 0_usize;

    fnv_mix_f64(&mut hash, fps);
    fnv_mix_f64(&mut hash, seq_len);
    fnv_mix(&mut hash, &session.revision().to_le_bytes());
    fnv_mix(
        &mut hash,
        &[
            u8::from(session.plugins.is_enabled("expressions-plus")),
            u8::from(precomp_enabled),
        ],
    );

    for clip in &flattened {
        let has_audio = decoded_audio.contains_key(&clip.asset_id)
            || session
                .project
                .assets
                .get(&clip.asset_id)
                .and_then(|r| r.bytes.as_ref())
                .is_some();
        if !has_audio {
            continue;
        }
        active_clips += 1;
        fnv_mix(&mut hash, &(clip.asset_id.len() as u64).to_le_bytes());
        fnv_mix(&mut hash, clip.asset_id.as_bytes());
        fnv_mix_f64(&mut hash, clip.root_start_frame);
        fnv_mix_f64(&mut hash, clip.root_length_frames);
        fnv_mix_f64(&mut hash, clip.own_start_frame);
        fnv_mix_f64(&mut hash, clip.own_length_frames);
        fnv_mix_f64(&mut hash, clip.own_fps);
        fnv_mix_f64(&mut hash, clip.media_offset_sec);
        fnv_mix(&mut hash, &(clip.depth as u64).to_le_bytes());
    }

    if active_clips == 0 {
        return 0;
    }

    // Fold audio track properties/keyframes/expressions + sequence controls + nested compositions
    // so direct mutations (including Bezier handle edits and Precomp+ edits) invalidate immediately.
    let _ = serde_json::to_writer(FnvWriter(&mut hash), &session.project.sequence.audio_tracks);
    let _ = serde_json::to_writer(FnvWriter(&mut hash), &session.project.sequence.controls);
    if !session.project.compositions.is_empty() {
        let mut comp_ids: Vec<Uuid> = session.project.compositions.keys().copied().collect();
        comp_ids.sort();
        for cid in comp_ids {
            if let Some(comp) = session.project.compositions.get(&cid) {
                fnv_mix(&mut hash, cid.as_bytes());
                let _ = serde_json::to_writer(FnvWriter(&mut hash), &comp.audio_tracks);
                let _ = serde_json::to_writer(FnvWriter(&mut hash), &comp.controls);
            }
        }
    }

    hash.max(1)
}

pub(crate) fn build_offline_export_audio(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    settings: &CoreExportSettings,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    if matches!(
        settings.container,
        CoreExportContainer::PngSequence | CoreExportContainer::Gif
    ) || (!settings.container.is_audio_only() && settings.audio_codec == AudioCodec::None)
    {
        return Ok(None);
    }
    let sr = settings.sample_rate.max(8_000);
    mix_sequence_audio_stereo(
        session,
        cached_buffers,
        sr,
        seq_fps,
        start_seq_frame,
        end_seq_frame,
    )
}

pub(crate) fn mix_sequence_audio_stereo(
    session: &Session,
    cached_buffers: &HashMap<String, Arc<AudioBuffer>>,
    sample_rate: u32,
    seq_fps: f64,
    start_seq_frame: u32,
    end_seq_frame: u32,
) -> Result<Option<Vec<[f32; 2]>>, String> {
    let range_start = f64::from(start_seq_frame);
    let range_end = f64::from(end_seq_frame);
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        range_start,
        range_end,
        session.plugins.is_enabled("precomp-plus"),
    );
    if flattened.is_empty() {
        return Ok(None);
    }

    let safe_fps = seq_fps.max(1.0);
    let mut buffers: HashMap<String, Arc<AudioBuffer>> = HashMap::new();
    let mut silent_assets: HashSet<String> = HashSet::new();
    let mut active_clips = Vec::new();

    for clip in flattened {
        if silent_assets.contains(&clip.asset_id) {
            continue;
        }
        if !buffers.contains_key(&clip.asset_id) {
            if let Some(cached) = cached_buffers.get(&clip.asset_id) {
                if cached.is_empty() {
                    silent_assets.insert(clip.asset_id.clone());
                    continue;
                }
                buffers.insert(clip.asset_id.clone(), Arc::clone(cached));
            } else {
                let Some(rec) = session.project.assets.get(&clip.asset_id) else {
                    continue;
                };
                let Some(bytes) = rec.bytes.as_deref() else {
                    continue;
                };
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str());
                match decode_audio_bytes(bytes, ext) {
                    Ok(decoded) => {
                        if decoded.is_empty() {
                            silent_assets.insert(clip.asset_id.clone());
                            continue;
                        }
                        buffers.insert(clip.asset_id.clone(), Arc::new(decoded));
                    }
                    Err(AudioError::NoAudioTrack) => {
                        tracing::debug!(
                            "audio asset `{}` has no audio track; treating as silent",
                            clip.asset_id
                        );
                        silent_assets.insert(clip.asset_id.clone());
                        continue;
                    }
                    Err(err) => {
                        return Err(format!(
                            "failed to decode audio asset `{}`: {err}",
                            clip.asset_id
                        ));
                    }
                }
            }
        }
        active_clips.push(clip);
    }

    if active_clips.is_empty() {
        return Ok(None);
    }
    let seq_span_frames = f64::from(end_seq_frame.saturating_sub(start_seq_frame).max(1));
    let sr = sample_rate.max(8_000);
    mix_flattened_audio_clips(
        &active_clips,
        range_start,
        seq_span_frames,
        safe_fps,
        sr,
        &buffers,
    )
    .map(Some)
    .map_err(|err| format!("failed to mix sequence audio: {err}"))
}

/// Reads a 512-sample stereo RMS meter window directly from the active pre-mixed
/// [`AudioMixWindow`] in O(1) without re-flattening or re-mixing clips on the UI thread.
pub(crate) fn sample_live_audio_meter_from_window(
    window: &AudioMixWindow,
    frame: f64,
    fps: f64,
) -> Option<(f32, f32)> {
    const METER_WINDOW: usize = 512;
    let samples = window.samples.as_slice();
    if samples.is_empty() {
        return None;
    }
    let safe_fps = fps.max(1.0);
    let rel_sec = (frame - window.origin_frame) / safe_fps;
    if !rel_sec.is_finite() || rel_sec < 0.0 {
        return None;
    }
    let start_idx = (rel_sec * f64::from(STANDARD_SAMPLE_RATE)).floor() as usize;
    if start_idx >= samples.len() {
        return None;
    }
    let end_idx = start_idx.saturating_add(METER_WINDOW).min(samples.len());
    let mut sum_sq_l = 0.0_f64;
    let mut sum_sq_r = 0.0_f64;
    for &[sl, sr] in &samples[start_idx..end_idx] {
        let l = f64::from(sl.clamp(-1.0, 1.0));
        let r = f64::from(sr.clamp(-1.0, 1.0));
        sum_sq_l += l * l;
        sum_sq_r += r * r;
    }
    let denom = METER_WINDOW as f64;
    let rms_l = (sum_sq_l / denom).sqrt().clamp(0.0, 1.0) as f32;
    let rms_r = (sum_sq_r / denom).sqrt().clamp(0.0, 1.0) as f32;
    Some((rms_l, rms_r))
}

pub(crate) fn sample_live_audio_meter(
    session: &Session,
    buffers: &HashMap<String, Arc<AudioBuffer>>,
    frame: f64,
    fps: f64,
) -> (f32, f32) {
    const METER_WINDOW: usize = 512;
    if buffers.is_empty() {
        return (0.0, 0.0);
    }
    let safe_fps = fps.max(1.0);
    let dt_frames = (METER_WINDOW as f64 / f64::from(STANDARD_SAMPLE_RATE)) * safe_fps;
    let flattened = collect_flattened_audio_clips_with_precomp(
        &session.project,
        frame,
        frame + dt_frames.max(1.0),
        session.plugins.is_enabled("precomp-plus"),
    );
    if flattened.is_empty() {
        return (0.0, 0.0);
    }

    let mut mix_l = [0.0_f32; METER_WINDOW];
    let mut mix_r = [0.0_f32; METER_WINDOW];
    let mut has_active_audio = false;

    for clip in &flattened {
        let local_f = frame - clip.root_start_frame;
        if local_f < 0.0 || local_f >= clip.root_length_frames {
            continue;
        }
        let Some(buf) = buffers.get(&clip.asset_id) else {
            continue;
        };
        let total_frames = buf.frame_count();
        if total_frames == 0 {
            continue;
        }
        let p0 = clip.evaluate_params(local_f);
        if !p0.media_time_sec.is_finite() || p0.media_time_sec < 0.0 {
            continue;
        }
        let probe_dt = dt_frames.max(1e-4);
        let p1 = clip.evaluate_params((local_f + probe_dt).min(clip.root_length_frames - 1e-6));
        let effective_rate = if p1.media_time_sec.is_finite() && p1.media_time_sec >= 0.0 {
            (p1.media_time_sec - p0.media_time_sec) / (probe_dt / safe_fps)
        } else {
            1.0
        };
        let sr = f64::from(buf.sample_rate.max(1));
        let base_pos = p0.media_time_sec * sr;
        let step_samples = (sr / f64::from(STANDARD_SAMPLE_RATE)) * effective_rate;

        for i in 0..METER_WINDOW {
            let pos = base_pos + (i as f64) * step_samples;
            if pos < 0.0 {
                continue;
            }
            let fi = pos as usize;
            if fi >= total_frames {
                break;
            }
            let stereo = buf.read_stereo_frame(fi);
            let [out_l, out_r] = apply_gain_and_pan(stereo, p0.volume, p0.pan);
            mix_l[i] += out_l;
            mix_r[i] += out_r;
            has_active_audio = true;
        }
    }

    if !has_active_audio {
        return (0.0, 0.0);
    }

    let mut sum_sq_l = 0.0_f64;
    let mut sum_sq_r = 0.0_f64;
    for i in 0..METER_WINDOW {
        let l = f64::from(mix_l[i].clamp(-1.0, 1.0));
        let r = f64::from(mix_r[i].clamp(-1.0, 1.0));
        sum_sq_l += l * l;
        sum_sq_r += r * r;
    }
    let rms_l = (sum_sq_l / METER_WINDOW as f64).sqrt().clamp(0.0, 1.0) as f32;
    let rms_r = (sum_sq_r / METER_WINDOW as f64).sqrt().clamp(0.0, 1.0) as f32;
    (rms_l, rms_r)
}

impl ZoidCraftApp {
    pub(crate) fn reset_live_audio_output(&mut self) {
        self.audio_clock.pause();
        self.audio_output = None;
        self.active_audio_window = None;
        self.audio_mix_generation = self.audio_mix_generation.wrapping_add(1);
        self.pending_audio_mix = None;
        self.pending_audio_mix_target = None;
        self.audio_mix_hash = 0;
        self.audio_window_start = 0;
        self.audio_window_end = 0;
        self.audio_stream_attempted_hash = 0;
        self.audio_failure_latch.on_playback_stopped();
        self.last_audio_playing = false;
        self.last_audio_speed = 0.0;
    }

    pub(crate) fn cached_audio_mix_hash(&mut self, fps: f64, seq_len: f64) -> u64 {
        let key = (
            self.session.revision(),
            self.decoded_audio.len(),
            fps.to_bits(),
            seq_len.to_bits(),
            self.session.plugins.is_enabled("precomp-plus"),
        );
        if let Some((cached_key, hash)) = self.cached_audio_mix_hash
            && cached_key == key
        {
            return hash;
        }
        let hash = compute_audio_mix_hash(&self.session, &self.decoded_audio, fps, seq_len);
        self.cached_audio_mix_hash = Some((key, hash));
        hash
    }

    pub(crate) fn spawn_live_audio_mix_job(
        &mut self,
        mix_hash: u64,
        win_start: u32,
        win_end: u32,
        fps: f64,
        total_u32: u32,
        loop_enabled: bool,
    ) {
        let (clips, buffers, raw_assets) = collect_audio_mix_inputs(
            &self.session,
            &self.decoded_audio,
            f64::from(win_start),
            f64::from(win_end),
        );
        let loop_head_inputs = if win_start > 0 && win_end >= total_u32 && loop_enabled {
            let loop_head_end = ((2.0 * fps).round().max(1.0) as u32).min(total_u32);
            let (lh_clips, lh_buffers, lh_raw) = collect_audio_mix_inputs(
                &self.session,
                &self.decoded_audio,
                0.0,
                f64::from(loop_head_end),
            );
            Some((lh_clips, lh_buffers, lh_raw, loop_head_end))
        } else {
            None
        };

        self.audio_mix_generation = self.audio_mix_generation.wrapping_add(1);
        let mix_gen = self.audio_mix_generation;
        self.audio_stream_attempted_hash = mix_hash;
        let (tx, rx) = mpsc::channel();
        let work = move || {
            let window =
                execute_audio_mix_work(clips, buffers, raw_assets, win_start, win_end, fps);
            let loop_head = loop_head_inputs.and_then(|(lc, lb, lr, lend)| {
                execute_audio_mix_work(lc, lb, lr, 0, lend, fps)
                    .ok()
                    .flatten()
                    .map(|w| w.samples)
            });
            let _ = tx.send(AudioMixJobResult {
                mix_hash,
                win_start,
                win_end,
                window,
                loop_head,
            });
        };
        #[cfg(not(target_arch = "wasm32"))]
        let spawn_res = thread::Builder::new()
            .name("zc-audio-mix".to_owned())
            .spawn(work);
        #[cfg(target_arch = "wasm32")]
        let spawn_res: Result<(), std::io::Error> = {
            work();
            Ok(())
        };
        match spawn_res {
            Ok(_) => {
                self.pending_audio_mix = Some((mix_gen, rx));
                self.pending_audio_mix_target = Some((mix_hash, win_start, win_end));
            }
            Err(err) => {
                tracing::error!("failed to spawn live audio mix thread: {err}");
                if self.audio_failure_latch.on_open_failure() {
                    self.show_toast(format!("Audio playback failed: {err}"));
                }
            }
        }
    }

    pub(crate) fn poll_pending_audio_mix(&mut self, ctx: &egui::Context) {
        let Some((job_gen, rx)) = self.pending_audio_mix.as_ref() else {
            return;
        };
        let job_gen = *job_gen;
        match rx.try_recv() {
            Ok(result) => {
                self.pending_audio_mix = None;
                self.pending_audio_mix_target = None;
                if job_gen != self.audio_mix_generation || !self.session.playback.is_playing() {
                    return;
                }
                match result.window {
                    Ok(maybe_window) => {
                        self.audio_mix_hash = result.mix_hash;
                        self.audio_window_start = result.win_start;
                        self.audio_window_end = result.win_end;
                        self.active_audio_window = maybe_window.clone();
                        if let Some(handle) = self.audio_output.as_ref() {
                            // Seamlessly hot-swap the active audio window without tearing down
                            // or reopening the OS CPAL audio stream.
                            handle.update_window(maybe_window);
                            handle.set_loop_head(result.loop_head);
                        } else if self.audio_failure_latch.should_attempt_open() {
                            self.audio_clock
                                .seek_preserving_playback(self.session.playback.current_frame);
                            self.audio_clock.set_speed(self.session.playback.speed);
                            match self.audio_clock.start_audio_output_windowed(maybe_window) {
                                Ok(handle) => {
                                    handle.set_loop_head(result.loop_head);
                                    self.audio_output = Some(handle);
                                    self.audio_failure_latch.on_open_success();
                                }
                                Err(err) => {
                                    tracing::warn!("live audio output unavailable: {err}");
                                    if self.audio_failure_latch.on_open_failure() {
                                        self.show_toast(format!(
                                            "Audio output unavailable ({err}); playing silently"
                                        ));
                                    }
                                }
                            }
                        }
                    }
                    Err(err) => {
                        tracing::error!("failed to mix live playback audio: {err}");
                        if self.audio_failure_latch.on_open_failure() {
                            self.show_toast(format!("Audio playback failed: {err}"));
                        }
                    }
                }
            }
            Err(TryRecvError::Empty) => {
                if self.session.playback.is_playing() {
                    ctx.request_repaint_after(Duration::from_millis(8));
                }
            }
            Err(TryRecvError::Disconnected) => {
                self.pending_audio_mix = None;
                self.pending_audio_mix_target = None;
            }
        }
    }

    /// Pauses live transport and tears down the OS CPAL audio stream immediately before
    /// blocking the main thread on a synchronous native file dialog so audio never
    /// stutters or loops behind the modal dialog.
    pub(crate) fn pause_playback_for_modal_dialog(&mut self) {
        if self.session.playback.is_playing() {
            self.session.playback.set_speed(0.0);
        }
        self.reset_live_audio_output();
        self.last_audio_epoch = self.session.playback.epoch_id;
    }

    pub(crate) fn tick_playback_and_meter(&mut self, ctx: &egui::Context) {
        self.poll_pending_audio_mix(ctx);

        let now = Instant::now();
        let dt = now
            .saturating_duration_since(self.last_tick)
            .as_secs_f64()
            .clamp(0.0, 0.1);
        self.last_tick = now;

        let fps = self.session.document().fps.max(1.0);
        let seq_len = self.session.document().sequence_length.max(1.0);
        let loop_enabled = self.session.playback.looping;

        if (fps - self.last_audio_fps).abs() > 1e-6
            || (seq_len - self.last_audio_seq_len).abs() > 1e-6
        {
            self.audio_clock.set_timeline_config(fps, seq_len);
            self.last_audio_fps = fps;
            self.last_audio_seq_len = seq_len;
        }
        if loop_enabled != self.last_audio_loop {
            self.audio_clock.set_loop(loop_enabled);
            self.last_audio_loop = loop_enabled;
        }

        let is_playing = self.session.playback.is_playing();
        let speed = self.session.playback.speed;
        let epoch = self.session.playback.epoch_id;

        if !is_playing && self.last_audio_playing {
            self.audio_failure_latch.on_playback_stopped();
        }

        // Lazily build and seamlessly hot-swap bounded 60-second CPAL live audio windows off
        // the UI thread so pressing Space, editing audio during playback, or crossing window
        // boundaries on long timelines never blocks the UI thread or tears down the CPAL stream.
        if is_playing && self.audio_failure_latch.should_attempt_open() {
            let mix_hash = self.cached_audio_mix_hash(fps, seq_len);
            if mix_hash == 0 {
                if self.audio_output.is_some()
                    || self.active_audio_window.is_some()
                    || self.pending_audio_mix.is_some()
                {
                    self.reset_live_audio_output();
                }
            } else {
                let cur_f = self.session.playback.current_frame.max(0.0);
                let total_u32 = seq_len.ceil().max(1.0) as u32;
                let window_frames = (LIVE_AUDIO_WINDOW_SECS * fps).round().max(1.0) as u32;

                let needs_rebuild = live_audio_needs_rebuild(
                    self.audio_mix_hash,
                    self.audio_window_start,
                    self.audio_window_end,
                    self.audio_stream_attempted_hash,
                    self.audio_output.is_some(),
                    mix_hash,
                    cur_f,
                );
                let should_prefetch = !needs_rebuild
                    && self.audio_output.is_some()
                    && live_audio_should_prefetch(
                        self.audio_mix_hash,
                        self.audio_window_start,
                        self.audio_window_end,
                        mix_hash,
                        cur_f,
                        speed,
                        total_u32,
                    );

                if needs_rebuild || should_prefetch {
                    let pending_covers_cur = matches!(
                        self.pending_audio_mix_target,
                        Some((p_hash, p_start, p_end))
                            if p_hash == mix_hash
                                && p_end > p_start
                                && cur_f >= f64::from(p_start)
                                && cur_f < f64::from(p_end)
                    );
                    if !pending_covers_cur {
                        let (win_start, win_end) =
                            compute_live_audio_window(cur_f, speed, total_u32, window_frames);
                        self.spawn_live_audio_mix_job(
                            mix_hash,
                            win_start,
                            win_end,
                            fps,
                            total_u32,
                            loop_enabled,
                        );
                    }
                }
            }
        }

        // Propagate transport changes (play, pause, speed, user seek) into the master audio clock.
        let transport_changed = epoch != self.last_audio_epoch
            || is_playing != self.last_audio_playing
            || (speed - self.last_audio_speed).abs() > 1e-6;
        if transport_changed {
            if is_playing {
                self.audio_clock
                    .seek_preserving_playback(self.session.playback.current_frame);
                self.audio_clock.set_speed(speed);
            } else {
                // If playback paused in-place (Space / K / Pause button) on a fractional
                // frame without an explicit seek to a different target frame, snap to the
                // nearest integer frame so viewport, inspector, and eyedropper agree.
                if self.last_audio_playing
                    && (self.session.playback.current_frame - self.last_audio_frame).abs() < 1e-9
                {
                    let max_frame = (seq_len - 1.0).max(0.0);
                    self.session.playback.current_frame = self
                        .session
                        .playback
                        .current_frame
                        .round()
                        .clamp(0.0, max_frame);
                }
                self.audio_clock
                    .set_frame(self.session.playback.current_frame);
            }
            self.last_audio_epoch = epoch;
            self.last_audio_playing = is_playing;
            self.last_audio_speed = speed;
        }

        if is_playing {
            if self.audio_output.is_some() && self.audio_clock.snapshot().hardware_clock_active {
                let exact = self.audio_clock.exact_frame();
                if !self.audio_clock.is_playing() {
                    let max_frame = (seq_len - 1.0).max(0.0);
                    self.session.playback.current_frame = exact.round().clamp(0.0, max_frame);
                    self.session.playback.speed = 0.0;
                    self.last_audio_playing = false;
                    self.last_audio_speed = 0.0;
                } else {
                    self.session.playback.current_frame = exact;
                }
            } else {
                self.session.playback.advance(dt, fps, seq_len);
                if !self.session.playback.is_playing() {
                    let max_frame = (seq_len - 1.0).max(0.0);
                    self.session.playback.current_frame = self
                        .session
                        .playback
                        .current_frame
                        .round()
                        .clamp(0.0, max_frame);
                }
            }
            let (rms_l, rms_r) = self
                .active_audio_window
                .as_ref()
                .and_then(|win| {
                    sample_live_audio_meter_from_window(
                        win,
                        self.session.playback.current_frame,
                        fps,
                    )
                })
                .unwrap_or_else(|| {
                    if self.decoded_audio.is_empty() {
                        (0.0, 0.0)
                    } else {
                        sample_live_audio_meter(
                            &self.session,
                            &self.decoded_audio,
                            self.session.playback.current_frame,
                            fps,
                        )
                    }
                });
            self.workspace
                .audio_meter
                .update_stereo(rms_l, rms_r, dt as f32);
            ctx.request_repaint();
        } else if self.workspace.audio_meter.peak_hold > 0.001
            || self.workspace.audio_meter.rms > 0.001
        {
            self.workspace
                .audio_meter
                .update_stereo(0.0, 0.0, dt as f32);
            ctx.request_repaint();
        }
        self.last_audio_frame = self.session.playback.current_frame;
    }

    pub(crate) fn rebuild_audio_caches(&mut self) {
        self.decoded_audio.clear();
        self.asset_waveform_peaks.clear();
        self.timeline.audio_peaks.clear();
        self.cached_audio_mix_hash = None;
        self.last_audio_peaks_sync_rev = None;
        self.last_audio_peaks_asset_count = 0;

        // Decode embedded audio assets (including those only present in Media/Assets so placing
        // them later onto the timeline immediately has waveforms and audio playback).
        let audio_asset_keys: Vec<String> = self
            .session
            .project
            .assets
            .iter()
            .filter_map(|(k, rec)| {
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(str::to_ascii_lowercase);
                let is_audio_ext =
                    matches!(ext.as_deref(), Some("wav" | "mp3" | "ogg" | "flac" | "aac"));
                (is_audio_ext && rec.bytes.is_some()).then(|| k.clone())
            })
            .collect();
        for asset_key in audio_asset_keys {
            if let Some(rec) = self.session.project.assets.get(&asset_key)
                && let Some(bytes) = rec.bytes.as_deref()
            {
                let ext = Path::new(&rec.filename)
                    .extension()
                    .and_then(|e| e.to_str());
                if let Ok(buf) = decode_audio_bytes(bytes, ext) {
                    let peaks = WaveformPeaks::from_audio_buffer(&buf);
                    self.asset_waveform_peaks.insert(
                        asset_key.clone(),
                        AudioWaveformPeaks {
                            points_per_sec: f64::from(peaks.points_per_sec()),
                            min_buf: peaks.min_peaks().to_vec(),
                            max_buf: peaks.max_peaks().to_vec(),
                        },
                    );
                    self.decoded_audio.insert(asset_key, Arc::new(buf));
                }
            }
        }

        let all_audio_tracks = self.session.project.sequence.audio_tracks.iter().chain(
            self.session
                .project
                .compositions
                .values()
                .flat_map(|c| c.audio_tracks.iter()),
        );
        for at in all_audio_tracks {
            for ac in &at.clips {
                let Some(asset_key) = ac.media.as_deref() else {
                    continue;
                };
                if !self.decoded_audio.contains_key(asset_key) {
                    let Some(rec) = self.session.project.assets.get(asset_key) else {
                        continue;
                    };
                    let Some(bytes) = rec.bytes.as_deref() else {
                        continue;
                    };
                    let ext = Path::new(&rec.filename)
                        .extension()
                        .and_then(|e| e.to_str());
                    if let Ok(buf) = decode_audio_bytes(bytes, ext) {
                        let peaks = WaveformPeaks::from_audio_buffer(&buf);
                        self.asset_waveform_peaks.insert(
                            asset_key.to_owned(),
                            AudioWaveformPeaks {
                                points_per_sec: f64::from(peaks.points_per_sec()),
                                min_buf: peaks.min_peaks().to_vec(),
                                max_buf: peaks.max_peaks().to_vec(),
                            },
                        );
                        self.decoded_audio
                            .insert(asset_key.to_owned(), Arc::new(buf));
                    }
                }
                if let Some(ui_peaks) = self.asset_waveform_peaks.get(asset_key) {
                    self.timeline
                        .audio_peaks
                        .insert(ac.id.to_string(), ui_peaks.clone());
                }
            }
        }
    }

    /// Ensures newly split, duplicated, or undone/redone audio clips inherit waveform peaks
    /// from their shared content-addressed `asset_id` without re-decoding audio bytes.
    pub(crate) fn sync_timeline_audio_peaks(&mut self) {
        let rev = self.session.revision();
        let asset_count = self.asset_waveform_peaks.len();
        if asset_count == 0 {
            self.last_audio_peaks_sync_rev = Some(rev);
            self.last_audio_peaks_asset_count = 0;
            return;
        }
        if self.last_audio_peaks_sync_rev == Some(rev)
            && self.last_audio_peaks_asset_count == asset_count
        {
            return;
        }
        for at in &self.session.project.sequence.audio_tracks {
            for ac in &at.clips {
                let cid = ac.id.to_string();
                if !self.timeline.audio_peaks.contains_key(&cid)
                    && let Some(asset_key) = ac.media.as_deref()
                    && let Some(peaks) = self.asset_waveform_peaks.get(asset_key)
                {
                    self.timeline.audio_peaks.insert(cid, peaks.clone());
                }
            }
        }
        self.last_audio_peaks_sync_rev = Some(rev);
        self.last_audio_peaks_asset_count = asset_count;
    }
}
