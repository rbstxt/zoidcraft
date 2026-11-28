import { createFile, DataStream, MP4BoxBuffer } from "mp4box";
import { ArrayBufferTarget as Mp4Target, Muxer as Mp4Muxer } from "mp4-muxer";
import { ArrayBufferTarget as WebmTarget, Muxer as WebmMuxer } from "webm-muxer";
import { Zip, ZipPassThrough } from "fflate";

const yieldToBrowser = () => new Promise(resolve => setTimeout(resolve, 0));

function pickFile(accept) {
    return new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = accept;
        input.hidden = true;
        document.body.append(input);
        const cleanup = () => input.remove();
        input.addEventListener("cancel", () => {
            cleanup();
            resolve(null);
        }, { once: true });
        input.addEventListener("change", async () => {
            const file = input.files[0];
            cleanup();
            try {
                resolve(file ? { name: file.name, bytes: new Uint8Array(await file.arrayBuffer()) } : null);
            } catch (error) {
                reject(error);
            }
        }, { once: true });
        try {
            input.click();
        } catch (error) {
            cleanup();
            reject(error);
        }
    });
}

function downloadBlob(filename, bytes, mime) {
    const blob = bytes instanceof Blob ? bytes : new Blob([bytes], { type: mime });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

let clipboardText = "";
let clipboardGeneration = 0;

function cacheClipboardText(text) {
    clipboardText = text;
    clipboardGeneration++;
}

function readText() {
    const generation = clipboardGeneration;
    globalThis.navigator?.clipboard?.readText?.().then(text => {
        if (generation === clipboardGeneration) cacheClipboardText(text);
    }, () => {});
    return clipboardText;
}

function writeText(text) {
    cacheClipboardText(text);
    globalThis.navigator?.clipboard?.writeText?.(text).catch(() => {});
}

let audioContext;
const audioWorkletModules = new WeakMap();

const AUDIO_WORKLET_PROCESSOR_SOURCE = `
class ZcAudioWorkletProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.capacity = 16384;
        this.left = new Float32Array(this.capacity);
        this.right = new Float32Array(this.capacity);
        this.readPos = 0;
        this.writePos = 0;
        this.bufferedFrames = 0;
        this.stopped = false;
        this.lowWaterFrames = 2048;
        this.needPending = false;
        this.port.onmessage = event => {
            const msg = event.data;
            if (!msg) return;
            if (msg.type === "stop") {
                this.stopped = true;
                this.bufferedFrames = 0;
                this.readPos = 0;
                this.writePos = 0;
                return;
            }
            if (msg.type === "push" && msg.samples) {
                const samples = msg.samples;
                const frameCount = samples.length >>> 1;
                for (let i = 0; i < frameCount; i++) {
                    if (this.bufferedFrames >= this.capacity) {
                        this.readPos = (this.readPos + 1) % this.capacity;
                        this.bufferedFrames--;
                    }
                    this.left[this.writePos] = samples[i * 2];
                    this.right[this.writePos] = samples[i * 2 + 1];
                    this.writePos = (this.writePos + 1) % this.capacity;
                    this.bufferedFrames++;
                }
                this.needPending = false;
            }
        };
    }

    process(_inputs, outputs) {
        if (this.stopped) return false;
        const output = outputs[0];
        if (!output || output.length === 0) return true;
        const outL = output[0];
        const outR = output[1] ?? output[0];
        const frames = outL.length;
        for (let i = 0; i < frames; i++) {
            if (this.bufferedFrames > 0) {
                outL[i] = this.left[this.readPos];
                if (outR !== outL) outR[i] = this.right[this.readPos];
                this.readPos = (this.readPos + 1) % this.capacity;
                this.bufferedFrames--;
            } else {
                outL[i] = 0;
                if (outR !== outL) outR[i] = 0;
            }
        }
        if (this.bufferedFrames <= this.lowWaterFrames && !this.needPending) {
            this.needPending = true;
            this.port.postMessage({ type: "need", bufferedFrames: this.bufferedFrames });
        }
        return true;
    }
}
registerProcessor("zc-audio-worklet", ZcAudioWorkletProcessor);
`;

function audioOutputContext() {
    const Context = globalThis.AudioContext ?? globalThis.webkitAudioContext;
    if (!Context) throw new Error("Web Audio is unavailable");
    return audioContext ??= new Context({ latencyHint: "interactive" });
}

function sampleRate() {
    const sr = audioOutputContext().sampleRate;
    return Number.isFinite(sr) && sr > 0 ? Math.round(sr) : 48_000;
}

function resumeAudio() {
    if (!(globalThis.AudioContext ?? globalThis.webkitAudioContext)) return;
    const context = audioOutputContext();
    if (context.state !== "running") context.resume().catch(() => {});
}

function ensureAudioWorklet(context) {
    let promise = audioWorkletModules.get(context);
    if (promise) return promise;
    const blob = new Blob([AUDIO_WORKLET_PROCESSOR_SOURCE], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    promise = context.audioWorklet.addModule(url).finally(() => {
        URL.revokeObjectURL(url);
    });
    audioWorkletModules.set(context, promise);
    promise.catch(() => {
        if (audioWorkletModules.get(context) === promise) audioWorkletModules.delete(context);
    });
    return promise;
}

function startFallbackScheduler(context, actualSampleRate, pull) {
    const sources = new Set();
    const blockFrames = 1024;
    const blockDuration = blockFrames / actualSampleRate;
    let nextTime = context.currentTime;

    function schedule() {
        if (context.state !== "running") return;
        const now = context.currentTime;
        nextTime = Math.max(nextTime, now + 0.006);
        while (nextTime < now + 0.055) {
            const samples = pull(blockFrames, Math.max(0, nextTime - now) + (context.outputLatency || context.baseLatency || 0));
            if (samples.length !== blockFrames * 2) return;
            const buffer = context.createBuffer(2, blockFrames, actualSampleRate);
            const left = buffer.getChannelData(0);
            const right = buffer.getChannelData(1);
            for (let i = 0; i < blockFrames; i++) {
                left[i] = samples[i * 2];
                right[i] = samples[i * 2 + 1];
            }
            const source = context.createBufferSource();
            source.buffer = buffer;
            source.connect(context.destination);
            sources.add(source);
            source.onended = () => sources.delete(source);
            source.start(nextTime);
            nextTime += blockDuration;
        }
    }

    resumeAudio();
    const timer = setInterval(schedule, 12);
    return () => {
        clearInterval(timer);
        for (const source of sources) {
            source.onended = null;
            source.stop();
            source.disconnect();
        }
        sources.clear();
    };
}

function start(requestedSampleRate, pull) {
    const context = audioOutputContext();
    const actualSampleRate = Number.isFinite(context.sampleRate) && context.sampleRate > 0
        ? Math.round(context.sampleRate)
        : (requestedSampleRate || 48_000);
    const canUseWorklet = !!context.audioWorklet
        && typeof AudioWorkletNode === "function"
        && typeof Blob === "function"
        && typeof URL?.createObjectURL === "function";
    if (!canUseWorklet) {
        return startFallbackScheduler(context, actualSampleRate, pull);
    }

    let stopped = false;
    let node = null;
    let timer = null;
    let stopFallback = null;
    const blockFrames = 512;
    const targetFrames = Math.max(2048, Math.round(actualSampleRate * 0.064));
    let estimatedBufferedFrames = 0;
    let lastFeedTime = context.currentTime;

    function currentBufferedFrames() {
        const now = context.currentTime;
        const elapsed = Math.max(0, now - lastFeedTime);
        lastFeedTime = now;
        estimatedBufferedFrames = Math.max(0, estimatedBufferedFrames - elapsed * actualSampleRate);
        return estimatedBufferedFrames;
    }

    function pumpWorklet(reportedBuffered) {
        if (stopped || !node || context.state !== "running") return;
        if (typeof reportedBuffered === "number") {
            estimatedBufferedFrames = reportedBuffered;
            lastFeedTime = context.currentTime;
        } else {
            currentBufferedFrames();
        }
        const hwLatency = context.outputLatency || context.baseLatency || 0;
        let blocksSent = 0;
        while (estimatedBufferedFrames < targetFrames && blocksSent < 8) {
            const delaySecs = (estimatedBufferedFrames / actualSampleRate) + hwLatency;
            const pulled = pull(blockFrames, delaySecs);
            if (!pulled || pulled.length !== blockFrames * 2) return;
            const copy = new Float32Array(pulled);
            node.port.postMessage({ type: "push", samples: copy }, [copy.buffer]);
            estimatedBufferedFrames += blockFrames;
            blocksSent++;
        }
    }

    resumeAudio();
    ensureAudioWorklet(context).then(() => {
        if (stopped) return;
        node = new AudioWorkletNode(context, "zc-audio-worklet", {
            numberOfInputs: 0,
            numberOfOutputs: 1,
            outputChannelCount: [2],
        });
        node.port.onmessage = event => {
            if (event.data?.type === "need") {
                pumpWorklet(event.data.bufferedFrames);
            }
        };
        node.connect(context.destination);
        lastFeedTime = context.currentTime;
        pumpWorklet(0);
        timer = setInterval(() => pumpWorklet(), 12);
    }).catch(() => {
        if (stopped) return;
        stopFallback = startFallbackScheduler(context, actualSampleRate, pull);
    });

    return () => {
        stopped = true;
        if (timer !== null) {
            clearInterval(timer);
            timer = null;
        }
        if (node) {
            try { node.port.postMessage({ type: "stop" }); } catch { /* Ignore closed port. */ }
            node.port.onmessage = null;
            node.disconnect();
            node = null;
        }
        if (stopFallback) {
            stopFallback();
            stopFallback = null;
        }
    };
}

async function decodeAudio(blob) {
    try {
        const Context = globalThis.OfflineAudioContext ?? globalThis.webkitOfflineAudioContext;
        if (!Context) return null;
        const context = new Context(2, 1, 48_000);
        let audio = await context.decodeAudioData(await blob.arrayBuffer());
        if (audio.length > 32 * 1024 * 1024) return null;
        if (audio.numberOfChannels > 2) {
            const stereo = new Context(2, audio.length, audio.sampleRate);
            const source = stereo.createBufferSource();
            source.buffer = audio;
            source.connect(stereo.destination);
            source.start();
            audio = await stereo.startRendering();
        }
        const samples = new Float32Array(audio.length * 2);
        const left = audio.getChannelData(0);
        const right = audio.getChannelData(Math.min(1, audio.numberOfChannels - 1));
        for (let i = 0; i < audio.length; i++) {
            samples[i * 2] = left[i];
            samples[i * 2 + 1] = right[i];
        }
        return { samples, sampleRate: audio.sampleRate };
    } catch {
        return null;
    }
}

const videoTypes = {
    mp4: "video/mp4", m4v: "video/mp4", mov: "video/quicktime",
    webm: "video/webm", mkv: "video/x-matroska", avi: "video/x-msvideo",
};

function resolveVideoMime(ext) {
    if (ext === "mkv") {
        if (typeof document !== "undefined") {
            const probe = document.createElement("video");
            if (typeof probe.canPlayType === "function") {
                if (probe.canPlayType("video/x-matroska") !== "") return "video/x-matroska";
                if (probe.canPlayType("video/webm") !== "") return "video/webm";
            }
        }
        return "video/x-matroska";
    }
    return videoTypes[ext] ?? "";
}
const videos = new Map();
const videoImports = new Map();
const cache = new Map();
const activeVideoSources = new Map();
const cacheBudget = 128 * 1024 * 1024;
const assetCacheBudget = 48 * 1024 * 1024;
let cacheBytes = 0;
let decoding = 0;
let dirty = false;
let frameRequests = null;
let preparedFrames = null;

function sampleTimestamp(sample, origin) {
    return Math.round((sample.cts - origin) * 1_000_000 / sample.timescale);
}

function demuxIsoVideo(bytes) {
    const file = createFile(false);
    let info;
    file.onReady = ready => { info = ready; };
    const buffer = bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer : bytes.slice().buffer;
    file.appendBuffer(MP4BoxBuffer.fromArrayBuffer(buffer, 0), true);
    file.flush();
    const track = info?.videoTracks[0];
    if (!track) return null;
    const trackSamples = file.getTrackSamplesInfo(track.id);
    if (!trackSamples?.length) return null;

    const timescale = track.timescale;
    const origin = trackSamples.reduce((minimum, sample) => Math.min(minimum, sample.cts), Infinity);
    let previousKey = -1;
    const samples = trackSamples.map((sample, index) => {
        if (sample.is_sync) previousKey = index;
        return {
            index,
            key: previousKey,
            offset: sample.offset,
            size: sample.size,
            timestamp: sampleTimestamp(sample, origin),
            duration: Math.max(1, Math.round(sample.duration * 1_000_000 / timescale)),
        };
    });
    const display = [...samples].sort((a, b) => a.timestamp - b.timestamp);
    const duration = track.duration / timescale || (display.at(-1).timestamp + display.at(-1).duration) / 1_000_000;
    const entry = trackSamples[0].description;
    const configBox = entry?.avcC ?? entry?.hvcC ?? entry?.vpcC ?? entry?.av1C;
    let description;
    if (configBox) {
        const stream = new DataStream(undefined, 0, DataStream.BIG_ENDIAN);
        configBox.write(stream);
        description = new Uint8Array(stream.buffer, 8).slice();
    }
    return {
        config: {
            codec: track.codec,
            codedWidth: track.video.width,
            codedHeight: track.video.height,
            ...(description ? { description } : {}),
        },
        samples,
        display,
        width: track.video.width,
        height: track.video.height,
        fps: samples.length / duration,
        duration,
        hasAudio: info.audioTracks.length > 0,
        hasTransform: !!track.matrix
            && (track.matrix[0] !== 65536 || track.matrix[1] !== 0 || track.matrix[3] !== 0 || track.matrix[4] !== 65536),
    };
}

// TrackEntry.DefaultDuration is expressed in nanoseconds. It is optional for VFR tracks.
function probeEbmlTracks(bytes) {
    const limit = Math.min(bytes.length, 2 * 1024 * 1024);
    let fps = 0;
    let hasAudio = false;

    function vint(offset, keepMarker) {
        const first = bytes[offset];
        let bit = 128;
        let width = 1;
        while (bit && !(first & bit)) {
            bit >>= 1;
            width++;
        }
        if (!bit || offset + width > limit) throw new Error("Incomplete EBML header");
        let value = keepMarker ? first : first & (bit - 1);
        for (let i = 1; i < width; i++) value = value * 256 + bytes[offset + i];
        return [value, width];
    }

    function elements(start, end, visit) {
        for (let offset = start; offset < end;) {
            const [id, idWidth] = vint(offset, true);
            const [size, sizeWidth] = vint(offset + idWidth, false);
            const begin = offset + idWidth + sizeWidth;
            const next = Math.min(end, begin + size);
            if (visit(id, begin, next)) break;
            offset = next;
        }
    }

    const uint = (start, end) => {
        let value = 0;
        for (let i = start; i < Math.min(end, start + 8); i++) value = value * 256 + bytes[i];
        return value;
    };

    try {
        elements(0, limit, (id, start, end) => {
            if (id !== 0x18538067) return false; // Segment
            elements(start, end, (child, tracksStart, tracksEnd) => {
                if (child !== 0x1654ae6b) return false; // Tracks
                elements(tracksStart, tracksEnd, (entry, entryStart, entryEnd) => {
                    if (entry !== 0xae) return false; // TrackEntry
                    let type = 0;
                    let duration = 0;
                    elements(entryStart, entryEnd, (field, from, to) => {
                        if (field === 0x83) type = uint(from, to);
                        if (field === 0x23e383) duration = uint(from, to);
                        return false;
                    });
                    if (type === 1 && duration > 0) fps = 1_000_000_000 / duration;
                    if (type === 2) hasAudio = true;
                    return false;
                });
                return true;
            });
            return true;
        });
    } catch {
        // The video element can still play a file with no readable track header here.
    }
    return { fps, hasAudio };
}

function waitForMedia(video, event, action) {
    return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => finish(new Error(`Video timed out waiting for ${event}`)), 10_000);
        const success = () => finish();
        const fail = () => finish(new Error(video.error?.message || "Browser could not decode this video"));
        function finish(error) {
            clearTimeout(timeout);
            video.removeEventListener(event, success);
            video.removeEventListener("error", fail);
            if (error) reject(error);
            else resolve();
        }
        video.addEventListener(event, success);
        video.addEventListener("error", fail);
        try { action(); } catch (error) { finish(error); }
    });
}

async function openVideoElement(blob) {
    const video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.preload = "auto";
    const url = URL.createObjectURL(blob);
    try {
        await waitForMedia(video, "loadeddata", () => { video.src = url; });
        return { video, url };
    } catch (error) {
        video.removeAttribute("src");
        video.load();
        URL.revokeObjectURL(url);
        throw error;
    }
}

function findSample(display, seconds) {
    // Pick the frame whose presentation interval contains the requested time.
    const timestamp = Math.round(seconds * 1_000_000);
    let first = 0;
    let end = display.length;
    while (first < end) {
        const mid = (first + end) >>> 1;
        if (display[mid].timestamp <= timestamp + 1) first = mid + 1;
        else end = mid;
    }
    return display[Math.max(0, first - 1)];
}

function closeIsoDecoder(asset) {
    if (asset.session && asset.session.decoder.state !== "closed") asset.session.decoder.close();
    for (const frame of asset.session?.frames.values() ?? []) frame.close();
    asset.session = null;
}

function resetIsoDecoder(asset, key) {
    closeIsoDecoder(asset);
    const frames = new Map();
    const session = { frames, next: key, failure: null, wanted: -1, lastOutput: -Infinity, wake: null };
    const decoder = new VideoDecoder({
        output(frame) {
            frames.get(frame.timestamp)?.close();
            frames.set(frame.timestamp, frame);
            session.lastOutput = Math.max(session.lastOutput, frame.timestamp);
            while (frames.size > 6) {
                const evict = [...frames.keys()].find(timestamp => timestamp !== session.wanted);
                frames.get(evict).close();
                frames.delete(evict);
            }
            session.wake?.();
        },
        error(error) {
            session.failure = error;
            session.wake?.();
        },
    });
    session.decoder = decoder;
    decoder.configure(asset.iso.config);
    asset.session = session;
    return session;
}

async function decodeIsoFrame(asset, time) {
    const { samples, display } = asset.iso;
    const sample = findSample(display, time);
    if (sample.key < 0) throw new Error("No keyframe for requested video time");
    let session = asset.session;
    if (!session || session.failure ||
        (!session.frames.has(sample.timestamp) && (session.lastOutput >= sample.timestamp || session.next < sample.key))) {
        session = resetIsoDecoder(asset, sample.key);
    }
    session.wanted = sample.timestamp;
    const { decoder, frames } = session;
    while (!frames.has(sample.timestamp)) {
        if (session.failure) throw session.failure;
        if (session.next === samples.length) {
            await decoder.flush();
            if (frames.has(sample.timestamp)) break;
            throw new Error("VideoDecoder did not return the requested frame");
        }
        while (session.next < samples.length && decoder.decodeQueueSize < 8 && !frames.has(sample.timestamp)) {
            const input = samples[session.next++];
            const data = await asset.blob.slice(input.offset, input.offset + input.size).arrayBuffer();
            decoder.decode(new EncodedVideoChunk({
                type: input.index === input.key ? "key" : "delta",
                timestamp: input.timestamp,
                duration: input.duration,
                data,
            }));
        }
        if (decoder.decodeQueueSize >= 8 && !frames.has(sample.timestamp)) {
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => finish(new Error("VideoDecoder stalled")), 4000);
                function finish(error) {
                    clearTimeout(timeout);
                    decoder.removeEventListener("dequeue", awake);
                    session.wake = null;
                    if (error) reject(error);
                    else resolve();
                }
                function awake() { finish(); }
                session.wake = awake;
                decoder.addEventListener("dequeue", awake, { once: true });
            });
        }
    }
    return frames.get(sample.timestamp);
}

function readRgba(asset, source, width, height) {
    const canvas = asset.canvas ??= typeof OffscreenCanvas === "function"
        ? new OffscreenCanvas(width, height)
        : document.createElement("canvas");
    if (canvas.width !== width) canvas.width = width;
    if (canvas.height !== height) canvas.height = height;
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) throw new Error("Could not open the video frame canvas");
    context.clearRect(0, 0, width, height);
    context.drawImage(source, 0, 0, width, height);
    return new Uint8Array(context.getImageData(0, 0, width, height).data.buffer);
}

async function decodeRgba(asset, request) {
    const { time, width, height } = request;
    if (asset.iso) {
        try {
            return readRgba(asset, await decodeIsoFrame(asset, time), width, height);
        } catch {
            closeIsoDecoder(asset);
            asset.iso = null;
        }
    }
    asset.element ??= await openVideoElement(asset.blob);
    const video = asset.element.video;
    const target = Math.max(0, Math.min(time + 0.000002, video.duration - 0.000002));
    if (Math.abs(video.currentTime - target) > 0.00001 || video.readyState < 2) {
        await waitForMedia(video, "seeked", () => { video.currentTime = target; });
    }
    if (typeof VideoFrame === "function") {
        let frame;
        try {
            frame = new VideoFrame(video);
            return readRgba(asset, frame, width, height);
        } catch {
            // Some WebCodecs implementations only accept a subset of CanvasImageSources.
        } finally {
            frame?.close();
        }
    }
    return readRgba(asset, video, width, height);
}

function releaseVideoElement(asset) {
    if (!asset?.element) return;
    const { video, url } = asset.element;
    try { video.pause(); } catch { /* Ignore detached/uninitialized video. */ }
    video.removeAttribute("src");
    try { video.load(); } catch { /* Ignore mock video elements. */ }
    if (typeof video.remove === "function") video.remove();
    if (url && typeof URL?.revokeObjectURL === "function") URL.revokeObjectURL(url);
    asset.element = null;
}

function touchVideoSource(asset) {
    activeVideoSources.delete(asset.hash);
    activeVideoSources.set(asset.hash, asset);
    for (const [hash, source] of activeVideoSources) {
        if (activeVideoSources.size <= 8) break;
        if (source === asset || source.pending.size > 0) continue;
        closeIsoDecoder(source);
        releaseVideoElement(source);
        source.canvas = null;
        activeVideoSources.delete(hash);
    }
}

function disposeVideo(hash) {
    videoImports.delete(hash);
    activeVideoSources.delete(hash);
    const asset = videos.get(hash);
    if (asset) {
        videos.delete(hash);
        asset.nextPreview = null;
        asset.previewActive = false;
        closeIsoDecoder(asset);
        releaseVideoElement(asset);
        asset.canvas = null;
        asset.blob = null;
        asset.pending.clear();
        asset.failed.clear();
        for (const [key, item] of [...asset.cached.entries()]) {
            dropCached(key, item);
        }
    }
    const prefix = `${hash}:`;
    if (preparedFrames) {
        for (const key of [...preparedFrames.keys()]) {
            if (key.startsWith(prefix)) preparedFrames.delete(key);
        }
    }
    if (frameRequests) {
        for (const key of [...frameRequests.keys()]) {
            if (key.startsWith(prefix)) frameRequests.delete(key);
        }
    }
}

function disposeAllVideos() {
    for (const hash of [...videos.keys()]) {
        disposeVideo(hash);
    }
    videos.clear();
    videoImports.clear();
    activeVideoSources.clear();
    cache.clear();
    cacheBytes = 0;
    preparedFrames = null;
    frameRequests = null;
    dirty = false;
}

function dropCached(key, item) {
    cache.delete(key);
    item.asset.cached.delete(key);
    item.asset.cachedBytes -= item.bytes.byteLength;
    cacheBytes -= item.bytes.byteLength;
}

function rememberFrame(asset, request, bytes) {
    const item = { ...request, asset, bytes };
    cache.set(request.key, item);
    asset.cached.set(request.key, item);
    asset.cachedBytes += bytes.byteLength;
    cacheBytes += bytes.byteLength;
    while (asset.cached.size > 1 && (asset.cached.size > 6 || asset.cachedBytes > assetCacheBudget)) {
        dropCached(...asset.cached.entries().next().value);
    }
    while (cache.size > 1 && cacheBytes > cacheBudget) {
        dropCached(...cache.entries().next().value);
    }
    return bytes;
}

function requestAt(asset, time, width, height) {
    const index = Math.round(time * asset.fps);
    return {
        index, time: Math.min(index / asset.fps, Math.max(0, asset.durationSecs - 0.000002)),
        width, height, key: `${asset.hash}:${index}:${width}:${height}`,
    };
}

function exactFrame(request) {
    return preparedFrames?.get(request.key) ?? cache.get(request.key)?.bytes;
}

function decodeRequest(asset, request) {
    const cached = exactFrame(request);
    if (cached) return Promise.resolve(cached);
    if (asset.pending.has(request.key)) return asset.pending.get(request.key);
    decoding++;
    const promise = asset.queue.then(async () => {
        const bytes = await decodeRgba(asset, request);
        asset.failed.delete(request.key);
        return rememberFrame(asset, request, bytes);
    }).finally(() => {
        asset.pending.delete(request.key);
        touchVideoSource(asset);
        decoding--;
    });
    asset.pending.set(request.key, promise);
    asset.queue = promise.catch(() => {});
    return promise;
}

function schedulePreview(asset, request) {
    asset.nextPreview = request;
    if (asset.previewActive) return;
    asset.previewActive = true;
    (async () => {
        while (asset.nextPreview) {
            const next = asset.nextPreview;
            asset.nextPreview = null;
            try {
                await decodeRequest(asset, next);
                dirty = true;
            } catch {
                if (asset.failed.size >= 16) asset.failed.clear();
                asset.failed.add(next.key);
            }
        }
        asset.previewActive = false;
    })();
}

export async function importVideo(hash, filename, bytes) {
    if (videoImports.has(hash)) return videoImports.get(hash);
    const work = (async () => {
        const known = videos.get(hash);
        if (known) {
            // Audio is returned on demand and held by the Rust project, which owns its lifetime.
            const audio = known.hasAudio ? await decodeAudio(known.blob) : null;
            return videoInfo(known, audio);
        }
        const ext = filename.split(".").at(-1).toLowerCase();
        const mime = resolveVideoMime(ext);
        let blob = new Blob([bytes], { type: mime });
        let probe;
        if (["mp4", "m4v", "mov"].includes(ext)) {
            try { probe = demuxIsoVideo(bytes); } catch { /* Use the browser's media demuxer. */ }
        }
        const ebml = ["webm", "mkv"].includes(ext) ? probeEbmlTracks(bytes) : null;
        let iso;
        if (probe && !probe.hasTransform && typeof VideoDecoder === "function") {
            try {
                const result = await VideoDecoder.isConfigSupported(probe.config);
                if (result.supported) iso = probe;
            } catch { /* Use the video element. */ }
        }
        let element = null;
        if (!iso) {
            try {
                element = await openVideoElement(blob);
            } catch (error) {
                if (ext === "mkv" && mime !== "video/webm") {
                    blob = new Blob([bytes], { type: "video/webm" });
                    element = await openVideoElement(blob);
                } else {
                    throw error;
                }
            }
        }
        const video = element?.video;
        const width = video?.videoWidth || iso?.width;
        const height = video?.videoHeight || iso?.height;
        const durationSecs = video?.duration || iso?.duration;
        if (!(width > 0 && height > 0 && Number.isFinite(durationSecs) && durationSecs > 0)) {
            throw new Error("This file has no playable video track");
        }
        const fps = probe?.fps || ebml?.fps || 30;
        const audio = probe?.hasAudio === false ? null : await decodeAudio(blob);
        const asset = {
            hash, width, height, fps, durationSecs, blob, element, iso,
            hasAudio: !!audio || !!probe?.hasAudio || !!ebml?.hasAudio || !!video?.audioTracks?.length,
            cached: new Map(), cachedBytes: 0, pending: new Map(), failed: new Set(), queue: Promise.resolve(),
        };
        videos.set(hash, asset);
        if (element) touchVideoSource(asset);
        return videoInfo(asset, audio);
    })();
    videoImports.set(hash, work);
    try { return await work; } finally { videoImports.delete(hash); }
}

function videoInfo(asset, audio) {
    return {
        width: asset.width, height: asset.height, fps: asset.fps, durationSecs: asset.durationSecs,
        hasAudio: asset.hasAudio, audioSamples: audio?.samples, sampleRate: audio?.sampleRate,
    };
}

function getVideoFrame(hash, time, width, height) {
    const asset = videos.get(hash);
    if (!asset) return null;
    const request = requestAt(asset, time, width, height);
    const bytes = exactFrame(request);
    if (bytes) {
        const item = cache.get(request.key);
        if (item) {
            cache.delete(request.key);
            cache.set(request.key, item);
        }
        return bytes;
    }
    if (frameRequests) {
        frameRequests.set(request.key, [asset, request]);
        return null;
    }
    if (!asset.failed.has(request.key)) schedulePreview(asset, request);
    let closest;
    for (const item of asset.cached.values()) {
        if (item.width !== width || item.height !== height) continue;
        if (!closest || Math.abs(item.index - request.index) < Math.abs(closest.index - request.index)) {
            closest = item;
        }
    }
    return closest?.bytes ?? null;
}

function isExactVideoFrame(hash, time, width, height) {
    const asset = videos.get(hash);
    return !!asset && !!exactFrame(requestAt(asset, time, width, height));
}

function beginFrameRequests() {
    frameRequests = new Map();
    preparedFrames = null;
}

async function endFrameRequests() {
    const requests = frameRequests;
    frameRequests = null;
    const frames = new Map();
    if (!requests) return frames;
    for (const [asset, request] of requests.values()) {
        frames.set(request.key, await decodeRequest(asset, request));
    }
    return frames;
}

async function prepareVideoFrames(requests) {
    const frames = new Map();
    if (Array.isArray(requests)) {
        for (const entry of requests) {
            if (!Array.isArray(entry) || entry.length < 4) continue;
            const [hash, time, width, height] = entry;
            const asset = videos.get(hash);
            if (!asset) continue;
            const request = requestAt(asset, time, width, height);
            if (!frames.has(request.key)) {
                frames.set(request.key, await decodeRequest(asset, request));
            }
        }
    }
    preparedFrames = frames.size > 0 ? frames : null;
    return frames;
}

function beginMissedFrameCapture() {
    frameRequests = new Map();
}

function hasPendingFrameRequests() {
    return !!(frameRequests && frameRequests.size > 0);
}

async function endMissedFrameCapture() {
    const requests = frameRequests;
    frameRequests = null;
    if (!requests || requests.size === 0) return preparedFrames;
    const merged = preparedFrames ? new Map(preparedFrames) : new Map();
    for (const [asset, request] of requests.values()) {
        merged.set(request.key, await decodeRequest(asset, request));
    }
    preparedFrames = merged;
    return merged;
}

function clearPreparedFrames() {
    frameRequests = null;
    preparedFrames = null;
}

async function encodeImageBlob(rgba, width, height, mime, quality = 0.92) {
    const pixelCount = width * height;
    const expected = pixelCount * 4;
    if (!rgba || rgba.length !== expected) {
        throw new Error(`Invalid RGBA frame size: expected ${expected}, got ${rgba?.length ?? 0}`);
    }
    const clamped = new Uint8ClampedArray(expected);
    if (mime === "image/jpeg") {
        // Composite straight-alpha RGBA onto opaque black for JPEG (no alpha channel).
        for (let i = 0; i < expected; i += 4) {
            const a = rgba[i + 3];
            clamped[i] = ((rgba[i] * a + 127) / 255) | 0;
            clamped[i + 1] = ((rgba[i + 1] * a + 127) / 255) | 0;
            clamped[i + 2] = ((rgba[i + 2] * a + 127) / 255) | 0;
            clamped[i + 3] = 255;
        }
    } else {
        clamped.set(rgba);
    }
    const imageData = new ImageData(clamped, width, height);
    if (typeof document !== "undefined") {
        const canvas = document.createElement("canvas");
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Could not create 2D canvas context for image export");
        ctx.putImageData(imageData, 0, 0);
        if (typeof canvas.toBlob === "function") {
            return await new Promise((resolve, reject) => {
                canvas.toBlob(blob => {
                    if (blob) resolve(blob);
                    else reject(new Error(`Browser failed to encode ${mime}`));
                }, mime, quality);
            });
        }
        const dataUrl = canvas.toDataURL(mime, quality);
        const comma = dataUrl.indexOf(",");
        const binary = atob(dataUrl.slice(comma + 1));
        const out = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
        return new Blob([out], { type: mime });
    }
    if (typeof OffscreenCanvas === "function") {
        const canvas = new OffscreenCanvas(width, height);
        const ctx = canvas.getContext("2d");
        if (!ctx) throw new Error("Could not create OffscreenCanvas 2D context for image export");
        ctx.putImageData(imageData, 0, 0);
        return await canvas.convertToBlob({ type: mime, quality });
    }
    throw new Error("Canvas image encoding is unavailable in this environment");
}

function consumeDirty() {
    const pending = dirty;
    dirty = false;
    return pending;
}

const videoEncoderCodecs = {
    h264: ["avc1.640028", "avc1.4d002a", "avc1.42001f", "avc1.640033", "avc1.640034"],
    h265: ["hvc1.1.6.L120.B0", "hvc1.1.6.L153.B0"],
    vp8: ["vp8"],
    vp9: ["vp09.00.10.08", "vp09.00.40.08", "vp09.00.51.08"],
    av1: ["av01.0.04M.08", "av01.0.08M.08", "av01.0.13M.08"],
};
const mp4Codecs = { h264: "avc", h265: "hevc", vp9: "vp9", av1: "av1" };
const webmCodecs = { vp8: "V_VP8", vp9: "V_VP9", av1: "V_AV1" };
const audioEncoders = { aac: "mp4a.40.2", opus: "opus" };
const exportsById = new Map();
let nextExportId = 0;

function isSupported() {
    return typeof VideoEncoder !== "undefined" && typeof VideoFrame !== "undefined";
}

async function selectVideoEncoder(options) {
    const { width, height, fps, bitrateKbps, videoCodec } = options;
    const base = { width, height, bitrate: bitrateKbps * 1000, framerate: fps };
    if (videoCodec === "h264") base.avc = { format: "avc" };
    for (const latencyMode of ["quality", "realtime"]) {
        for (const hardwareAcceleration of ["prefer-hardware", "no-preference"]) {
            for (const codec of videoEncoderCodecs[videoCodec] ?? []) {
                const config = { ...base, codec, hardwareAcceleration, latencyMode };
                try {
                    if ((await VideoEncoder.isConfigSupported(config)).supported) return config;
                } catch { /* Try the next browser encoder profile. */ }
            }
        }
    }
    throw new Error(`No ${videoCodec.toUpperCase()} encoder is available at ${width}×${height}, ${fps} fps`);
}

function checkExport(state) {
    if (!state || state.cancelled) throw new Error("Export cancelled");
    if (state.error) throw state.error;
}

function exportState(id) {
    const state = exportsById.get(id);
    checkExport(state);
    return state;
}

function onEncoded(state, consume) {
    return (chunk, metadata) => {
        if (state.cancelled || state.error) return;
        try {
            consume(chunk, metadata);
        } catch (error) {
            state.error = error;
            state.wakeQueue?.();
        }
    };
}

function createEncoder(Type, state, config, output) {
    const encoder = new Type({
        output: onEncoded(state, output),
        error(error) {
            state.error = error;
            state.wakeQueue?.();
        },
    });
    try {
        encoder.configure(config);
        return encoder;
    } catch (error) {
        if (encoder.state !== "closed") encoder.close();
        throw error;
    }
}

function emitAlphaChunks(state) {
    for (const [timestamp, [chunk, metadata]] of state.colorChunks) {
        const alpha = state.alphaChunks.get(timestamp);
        if (!alpha) break;
        state.muxer.addVideoChunk(chunk, { ...metadata, zoidcraftAlpha: alpha });
        state.alphaChunks.delete(timestamp);
        state.colorChunks.delete(timestamp);
    }
}

function configureAlphaEncoder(state, config) {
    if (state.container !== "webm" || state.videoCodec !== "vp9") {
        throw new Error("Video transparency requires VP9 WebM. PNG also preserves transparency.");
    }
    state.colorChunks = new Map();
    state.alphaChunks = new Map();
    state.alpha = createEncoder(VideoEncoder, state, {
        ...config,
        hardwareAcceleration: "no-preference",
    }, chunk => {
        const bytes = new Uint8Array(chunk.byteLength);
        chunk.copyTo(bytes);
        state.alphaChunks.set(chunk.timestamp, bytes);
        emitAlphaChunks(state);
    });
}

async function configureMovie(state) {
    const webm = state.container === "webm";
    const audioOnly = state.container === "m4a";
    if (!audioOnly && !isSupported()) throw new Error("This browser does not support WebCodecs video encoding");
    const muxerCodec = (webm ? webmCodecs : mp4Codecs)[state.videoCodec];
    if (!audioOnly && !muxerCodec) throw new Error("The selected video codec does not support this container");
    const videoConfig = audioOnly ? null : await selectVideoEncoder(state);
    const withAudio = state.audioCodec !== "none" && state.audioSamples?.length > 0;
    let audioConfig;
    if (withAudio) {
        if (typeof AudioEncoder !== "function" || typeof AudioData !== "function") {
            throw new Error("This browser does not support WebCodecs audio encoding");
        }
        audioConfig = {
            codec: audioEncoders[state.audioCodec],
            sampleRate: state.sampleRate,
            numberOfChannels: 2,
            bitrate: state.audioBitrateKbps * 1000,
        };
        if (!(await AudioEncoder.isConfigSupported(audioConfig)).supported) {
            throw new Error(`No ${state.audioCodec.toUpperCase()} encoder is available at ${state.sampleRate} Hz`);
        }
    }

    state.target = webm ? new WebmTarget() : new Mp4Target();
    const tracks = {
        target: state.target,
        ...(videoConfig ? { video: {
            codec: muxerCodec,
            width: state.width, height: state.height, frameRate: state.fps,
            ...(webm ? { alpha: state.transparentAlpha } : {}),
        } } : {}),
        ...(audioConfig ? { audio: {
            codec: webm ? "A_OPUS" : state.audioCodec,
            numberOfChannels: 2,
            sampleRate: state.sampleRate,
        } } : {}),
        firstTimestampBehavior: "offset",
    };
    state.muxer = webm ? new WebmMuxer(tracks) : new Mp4Muxer({ ...tracks, fastStart: "in-memory" });
    if (videoConfig) {
        if (state.transparentAlpha) configureAlphaEncoder(state, videoConfig);
        state.video = createEncoder(VideoEncoder, state, videoConfig, (chunk, metadata) => {
            if (state.alpha) {
                state.colorChunks.set(chunk.timestamp, [chunk, metadata]);
                emitAlphaChunks(state);
            } else {
                state.muxer.addVideoChunk(chunk, metadata);
            }
        });
    }
    if (audioConfig) {
        state.audio = createEncoder(AudioEncoder, state, audioConfig, (chunk, metadata) => {
            state.muxer.addAudioChunk(chunk, metadata);
        });
    }
}

async function startExport(options) {
    const state = { ...options, audioOffset: 0, framesWritten: 0 };
    try {
        if (state.container === "png") {
            state.pngParts = [];
            if (state.totalFrames > 1) {
                if (state.totalFrames >= 65_535) throw new Error("A PNG ZIP export supports fewer than 65,535 frames");
                state.zip = new Zip((error, bytes) => {
                    if (error) state.error = error;
                    else state.pngParts.push(bytes);
                });
            }
        } else {
            await configureMovie(state);
        }
        const id = ++nextExportId;
        exportsById.set(id, state);
        return id;
    } catch (error) {
        closeExport(state);
        throw error;
    }
}

async function waitForQueue(state, encoder, limit) {
    while (encoder.encodeQueueSize >= limit) {
        checkExport(state);
        // A timer also wakes this wait after cancellation or an encoder error.
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => finish(new Error("WebCodecs encoder stalled")), 15_000);
            function finish(error) {
                clearTimeout(timeout);
                encoder.removeEventListener("dequeue", ready);
                state.wakeQueue = null;
                if (error) reject(error);
                else resolve();
            }
            function ready() { finish(); }
            state.wakeQueue = ready;
            encoder.addEventListener("dequeue", ready, { once: true });
        });
    }
    checkExport(state);
}

async function encodeAudioThrough(state, targetSeconds) {
    if (!state.audio) return;
    const frameCount = state.audioSamples.length / 2;
    const target = Math.min(frameCount, Math.ceil(targetSeconds * state.sampleRate));
    while (state.audioOffset < target) {
        await waitForQueue(state, state.audio, 8);
        const offset = state.audioOffset;
        const numberOfFrames = Math.min(frameCount - offset, state.audioCodec === "aac" ? 1024 : state.sampleRate / 50);
        const data = new AudioData({
            format: "f32", sampleRate: state.sampleRate, numberOfChannels: 2, numberOfFrames,
            timestamp: Math.round(offset * 1_000_000 / state.sampleRate),
            data: state.audioSamples.subarray(offset * 2, (offset + numberOfFrames) * 2),
        });
        try { state.audio.encode(data); } finally { data.close(); }
        state.audioOffset += numberOfFrames;
    }
}

function encodeAlphaFrame(state, rgba, init, keyFrame) {
    const pixelCount = state.width * state.height;
    const plane = new Uint8Array(pixelCount * 3 / 2);
    for (let i = 0; i < pixelCount; i++) plane[i] = rgba[4 * i + 3];
    plane.fill(128, pixelCount);
    const frame = new VideoFrame(plane, {
        ...init,
        format: "I420",
        colorSpace: { fullRange: true, matrix: "bt709", primaries: "bt709", transfer: "bt709" },
    });
    try { state.alpha.encode(frame, { keyFrame }); } finally { frame.close(); }
}

async function encodeExportFrame(id, rgba, offset) {
    const state = exportState(id);
    if (offset !== state.framesWritten) throw new Error("Export frames must arrive in order");
    await encodeAudioThrough(state, (offset + 1) / state.fps);
    await waitForQueue(state, state.video, 3);
    if (state.alpha) await waitForQueue(state, state.alpha, 3);

    const timestamp = Math.round(offset * 1_000_000 / state.fps);
    const init = {
        timestamp,
        duration: Math.round((offset + 1) * 1_000_000 / state.fps) - timestamp,
        codedWidth: state.width,
        codedHeight: state.height,
    };
    const keyFrame = offset % Math.max(1, Math.round(state.fps * 2)) === 0;
    const frame = new VideoFrame(rgba, { ...init, format: state.alpha ? "RGBA" : "RGBX" });
    try { state.video.encode(frame, { keyFrame }); } finally { frame.close(); }
    if (state.alpha) encodeAlphaFrame(state, rgba, init, keyFrame);
    state.framesWritten++;
}

function encodeExportPng(id, png, offset) {
    const state = exportState(id);
    if (offset !== state.framesWritten) throw new Error("PNG frames must arrive in order");
    if (state.zip) {
        state.pngBytes = (state.pngBytes ?? 0) + png.byteLength + 256;
        if (state.pngBytes >= 0xffffffff) throw new Error("The PNG sequence exceeds the 4 GiB ZIP limit");
        const entry = new ZipPassThrough(`frame_${String(offset + 1).padStart(6, "0")}.png`);
        state.zip.add(entry);
        entry.push(png, true);
    } else {
        state.pngParts.push(png);
    }
    state.framesWritten++;
}

function closeExport(state) {
    state.cancelled = true;
    state.wakeQueue?.();
    for (const encoder of [state.video, state.audio, state.alpha]) {
        if (encoder && encoder.state !== "closed") encoder.close();
    }
    state.zip?.terminate();
}

async function finishExport(id, download = true) {
    const state = exportState(id);
    try {
        if (state.container !== "m4a" && state.framesWritten !== state.totalFrames) {
            throw new Error("Export stopped before all frames were written");
        }
        if (state.container === "png") {
            state.zip?.end();
            checkExport(state);
            const blob = new Blob(state.pngParts, { type: state.zip ? "application/zip" : "image/png" });
            if (download) downloadBlob(state.filename, blob);
            return blob;
        }
        await encodeAudioThrough(state, Infinity);
        await Promise.all([state.video, state.audio, state.alpha].filter(Boolean).map(encoder => encoder.flush()));
        checkExport(state);
        if (state.colorChunks?.size || state.alphaChunks?.size) throw new Error("VP9 alpha frames were lost");
        state.muxer.finalize();
        const mime = { m4a: "audio/mp4", webm: "video/webm", mp4: "video/mp4" }[state.container];
        const blob = new Blob([state.target.buffer], { type: mime });
        if (download) downloadBlob(state.filename, blob);
        return blob;
    } finally {
        exportsById.delete(id);
        closeExport(state);
    }
}

function cancelExport(id) {
    const state = exportsById.get(id);
    exportsById.delete(id);
    if (state) closeExport(state);
}

function evaluateWebUpdate(currentBuildId, payload, dismissedBuildId = null) {
    if (typeof currentBuildId !== "string" || !currentBuildId.trim() || currentBuildId === "__ZC_BUILD_ID__") {
        return null;
    }
    if (!payload || typeof payload !== "object") return null;
    const remoteBuildId = typeof payload.buildId === "string" ? payload.buildId.trim() : "";
    if (!remoteBuildId || remoteBuildId === currentBuildId || remoteBuildId === dismissedBuildId) {
        return null;
    }
    const version = typeof payload.version === "string" && payload.version.trim()
        ? payload.version.trim()
        : null;
    return { buildId: remoteBuildId, version };
}

function createWebUpdateMonitor({
    currentBuildId,
    fetchVersion,
    onUpdate,
    intervalMs = 10 * 60 * 1000,
    minCheckGapMs = 60 * 1000,
    now = () => Date.now(),
}) {
    let dismissedBuildId = null;
    let lastCheckAt = null;
    let inFlight = false;
    let timerId = null;

    async function checkNow({ force = false } = {}) {
        if (inFlight) return null;
        if (typeof currentBuildId !== "string" || !currentBuildId.trim() || currentBuildId === "__ZC_BUILD_ID__") {
            return null;
        }
        const currentTime = now();
        if (!force && lastCheckAt !== null && (currentTime - lastCheckAt) < minCheckGapMs) {
            return null;
        }
        inFlight = true;
        lastCheckAt = currentTime;
        try {
            const payload = await fetchVersion();
            const update = evaluateWebUpdate(currentBuildId, payload, dismissedBuildId);
            if (update) onUpdate?.(update);
            return update;
        } catch {
            return null;
        } finally {
            inFlight = false;
        }
    }

    function dismiss(buildId) {
        dismissedBuildId = typeof buildId === "string" && buildId.trim() ? buildId.trim() : null;
    }

    function start() {
        if (timerId !== null) return;
        timerId = setInterval(() => { void checkNow(); }, intervalMs);
        timerId?.unref?.();
    }

    function stop() {
        if (timerId === null) return;
        clearInterval(timerId);
        timerId = null;
    }

    return { checkNow, dismiss, start, stop };
}

globalThis.__zcFiles = { pickFile, downloadBlob, yieldToBrowser };
globalThis.__zcClipboard = { readText, writeText };
globalThis.__zcWebAudio = { start, sampleRate };
globalThis.__zcWebUpdate = { evaluateWebUpdate, createWebUpdateMonitor };
globalThis.__zcWebCodecs = {
    isSupported, importVideo, disposeVideo, disposeAllVideos, getVideoFrame, isExactVideoFrame,
    beginFrameRequests, endFrameRequests, usePreparedFrames: frames => { preparedFrames = frames; },
    prepareVideoFrames, beginMissedFrameCapture, hasPendingFrameRequests, endMissedFrameCapture,
    clearPreparedFrames, encodeImageBlob,
    consumeDirty, isDecoding: () => decoding > 0,
    startExport, encodeExportPng, encodeExportFrame, finishExport, cancelExport,
};

if (typeof document !== "undefined") {
    document.addEventListener("paste", event => {
        const text = event.clipboardData?.getData("text/plain");
        if (typeof text === "string") cacheClipboardText(text);
    }, { capture: true });
    document.addEventListener("pointerdown", resumeAudio, { capture: true });
    document.addEventListener("keydown", resumeAudio, { capture: true });
}

export {
    createWebUpdateMonitor,
    demuxIsoVideo,
    disposeAllVideos,
    disposeVideo,
    evaluateWebUpdate,
    findSample,
    probeEbmlTracks,
    resolveVideoMime,
};
