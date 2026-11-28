import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { unzipSync } from "fflate";
import {
    createWebUpdateMonitor,
    demuxIsoVideo,
    evaluateWebUpdate,
    findSample,
    probeEbmlTracks,
} from "./zc_web_runtime.js";

const codecs = globalThis.__zcWebCodecs;

test("video sample selection uses presentation order and a frame's interval", () => {
    const display = [{ timestamp: 0 }, { timestamp: 40_000 }, { timestamp: 83_333 }];
    assert.equal(findSample(display, 0.039), display[0]);
    assert.equal(findSample(display, 0.04), display[1]);
    assert.equal(findSample(display, 0.083333), display[2]);
    assert.equal(findSample(display, 2), display[2]);
});

test("single PNG export returns a PNG download", async () => {
    const id = await codecs.startExport({ container: "png", totalFrames: 1 });
    const png = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
    codecs.encodeExportPng(id, png, 0);
    const blob = await codecs.finishExport(id, false);
    assert.equal(blob.type, "image/png");
    assert.deepEqual(new Uint8Array(await blob.arrayBuffer()), png);
});

test("PNG sequences are ordered and stored without ZIP compression", async () => {
    const id = await codecs.startExport({ container: "png", totalFrames: 2 });
    const first = Uint8Array.of(1, 2, 3);
    const second = Uint8Array.of(4, 5, 6);
    codecs.encodeExportPng(id, first, 0);
    codecs.encodeExportPng(id, second, 1);
    const blob = await codecs.finishExport(id, false);
    const zip = new Uint8Array(await blob.arrayBuffer());
    assert.equal(blob.type, "application/zip");
    assert.equal(new DataView(zip.buffer).getUint16(8, true), 0);
    assert.deepEqual(Object.entries(unzipSync(zip)), [
        ["frame_000001.png", first], ["frame_000002.png", second],
    ]);
});

test("cancelling an export drops its state", async () => {
    const id = await codecs.startExport({ container: "png", totalFrames: 2 });
    codecs.cancelExport(id);
    await assert.rejects(codecs.finishExport(id, false), /cancelled/);
});

test("clipboard writes have an in-memory fallback", () => {
    globalThis.__zcClipboard.writeText("clipboard sample");
    assert.equal(globalThis.__zcClipboard.readText(), "clipboard sample");
});

test("evaluateWebUpdate ignores matching, placeholder, invalid, and dismissed builds", () => {
    assert.equal(globalThis.__zcWebUpdate.evaluateWebUpdate, evaluateWebUpdate);
    assert.equal(globalThis.__zcWebUpdate.createWebUpdateMonitor, createWebUpdateMonitor);

    assert.equal(evaluateWebUpdate("abc123", { buildId: "abc123", version: "0.2.6" }), null);
    assert.equal(evaluateWebUpdate("__ZC_BUILD_ID__", { buildId: "def456", version: "0.2.7" }), null);
    assert.equal(evaluateWebUpdate("", { buildId: "def456", version: "0.2.7" }), null);
    assert.equal(evaluateWebUpdate("abc123", null), null);
    assert.equal(evaluateWebUpdate("abc123", { version: "0.2.7" }), null);
    assert.equal(evaluateWebUpdate("abc123", { buildId: "   ", version: "0.2.7" }), null);
    assert.equal(evaluateWebUpdate("abc123", { buildId: "def456", version: "0.2.7" }, "def456"), null);
});

test("evaluateWebUpdate returns update details when remote build differs", () => {
    assert.deepEqual(
        evaluateWebUpdate("abc123", { buildId: "def456", version: " 0.2.7 " }),
        { buildId: "def456", version: "0.2.7" },
    );
    assert.deepEqual(
        evaluateWebUpdate("abc123", { buildId: "def456", version: "   " }),
        { buildId: "def456", version: null },
    );
    assert.deepEqual(
        evaluateWebUpdate("abc123", { buildId: "def456" }),
        { buildId: "def456", version: null },
    );
});

test("createWebUpdateMonitor throttles checks, honors force and dismiss, and survives fetch errors", async () => {
    let clock = 1_000;
    let fetches = 0;
    let remote = { buildId: "build-2", version: "0.2.7" };
    let shouldFail = false;
    const updates = [];

    const monitor = createWebUpdateMonitor({
        currentBuildId: "build-1",
        minCheckGapMs: 60_000,
        intervalMs: 600_000,
        now: () => clock,
        async fetchVersion() {
            fetches++;
            if (shouldFail) throw new Error("network down");
            return remote;
        },
        onUpdate(update) {
            updates.push(update);
        },
    });

    const first = await monitor.checkNow();
    assert.deepEqual(first, { buildId: "build-2", version: "0.2.7" });
    assert.equal(fetches, 1);
    assert.equal(updates.length, 1);

    // Throttled within minCheckGapMs.
    clock += 30_000;
    assert.equal(await monitor.checkNow(), null);
    assert.equal(fetches, 1);

    // Force bypasses minCheckGapMs, but dismissal suppresses the same buildId.
    monitor.dismiss("build-2");
    assert.equal(await monitor.checkNow({ force: true }), null);
    assert.equal(fetches, 2);
    assert.equal(updates.length, 1);

    // A newer buildId after dismissal triggers a new notification once gap elapses.
    clock += 60_000;
    remote = { buildId: "build-3", version: "0.2.8" };
    assert.deepEqual(await monitor.checkNow(), { buildId: "build-3", version: "0.2.8" });
    assert.equal(fetches, 3);
    assert.equal(updates.length, 2);

    // Fetch errors are swallowed cleanly.
    clock += 60_000;
    shouldFail = true;
    assert.equal(await monitor.checkNow(), null);
    assert.equal(fetches, 4);
    assert.equal(updates.length, 2);

    // start/stop manages interval cleanly.
    monitor.start();
    monitor.stop();
});

test("MP4 sample index yields AVC configuration and bounded sample offsets", {
    skip: !process.env.ZOIDCRAFT_TEST_MP4,
}, async () => {
    const bytes = new Uint8Array(await readFile(process.env.ZOIDCRAFT_TEST_MP4));
    const video = demuxIsoVideo(bytes);
    assert.equal(video.width, 64);
    assert.equal(video.height, 48);
    assert.equal(video.fps, 24);
    assert.equal(video.hasAudio, true);
    assert.match(video.config.codec, /^avc1\./);
    assert.equal(video.config.description[0], 1);
    assert.equal(video.samples.length, 24);
    assert.equal(video.samples[0].key, 0);
    assert.ok(video.samples.every(sample => sample.size > 0 && sample.offset + sample.size <= bytes.byteLength));
    assert.equal(findSample(video.display, 0.5).timestamp, 500_000);
});

test("WebM track headers report video frame rate and audio presence", {
    skip: !process.env.ZOIDCRAFT_TEST_WEBM,
}, async () => {
    const bytes = new Uint8Array(await readFile(process.env.ZOIDCRAFT_TEST_WEBM));
    const info = probeEbmlTracks(bytes);
    assert.ok(Math.abs(info.fps - 24) < 0.001);
    assert.equal(info.hasAudio, true);
});

test("the browser bridge replaces a nearby cached frame after an exact seek", {
    skip: !process.env.ZOIDCRAFT_TEST_MP4,
}, async t => {
    class TestDecoder {
        static async isConfigSupported() { return { supported: true }; }
        state = "unconfigured";
        decodeQueueSize = 0;
        constructor(callbacks) { this.callbacks = callbacks; }
        configure() { this.state = "configured"; }
        decode(chunk) {
            queueMicrotask(() => this.callbacks.output({ timestamp: chunk.timestamp, close() {} }));
        }
        async flush() {}
        close() { this.state = "closed"; }
    }
    class TestChunk {
        constructor(data) { Object.assign(this, data); }
    }
    class TestCanvas {
        constructor(width, height) { Object.assign(this, { width, height }); }
        getContext() {
            return {
                clearRect() {},
                drawImage: image => { this.timestamp = image.timestamp; },
                getImageData: () => {
                    const data = new Uint8Array(this.width * this.height * 4);
                    data[0] = this.timestamp === 0 ? 0 : 255;
                    return { data };
                },
            };
        }
    }
    for (const [name, value] of Object.entries({
        VideoDecoder: TestDecoder,
        EncodedVideoChunk: TestChunk,
        OffscreenCanvas: TestCanvas,
    })) {
        globalThis[name] = value;
        t.after(() => { delete globalThis[name]; });
    }

    const mp4 = new Uint8Array(await readFile(process.env.ZOIDCRAFT_TEST_MP4));
    const hash = "test-mp4-frames";
    await codecs.importVideo(hash, "test.mp4", mp4);
    assert.equal(codecs.getVideoFrame(hash, 0, 2, 2), null);
    for (let i = 0; i < 200 && codecs.isDecoding(); i++) {
        await new Promise(resolve => setTimeout(resolve, 2));
    }
    assert.equal(codecs.consumeDirty(), true);
    assert.equal(codecs.getVideoFrame(hash, 0, 2, 2)[0], 0);
    assert.equal(codecs.getVideoFrame(hash, 0.5, 2, 2)[0], 0);
    codecs.beginFrameRequests();
    assert.equal(codecs.getVideoFrame(hash, 0.5, 2, 2), null);
    codecs.usePreparedFrames(await codecs.endFrameRequests());
    assert.equal(codecs.getVideoFrame(hash, 0.5, 2, 2)[0], 255);
    assert.equal(codecs.isExactVideoFrame(hash, 0.5, 2, 2), true);
    codecs.usePreparedFrames(null);
});
