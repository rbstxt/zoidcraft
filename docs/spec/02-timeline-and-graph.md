# 02 — Timeline, Playback Clock, Keyframes, Easing, and Graph Editor Specification

**Status:** Clean-Room Behavioral Specification  
**Scope:** Panzoid ClipMaker 3 (CM3) Timeline, Keyframe Dope Sheet, Transport Toolbar, Keyframes & Easing Mathematics, Vanilla Graph Editor, Zoidium Graph Editor 2 (`graph-plus`), Zoidium Easing+ (`easing-plus`), and Playback / Audio Synchronization Engine.

This document specifies exact pixel geometry, colors, hitboxes, mathematical formulas, state machines, and interaction rules required for 1:1 behavioral parity in ZoidCraft while eliminating known CM3 playback/audio synchronization defects.

---

## 1. Timeline Visual Anatomy (Pixel-Exact)

### 1.1 Panel Frame & Global Typography

| Property | Value |
| --- | --- |
| Panel background | `#2a2a2a` |
| Panel border (unfocused) | `1px solid #333333` |
| Panel border (focused / focus-within) | `1px solid #384668` |
| Font family | `'Source Code Pro', monospace` |
| Duplicatable panel | Yes (multiple timeline panels may observe the same sequence) |
| Left column width (timestamp + track/keyframe headers) | `200px` (fixed) |

Vertical stacking from top to bottom inside the timeline panel:

```text
+-----------------------+-----------------------------------------------+---+
| Timestamp Box         | Time Ruler (ticks, labels, markers, sel bar)  |   | 35px
| 200px × 35px (#1d1d1d)| left: 200px, right: S, height: 35px (#242424) |   |
+-----------------------+-----------------------------------------------+---+
| Keyframe Labels       | Keyframe Dope-Sheet Lanes                     | V | default 60px
| 200px (scroll-synced) | left: 200px, rows 20px, overflow-y: scroll    | S | (min 20px, step 20px)
+-----------------------+-----------------------------------------------+---+
| Keyframe / Tracks Horizontal Splitter (100% × 3px, #222, ns-resize)       | 3px (default top: 95px)
+-----------------------+-----------------------------------------------+---+
| Video Track Labels    | Video Track Lanes (V_n top ... V_1 bottom)    | V | default 50% - 2px
| 200px (#2b2b2b)       | left: 200px, background #272727               | S |
+-----------------------+-----------------------------------------------+---+
| Video / Audio Splitter (100% × 4px, top/bottom 1px #212121, ns-resize)    | 4px
+-----------------------+-----------------------------------------------+---+
| Audio Track Labels    | Audio Track Lanes (A_1 top ... A_n bottom)    | V | remainder
| 200px (#2b2b2b)       | + Waveform Canvas Overlay (pointer-events:no) | S |
+-----------------------+-----------------------------------------------+---+
| (empty 200px corner)  | Horizontal Scrollbar (height: S, right: S)    |   | S (native scrollbar thickness)
+-----------------------+-----------------------------------------------+---+
   Overlay across [left: 200px, top: 0, right: S, bottom: S]:
   - Playhead vertical stem (1px wide, full height, #b71e1e)
   - Keyframe drag snap guide (1px wide, full height, #a6a6a6 @ 50% opacity)
```

*(Where `S` is the scrollbar gutter thickness, initialized to `20px` and calibrated dynamically to scrollbar width/height.)*

### 1.2 Transport & Action Toolbar Panel

In CM3, transport and global editor controls live in a dedicated toolbar panel separate from the timeline panel (though ZoidCraft may dock it directly above the timeline).

| Element | Dimensions / Styling |
| --- | --- |
| Container | `background: #242424`, `padding: 5px`, centered horizontal flow, `min-width: 40px`, `min-height: 40px` |
| Toolbar button | `height: 29px`, `border: 2px solid transparent` (`#384668` on focus), `margin-right: 5px`, `padding: 0`, transparent background |
| Button SVG icon | `25px × 25px`, fill `#acacac` (`rgb(172, 172, 172)`); active toggle state fill `#a92424` |
| Group separator | `15px × 25px` blank spacer |
| Standard buttons & actions | Play / Pause (`Space`), Step Back / Step Forward, Jump to Start / End, Loop toggle, Prev / Next Marker, Split Clip (`C`), Graph Editor toggle (`Ctrl+G`), Undo (`Ctrl+Z`), Redo (`Ctrl+Y`) |

**Toolbar Shortcut Dispatch Rule:**
- Keydowns are ignored whenever the active focused target is an `input`, `select`, `textarea`, or `contentEditable` element, or when the target is a `button` and the key is `Enter` or `Space`.
- Modifiers are matched strictly via a 3-bit mask: `(Shift << 2) | (Ctrl << 1) | Alt`.

### 1.3 Top-Left Timestamp Display

| Property | Value |
| --- | --- |
| Position & size | `top: 0`, `left: 0`, `width: 200px`, `height: 35px` |
| Background | `#1d1d1d` |
| Text style | `#cccccc`, `line-height: 30px`, `padding-left: 10px` |
| Format cycle | Right-clicking the box advances `timeFormat = (timeFormat + 1) % 3` |

| `timeFormat` | Format | Formula for time `t` (seconds) at sequence rate `fps` |
| --- | --- | --- |
| `0` (default) | `HH:MM:SS.mmm` | `H = floor(t / 3600)`, `M = floor((t / 60) % 60)`, `S = (t % 60)` formatted to 3 decimals, zero-padded to width 6 (`00.000`) |
| `1` | `HH:MM:SS:FF` | `H = floor(t / 3600)`, `M = floor((t / 60) % 60)`, `S = floor(t % 60)`, `F = round(t * fps) % fps`, each 2-digit zero-padded |
| `2` | Frame number | `round(t * fps)` (integer string) |

### 1.4 Time Ruler, Ticks, Markers, Selection Shadow & Playhead

| Element | Visual Specification |
| --- | --- |
| Ruler container | `left: 200px`, `top: 0`, `right: S`, `height: 35px`, `background: #242424` (`rgb(36, 36, 36)`), `overflow: hidden` |
| Major/minor tick label slot | `height: 100%` (`35px`), `font-size: 14px`, `color: #656565`, `padding-left: 5px`, `border-left: 1px solid #555555` (major tick), `border-right: 1px solid #3a3a3a` (midpoint minor tick) |
| Selection shadow bar | `bottom: 0`, `height: 2px`, `background: #526183`, `opacity: 0.8`. Visible only when `>= 1` clip is selected. Spans `[min(c.start), max(c.start + c.length)]` |
| Sequence markers | `top: 50%`, `height: 50%` (`17.5px`), `border-left: 1px dashed rgb(112, 194, 157)` (`#70c29d`), `padding-left: 5px`, `color: rgb(112, 194, 157)`, `font-size: 14px`, non-interactive (`pointer-events: none`), text = `marker.value` |
| Playhead stem | `width: 1px`, `height: 100%` (full panel height from `y = 0` to `bottom - S`), `background: rgb(183, 30, 30)` (`#b71e1e`), positioned at `x = frameToPx(currentFrame) - scrollLeft` |
| Sequence duration boundary | Dynamic: `sequence.length = max(clip.start + clip.length)` over the final clip on every video and audio track (`0` if all tracks are empty). Scrollbar spacer width = `frameToPx(sequence.length)`. Track lanes span `max(viewportWidth, frameToPx(sequence.length))`. |

**Exact Ruler Tick Interval Algorithm:**
Let `timeUnits = [1, 5, 15, 300, 1800]` (in seconds: 1s, 5s, 15s, 5m, 30m). Given current `zoom` (where `zoom = 1.0` equals `400 px/sec`):

```text
invZoom  = 1 / zoom
unitIdx  = clamp(floor(log10(0.5 * invZoom)) + 1, 0, 4)
baseUnit = timeUnits[unitIdx]
stepSec  = baseUnit * 2^(floor(log2(invZoom / baseUnit)))
majorPx  = 400 * zoom * stepSec
minorPx  = 0.5 * majorPx
```

- Pool size of visible label slots: `N = ceil(rulerClientWidth / 200)`.
- Anchor offset: `offsetPx = -(scrollLeft mod majorPx)`.
- For slot `k in [0, N - 1]`:
  - `leftPx = offsetPx + k * majorPx`
  - `widthPx = 0.5 * majorPx` (left border draws major tick; right border draws midpoint minor tick at `+0.5 * stepSec`)
  - `labelTimeSec = (k + floor(scrollLeft / majorPx)) * stepSec`, formatted with the active `timeFormat`.

### 1.5 Keyframe Dope-Sheet Area

| Property | Value |
| --- | --- |
| Default geometry | `top: 35px`, `left: 0`, `right: 0`, `height: 60px` (3 rows) |
| Bottom resize bar | `height: 3px`, `background: #222222`, `cursor: ns-resize`, positioned at `top = keyframesHeight + 35px` |
| Resize quantization | Height snaps to `20 * max(round(dragHeight / 20), 1)` (minimum `20px`, multiples of `20px`); tracks area top shifts to `keyframesHeight + 38px` |
| Row height | `20px` (both label and keyframe lane) |
| Label styling | `width: 200px`, `line-height: 20px`, `color: #848484`, `font-size: 15px`, `padding: 0 5px`, non-selectable, tooltip = full name |
| Alternating lane colors | Visible rows alternate `#343434` (even) and `#3f3f3f` (odd) |
| Alternating label colors | Visible rows alternate `#2e2e2e` (even) and `#3b3b3b` (odd) |
| Visibility rule | Keyframe rows with zero keyframes are hidden (`display: none`) and skipped in alternation parity |
| Keyframe diamond | `12px × 12px` square rotated `45°`, `margin-left: -6px` (horizontally centered at `frameToPx(absFrame)`), `margin-top: 4px` (vertically centered in `20px` row), `cursor: move` |
| Keyframe diamond color | Deterministic hue `H = (40 * stringHash(trackName)) mod 360`. Unselected fill `hsl(H, 40%, 45%)`, border `1px solid hsl(H, 40%, 60%)`. Selected fill overridden to `#292929` (retaining colored border). |

### 1.6 Left Track Header Column & Video/Audio Split

| Property | Value |
| --- | --- |
| Column width | `200px` |
| Default row heights | `videoTrackSize = 40px`, `audioTrackSize = 40px` |
| Track height resize | `Ctrl + Wheel` over video labels adjusts `videoTrackSize` by `±5px`, clamped to `[20px, 100px]`; same over audio labels for `audioTrackSize` |
| Video / Audio divider | `height: 4px`, `cursor: ns-resize`, `border-top: 1px solid #212121`, `border-bottom: 1px solid #212121` |
| Initial divider split | `splitY = tracksClientHeight * (videoTrackSize / (videoTrackSize + audioTrackSize)) - 2` (50% minus 2px at default 40/40) |
| Divider drag bounds | `splitY` clamped to `[0, tracksClientHeight - 4]`; audio section starts at `splitY + 4px` |
| **Video track ordering** | **Bottom-up** from the divider: `V1` is immediately above the divider (lowest compositor z-order), `V2` above `V1`, ..., `Vn` at top. Vertical scroll preserves `scrollBottom` anchor (`scrollTop = round((totalHeight - boundHeight - scrollBottom) * dpr) / dpr`). |
| **Audio track ordering** | **Top-down** from the divider: `A1` at `top = 0`, `A2` at `top = audioTrackSize`, etc. |
| Header cell styling | `background: rgb(43, 43, 43)` (`#2b2b2b`), `color: #848484`, `font-size: 15px`, `padding: 0 5px`, `border-right: 1px solid #212121`, grid layout `1fr auto auto` (or additional columns for extension toggles), vertically centered |
| Row separator border | Video tracks: `border-top: 1px solid #212121`. Audio tracks: `border-bottom: 1px solid #212121`. |
| Header controls (Video) | 1) Label `"V1"`, `"V2"`, ... 2) Visibility/Enable eye icon (`16px × 16px`, fill `#cccccc` when enabled, `#8a2828` when disabled) 3) Delete track icon (`16px × 16px`, fill `#cccccc`). *(Zoidium extensions add: Camera+ **3D** toggle button for `track3d` shared-camera tracking, plus Lock / Mute / Solo switches.)* |
| Header controls (Audio) | 1) Label `"A1"`, `"A2"`, ... 2) Delete track icon (`16px × 16px`, fill `#cccccc`). *(Zoidium Track Switches extension adds Mute / Solo / Lock toggles.)* |
| Track lane background | Solid `rgb(39, 39, 39)` (`#272727`) with matching `1px solid #212121` top (video) or bottom (audio) separator border. No vertical frame grid lines inside vanilla CM3 lanes. |

### 1.7 Clip Visual Design & Audio Waveform Rendering

| Element | Visual Specification |
| --- | --- |
| Clip box | `height: 100%` of track lane, `border-radius: 0`, `overflow: hidden`, `cursor: default`, `box-sizing: border-box` |
| Clip fill (unselected) | `#526183` (same steel-blue base for video and audio clips in vanilla CM3) |
| Clip fill (selected) | `#7f94c7` |
| Clip borders | `border-top: 0`, `border-bottom: 0`, `border-right: 1px solid #232323`, `border-left: 1px solid #838383` (subtle bright left edge distinguishes adjacent clips) |
| Clip label | `position: absolute`, `left: 3px`, `top: 0px`, `color: #2c323f`, `white-space: nowrap` (hard-clipped by clip box; full text in tooltip) |
| Inline rename input | Clip gains `#cccccc` background, `1px solid #868686` border; `16px` black monospace input fills clip width |
| Move drag proxy | `opacity: 0.5`, `border: 1px solid #000000`, `z-index: 9999` |
| Trim drag proxy | `opacity: 0.5`, `border: 2px solid #000000`, `cursor: ew-resize`, `padding: 0`, `z-index: 9999` |
| Trim edge handles | Invisible overlays at `left: 0` and `right: 0`, `height: 100%`, `width: min(50px, 0.2 * clipWidthPx)`, `cursor: ew-resize`. When selected, displays a `3px` solid `#982b2b` vertical bar along the corresponding left or right edge. |

**Audio Waveform Rendering Specification:**
- Rendered onto a single 2D canvas positioned over `audioContainer` (`top = splitY + 4px`, `width = audioContainer.clientWidth`, `height = audioContainer.clientHeight`, `pointer-events: none`).
- **Peak Analysis Cache:** Keyed per audio asset. Audio is decoded into `pointsPerSec = 200` buckets per second (`samplesPerBucket = floor(sampleRate / 200)`). For each bucket, channels are averaged (`sum(ch[j]) / numChannels`), recording `minBuf[k] = Int8(127 * minAvg)` and `maxBuf[k] = Int8(127 * maxAvg)` (signed 8-bit arrays).
- **Canvas Draw Algorithm:**
  1. Clear canvas; set fill style `#242d42`.
  2. Translate by `(-scrollLeft, -audioContainer.scrollTop)`.
  3. For each audio track `i`: save transform, translate `(0, trackTop + 1)`, scale `(1, (trackHeight - 2) / 256)`.
  4. For each non-proxy clip in track `i`:
     - Visible local X start: `xStart = max(scrollLeft - clipLeftPx, 0) + 1`
     - Visible pixel span: `xSpan = min(canvasWidth, clipWidthPx - xStart) - 2`; skip if `xSpan < 1`.
     - Media time range: `t0 = clip.time.get(0)`, `t1 = clip.time.get(clip.length)`.
     - Starting peak index: `baseIdx = floor(t0 * 200)`.
     - Peaks per pixel: `step = floor((t1 - t0) * 200) / (clipWidthPx - 2)`.
     - For each integer pixel column `x` in `[xStart, xStart + xSpan)`:
       - `idx = floor(step * x) + baseIdx`
       - Scan `p in [0, step)` to find `hi = max(0, maxBuf[idx + p])` and `lo = min(0, minBuf[idx + p])`.
       - Amplitude bar height `d = |hi - lo|` (range `0..255`).
       - Draw bottom-aligned 1px bar: `fillRect(x, 256 - d, 1, d)` (scaled into track height).

---

## 2. Timeline Dynamic & Invisible Interaction Mechanics

### 2.1 Coordinate Mapping, Zooming & Scrolling

Let `fps = sequence.properties.rate` and `zoom` be a positive float (default `1.0`).

```text
frameToPx(frame) = (frame / fps) * 400 * zoom
pxToFrame(px)    = px * fps / (400 * zoom)
pxPerFrame       = (400 * zoom) / fps
```

- **Zoom Step Factor:** `Z = 1.03125` (`33 / 32`, i.e., `3.125%` per step).
- **Anchor-Preserving Zoom Formula (`updateZoom(newZoom, anchorViewportX)`):**
  ```text
  newScrollLeft = (oldScrollLeft + anchorViewportX) * (newZoom / oldZoom) - anchorViewportX
  ```
  Recomputes scroll spacer width `frameToPx(totalFrames)`, selection shadow geometry, all keyframe positions, and all clip positions/handle widths.

| Input / Shortcut | Scope | Exact Behavior |
| --- | --- | --- |
| `Ctrl + Wheel Up / Down` | Anywhere in timeline (except track headers) | Zoom in (`zoom * 1.03125`) / out (`zoom / 1.03125`) anchored at mouse pointer X (`pageX - scrollBarRect.left`) |
| `=` / `-` | Timeline focused | Zoom in (`* 1.03125`) / out (`/ 1.03125`) anchored at **playhead X** (`frameToPx(currentFrame) - scrollLeft`) |
| `\` | Timeline focused | Fit entire sequence to viewport: `zoom = (scrollBar.clientWidth * fps) / (400 * totalFrames)`, `scrollLeft = 0` (no-op if `totalFrames == 0`) |
| `Shift + Wheel` or `Wheel deltaX` | Timeline | Horizontal scroll: `scrollLeft += (shiftKey ? deltaY : deltaX)` |
| `Wheel` (vertical, no Ctrl/Shift) | Video or audio lanes/labels | Vertical scroll of hovered pane: `scrollTop += sign(deltaY) * 0.5 * trackSize` |
| `Ctrl + Wheel` | Video labels / Audio labels | Resize video or audio track row height by `±5px`, clamped to `[20px, 100px]` |

*(Note: ZoidCraft should also clamp `zoom` to reasonable min/max bounds such as `[1e-4, 500]` to prevent floating-point degeneration while preserving unbounded feel for normal sequences.)*

### 2.2 Playhead Scrubbing & Auto-Scrolling

- **Triggers:**
  1. Left-click / drag anywhere on the `35px` Time Ruler (`initialMove = true`, `finalUpdate = true`).
  2. Right-click (`button === 2`) on empty track lane area or empty keyframe dope-sheet area (immediate single-shot jump; does not start marquee).
- **Scrub State Machine (`setFrame(contentX)`):**
  1. Run edge auto-scroll `keepInView(contentX)`:
     - Let `relX = contentX - scrollLeft`, `W = scrollBar.clientWidth`.
     - If `relX < 50`: `delta = relX - 50` (negative).
     - Else if `relX > W - 50`: `delta = relX - W + 50` (positive).
     - Else `delta = 0`.
     - Advance `scrollLeft += delta / 5` (evaluated each animation frame while dragging).
  2. Pause playback immediately: `playback.speed = 0`.
  3. Set `playback.currentFrame = clamp(round(pxToFrame(contentX)), 0, max(totalFrames - 1, 0))`.
- **Playback Follow Auto-Scroll (`scrollToShow(playheadPx)`):**
  - Runs on every UI frame while `playback.speed !== 0`.
  - If `playheadPx < scrollLeft` or `playheadPx > scrollLeft + W`:
    - Page-flip jump: `scrollLeft = floor(playheadPx / W) * W`.

### 2.3 Clip Selection & Clip Linking

Clips can belong to a **Link Group** (`clip.link`, shared integer key in `sequence.clipLinks`, typically linking a video clip with its paired audio clip).

| Action | Behavior |
| --- | --- |
| Left-click unselected clip | Clears handle selection and previous clip selection; selects clip **plus all clips in its link group** (unless `Alt` is held, which selects only the clicked clip). |
| `Shift + Click` unselected clip | Adds clip (and its link group unless `Alt` held) to current selection. |
| `Shift + Click` selected clip (no drag movement) | Deselects clip (and its link group unless `Alt` held) on mouse-up. |
| `Ctrl + Mousedown` on clip | Ignored (prevents clip drag start). |
| Left-click empty lane | Clears clip selection and handle selection (unless `Shift` held). |
| Left-drag empty lane (marquee) | Activates when Manhattan distance `|dx| + |dy| > 2px`. Draws `1px dashed rgb(151, 151, 151)` rectangle. Selects all video & audio clips intersecting the marquee rect (expanding link groups unless `Alt` held). Holding `Shift` preserves prior selection (`addOnly`). |
| `L` key | Links all currently selected clips (`>= 2`) under a fresh shared link key (unlinking any prior group memberships), in one undo operation. |
| `U` key | Unlinks all currently selected clips; if a link group drops to 1 remaining clip, automatically dissolves that singleton link. |

### 2.4 Snapping Specification

| Context | Threshold | Modifier to Disable | Priority / Tie-Break | Snap Targets |
| --- | --- | --- | --- | --- |
| Clip move & clip trim handles | `< 15px` (screen px) | Hold `Shift` | **Nearest target** wins | Frame `0`, Playhead `currentFrame`, all Sequence Markers, every Clip `start` and `start + length` across all video & audio tracks |
| Timeline keyframe drag | `< 10px` (screen px) | Hold `Shift` | **First match in priority order** | 1) Playhead `currentFrame`, 2) Sequence Markers (`dist > 0`), 3) Any unselected keyframe `.kfobject` across all visible keyframe tracks |
| Graph Editor 2 keyframe drag | `< 8px` (`8 * zoomX` frames) | Toolbar `Snap` toggle (`S`) | **Nearest target** wins | Playhead `currentFrame`, any non-dragged keyframe (`dist > 0`) on loaded curves |

### 2.5 Clip Dragging (Move, Clone, Track Transfer & Collision Overwrite)

**Drag Lifecycle State Machine:**
1. **`IDLE -> ARMED` (mousedown on clip):**
   - Resolve selection per Section 2.3.
   - Spawn one semi-transparent `.clip.proxy` per selected clip and record `minFrame = min(c.start)`, `minTrackOffset`, `maxTrackOffset` relative to the anchor clip's track index.
   - Append one temporary empty `40px` track proxy at the outer end of `videoContainer` (above top video track) and/or `audioContainer` (below bottom audio track) for every track containing selected clips — enabling drag-to-create-new-track!
2. **`ARMED -> MOVING` (first frame where `deltaFrames != 0` or track index changes):**
   - Reveal proxies.
   - Compute horizontal delta:
     ```text
     targetPx    = (pointerX + scrollLeft) + anchorGrabOffset
     snapped     = snap(targetPx, !shiftKey)
     deltaFrames = max(snapped - anchorOrigStart, -minFrame)
     ```
   - Compute hovered `videoTrackIndex` (scanning `videoContainer` bottom-up) and `audioTrackIndex` (scanning `audioContainer` top-down), clamping each to `[-minTrackOffset, containerCount - maxTrackOffset - 1]`.
   - Video clips stay within `videoContainer`; audio clips stay within `audioContainer` (no cross-type transfer).
   - Edge auto-scroll via `keepInView(pointerContentX)`.
3. **`MOVING -> COMMIT` (mouseup):**
   - Begin single undo operation (`history.startOperation()`).
   - Materialize any temporary track proxy that received `>= 1` clip proxy via `createTrack`.
   - If `Alt` is held (or dragging new media from Media panel): **clone** clips at proxy positions (remapping internal clip links to new keys) instead of moving originals.
   - Execute **Non-Overlapping Overwrite Resolution (`moveClipProxies`)**:
     - Step A: Insert/move all dragged clips into destination tracks in sorted `start` order (iterating in reverse order when `deltaFrames > 0` so earlier mutations do not shift later indices).
     - Step B: For each placed clip occupying `[a, a + L)` at index `b` on its destination track:
       1. **Check left neighbor `prev = clips[b - 1]`:** If `prev` exists and `prev.start + prev.length > a`:
          - **Straddle split case** (`prev.start + prev.length > a + L`): Create a new right-hand remainder clip at index `b + 1` cloned from `prev` with `start = a + L`, `length = (prev.start + prev.length) - (a + L)`, shifting its keyframes by `-(a + L - prev.start)` and updating its non-animated `timeStart` to `prev.time.get(a + L - prev.start)` *(Note: vanilla CM3 passes positive `a + o - y.start` in `moveClipProxies` which is a known sign-inversion bug relative to `splitClips`'s `-l`; ZoidCraft MUST shift keyframes by `-(a + L - prev.start)` so the split tail stays aligned in sequence time)*.
          - If `prev.start < a`: Right-trim `prev` to `length = a - prev.start` (and if `!prev.time.animated`, set `timeEnd = prev.time.get(a - prev.start)`).
          - Else (`prev.start == a`): Unlink and delete `prev`.
       2. **Check right neighbors `next = clips[b + 1, ...]`:**
          - If completely engulfed (`next.start + next.length <= a + L`): Unlink and delete `next`.
          - Else if partially overlapped (`next.start < a + L`): Left-trim `next` to `start = a + L`, `length = (next.start + next.length) - (a + L)`, shifting keyframes by `-(a + L - next.start)` and advancing non-animated `timeStart` by `(a + L - next.start) / fps`.
          - Else: Stop scanning.
   - Recompute `sequence.length` and rebuild media playback schedules (`analyzeSequence`).
   - End undo operation (`history.finishOperation()`).

### 2.6 Clip Trimming & Rate-Stretching (Left & Right Edge Handles)

- **Hitbox:** Invisible `.handle` zones on both left and right edges of every clip.
  ```text
  handleWidthPx = min(50, 0.2 * clipWidthPx)
  cursor        = ew-resize
  ```
- **Multi-Handle Selection:** Clicking a handle selects it (plus matching left/right handles of linked clips unless `Alt` held). `Shift + Click` adds/toggles handles across clips (including both handles on one clip).
- **Mode Selection (locked at drag start):**
  - **Standard Trim (`!ctrlKey && !clip.properties.time.animated`)**: `adjustTime = true`, `rateStretch = false`.
  - **Rate Stretch (`ctrlKey` held at drag start)**: `adjustTime = false`, `rateStretch = true`.
  - **Animated-Time Trim (`!ctrlKey && clip.properties.time.animated`)**: `adjustTime = false`, `rateStretch = false`.
- **Delta Calculation & Clamping:**
  - Snap pointer X (using `tracks.snap`, disabled while `Shift` held) to frame `f`.
  - Raw delta: `d = isRightHandle ? (f - (clip.start + clip.length)) : (f - clip.start)`.
  - Clamping across every selected clip proxy `r`:
    - If `adjustTime`:
      - If left handle selected: `d = max(d, -r.timeStart * fps, -r.start)` (cannot trim before media `0.0s` or sequence frame `0`).
      - If right handle selected: `d = min(d, r.mediaLength - r.timeEnd * fps)` (cannot trim past source media length).
    - Else:
      - If left handle selected: `d = max(d, -r.start)`.
    - **Minimum clip length = `1` frame**:
      - `d = min(d, r.length + rightContrib - 1)`
      - `d = max(d, -r.length + leftContrib + 1)`
- **Exact Property & Keyframe Update Formulas:**

| Mode | Left Handle (`d` frames) | Right Handle (`d` frames) |
| --- | --- | --- |
| **Standard Trim** (`adjustTime = true`) | `start' = start + d`<br>`length' = length - d`<br>**All dynamic keyframes shifted by `offset = -d`** (`kf.frame -= d`, preserving sequence-time alignment)<br>`time.kf[0].frame = 0`, `time.kf[1].frame = length'`<br>`time.kf[0].value = timeStart + d / fps` | `start' = start`<br>`length' = length + d`<br>Keyframes unchanged<br>`time.kf[1].frame = length'`<br>`time.kf[1].value = timeEnd + d / fps` |
| **Rate Stretch** (`ctrlKey` held) | `start' = start + d`<br>`length' = length - d`<br>`scale = length' / length`<br>**All dynamic keyframes scaled:** `kf.frame *= scale`<br>`time.kf[0].frame = 0`, `time.kf[1].frame = length'` (media `timeStart`/`timeEnd` unchanged -> changes playback speed!) | `start' = start`<br>`length' = length + d`<br>`scale = length' / length`<br>**All dynamic keyframes scaled:** `kf.frame *= scale`<br>`time.kf[0].frame = 0`, `time.kf[1].frame = length'` |
| **Animated-Time Trim** | `start' = start + d`, `length' = length - d` (keyframes & animated `time` untouched) | `length' = length + d` |

- On mouse-up: commits via `moveClipProxies` (so extending a clip into a neighbor trims/overwrites that neighbor) inside one undo operation.

### 2.7 Clip Splitting (`C` Key)

Operates on every selected clip inside a single undo operation:
1. Compute local split offset: `l = currentFrame - clip.start`.
2. Guard: skip clip if `l <= 0 || l >= clip.length` *(note: vanilla CM3 checks `l < 0 || l >= clip.length`, allowing a degenerate 0-frame split at `l == 0`; ZoidCraft requires `0 < l < clip.length`)*.
3. Compute split media time (when `!clip.properties.time.animated`):
   `tSplit = clip.properties.time.get(l)`.
4. **Right half** (created at `trackIdx`, index `oldIdx + 1`, cloned from original clip data):
   - `start = clip.start + l`
   - `length = clip.length - l`
   - Shift all dynamic keyframes by `offset = -l` (`kf.frame -= l`), keeping them at identical sequence frames.
   - If `!time.animated`: `time.kf[0] = (frame: 0, value: tSplit)`, `time.kf[1] = (frame: length - l, value: oldTimeEnd)`.
5. **Left half** (mutated in place at `oldIdx`):
   - `start` unchanged, `length = l`.
   - If `!time.animated`: `time.kf[1] = (frame: l, value: tSplit)`.
6. **Link preservation:** Right-half clips originating from linked clips are linked together under a newly generated link key (independent of the left-half link group).

### 2.8 Clipboard & Context Menus

| Action | Trigger | Exact Behavior |
| --- | --- | --- |
| Delete clips | `Delete` / `Backspace` (tracks focused) | Unlinks and deletes all selected clips in one undo operation; updates `sequence.length` and schedules. |
| Copy / Cut clips | `Ctrl+C` / `Ctrl+X` (tracks focused) | Packages selected clips grouped by video/audio track into JSON `[{ baseType: "track", data: [...] }]`. Cut also deletes them in one undo operation. |
| Paste clips | `Ctrl+V` (timeline or tracks focused) | Parses `"track"` payload, shifts all clip starts by `currentFrame - minFrame`, auto-creates missing tracks, inserts via `moveClipProxies` (overwriting collisions) in one undo operation. |
| Copy / Cut keyframes | `Ctrl+C` / `Ctrl+X` (keyframes focused) | Packages selected `.kfobject` keyframes keyed by `property.definition.name` into `[{ baseType: "propertyList", data: {...} }]`. |
| Paste keyframes | `Ctrl+V` (timeline or keyframes focused) | Pastes matching property names onto visible keyframe tracks shifted by `currentFrame - minFrame - prop.frameOffset`. |
| Inline clip rename | Right-click on a `.clip` | Replaces clip label with inline `<input>`; `Enter` or blur commits undoable `setValue` on `clip.properties.name`; `Escape` cancels. |
| Cycle time format | Right-click on timestamp box | Cycles `HH:MM:SS.mmm` -> `HH:MM:SS:FF` -> `Frame`. |
| Quick playhead seek | Right-click on empty track or keyframe lane | Immediately calls `setFrame(contentX)` (pauses playback and jumps playhead). |
| Property context menu *(Graph Editor 2)* | Right-click animatable property row or timeline keyframe label | Opens context menu with `Rename…` (if renamable) and `Show in Graph Editor (Alt+Shift+G)`. |

---

## 3. Keyframes & Easing Exact Mathematics

### 3.1 Keyframe Data Model & `tween` Bitfield

Each keyframe on a `PZ.property.dynamic.keyframes` channel stores:

| Field | Type | Default | Semantics |
| --- | --- | --- | --- |
| `frame` | `number` | `0` | Clip-local frame index (`absoluteFrame = frame + property.frameOffset`, where `frameOffset = clip.start` or `0` for sequence properties) |
| `value` | `number \| string \| array \| ...` | Property default | Keyframed value at `frame` |
| `tween` | `uint16` | `1` (interpolated) or `0` (non-interpolated) | Bitfield for the **incoming segment** `[prevKf, thisKf]` |
| `controlPoints` | `[[inX, inY], [outX, outY]]` | `[[-10, 0], [10, 0]]` | Relative Bezier handle offsets in `(frames, propertyUnits)`. `controlPoints[0]` is incoming handle (`inX <= 0`), `controlPoints[1]` is outgoing handle (`outX >= 0`). |
| `continuousTangent` | `bool` | `true` | When `true`, dragging one Bezier handle mirrors its angle (`+ π`) and length onto the opposite handle. |

**`tween` Bitfield Encoding:**
- `easeIndex = tween & 0xFF` (`0..32`): index into the 33-entry easing table below.
- `interpMode = (tween >> 8) & 1`:
  - `0` = Standard easing curve (`easingList[easeIndex]`).
  - `1` = Cubic Bezier curve (e.g. `257 = (1 << 8) | 1`).
- **Evaluation Precedence (exact CM3 order):**
  1. If `property.expression !== null` (or parent group has an expression): evaluate expression and return.
  2. Binary-search sorted keyframes for bracketing pair `(k0, k1)` around requested local frame `f`:
     - If `f <= keyframes[0].frame`: return `keyframes[0].value`.
     - If `f >= keyframes[last].frame`: return `keyframes[last].value`.
     - If `f == k0.frame`: return `k0.value`.
  3. Look up `easeFn = easingList[k1.tween & 0xFF].fn`. If `easeFn === null` (i.e., `easeIndex == 0` Hold/None): return `k0.value` (**even if `interpMode == 1`**).
  4. If `(k1.tween >> 8) == 1`: evaluate **Cubic Bezier** (Section 3.3).
  5. Otherwise: compute normalized time `u = max((f - k0.frame) / (k1.frame - k0.frame), 0)`, `e = easeFn(u)`, and return `k0.value + (k1.value - k0.value) * e`.

### 3.2 Complete Table of All 33 CM3 Easing Functions

For `u in [0, 1]`, every function `f(u)` in `PZ.tween.easingList[0..32]`:

| Idx | Name | Exact Mathematical Formula `f(u)` |
| --- | --- | --- |
| `0` | `none` (Hold/Step) | `null` (step function: returns `v0` for `f < k1.frame`, `v1` at `k1.frame`) |
| `1` | `linear` | `u` |
| `2` | `quadratic in` | `u^2` |
| `3` | `quadratic out` | `u * (2 - u)` |
| `4` | `quadratic in-out` | Let `w = 2u`: if `w < 1`, `0.5 * w^2`; else `-0.5 * ((w - 1) * (w - 3) - 1)` |
| `5` | `cubic in` | `u^3` |
| `6` | `cubic out` | `(u - 1)^3 + 1` |
| `7` | `cubic in-out` | Let `w = 2u`: if `w < 1`, `0.5 * w^3`; else `0.5 * ((w - 2)^3 + 2)` |
| `8` | `quartic in` | `u^4` |
| `9` | `quartic out` | `1 - (u - 1)^4` |
| `10` | `quartic in-out` | Let `w = 2u`: if `w < 1`, `0.5 * w^4`; else `-0.5 * ((w - 2)^4 - 2)` |
| `11` | `quintic in` | `u^5` |
| `12` | `quintic out` | `(u - 1)^5 + 1` |
| `13` | `quintic in-out` | Let `w = 2u`: if `w < 1`, `0.5 * w^5`; else `0.5 * ((w - 2)^5 + 2)` |
| `14` | `sinusoidal in` | `1 - cos(u * pi / 2)` |
| `15` | `sinusoidal out` | `sin(u * pi / 2)` |
| `16` | `sinusoidal in-out` | `0.5 * (1 - cos(pi * u))` |
| `17` | `exponential in` | `0` if `u == 0`, else `1024^(u - 1)` *(note: `1024 = 2^10`, so `2^(10(u - 1))`)* |
| `18` | `exponential out` | `1` if `u == 1`, else `1 - 2^(-10 * u)` |
| `19` | `exponential in-out` | `0` if `u == 0`; `1` if `u == 1`; let `w = 2u`: if `w < 1`, `0.5 * 1024^(w - 1)`; else `0.5 * (2 - 2^(-10 * (w - 1)))` |
| `20` | `circular in` | `1 - sqrt(1 - u^2)` |
| `21` | `circular out` | `sqrt(1 - (u - 1)^2)` |
| `22` | `circular in-out` | Let `w = 2u`: if `w < 1`, `-0.5 * (sqrt(1 - w^2) - 1)`; else `0.5 * (sqrt(1 - (w - 2)^2) + 1)` |
| `23` | `elastic in` | `0` if `u == 0`; `1` if `u == 1`; else `-2^(10 * (u - 1)) * sin((u - 1.1) * 2*pi / 0.4)` *(amplitude `a = 1`, period `p = 0.4`, phase shift `s = 0.1`)* |
| `24` | `elastic out` | `0` if `u == 0`; `1` if `u == 1`; else `2^(-10 * u) * sin((u - 0.1) * 2*pi / 0.4) + 1` |
| `25` | `elastic in-out` | `0` if `u == 0`; `1` if `u == 1`; let `w = 2u`: if `w < 1`, `-0.5 * 2^(10 * (w - 1)) * sin((w - 1.1) * 2*pi / 0.4)`; else `0.5 * 2^(-10 * (w - 1)) * sin((w - 1.1) * 2*pi / 0.4) + 1` |
| `26` | `back in` | With `c = 1.70158`: `u^2 * ((c + 1) * u - c)` |
| `27` | `back out` | With `c = 1.70158`: `(u - 1)^2 * ((c + 1) * (u - 1) + c) + 1` |
| `28` | `back in-out` | With `c = 2.5949095` (`1.70158 * 1.525`), `w = 2u`: if `w < 1`, `0.5 * w^2 * ((c + 1) * w - c)`; else `0.5 * ((w - 2)^2 * ((c + 1) * (w - 2) + c) + 2)` |
| `29` | `bounce in` | `1 - BounceOut(1 - u)` |
| `30` | `bounce out` | `7.5625 * u^2` if `u < 1/2.75`;<br>`7.5625 * (u - 1.5/2.75)^2 + 0.75` if `u < 2/2.75`;<br>`7.5625 * (u - 2.25/2.75)^2 + 0.9375` if `u < 2.5/2.75`;<br>`7.5625 * (u - 2.625/2.75)^2 + 0.984375` otherwise |
| `31` | `bounce in-out` | `0.5 * BounceIn(2u)` if `u < 0.5`; else `0.5 * BounceOut(2u - 1) + 0.5` |
| `32` | `cubic out-in` | `4 * (u - 0.5)^3 + 0.5` |

### 3.3 Cubic Bezier Evaluation, Curve Correction & Analytical Root Solver

Given segment `[k0, k1]` with `duration = k1.frame - k0.frame`:

1. **Handle Scale Correction (`correctCurve(k0, k1)`):**
   - **Vanilla CM3 rule:**
     ```text
     sumX = |k0.controlPoints[1][0]| + |k1.controlPoints[0][0]|
     s    = (sumX > duration) ? (duration / sumX) : 1.0
     ```
     This keeps `X(t)` monotonic (single-valued) even if handles are wider than the segment duration.
   - **Easing+ Crossing-Handle Preservation Override:**
     If `duration > 0`, let `x1 = k0.controlPoints[1][0] / duration` and `x2 = 1 + k1.controlPoints[0][0] / duration`.
     If `0 <= x1 <= 1`, `0 <= x2 <= 1`, and `x1 > x2 + 1e-8`: `s = 1.0` (both handles remain inside `[k0.frame, k1.frame]` so `X(t)` stays monotonic even when `x1 > x2`, e.g., sharp S-curves like `cubic-bezier(0.8, 0, 0.2, 1)`).
2. **Effective Control Points:**
   ```text
   X0 = k0.frame,                          Y0 = k0.value
   X1 = k0.frame + s * k0.controlPoints[1][0],  Y1 = k0.value + s * k0.controlPoints[1][1]
   X2 = k1.frame + s * k1.controlPoints[0][0],  Y2 = k1.value + s * k1.controlPoints[0][1]
   X3 = k1.frame,                          Y3 = k1.value
   ```
3. **Analytical Cubic Root Solver (`findZero` for `X(t) - f = 0`):**
   Write `X(t) - f = A*t^3 + B*t^2 + C*t + D = 0` where:
   ```text
   D = X0 - f
   C = 3 * (X1 - X0)
   B = 3 * (X0 - 2*X1 + X2)
   A = X3 - X0 + 3 * (X1 - X2)
   ```
   - Accept any real root `t` satisfying `-1e-10 <= t <= 1.000001`.
   - **If `A != 0` (depressed cubic via Cardano / trigonometric form):**
     - `h = B / (3A)`, `c = C / A`, `u = D / A`
     - `q = (2*h^3 - h*c + u) / 2`, `p = c/3 - h^2`, discriminant `m = q^2 + p^3`
     - If `m > 0`: single real root `t0 = cbrt(-q + sqrt(m)) + cbrt(-q - sqrt(m)) - h`.
     - If `m == 0`: `y = cbrt(-q)`, roots `2y - h` and `-y - h`.
     - If `m < 0`: `theta = acos(-q / sqrt(-p^3))`, `y = sqrt(-p)`, `d = cos(theta / 3)`, `e = sqrt(3 - 3*d^2)`, three roots: `2*y*d - h`, `-y*(d + e) - h`, `-y*(d - e) - h`.
   - **Else if `B != 0` (quadratic):** standard discriminant `C^2 - 4*B*D`.
   - **Else if `C != 0` (linear):** `t0 = -D / C`.
   - **Else if `D == 0`:** `t0 = 0`.
4. **Value Evaluation:**
   If at least one valid root `t` is found, evaluate the cubic Bernstein polynomial at the first valid root `t = roots[0]`:
   ```text
   Y(t) = Y0 + 3*(Y1 - Y0)*t + 3*(Y0 - 2*Y1 + Y2)*t^2 + (Y3 - Y0 + 3*(Y1 - Y2))*t^3
   ```
   If no valid root exists, return `k0.value`.

**Auxiliary Core Tween Helpers (used by expressions & integration):**
- `catmullRom(p0, p1, p2, p3, t)`: standard centripetal-uniform Catmull-Rom with tangents `m0 = 0.5*(p2 - p0)`, `m1 = 0.5*(p3 - p1)`.
- `integrate(frame)`: piecewise linear/trapezoidal integral of a keyframed scalar channel from frame `0` to `frame` (used for speed-to-phase accumulation: Hold segments add `v0 * dt`; interpolated segments add `0.5 * ((v1 - v0)/(f1 - f0)) * dt^2 + v0 * dt`).

### 3.4 Property Type Interpolation Matrix

| Property Type ID | Name | Container Class | `interpolated` | Default `tween` | Interpolation Semantics |
| --- | --- | --- | --- | --- | --- |
| `0` | `NUMBER` | `dynamic.keyframes` | `true` | `1` (Linear) | Scalar float; full easing & Bezier support; optional `min`, `max`, `step`, `scaleFactor` |
| `1` | `VECTOR2` | `dynamic.group` of 2 `NUMBER` | `true` | `1` per axis | Independent `X`, `Y` scalar keyframe channels |
| `2` | `VECTOR3` | `dynamic.group` of 3 `NUMBER` | `true` | `1` per axis | Independent `X`, `Y`, `Z` channels (includes 3D Position, Rotation Euler angles, Scale) |
| `3` | `VECTOR4` | `dynamic.group` of 4 `NUMBER` | `true` | `1` per axis | Independent `X`, `Y`, `Z`, `W` channels |
| `4` | `COLOR` | `dynamic.group` of 3 `NUMBER` | `true` | `1` per channel | Independent `R`, `G`, `B` channels clamped to `[0, 1]`, interpolated in RGB space |
| `5` | `GRADIENT` | `dynamic.keyframes` | `false` | `0` (Hold) | Stepped keyframe value |
| `6` | `OPTION` (Enum / Bool) | `dynamic.keyframes` | `false` | `0` (Hold) | Integer option index; stepped (no blending) |
| `7` | `TEXT` (String) | `dynamic.keyframes` | `false` | `0` (Hold) | Stepped string snapshot |
| `8` | `ASSET` | `dynamic.keyframes` | `false` | `0` (Hold) | Stepped asset reference |
| `9` | `LIST` | `dynamic.keyframes` | `false` | `0` (Hold) | Stepped list value |
| `10` | `CURVE` | `dynamic.keyframes` | `false` | `0` (Hold) | Stepped curve table snapshot |
| `11` | `SHADER` | `dynamic.keyframes` | `false` | `0` (Hold) | Stepped shader source |

### 3.5 Property-Row Keyframe Controls & Timeline Keyframe Editing

**Inspector Property Row Controls:**
- **Stopwatch button:** `17px × 17px` icon left of property label; `#2a2a2a` when static, `hsl((40 * hash(name)) % 360, 60%, 45%)` when animated. `Alt + Click` toggles Expression mode instead of keyframes.
- **Keyframe control strip** (`height: 27px`, `margin-bottom: 5px`, shown when `animated && !expression`):
  1. **Previous Keyframe** (`17×17px`, `#ccc`): jumps playhead to previous keyframe across channel(s).
  2. **Add / Delete Keyframe** (`17×17px` diamond): hollow stroke `#ccc` when playhead has no keyframe; solid `#ccc` fill when playhead is on a keyframe. Toggles keyframe at `currentFrame - frameOffset` (never deletes the sole remaining keyframe unless `definition.allowEmpty`).
  3. **Next Keyframe** (`17×17px`, `#ccc`): jumps playhead to next keyframe.
  4. **Interpolation Toggle** (`.pz-tweens`, `height: 27px`, `35×23px` SVG `interp_0` / `interp_1`): toggles bit 8 (`0` Standard vs `1` Bezier) on the keyframe at the playhead. Hidden when not on a keyframe, when non-interpolated, or when a vector group's channels disagree on `tween`.
  5. **Easing Picker** (`.pz-tweens`, `height: 27px`, `35×23px` SVG `ease_0..32`): opens a `250px`-wide, `max-height: 300px` scrollable dropdown (`#202020`, `1px solid #384668`) listing all 33 easings with icons (`Up`/`Down`/`Enter`/`Escape` navigation). When **Easing+** is active and `interp_1` is set on a non-first keyframe, this button displays `"Ez+"` (`min-width: 42px`) and opens the Easing+ curve window instead.

**Timeline Keyframe Interactions:**
- **Select:** Click selects diamond (expanding downward to all child keyframes on summary rows and upward to parent summary diamonds). `Shift + Click` adds/toggles. Marquee drag on empty dope-sheet lane (`> 2px` Manhattan threshold, `1px dashed rgb(151, 151, 151)`) selects intersecting keyframes (`Shift` = additive).
- **Drag:** Moves all selected keyframes horizontally by integer `deltaFrames`, snapping within `10px` (disabled by `Shift`) and displaying a vertical `#a6a6a6` 50%-opacity snap guide across the entire timeline.
- **Clone Drag:** Holding `Alt` at drag movement start clones the selected keyframes instead of moving originals (overwriting any existing keyframe at the destination frame).
- **Delete:** `Delete` / `Backspace` deletes all selected keyframes (preserving at least 1 keyframe per channel unless `allowEmpty`).

---

## 4. Keyframe Graph Editor Specification

### 4.1 Vanilla CM3 Graph Editor (`PZ.ui.graph` + `PZ.ui.graphEditor`)

- **Container:** `PZ.ui.splitPanel` with `33%` left property tree (`PZ.ui.edit` filtering `PZ.property.dynamic.keyframes`, no inline controls/buttons) and `67%` right SVG graph (`PZ.ui.graph`).
- **Coordinate Mapping:**
  - SVG `viewBox` = `[viewX, viewY, viewW, viewH]` where:
    - `viewX = centerX - 0.5 * width * zoomX`, `viewW = width * zoomX` (units: **frames**, `zoomX` = frames per CSS pixel)
    - `viewY = centerY - 0.5 * height * zoomY`, `viewH = height * zoomY` (units: **inverted scaled values**, `svgY = -value * valueScale` where `valueScale = 1 / (definition.scaleFactor || 1)`)
  - Each clip-owned property group applies `transform="translate(clip.start 0)"`.
- **Visual Parameters:**
  - Curve stroke: `2px` (`vector-effect: non-scaling-stroke`), color `hsl((40 * stringHash(name)) % 360, 40%, 45%)`.
  - **Limitation in vanilla CM3:** Non-Bezier segments (`(tween >> 8) == 0`) render as straight line segments `M x0 y0 L x1 y1`; only Bezier segments render as cubic SVG `C` paths.
  - Keyframe square: `8px × 8px` visual (`#000000` unselected, `#cccccc` selected) with transparent `16px` stroke hitbox (`2 * keyframeSize`).
  - Bezier handles (visible on selected keys only): `8px` diameter `#cccccc` circles with transparent `16px` stroke hitbox, connected by `1px` `#cccccc` tangent lines.
  - Playhead line: `1px` vertical line `#ff0000aa` at `x = currentFrame`.
  - Grid: `gridUnit = 20`, spacing `20 * 2^ceil(log2(zoom))`, odd gridlines `#333333`, even major gridlines `#1e1e1e` with `10px` `#666666` labels.
- **Vanilla Interactions & Shortcuts:**
  - `LMB drag` empty canvas: marquee select (`1px` dashed `#ccc`, `Shift` = additive).
  - `MMB drag`: pan view (`centerX -= dx * zoomX`, `centerY -= dy * zoomY`).
  - `RMB drag`: scrub playhead (`speed = 0`, clamped `[0, totalFrames - 1]`).
  - `Wheel`: zoom by `0.95` (up) / `1.05` (down); `Ctrl` locks X zoom, `Shift` locks Y zoom.
  - Keyframe drag (`4px` Manhattan deadzone): `Ctrl` locks frame (value-only), `Shift` locks value (frame-only).
  - Handle drag (`4px` deadzone): clamps `inX <= 0` and `outX >= 0`; `Shift` breaks `continuousTangent` (commits `continuous = false` on mouse-up); continuous mirror sets opposite handle to same Euclidean length at `angle + pi`.
  - Shortcuts: `=`/`+` zoom in (`* 0.95`), `-`/`_` zoom out (`* 1.05`), `\` fit all keys with `1.3×` bounding padding, `Del`/`Backspace` delete selected, `T` toggle continuous tangents, `I` toggle Bezier/Standard, `R` reset handles to `[[-10, 0], [10, 0]]`, `Ctrl+C`/`X`/`V` copy/cut/paste.

### 4.2 Zoidium Graph Editor 2 (`graph-plus`)

Graph Editor 2 replaces the vanilla graph editor with a full F-curve workstation while keeping 100% compatibility with CM3's data model and single-step undo history.

#### Window & Layout Geometry

| Region | Exact Dimensions & Behavior |
| --- | --- |
| Floating window | Centered, `75% × 75%` of viewport (default `900×600` on `1200×800`), `minWidth: 480px`, `minHeight: 300px`, resizable, maximizable, persisted geometry. Opened by `Ctrl+G`, toolbar button, `Alt+Shift+G` on focused property row, or right-click `Show in Graph Editor`. |
| Outer split | `33%` left property picker (`PZ.ui.edit`), `67%` right Graph Editor 2 workspace (`.g2-wrap`). |
| Top Toolbar (`.g2-toolbar`) | `min-height: 104px`, `max-height: 46%`, `padding: 15px 26px 10px`, `gap: 6px`, wraps cleanly. Switches to `.g2-narrow` compact mode when workspace width `< 700px` (`padding: 10px 12px 8px`, `min-height: 0`). |
| Big Preset Tiles | `68px × 54px` (`44px × 38px` in narrow mode), `2px` radius, `1px` border, `2px` accent outline when active. Group gap `13px` (`8px` narrow), separator `40px` (`12px` narrow). |
| Small Tool Tiles | `38px × 32px` (`32px × 28px` in narrow mode), `14px × 14px` icon + compact label beneath. |
| Track List Column (`.g2-tracks`) | Fixed `width: 188px`, right border `1px`, vertical scroll. Each row shows `12px × 4px` color swatch, truncated track/axis name, live value pill (refreshed every 20 rAF frames), Eye visibility toggle, and Solo (`S`) button on vector/color axes. |
| Top Frame Ruler (`.g2-ruler`) | `height: 30px`, `cursor: ew-resize`, 2D canvas synced to `viewX`/`zoomX` and `currentFrame`. LMB or RMB drag scrubs playhead. |
| Curve Canvas (`.g2-canvas`) | `flex: 1`, `min-width: 48px`, `min-height: 48px`, `touch-action: none`. Contains SVG grids, active sequence range highlight rect (`accent @ 8%`), curves, playhead, marquee overlay (`accent @ 20%`, `1px` dashed `4 3`), hover tooltip, and empty-state card (`NO CURVES`). |
| Status Bar (`.g2-status`) | `padding: 4px 10px`, `gap: 8px`, single-row non-wrapping pill badges (active tool, selected key frame/value, snap state, zoom/frame info). |

#### Toolbar Actions & Custom Easings Popup

1. **Big Interpolation & Tangent Tiles:**
   - **Linear** -> sets `tween = 1` on selected keys (clears Bezier bit).
   - **Step** -> sets `tween = 0` (Hold) on selected keys.
   - **Ease** -> sets `tween = 7` (Cubic In-Out) on selected keys.
   - **Break Tangents** -> sets `continuousTangent = false` on selected keys (recording exact pre-edit handle coordinates in undo stack so undo restores exact handles).
   - **Auto Tangents** -> sets `continuousTangent = true` on selected keys.
2. **Custom Easings Button & Popup:**
   - `50px × 30px` preview tile (`40px × 26px` narrow) + `"Custom easings"` label.
   - Opens `.g2-popup` (`min-width: 200px`, `max-height: 330px`, scrollable) listing all 33 `PZ.tween.easingList` entries (`0: none` .. `32: cubic out-in`), each with a live `64px × 48px` (`viewBox="0 -12 64 64"`) SVG thumbnail curve sampled via `easingPoints`.
3. **Right-Aligned Tool & View Tiles:**
   - **Follow selection** (persisted in `localStorage`, default **ON**): automatically tracks clicked property rows (`Ctrl`/`Shift` additive), selected timeline keyframes, or selected clips/objects (showing all animated/expression curves recursively). Turning OFF pins current curves; explicit `Show in Graph Editor` still loads & fits curves while pinned.
   - **Link axes** (default OFF): selecting or time-dragging a keyframe on one axis of a Vector/Color group also selects/moves same-frame sibling keys on the other axes, while value drags remain per-axis.
   - **Pan (`H`)** / **Marquee (`V`, default active)**: switches LMB canvas drag mode.
   - **Fit (`\`)**: fits all visible curves to viewport.
   - **Frame (`F`)**: frames currently selected keys.
   - **Norm (`N`)**: normalizes all visible curves to a shared vertical height for simultaneous comparison.
   - **Snap (`S`, default ON)**: toggles `8px` frame snapping during keyframe drags.
   - **+Key (`K`)**: inserts a keyframe at `currentFrame` on all visible editable (non-locked, non-expression) curves (auto-enabling stopwatch animation first if static).
   - **Del (`Delete`/`Backspace`)**: deletes selected keys.
   - **Bezier (`I`)**: toggles Bezier bit (`tween ^ 256`) on selected keys.
   - **Link (`T`)**: toggles `continuousTangent` on selected keys.
   - **Flat (`R`)**: resets handles of selected keys to `[[-10, 0], [10, 0]]`.
   - **Copy (`Ctrl+C`) / Paste (`Ctrl+V`)**: copies/pastes keyframes at playhead, preserving axis names; a single copied axis can paste onto any single selected destination axis.
   - **Labels** (default ON): toggles inline curve name labels.
   - **`?` button**: toggles gesture quick-reference card.

#### Multi-Channel Color Coding

| Channel / Axis | Hex Color |
| --- | --- |
| `X` or `R` | `#ed6a68` (Red) |
| `Y` or `G` | `#81c784` (Green) |
| `Z` or `B` | `#64b5f6` (Blue) |
| `W` or `A` | `#c792ea` (Purple) |
| Standalone Scalar Property | `hsl((47 * stringHash(name)) mod 360, 70%, 62%)` |

#### Coordinate Mapping, Zoom Clamps & Adaptive 1-2-5-10 Grid

- **Zoom Bounds (graph units per CSS pixel):**
  - `zoomX in [0.0001, 500]`
  - `zoomY in [0.000001, 10000]`
  - `centerX, centerY in [-1e12, 1e12]`
- **Adaptive Grid Ticks (`ticks(zoom)`):**
  ```text
  target = zoom * 64
  power  = 10 ^ floor(log10(target))
  factor = first v in [1, 2, 5, 10] where v * power >= target
  major  = factor * power
  minor  = major / 5
  ```
  Every 5th line (`idx % 5 == 0`) is a major line (`--zui-hover`); `idx == 0` is the zero axis (`--zui-text-faint`); other lines are minor (`--zui-bg`). Y major lines carry `10px` labels along the left edge (`v = -a`, formatted to 12 significant digits); X frame labels live exclusively in the top `30px` ruler.

#### True Curve Rendering & Progressive Expression Sampling

1. **Hold / Step Segments (`easingList[tween & 255].fn === null`):**
   Drawn as an exact step path: `M x0 y0 L x1 y0 L x1 y1`.
2. **Eased Segments (`(tween >> 8) == 0`, `fn !== null`):**
   Adaptively subdivided in screen space (`easingPoints(fn, widthPx, heightPx, tol = 0.35)`):
   - Detects endpoint discontinuities at `1e-10` and `1 - 1e-10` (`|jump| * |heightPx| > 0.35`).
   - Seeds `32` equal intervals `[i/32, (i+1)/32]` (preventing symmetric oscillations like Elastic from aliasing away).
   - Recursively bisects any interval `[a, b]` (max depth `24`, max `4096` total subdivisions) when either:
     - Vertical midpoint/quarter-point deviation from chord `* |heightPx| > 0.35px`, OR
     - Horizontal span `(b - a) * widthPx > 8px`.
3. **Bezier Segments (`(tween >> 8) == 1`):**
   Exact SVG cubic `C` using `correctCurve(k0, k1)` (preserving crossing handles when Easing+ is active).
4. **Expression-Driven Tracks:**
   - Displayed as **dashed, read-only** evaluated curves (keys cannot be selected/dragged).
   - Progressively sampled across animation frames with a shared budget of **at most `1024` evaluations and `6ms` per redraw** across all tracks.
   - Sample step: `step = min(1, max(2 * zoomX, 1/64))` frames (up to 64 samples/frame when zoomed in).
   - **Sub-pixel Column Collapse (`expressionPath`):** When multiple samples fall into the same integer screen pixel column `floor(screenX)`, that column collapses into a clean min–max envelope stroke (`x+0.5`: `first -> low -> high -> last`) instead of noisy zigzags. `NaN`/non-finite values break the path cleanly.

#### Keyframe, Handle Hitboxes & Gesture Rules

- **Visual Geometry:**
  - Curve stroke: `2.5px`, round caps & joins.
  - Keyframe dot: `11px` diameter circle (`rx = 0.5 * 11 * zoomX`), `2.5px` stroke unselected (`3px` selected), `22px` transparent hit target; broken Bezier keys use `stroke-dasharray: 2 1`.
  - Bezier handle dot: `9px` diameter ellipse (`rx = 0.5 * 9 * zoomX`), `1.5px` border, `1px` stem line.
  - **Minimum Visible Handle Length (`24px`):** If a handle's screen length `hypot(dxPx, dyPx) < 24px`, its displayed position is extended radially to `24px` (defaulting to `-24px` for incoming or `+24px` for outgoing if length `< 1e-9`) **without altering stored `controlPoints`**.
  - **Target Priority (`nearestTarget`):** When picking under the pointer, handles receive a `-4px` effective distance priority bonus over keyframe dots so handles lying near a key are always grabbable.
  - **Playhead Line Grab:** Left-dragging within `<= 7px` horizontally of the playhead line on the canvas switches cursor to `col-resize` and scrubs the playhead.

| Input Gesture | Exact Graph Editor 2 Behavior |
| --- | --- |
| Mouse wheel (discrete) | Zooms X around pointer (`factor = exp(clamp(dy * 0.002, -2, 2))`) |
| `Alt + Wheel` | Zooms Y around pointer (on ruler: zooms X only) |
| `Shift + Wheel` | Pans X (`centerX += (dx \|\| dy) * zoomX`) |
| Trackpad 2-finger scroll | Pans X and Y continuously (`classifyWheel` retains device mode across `220ms` inertial tails) |
| Pinch or `Ctrl + Wheel` | Zooms X and Y around pointer (`gain = 0.01` when `|dy| < 40`, else `0.002`; on ruler: X only) |
| `Alt + Pinch` | Zooms Y around pointer |
| `MMB drag` or `Space + LMB drag` | Pans X and Y (works regardless of active tool) |
| `LMB drag` on ruler / `RMB drag` on canvas | Scrubs playhead (`speed = 0`) |
| Double-click on canvas | Adds keyframe at clicked frame on active/nearest curve |
| `Ctrl` during key drag | Locks frame (vertical value drag only); clamps to `[definition.min, definition.max]` |
| `Shift` during key drag | Locks value (horizontal frame drag only) |
| Drag Bezier handle on preset segment | Automatically converts that segment's right-hand keyframe to Bezier (`tween = 257`) in the same undo step |
| `Alt` during handle drag | Pressing/holding `Alt` breaks tangent (`continuousTangent = false`); releasing `Alt` rejoins it |
| `Shift` during handle drag | Constrains handle screen angle to `45°` (`pi/4`) increments |
| `Ctrl` / `Cmd` during handle drag | Snaps handle angle within `15°` (`pi/12`) of horizontal (`0°`) or neighbor keyframe's slope |
| Double-click Bezier handle | Resets both handles of that keyframe to `[[-10, 0], [10, 0]]` |
| Arrow keys / `Ctrl+A` / `Esc` | Nudge selected keys / Select all keys / Deselect or close popup |

### 4.3 Zoidium Easing+ (`easing-plus`)

Easing+ provides a dedicated normalized `[0, 1] × [0, 1]` cubic-Bezier editor for any Bezier (`interp_1`) incoming segment `[prevKf, curKf]`, opening as a non-modal floating window above the live viewport and committing edits as one undo operation on close (`Escape` or close button).

#### Window Layout & Plot Geometry

- **2-Column Layout:** `minmax(0, 1fr)` left curve editor | `190px` right preset grid (`150px` when viewport `<= 620px`).
- **SVG Plot:** Virtual width `520`, height scaled to container aspect (default `420`), plot insets `left: 28, right: 28, top: 18, bottom: 18`.
  - `4 × 4` unit grid over `[0, 1] × [0, 1]`.
  - Curve: `2.25px` stroke (`--zui-text-bright`) over `5px` drop shadow.
  - Guides: `1.35px` accent/value lines from anchor to handle.
  - Anchors: endpoints `r = 7`, interior `r = 6`, drawn **before** handles.
  - Handles: `r = 5`, `cursor: grab`, drawn **last** so a handle dragged onto `(0, 0)` or `(1, 1)` still receives pointer hits.
- **Coordinate Inputs:** 2×2 grid of numeric fields (`Out X`, `Out Y`, `In X`, `In Y`), rounded to 4 decimals. X clamps to `[0, 1]`; Y is unclamped (supports overshoot).
- **Clipboard (`Ctrl/Cmd+C`, `Ctrl/Cmd+V`, Copy/Paste buttons):**
  - Formats as CSS `cubic-bezier(x1, y1, x2, y2)`.
  - Accepts `cubic-bezier(...)`, 4 comma/space-separated numbers, or a JSON 4-number array. X clamps to `[0, 1]`; Y preserved verbatim. Keeps an in-memory fallback copy when clipboard read is unavailable.

#### Normalized `<->` Native `controlPoints` Conversion

Let `duration = end.frame - start.frame` (`> 0`) and `delta = end.value - start.value`:

- **Reading (`curveFromSegment`):**
  1. If `start.controlPoints[1] == [10, 0]` and `end.controlPoints[0] == [-10, 0]` (untouched native CM3 defaults): present normalized Linear `(0.333, 0.333, 0.667, 0.667)` so initial curve does not depend on segment duration.
  2. Otherwise, let `c = correctCurve(start, end)`:
     ```text
     x1 = clamp((start.controlPoints[1][0] * c) / duration, 0, 1)
     x2 = clamp(1 + (end.controlPoints[0][0] * c) / duration, 0, 1)
     ```
     - If `|delta| > 1e-8`:
       ```text
       y1 = (start.controlPoints[1][1] * c) / delta
       y2 = 1 + (end.controlPoints[0][1] * c) / delta
       ```
     - If `|delta| <= 1e-8` (equal-value segment): map via fallback `valueScale` (`keyframeRange`, else `def.max - def.min`, else `|def.step| * 10`, else `max(|defaultVal|, 1)`):
       `y1 = x1 + (outY * c) / valueScale`, `y2 = x2 + (inY * c) / valueScale`.
- **Writing (`planCurve`):**
  ```text
  outX = clamp(x1, 0, 1) * duration
  inX  = -clamp(1 - x2, 0, 1) * duration
  ```
  - If `|delta| <= 1e-8`: write `outY = 0`, `inY = 0` (guarantees equal-value segments always play flat instead of bouncing out-and-back!).
  - Otherwise: `outY = y1 * delta`, `inY = (y2 - 1) * delta`.

#### Shape Preservation on Keyframe Edits (`rescaleSegmentHandles`)

When a keyframe is moved, edited in value, or rate-stretched (`moveKeyframe`, `setValue`, `startMove`/`finishMove`, `scaleKeyframes`), Easing+ rescales handles of every affected non-default Bezier segment `[beforeDuration, beforeDelta] -> [afterDuration, afterDelta]`:
- `xScale = afterDuration / beforeDuration` -> `outX' = outX * xScale`, `inX' = inX * xScale`.
- If `|beforeDelta| > 1e-8` and `|afterDelta| > 1e-8`: `yScale = afterDelta / beforeDelta` -> `outY' = outY * yScale`, `inY' = inY * yScale`.
- If `|beforeDelta| > 1e-8` and `|afterDelta| <= 1e-8` (collapsed to equal values): `outY' = 0`, `inY' = 0`.
- If `|beforeDelta| <= 1e-8`: keep `outY`, `inY` unchanged (flat zeros stay zero).
- Segments with untouched native defaults `[10, 0]` / `[-10, 0]` or newly reordered keyframe pairs retain native behavior.

#### Two-Stage Hysteresis Vertical Overshoot Gesture

Overshoot above `y = 1` (`top`) and below `y = 0` (`bottom`) unlock independently so dragging a handle to exact `0` or `1` never triggers accidental overshoot:

| Constant / Rule | Exact Value |
| --- | --- |
| Normal Y viewport bounds | `[-0.12, 1.12]` |
| Expanded bottom bound (when `bottom` unlocked) | `min(-0.55, minPointY - 0.12)` |
| Expanded top bound (when `top` unlocked) | `max(1.55, maxPointY + 0.12)` |
| Arm threshold (`OVERSHOOT_ARM_PX`) | `40px` past the `y = 0` or `y = 1` line in screen pixels |
| Disarm hysteresis (`OVERSHOOT_RELEASE_PX`) | `24px` (retreating within `< 24px` of boundary disarms) |
| State machine | 1) Dragging `0..40px` past boundary clamps handle to exact `0` or `1`.<br>2) Crossing `>= 40px` arms that side (`pendingOvershoot[side] = true`, highlights dashed zone, captures `offset = rawY - boundary` to avoid jumping).<br>3) Retreating `< 24px` disarms.<br>4) Releasing pointer while armed unlocks & expands that side's bounds for subsequent drags. Opening a curve that already has `y < -1e-8` or `y > 1 + 1e-8` immediately opens with the matching side(s) expanded. |

#### Complete Table of 12 Easing+ Presets

| # | Preset Name | `(x1, y1, x2, y2)` |
| --- | --- | --- |
| 1 | `Linear` | `(0.333, 0.333, 0.667, 0.667)` |
| 2 | `Ease` | `(0.25, 0.1, 0.25, 1)` |
| 3 | `Ease in` | `(0.42, 0, 1, 1)` |
| 4 | `Ease out` | `(0, 0, 0.58, 1)` |
| 5 | `Ease in-out` | `(0.42, 0, 0.58, 1)` |
| 6 | `Soft` | `(0.22, 0.08, 0.28, 1)` |
| 7 | `Swift in` | `(0.72, 0, 0.92, 0.38)` |
| 8 | `Swift out` | `(0.08, 0.62, 0.28, 1)` |
| 9 | `Sharp` | `(0.5, 0, 0.5, 1)` |
| 10 | `Expo in` | `(0.9, 0.02, 0.98, 0.4)` |
| 11 | `Expo out` | `(0.02, 0.6, 0.1, 0.98)` |
| 12 | `Sine` | `(0.37, 0, 0.63, 1)` |

---

## 5. Playback Engine & Audio Synchronization Rules

### 5.1 Playhead State & Stepping Semantics (Exact CM3 Parity)

| Property | Definition |
| --- | --- |
| `exactFrame` | High-precision floating-point timeline frame position |
| `currentFrame` | `round(exactFrame)` — integer frame passed to sequence evaluation, compositor, and UI overlays |
| `frameRate` | `sequence.properties.rate.get()` (e.g. `30`, `60`, `24`, `29.97`, `120`) |
| `totalFrames` | `sequence.length` |
| `currentTime` | `currentFrame / frameRate` (seconds) |
| `speed` | Playback multiplier (`0` = paused, `1` = normal forward, `-1` = reverse, etc.) |
| `loop` | Boolean observable |

**Boundary & Loop Behavior per Tick:**
- If `exactFrame >= totalFrames` or `exactFrame < 0`:
  - If `loop && speed != 0`: wrap positively:
    `exactFrame = ((exactFrame % totalFrames) + totalFrames) % totalFrames` (or `0` if `totalFrames == 0`).
  - Else: clamp `exactFrame = clamp(exactFrame, 0, max(totalFrames - 1, 0))` and stop (`speed = 0`).
- Scrubbing (`setFrame`) always sets `speed = 0` and clamps `currentFrame` to `[0, max(totalFrames - 1, 0)]`.

### 5.2 Audio Routing & Audio Meter (`PZ.ui.audioMeter`)

- **Per-Schedule Audio Chain:** `Source -> GainNode (animated clip.volume) -> StereoPannerNode (animated clip.pan) -> Shared AnalyserNode (fftSize = 1024) -> Output`.
- **Audio Meter Panel:**
  - `minWidth: 45px`, `minHeight: 45px`, background `#242424`, inner bar `4px` inset with vertical gradient `bottom: green -> 90%: yellow -> top: red`.
  - Dark mask (`#111111`, `opacity: 0.95`, `transform-origin: top`) scaled vertically by `1 - rms` where `rms = sqrt(mean((2 * |sample[i]/256 - 0.5|)^2))`.
  - Peak-hold indicator: `1px` horizontal line `#cece62` at `bottom = 100 * peak%`, decaying each animation frame via `peak = max(framePeak, peak - 0.01 / exp(peak))`.

### 5.3 Known CM3 Audio Desync Defects (Root Causes to Avoid)

Inspecting CM3's runtime reveals four specific bugs that caused editor and export desynchronization:

1. **Missing Seek on Scrub Inside an Active Clip:** `PZ.schedule.update` only assigns `el.currentTime = time.get(0)` when `currentItem` changes. Scrubbing to another frame inside the same audio clip leaves `el.currentTime` at its old value; starting playback then plays from the stale media offset while holding the playhead frozen or separated by seconds.
2. **Discarded Drift-Correction Rate Term:** In `PZ.ui.playback.updateSchedule`, the rate-correction computation `(Math.max(40 * o, 1), s)` uses the comma operator and discards the drift term, so small drift (`< 0.5s`) is never corrected.
3. **Hard-Resync Unit Mismatch (`s + t`):** When drift exceeds `0.5s`, CM3 seeks to `clip.time.get(s + t)` where `s` is a frame offset and `t` is the per-tick frame delta, overshooting into the end-of-clip `ended` boundary on short tails.
4. **Preload Padding Early Return & Export Tail Bug:** Active items load `padding = 60` frames early (`start - 60`); early-returning while `exactFrame < start` leaves stale seek state. On offline export, short final audio feeds where `totalSamples - sample < num` can leave unwritten tail samples in the transfer ring buffer if not zero-filled; and server/Node paths must apply `startOffset * sampleRate` after remuxing.

### 5.4 ZoidCraft Authoritative Playback & Audio Sync Architecture

ZoidCraft MUST preserve CM3's exact playhead semantics (Section 5.1) while replacing CM3's `HTMLMediaElement` hack with a deterministic native audio-clock architecture:

1. **Native Sample-Accurate Mixing (No `HTMLMediaElement` Clock):**
   - Decode audio assets into sample buffers / streaming decoders feeding a single native audio output stream (e.g., `cpal` / ring-buffered mixer at device sample rate `SR`, typically `48000 Hz`).
   - Track a monotonic **audio playback epoch** `(epochId, anchorExactFrame, anchorSampleCount, speed)` whenever playback starts, pauses, scrubs, loops, or changes speed.
2. **Authoritative Master Clock:**
   - While playing (`speed != 0`):
     - Advance `exactFrame = anchorExactFrame + ((currentHardwareSamples - anchorSampleCount) / SR) * frameRate * speed`.
     - If no audio device is running (headless / muted hardware fallback), advance from monotonic wall clock `dt * frameRate * speed`.
   - Every audio clip active at sample `n` evaluates its exact timeline frame `f(n)`, local offset `s = f(n) - clip.start`, media time `tMedia = clip.properties.time.get(s)`, volume `clip.object.properties.volume.get(s)`, and stereo pan `clip.object.properties.pan.get(s)` directly from the same timeline clock — making audio-playhead desync structurally impossible (`desync == 0` samples).
3. **Instantaneous Seek & Loop Invalidation:**
   - Any call to `setFrame` (scrub), loop wrap, clip move/trim/split, or speed change increments `epochId`, flushes/repositions all active audio and video decoders to `clip.properties.time.get(exactFrame - clip.start)`, and resets `(anchorExactFrame, anchorSampleCount)`.
   - Video clip decoders synchronize to the same `exactFrame` clock with a tolerance of `<= 0.5` frames (hard-seeking immediately if `|videoPts - targetMediaTime| > 0.5 / frameRate` or on `epochId` change).
4. **Exact Offline Export Contract:**
   - Offline audio export evaluates every clip's `time(s)`, `volume(s)`, and `pan(s)` in floating-point sample space at `48000 Hz` (applying `startOffset` when decoding compressed streams) and explicitly **zero-fills** any trailing buffer region beyond `totalSamples - sample` so no stale ring-buffer audio or dropped tail frames can ever reach the encoder.
