# ZoidCraft Clean-Room Specification 03: Scene, Layers, Materials, Expressions & Project Data Model

This document specifies the clean-room behavioral model for ZoidCraft's project hierarchy, 2D compositing layers, 3D scene graph, materials, coordinate conventions, deterministic expression engine, best-effort Panzoid ClipMaker 3 (`.pz`) import schema, and native `.zcraft` storage format.

All behaviors are expressed purely as data schemas, type tables, and mathematical specifications for implementation across `zc-time`, `zc-geom`, `zc-color`, `zc-anim`, `zc-expr`, `zc-project`, `zc-scene`, `zc-shader`, and `zc-render`.

---

## 1. Project & Sequence Hierarchy

### 1.1 Document Tree

A project document is rooted at a `Project` node containing an asset registry, a media library, and a primary `Sequence`:

```text
Project
├── assets: Map<AssetKey, AssetRecord>
├── media:  List<MediaItem>
└── sequence: Sequence
    ├── properties: SequenceProperties
    ├── length: f64 (frames)
    ├── videoTracks: List<VideoTrack>      // rendered bottom-to-top: index 0 -> N-1
    │   └── clips: List<VideoClip>
    │       ├── start, length, link
    │       ├── properties: { name, media, time }
    │       └── object: Layer (types 0..8)
    │           ├── properties: LayerProperties
    │           ├── [objects]: child Layers / 3D Objects / 2D Shapes
    │           └── effects: List<Effect>
    ├── audioTracks: List<AudioTrack>
    │   └── clips: List<AudioClip>
    │       ├── start, length, link
    │       ├── properties: { name, media, time }
    │       └── object: AudioObject { volume, pan }
    └── clipLinks: ClipLinkRegistry
```

### 1.2 Project Metadata & Asset Table

Every external or imported binary resource is stored once in `Project.assets`, deduplicated by `AssetKey`.

- **CM3 Asset Key Derivation**:
  - For user files (`source = FILE`): SHA-256 digest of the first `268,435,456` bytes (256 MiB) of the file, lowercase hex, **truncated to the first 32 hex characters** (128 bits).
  - For built-in presets (`source = PRESET`): canonical relative URL path (e.g., `/assets/fonts/2d/bebas.ttf`).
- **Asset Types (`AssetType` enum)**:

| Numeric ID | Name | Typical Formats |
|---:|---|---|
| `0` | `IMAGE` | PNG, JPEG, GIF, WebP |
| `1` | `FONT` | TTF, OTF, WOFF, WOFF2 |
| `2` | `GEOMETRY` | OBJ, Three JSON geometry, glTF/GLB (ZoidCraft extension) |
| `3` | `AV` | MP4, WebM, MOV, MP3, WAV, OGG, FLAC, AAC |
| `4` | `SHADER` | GLSL ES 1.00 / WGSL shader source |
| `5` | `MISC` | `.cube` 3D LUTs and auxiliary data files |

- **Asset Sources (`AssetSource` enum)**:

| Numeric ID | Name | Semantics |
|---:|---|---|
| `1` | `FILE` | Local user-imported file; embedded in `.pz` archive if `size <= 16,777,216` bytes (16 MiB) and `external == false`. |
| `2` | `ARCHIVE` | Extracted from a project archive. |
| `3` | `PRESET` | Built-in/external preset path; `external = true`, not embedded in archive. |
| `4` | `CLOUD` | Remote cloud-referenced asset. |

- **Asset Record Fields**:
  - `sha256: string` (32-hex key in `.pz`; full 64-hex in `.zcraft`)
  - `url?: string` (present when `source != FILE`)
  - `source: AssetSource`
  - `external: bool`
  - `size: u64` (byte length)
  - `filename: string` (original display filename)
- **Reference Counting**: Tracked separately for media library references (`mediaReferences`) and active timeline/property references (`references`). Missing assets (referenced in `project` but absent from archive bytes) MUST NOT abort project open or save; they load as unresolved placeholders and preserve their keys for relinking.

### 1.3 Sequence Properties & Global Rendering Defaults

| Key | UI Name | Prop Type | Animated | Default | Constraints / Notes |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | `"Sequence"` | Hidden in inspector |
| `resolution` | Resolution | `VECTOR2` (1) | No | `[1920, 1080]` | Integer `[1, 4000]` per axis |
| `rate` | Frame rate | `NUMBER` (0) | No | `30` | Integer `[1, 120]` fps |
| `markers` | Markers | `TEXT` (7) | Yes (always) | Empty | Keyframe default text = `keyframes.length + 1` |
| `motionBlur` | Multisample motion blur | `OPTION` (6) | No | `0` (`off`) | `0: off`, `1: on` |
| `motionBlurSamples` | Samples | `NUMBER` (0) | Yes | `16` | Integer `[1, 128]` |
| `motionBlurShutter` | Shutter | `NUMBER` (0) | Yes | `0.5` | Shutter angle fraction in frames (`0.5` = 180°) |

- **Sequence Duration**: `sequence.length` in frames (default `0` on empty sequence).
- **Canvas Background Color & Alpha**:
  - Root composition buffer (`screenBuffers[0]`) is cleared at the start of each frame/sub-sample to **opaque black `rgba(0, 0, 0, 1)`**.
  - Intermediate layer buffers and nested composite buffers (`screenBuffers[depth > 0]`) are cleared to **transparent black `rgba(0, 0, 0, 0)`**.
  - ZoidCraft MAY expose an optional sequence-level `backgroundColor` / transparent-alpha export flag in `.zcraft`, defaulting to opaque black `(0, 0, 0, 1)` for CM3 parity.
- **Audio Sample Rate**: Fixed **`48,000 Hz`**, stereo (2 channels, 32-bit float PCM). Export total samples = `floor((exportLength / exportRate) * 48000)`.
- **Multisample Motion Blur Accumulation**:
  - When `motionBlur == 1` (`on`), let $N = \max(1, \lfloor \text{motionBlurSamples}(f) \rfloor)$ and $S = \text{motionBlurShutter}(f)$.
  - For each sub-sample $s \in \{0, 1, \dots, N-1\}$, evaluate every active layer at sub-frame:
    $$f_{\text{local}}(s) = f - \text{clip.start} + \frac{s}{N} \cdot S$$
  - Render the entire track stack into `screenBuffers[0]`, accumulate additively into a floating-point RGBA buffer (`accumBuffer`), and scale the final accumulated buffer by $1/N$.
  - When `motionBlur == 0` (`off`), $N = 1$ and $s = 0$ (no sub-frame offset).

### 1.4 Tracks, Render Order & Clips

#### Track Types

| Numeric Type | Class | Clips Container | Notes |
|---:|---|---|---|
| `0` | Video Track | `List<VideoClip>` | Runtime `enabled: bool` (default `true`), `skip: bool`. |
| `1` | Audio Track | `List<AudioClip>` | Mixed into 48 kHz stereo output. |

#### Exact Render & Compositing Order (Critical Invariant)

CM3 uses **opposite** index orders at the Sequence track level versus inside Composite layers:

1. **Sequence `videoTracks`**: Iterated **ascending from index `0` to `videoTracks.length - 1`**.
   - **`videoTracks[0]` is the bottom-most (background) track**; it renders first onto the opaque black canvas.
   - **`videoTracks[N - 1]` is the top-most (foreground) track**; it composites last over all lower tracks.
2. **Composite Layer `objects`**: Iterated **descending from `objects.length - 1` down to `0`**.
   - **`objects[M - 1]` is the bottom-most child**; it renders first onto the transparent nested buffer.
   - **`objects[0]` is the top-most child**; it composites last on top of its siblings.

ZoidCraft MUST preserve this exact ordering semantics when importing `.pz` files and evaluating composites.

#### Clip Model

A track selects at most one active clip at sequence frame $f$: the first clip satisfying `clip.start <= f < clip.start + clip.length`. Clips on a single track are ordered by ascending `start` and do not overlap.

- **Clip Fields**:
  - `start: f64` — inclusive start frame on sequence timeline.
  - `length: f64` — duration in frames (`endFrame = start + length`).
  - `link: Option<u32>` — link group ID in `sequence.clipLinks` (links audio + video clips moved/trimmed together).
  - `properties`:
    - `name`: `TEXT` (hidden, default `"Clip"`)
    - `media`: `ASSET` (`AV`, read-only, default `null`)
    - `time`: dynamic `NUMBER` (`decimals: 3`, `staticBehavior: LINEAR (1)`). Defaults to two linear keyframes: `(frame: 0, value: 0.0)` and `(frame: clip.length, value: clip.length / sequence.rate)`. Maps clip-local frame to media timestamp in seconds (enabling speed ramps, freeze frames, and reverse playback).
  - `object`: either a `Layer` (for `VideoClip`) or an `AudioObject` (for `AudioClip`).

#### Audio Clip Object & Mixing Law

| Key | UI Name | Type | Animated | Default | Range |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | `"Audio"` | Hidden |
| `volume` | Volume | `NUMBER` (0) | Yes | `1.0` | `[0, ∞)` |
| `pan` | Pan | `NUMBER` (0) | Yes | `0.0` | `[-1.0, +1.0]` |

- **Effective Playback Rate** (for offline segment decoding):
  $$r_{\text{play}} = \frac{\text{time}(\text{clip.length}) - \text{time}(0)}{\text{clip.length}} \cdot \text{sequence.rate}$$
- **Offline Stereo Gain & Equal-Power Panning Law** (given input stereo sample $(L, R)$, gain $g = \text{volume}(f_{\text{local}})$, pan $p = \text{pan}(f_{\text{local}}) \in [-1, 1]$):
  1. Apply gain: $L_1 = g L$, $R_1 = g R$.
  2. Apply pan:
     - If $p \le 0$: let $\theta = (p + 1)\frac{\pi}{2}$; then $L_{\text{out}} = L_1 + R_1 \cos\theta$, $R_{\text{out}} = R_1 \sin\theta$.
     - If $p > 0$: let $\theta = |p|\frac{\pi}{2}$; then $L_{\text{out}} = L_1 \cos\theta$, $R_{\text{out}} = R_1 + L_{\text{out}} \sin\theta$.
  3. Sum all active audio clips at each sample and hard-clamp each output channel to $[-1.0, +1.0]$.

---

## 2. CM3 2D Layers, Effects & Compositing Pipeline

### 2.1 Layer Type Registry (Exact `.pz` Numeric Type IDs)

| Type ID | Internal Class | Default Name | Content Source | Extra Serialized Fields |
|---:|---|---|---|---|
| `0` | `layer.texture` | `"Layer"` | Decoded video frame texture from clip `media` | — |
| `1` | `layer.adjustment` | `"Adjustment"` | Snapshot copy of accumulated tracks below it (`screenBuffers[depth]`) | — |
| `2` | `layer.composite` | `"Composite"` | Nested stack of child `Layer`s rendered into `screenBuffers[depth + 1]` | `objects: List<Layer>` |
| `3` | `layer.image` | `"Image"` | Static decoded image texture (`premultiplyAlpha = true`) | `properties.childProperties.image` |
| `4` | `layer.scene` | `"Scene"` | 3D Scene graph rendered via its active Camera | `objects: List<Object3D>`, optional `zoidiumShadowMode` |
| `5` | `layer` | `"Layer"` | Empty transparent buffer (generator host for shader effects) | — |
| `6` | `layer.shape` | `"Shape"` | 2D vector shape tree rasterized to premultiplied RGBA canvas | `objects: List<Shape2D>` |
| `7` | `layer.shape.text` | `"Text"` | 2D outline text converted to per-glyph `Shape2D` paths | `properties.childProperties.{text, font, size}`, `objects: [group]` |
| `8` | `layer.shape.preset` | `"Preset shape"` | Built-in 2D vector preset (`ellipse` / `rectangle`) | `properties.childProperties.preset`, `objects: [group]` |

> **Shader / Generator Layers**: In CM3, procedural generators (noise, fractals, Shader+ templates, Node Effects) are **Effects** (`PZ.effect`, specifically Custom Shader `type: 1` or built-in named effects) attached to a Base Layer (`type: 5`), Adjustment Layer (`type: 1`), or Solid/Shape layer. ZoidCraft treats shader generators identically via the layer `effects` stack, and may offer a UI creation shortcut that instantiates a `Layer (5)` with a Custom Shader effect (`type: 1`).

### 2.2 Common Layer Properties & Transform

Every layer owns a `properties` list and an `effects` list (`List<Effect>`):

| Key | UI Name | Type | Animated | Default | Units / Constraints |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | Type default | Hidden |
| `childProperties` | — | `PropertyList` | — | Type-specific | Nested sub-properties |
| `resolution` | Resolution | `VECTOR2` (1) | No | Sequence res or media res | Read-only integer `[1, ∞)` |
| `position` | Position | `VECTOR2` (1) | Yes (group) | `[0, 0]` | Pixels, center-origin, +X right, +Y up |
| `scale` | Scale | `VECTOR2` (1) | Yes (group) | `[1, 1]` | Multiplier, min `0.001`, `linkRatio: true` |
| `rotation` | Rotation | `NUMBER` (0) | Yes | `0` | Radians stored (UI displays degrees), CCW around Z |
| `opacity` | Opacity | `NUMBER` (0) | Yes | `1.0` | `[0.0, 1.0]` |
| `blending` | Blending mode | `LIST` (9) | No | `"BLEND_SRC_OVER"` | One of 43 blend mode tokens (§2.6) |

- **Anchor / Pivot**:
  - Native CM3 layers pivot strictly around the **layer center `(0, 0)`** (`PlaneGeometry(1, 1)` scaled by `(resW, resH, 1)` inside an `Object3D` group with `position`, `rotation.z`, `scale`).
  - ZoidCraft MAY add an optional `anchor` (`VECTOR2`, default `[0, 0]`) in `.zcraft` that reduces to identity when `[0, 0]` for full CM3 parity.
- **Hidden Layer Transforms on Shape Layers**: For `layer.shape` (`6`), `layer.shape.text` (`7`), and `layer.shape.preset` (`8`), layer-level `position`, `scale`, and `rotation` are hidden in UI because transforms are authored on the child `Shape2D` group/path nodes instead.

### 2.3 Type-Specific Layer Details

#### Adjustment Layer (`type: 1`)
1. Inherits sequence resolution (`resolution = sequence.resolution`).
2. During `renderLayer(layer, depth)`:
   - Copies the current accumulated backdrop `screenBuffers[depth]` (all enabled tracks rendered below this layer) verbatim into `readBuffer` via `copyPass`.
   - Executes `layer.effects` sequentially on `readBuffer` ↔ `writeBuffer`.
   - Composites the resulting `readBuffer` back over `screenBuffers[depth]` using the adjustment layer's own `position`, `scale`, `rotation`, `opacity`, and `blending` mode.
3. Because it samples `screenBuffers[depth]`, an Adjustment Layer inside a Composite layer (`depth >= 1`) affects **only the sibling layers below it within that Composite**, not tracks outside the Composite.

#### Composite Layer (`type: 2`)
1. Inherits sequence resolution.
2. Skipped if `objects.length == 0`.
3. Allocates (or reuses) `screenBuffers[depth + 1]` matching the active render target dimensions, clears it to transparent black `(0, 0, 0, 0)`.
4. Iterates `i = objects.length - 1` down to `0`, recursively calling `renderLayer(objects[i], depth + 1)` (skipping scenes without a camera and empty composites).
5. Swaps `screenBuffers[depth + 1]` into `readBuffer`, applies the composite's own `effects` stack, then blends into `screenBuffers[depth]`.
6. **Nested Video Material Scheduling**: Vanilla CM3 failed to scan nested `Composite -> Scene` trees for `Video` materials. ZoidCraft MUST recursively traverse all descendant `Composite` layers (with cycle/depth guard) when scheduling video decoders for `material.video` instances.

#### Image Layer (`type: 3`)
- `childProperties.image`: static `ASSET` (`IMAGE`), default `null`.
- Decoded texture uploaded with `premultiplyAlpha = true`.
- `resolution` stores the image's natural pixel dimensions `[width, height]` (so the composite quad renders at 1:1 pixel size before `scale`).

#### 2D Shape Layers (`type: 6, 7, 8`) & `Shape2D` / `DrawOp` Model

2D vector layers rasterize onto an offscreen RGBA8 canvas (scaled by preview ratio `canvasW / layerResW`, `canvasH / layerResH`), translating the canvas origin to `(canvasW / 2, canvasH / 2)` before evaluating child `Shape2D` nodes, then uploading as a premultiplied-alpha texture.

- **`Shape2D` Types**:
  - `0`: `shape.group` — contains `objects: List<Shape2D>`, `operations: List<DrawOp>`, `properties`.
  - `1`: `shape.path` — contains vector `path`, `operations: List<DrawOp>`, `properties`.
- **`Shape2D` Properties**:
  - `name`: `TEXT` (hidden)
  - `position`: dynamic `VECTOR2`, default `[0, 0]` (pixels, +X right, +Y up)
  - `rotation`: dynamic `NUMBER`, default `0` (radians stored, degrees UI, CCW positive)
  - `scale`: dynamic `VECTOR2`, default `[1, 1]`, min `0.001`, `linkRatio: true`
- **2D Shape Canvas Transform Matrix** (mapping Y-up shape space to Y-down 2D canvas space around pixel center `(0.5, 0.5)`):
  Given position $(p_x, p_y)$, scale $(s_x, s_y)$, rotation $r$ (radians), $c = \cos r$, $s = \sin r$:
  $$\begin{bmatrix} s_x c & s_y s & -0.5 s_x c - 0.5 s_y s + p_x + 0.5 \\ -s_x s & s_y c & 0.5 s_x s - 0.5 s_y c - p_y + 0.5 \\ 0 & 0 & 1 \end{bmatrix}$$
- **Group Draw Semantics**: For each child `obj` in `group.objects`, draws `obj` and then executes every `op` in `group.operations` (allowing group-level fill/stroke to paint each child path).
- **`DrawOp` Types**:
  - `0`: `draw.fill`:
    - `color`: dynamic `COLOR`, default `[1, 1, 1]` (opaque fill `rgb(floor(255*r), floor(255*g), floor(255*b))`).
    - *CM3 JSON quirk*: In CM3's property definition, the 2nd and 3rd scalar channel display names are swapped (`Fill.R`, `Fill.B`, `Fill.G`), while indices `0, 1, 2` are always read as `(R, G, B)`. Importers MUST map by array index `[0]=R, [1]=G, [2]=B`.
  - `1`: `draw.stroke`:
    - `color`: dynamic `COLOR`, default `[0, 0, 0]` (same index order `[0]=R, [1]=G, [2]=B`).
    - `size`: dynamic `NUMBER`, default `1.0`, min `0.1` (stroke width in pixels).
    - `opacity`: dynamic `NUMBER`, default `1.0`, range `[0, 1]`.
- **Preset Shape Layer (`type: 8`)**:
  - Creates one implicit `shape.group` with `[draw.fill, draw.stroke]` and exposes `childProperties`:
    - `preset`: static `OPTION`, default `0`, items: `0: "ellipse"` (circle radius `100` centered at `(0,0)`), `1: "rectangle"` (`200 x 200` centered at `(0,0)`, i.e. `[-100, +100]^2`).
    - Aliases `transform`, `fill`, `stroke` referencing `objects[0]`.
- **2D Text Layer (`type: 7`)**:
  - Creates one implicit `shape.group` with `[draw.fill, draw.stroke]` and exposes `childProperties`:
    - `text`: static `TEXT`, default `"text"`.
    - `font`: static `ASSET` (`FONT`), default `"/assets/fonts/2d/bebas.ttf"`.
    - `size`: static `NUMBER`, default `100`, min `1`, integer (em size in pixels).
    - Aliases `transform`, `fill`, `stroke` referencing `objects[0]`.
  - Layout: converts `text` to glyphs at scale `s = size / unitsPerEm`, computes tight glyph bounding box `[minX, maxX] x [minY, maxY]`, offsets starting pen by `(-0.5*(maxX - minX) - minX, -0.5*(maxY - minY) - minY)` so the text block centers on `(0, 0)`, and emits one `shape.path` per glyph positioned at `(penX, penY)` with scale `(s, s)`.

### 2.4 Effects Stack (`PZ.effect`)

- **Numeric Built-in Effect Types**:
  - `0`: `effect.group` — macro container holding `enabled`, `customProperties` (wired via expressions to child effects), and child `objects: List<Effect>` (e.g., Twitch).
  - `1`: `effect.shader` — Custom GLSL ES 1.00 fragment shader effect with `properties.fragShader` and dynamic/static `customProperties`.
- **String-Typed Standard Effects**: Loaded by alphanumeric identifier string (e.g., `"coloroverlay"`, `"gradientoverlay"`, `"overlay"`, `"negative"`, `"grayscale"`, `"brightnesscontrast"`, `"colorgrading"`, `"colorcorrection"`, `"colorize"`, `"posterize"`, `"chromakey"`, `"lumakey"`, `"technicolor"`, `"hueshift"`, `"exposure"`, `"cubelut"`, `"fxaa"`, `"bloom"`, `"anamorphicflare"`, `"edgedetection"`, `"rgbshift"`, `"fisheye"`, `"pulse"`, `"wavy"`, `"pixelated"`, `"swirl"`, `"static"`, `"glitch"`, `"displacementmap"`, etc.).
- **Execution**: Effects run in list order (`0 .. effects.length - 1`) on `readBuffer` → `writeBuffer` within viewport `[0, 0, layerW, layerH]`, receiving `uvScale = (layerW / bufferW, layerH / bufferH)`. If `pass.needsSwap == true`, `readBuffer` and `writeBuffer` swap after the pass.

### 2.5 Premultiplied Alpha Compositing Pipeline

All internal composition buffers (`readBuffer`, `writeBuffer`, `screenBuffers[*]`) use **straight-encoded sRGB primaries stored in premultiplied-alpha form**:
$$(C_{\text{pre}}, \alpha) = (C_{\text{straight}} \cdot \alpha, \; \alpha)$$

1. **Source Ingestion**:
   - `layer.image` and `layer.shape` textures are uploaded with alpha premultiplied (`rgb *= a`).
   - `layer.texture` (video) has opaque RGB (`a = 1.0`).
   - `layer.scene` (3D WebGL/wgpu render pass) clears to `(0, 0, 0, 0)` and outputs premultiplied alpha.
2. **Effects Pass**: Runs on `readBuffer` in premultiplied RGBA; effects that manipulate color channels must un-premultiply safely (`rgb / max(a, eps)`) and re-premultiply on output.
3. **Layer Blend Pass**:
   - Renders the layer's transformed quad into `writeBuffer` with `scene.background = screenBuffers[depth]` (so untouched pixels outside the transformed quad preserve the backdrop unmodified).
   - Fragment shader samples backdrop `dst = tBG` and layer `src = tDiffuse * opacity` (scaling all 4 premultiplied channels by `opacity`), applies the active `BLEND_*` equation, and writes the premultiplied result.
   - `writeBuffer` swaps with `screenBuffers[depth]`.

### 2.6 Complete CM3 Layer Blend Modes (43 Modes)

Let $C_s, \alpha_s$ be premultiplied source (after layer `opacity`) and $C_d, \alpha_d$ be premultiplied destination (`tBG`). Straight colors are $c_s = C_s / \alpha_s$ and $c_d = C_d / \alpha_d$ (zero when $\alpha = 0$). Standard Porter-Duff & W3C Compositing / Cairo / OpenVG blend formulas apply:

| Token (`blending` value) | UI Name | Category | Output / Blend Function |
|---|---|---|---|
| `BLEND_ZERO` | zero | Porter-Duff | $(0, 0)$ |
| `BLEND_SRC` | src | Porter-Duff | $(C_s, \alpha_s)$ |
| `BLEND_DST` | dest | Porter-Duff | $(C_d, \alpha_d)$ |
| `BLEND_SRC_OVER` | src over | Porter-Duff (Default) | $C_s + C_d(1 - \alpha_s)$, $\alpha = \alpha_s + \alpha_d(1 - \alpha_s)$ |
| `BLEND_DST_OVER` | dest over | Porter-Duff | $C_d + C_s(1 - \alpha_d)$, $\alpha = \alpha_d + \alpha_s(1 - \alpha_d)$ |
| `BLEND_SRC_IN` | src in | Porter-Duff | $C_s \alpha_d$, $\alpha = \alpha_s \alpha_d$ |
| `BLEND_DST_IN` | dest in | Porter-Duff | $C_d \alpha_s$, $\alpha = \alpha_d \alpha_s$ |
| `BLEND_SRC_OUT` | src out | Porter-Duff | $C_s(1 - \alpha_d)$, $\alpha = \alpha_s(1 - \alpha_d)$ |
| `BLEND_DST_OUT` | dest out | Porter-Duff | $C_d(1 - \alpha_s)$, $\alpha = \alpha_d(1 - \alpha_s)$ |
| `BLEND_SRC_ATOP` | src atop | Porter-Duff | $C_s \alpha_d + C_d(1 - \alpha_s)$, $\alpha = \alpha_d$ |
| `BLEND_DST_ATOP` | dest atop | Porter-Duff | $C_d \alpha_s + C_s(1 - \alpha_d)$, $\alpha = \alpha_s$ |
| `BLEND_XOR` | xor | Porter-Duff | $C_s(1 - \alpha_d) + C_d(1 - \alpha_s)$, $\alpha = \alpha_s + \alpha_d - 2\alpha_s\alpha_d$ |
| `BLEND_MULTIPLY` | multiply | Separable | $B(c_d, c_s) = c_d \cdot c_s$ |
| `BLEND_SCREEN` | screen | Separable | $B = c_d + c_s - c_d c_s$ |
| `BLEND_OVERLAY` | overlay | Separable | $\text{HardLight}(c_s, c_d)$ |
| `BLEND_DARKEN` | darken | Separable | $\min(c_d, c_s)$ |
| `BLEND_LIGHTEN` | lighten | Separable | $\max(c_d, c_s)$ |
| `BLEND_COLORDODGE` | color dodge | Separable | $c_d = 0 \to 0$; $c_s = 1 \to 1$; else $\min(1, c_d / (1 - c_s))$ |
| `BLEND_COLORBURN` | color burn | Separable | $c_d = 1 \to 1$; $c_s = 0 \to 0$; else $1 - \min(1, (1 - c_d)/c_s)$ |
| `BLEND_HARDLIGHT` | hard light | Separable | $c_s \le 0.5 \to 2 c_d c_s$; else $1 - 2(1 - c_d)(1 - c_s)$ |
| `BLEND_SOFTLIGHT` | soft light | Separable | W3C Pegtop/Photoshop soft light $B(c_d, c_s)$ |
| `BLEND_DIFFERENCE` | difference | Separable | $|c_d - c_s|$ |
| `BLEND_EXCLUSION` | exclusion | Separable | $c_d + c_s - 2 c_d c_s$ |
| `BLEND_INVERT` | invert | Extended | $C = (1 - c_d)\alpha_s\alpha_d + C_d(1-\alpha_s)$ (inverts destination where source has alpha) |
| `BLEND_INVERT_RGB` | invert rgb | Extended | Channel-wise $c_s(1 - c_d) + (1 - c_s)c_d$ |
| `BLEND_LINEARDODGE` | linear dodge | Separable | $\min(1, c_d + c_s)$ |
| `BLEND_LINEARBURN` | linear burn | Separable | $\max(0, c_d + c_s - 1)$ |
| `BLEND_VIVIDLIGHT` | vivid light | Separable | $c_s \le 0.5 \to \text{ColorBurn}(c_d, 2c_s)$; else $\text{ColorDodge}(c_d, 2c_s - 1)$ |
| `BLEND_LINEARLIGHT` | linear light | Separable | $\text{clamp}(c_d + 2c_s - 1, 0, 1)$ |
| `BLEND_PINLIGHT` | pin light | Separable | $c_s \le 0.5 \to \min(c_d, 2c_s)$; else $\max(c_d, 2c_s - 1)$ |
| `BLEND_HARDMIX` | hard mix | Separable | $\text{step}(1.0, c_d + c_s)$ (0 or 1 per channel) |
| `BLEND_HUE` | hue | Non-separable HSL | $\text{SetLum}(\text{SetSat}(c_s, \text{Sat}(c_d)), \text{Lum}(c_d))$ |
| `BLEND_SATURATION` | saturation | Non-separable HSL | $\text{SetLum}(\text{SetSat}(c_d, \text{Sat}(c_s)), \text{Lum}(c_d))$ |
| `BLEND_COLOR` | color | Non-separable HSL | $\text{SetLum}(c_s, \text{Lum}(c_d))$ |
| `BLEND_LUMINOSITY` | luminosity | Non-separable HSL | $\text{SetLum}(c_d, \text{Lum}(c_s))$ |
| `BLEND_PLUS` | add | Arithmetic | $\min(1, C_s + C_d)$, $\alpha = \min(1, \alpha_s + \alpha_d)$ |
| `BLEND_PLUS_DARKER` | add darker | Arithmetic | $\max(0, \alpha_s + \alpha_d - 1)$ Porter-Duff PlusDarker |
| `BLEND_MINUS` | subtract | Arithmetic | $\max(0, C_d - C_s)$ |
| `BLEND_CONTRAST` | contrast | Extended | W3C/Cairo contrast blend around 0.5 midpoint |
| `BLEND_INVERT_OVG` | invert ovg | OpenVG | OpenVG invert: $\alpha_s(\alpha_d - C_d) + C_d(1 - \alpha_s)$ |
| `BLEND_RED` | red | Channel | Replaces R channel with source over; preserves G, B from dest |
| `BLEND_GREEN` | green | Channel | Replaces G channel with source over; preserves R, B from dest |
| `BLEND_BLUE` | blue | Channel | Replaces B channel with source over; preserves R, G from dest |

For all separable/HSL blend modes $B(c_d, c_s)$, the premultiplied compositing formula is:
$$C_{\text{out}} = (1 - \alpha_d) C_s + (1 - \alpha_s) C_d + \alpha_s \alpha_d B(c_d, c_s), \qquad \alpha_{\text{out}} = \alpha_s + \alpha_d(1 - \alpha_s)$$

---

## 3. CM3 3D Object Types

### 3.1 3D Object Type Registry (Exact `.pz` Numeric Type IDs)

| Type ID | Internal Class | Default Name | Description |
|---:|---|---|---|
| `0` | `object3d.shape` | `"Shape"` (or sub-type name) | Procedural 3D primitive mesh or custom geometry (`objectType` 1..7, 99) |
| `1` | `object3d.text` | `"Text"` | Extruded 3D glyph mesh with bevels and alignment |
| `2` | `object3d.model` | `"Model"` | Multi-mesh 3D model (CM3 presets `1, 2`; ZoidCraft OBJ/glTF) |
| `3` | `object3d.light` | `"Light"` (or sub-type name) | Spot (`1`), Point (`2`), Directional (`3`), Hemisphere (`4`) light |
| `4` | `object3d.particles` | `"Particles"` | Stateless deterministic GPU point-sprite particle system |
| `5` | `object3d.group` | `"Group"` | Transform hierarchy container, visibility mask, custom properties |
| `6` | `object3d.camera` | `"Camera"` | Perspective (`1`) or Orthographic (`2`) camera with procedural shake |
| `99` | `object3d.dummy` | `"3D Object"` | Fallback placeholder for unknown/missing object types |

> **Zoidium `enabled` Extension (All 3D Objects)**: Any 3D object may carry a dynamic `enabled` property (`OPTION`: `0: off`, `1: on`, default `1` when absent). When `0`, the object is hidden and its per-frame update and asset preparation are skipped. A disabled `Group` skips its children unless the subtree contains an active Camera.

### 3.2 Common 3D Transform Properties

Shared by `Group`, `Shape`, `Text`, `Model`, and `Camera` (Camera omits `scale`; Model omits `eulerOrder` in CM3 1.0.102 and defaults to `XYZ`):

| Key | UI Name | Type | Animated | Default | Constraints / Units |
|---|---|---|---|---|---|
| `position` | Position | `VECTOR3` (2) | Yes (group) | `[0, 0, 0]` (`[0, 0, 80]` for Camera) | World/local 3D units |
| `rotation` | Rotation | `VECTOR3` (2) | Yes (group) | `[0, 0, 0]` | **Radians stored**, UI degrees (`scaleFactor = π/180`) |
| `scale` | Scale | `VECTOR3` (2) | Yes (group) | `[1, 1, 1]` | Min `0.001`, `linkRatio: true` |
| `eulerOrder` | Rotation order | `LIST` (9) | No | `"XYZ"` | `"XYZ"`, `"YZX"`, `"ZXY"`, `"XZY"`, `"YXZ"`, `"ZYX"` |

- **Transform Composition**: Local matrix is $M_{\text{local}} = T(\text{position}) \cdot R_{\text{eulerOrder}}(\text{rotation}) \cdot S(\text{scale})$ (intrinsic Three.js r91 Euler order, default `XYZ`: $R = R_x R_y R_z$). World matrix is $M_{\text{world}} = M_{\text{parent}} \cdot M_{\text{local}}$.

### 3.3 Group (`type: 5`)

| Key | UI Name | Type | Animated | Default | Values / Notes |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | `"Group"` | Hidden |
| `enabled` | Enabled | `OPTION` (6) | Yes | `1` (`on`) | `0: off`, `1: on` |
| `reflectionVisibility` | Visibility | `OPTION` (6) | No | `0` (`normal`) | `0`: main layer 0 only; `1`: reflection layer 10 only; `2`: layers 0 & 10 |
| `position`, `rotation`, `scale`, `eulerOrder` | — | — | — | Standard | See §3.2 |

- **Extra Serialized Fields**:
  - `customProperties: List<DynamicProperty>` — user-defined dynamic properties accessible by child expressions via `properties.<name>`.
  - `objects: List<Object3D>` — child 3D objects.
- **Layer Mask Propagation**: Non-group direct children inherit their parent group's Three.js visibility layer mask (`bit 0` = main camera, `bit 10` = reflection `CubeCamera`). Nested groups control their own mask independently.

### 3.4 3D Text (`type: 1`)

| Key | UI Name | Type | Animated | Default | Constraints / Notes |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | `"Text"` | Hidden |
| `text` | Text | `TEXT` (7) | No | `"text"` | UTF-8 string |
| `font` | Font | `ASSET` (8) | No | `"/assets/fonts/2d/bebas.ttf"` | `AssetType.FONT` (`.ttf,.otf,.woff,.woff2`) |
| `size` | Size | `VECTOR2` (1) | No | `[20, 3]` | `[0]=height` (glyph cap size), `[1]=thickness` (extrude depth); int `[1, 200]` |
| `detail` | Detail | `NUMBER` (0) | No | `5` | `curveSegments` per Bezier glyph curve; int `[1, 200]` |
| `bevel` | Bevel | `OPTION` (6) | No | `0` (`off`) | `0: off`, `1: on` |
| `bevelSize` | Bevel size | `VECTOR2` (1) | No | `[0.1, 0.5]` | Min `0.01`, step `0.1`. **CM3 mapping quirk**: `[0]` → `bevelThickness` (Z depth, UI subtitle `"size"`), `[1]` → `bevelSize` (XY outset, UI subtitle `"thickness"`). |
| `positionMode` | Center point | `OPTION` (6) | No | `0` (`center`) | `0: center`, `1: left`, `2: right`, `3: top`, `4: bottom`, `5: front`, `6: back` |
| `position`, `rotation`, `scale`, `eulerOrder` | — | — | — | Standard | See §3.2 |

- **Shadow Flags**: `castShadow = true`, `receiveShadow = true`.
- **Material Slots**:
  - CM3 serializes a single `material` object (default `"singlecolor"`), while extruded geometry tags front/back caps as face material index `0` and side walls/bevel as index `1`.
  - ZoidCraft supports a single material (applied to all faces for CM3 parity) AND optional per-slot overrides (`front`, `side`, `bevel` material slots in `.zcraft`).
- **Center Point / Alignment Math** (given raw extruded mesh AABB $[\mathbf{b}_{\min}, \mathbf{b}_{\max}]$):
  Compute base centering vector $\mathbf{t} = -\frac{1}{2}(\mathbf{b}_{\min} + \mathbf{b}_{\max})$, then adjust by `positionMode`:

| `positionMode` | Label | Final Geometry Translation Vector |
|---:|---|---|
| `0` | `center` | $(t_x, t_y, t_z)$ |
| `1` | `left` | $(0, t_y, t_z)$ |
| `2` | `right` | $(2t_x, t_y, t_z)$ |
| `3` | `top` | $(t_x, 2t_y, t_z)$ |
| `4` | `bottom` | $(t_x, 0, t_z)$ |
| `5` | `front` | $(t_x, t_y, 2t_z)$ |
| `6` | `back` | $(t_x, t_y, 0)$ |

- **Normal & UV Generation (Avoiding `WorldUVGenerator` Shear)**:
  - *CM3 Legacy Bug (`uvMode = "legacy"`)*: Three.js r91 `ExtrudeGeometry` defaults to `WorldUVGenerator`, which emits raw object-space cap UVs $(x, y)$ (causing ~12×13 tiles at `Repeat = (1,1)` on 20-unit text) and planar side-wall UVs $(x \text{ or } y, 1 - z)$ (producing negative $V \in [1-\text{thickness}, 1]$ and up to **7.66× texture shear** on curved glyph outlines).
  - *ZoidCraft Default (`uvMode = "current"`, matching Zoidium Material+ UV Layout)*: Use a **single uniform texel density** across caps, bevels, and side walls scaled by the mesh's front height $H = b_{\max,y} - b_{\min,y}$ (falling back to $1.0$ if $H \le 0$):
    - **Front / Back Caps**: $u = (x - b_{\min,x}) / H$, $v = (y - b_{\min,y}) / H$.
    - **Side Walls & Bevels**: Walk each closed glyph contour loop to accumulate Euclidean arc length $s$ along the perimeter, mapping $u = s / H$ and $v = (z - b_{\min,z}) / H$.
    - **Normals**: Flat face normals on caps $(0, 0, \pm 1)$ with sharp crease split between cap and wall/bevel; smooth averaged normals along curved wall segments (or flat normals when material sets flat shading).
  - Objects imported from `.pz` without `zoidiumUvMode` MAY select `"legacy"` when pixel-exact legacy texture tiling is requested, while new ZoidCraft objects always default to `"current"`.

### 3.5 Basic 3D Shapes (`type: 0`)

All shapes serialize `objectType`, `properties` (including nested `properties.geometryProperties`), and `material` (default `"singlecolor"`). All generated geometries are auto-centered around their AABB center and have `castShadow = true`, `receiveShadow = true`.

| `objectType` | Default Name | Geometry Properties & Exact Defaults | Geometry Specification |
|---:|---|---|---|
| `1` | `"Box"` | `box_size`: V3 `[10, 10, 10]` (width, height, depth; int, min 0) | Axis-aligned box centered at origin |
| `2` | `"Cylinder"` | `cylinder_size`: V3 `[10, 10, 10]` (topRadius, bottomRadius, height; int, min 0)<br>`cylinder_detail`: NUM `10` (radial segments; int, min 0)<br>`cylinder_openEnded`: OPT `0` (`0: open`, `1: closed`) | Cylinder/frustum/cone along Y. **Note inversion**: `openEnded = (cylinder_openEnded == 0)` (`0` omits caps; `1` includes caps; heightSegments = 1). Setting `topRadius = 0` yields a Cone. |
| `3` | `"Rectangle"` | `rect_size`: V2 `[10, 10]` (width, height; int, min 0) | 1×1 subdivided quad in XY plane (`z = 0`, normal `+Z`) |
| `4` | `"Circle"` | `circle_size`: NUM `10` (radius; int, min 0)<br>`circle_detail`: NUM `10` (segments; int, min 1) | Disk in XY plane (`θ ∈ [0, 2π]`, normal `+Z`) |
| `5` | `"Sphere"` | `sphere_size`: NUM `10` (radius; int, min 0)<br>`sphere_detail`: V2 `[10, 10]` (widthSegs, heightSegs; int, min 1) | UV sphere (`φ ∈ [0, 2π]`, `θ ∈ [0, π]`) |
| `6` | `"Donut"` | `donut_size`: V2 `[10, 10]` (major radius, tube radius; int, min 0)<br>`donut_detail`: V2 `[10, 10]` (radialSegs, tubularSegs; int, min 1) | Torus in XY plane (`arc = 2π`) |
| `7` | `"Wire"` | `path_size`: V3 `[10, 10, 10]` (horiz, vert, stretch)<br>`path_thickness`: NUM `2` (min 1)<br>`path_loops`: NUM `5` (min 1)<br>`path_detail`: V2 `[10, 10]` (innerSegs, outerSegs; min 1) | Circular tube of radius `path_thickness` (`innerSegs` vertices) extruded along a Catmull-Rom helix of `loops * outerSegs` points $(S_x \cos\frac{2\pi k}{\text{outer}}, S_y \sin\frac{2\pi k}{\text{outer}}, S_z \frac{k}{\text{outer}})$ with `10 * outerSegs` steps |
| `99` | `"Geometry"` | `customGeometry`: `ASSET` (`GEOMETRY`, default `null`) | Imported custom mesh asset (OBJ / Three JSON / glTF) |

> **ZoidCraft Native Shape Extensions** (serialized in `.zcraft` by string kind, not conflicting with CM3 numeric IDs): `Cone` (explicit alias of Cylinder with `topRadius = 0`), `Icosahedron` (`radius`, `detail`), `Octahedron`, `Dodecahedron`, `Tetrahedron`, `Capsule`.

### 3.6 Lights & Shadows (`type: 3`)

Serialized fields: `type: 3`, `objectType`, `properties`.

| `objectType` | Default Name | Properties & Exact Defaults | Fixed / Implicit Parameters | Casts Shadow |
|---:|---|---|---|---|
| `1` | `"Light"` (Spot) | `color`: RGB `[1,1,1]`<br>`position`: V3 `[0, 10, 0]`<br>`target`: V3 `[0, 0, 0]`<br>`intensity`: NUM `1.0` (`[0, 100]`)<br>`angle`: NUM `60` (**degrees**, `[0, 90]`) | `distance = 0` (infinite), `penumbra = 0.5`, `decay = 1` | Yes |
| `2` | `"Point Light"` | `color`: RGB `[1,1,1]`<br>`position`: V3 `[0, 10, 0]`<br>`intensity`: NUM `1.0` (`[0, 100]`) | `distance = 0` (infinite), `decay = 1` | Yes |
| `3` | `"Directional Light"` | `color`: RGB `[1,1,1]`<br>`position`: V3 `[0, 10, 0]`<br>`target`: V3 `[0, 0, 0]`<br>`intensity`: NUM `1.0` (`[0, 100]`) | Parallel rays from `position` toward `target` | Yes |
| `4` | `"Hemisphere Light"` | `color` (UI `"Sky color"`): RGB `[1,1,1]`<br>`groundColor` (UI `"Ground color"`): RGB `[1,1,1]`<br>`intensity`: NUM `1.0` (`[0, 100]`) | Up-axis `+Y` gradient between sky and ground color | No |

> **Ambient Light**: CM3 1.0.102 does not define a separate `AmbientLight` `objectType` (a `HemisphereLight` with identical sky and ground colors produces uniform ambient lighting). ZoidCraft MAY expose an `Ambient` light kind in `.zcraft` (`color`, `intensity`) and map it cleanly on export.

#### Shadow Map Specification (CM3 Defaults & Quality Modes)

- **Shadow Filter**: Percentage-Closer Filtering Soft Shadow Map (`PCFSoftShadowMap`, 3×3 / 5×5 kernel).
- **Default Shadow Parameters** (applied to Spot, Point, Directional lights):
  - `mapSize`: `1024 × 1024` (ZoidCraft `.zcraft` may allow per-light overrides: `512`, `1024`, `2048`, `4096`).
  - `shadow.camera.near = 5.0`, `shadow.camera.far = 1500.0`, `shadow.camera.fov = 60°` (perspective shadow frustum for SpotLight; orthographic frustum for DirectionalLight; 6-face cubemap for PointLight).
  - `shadow.bias = 0.0` (ZoidCraft `.zcraft` exposes configurable `shadowBias`, `shadowNormalBias`, `shadowRadius`, and `castShadow` toggle).
- **Scene Shadow Quality Mode (`zoidiumShadowMode`)**:
  - `"legacy"`: Raw depth pass with zero polygon offset (exact CM3 r91 depth behavior).
  - `"current"` (ZoidCraft default for new scenes): Slope-scaled polygon offset + packed depth bias correction to eliminate self-shadowing acne while keeping contact shadows tight.

### 3.7 Camera & Procedural Shake (`type: 6`)

Serialized fields: `type: 6`, `objectType`, `properties` (including nested `properties.shake`).

| `objectType` | Default Name | Projection Parameters |
|---:|---|---|
| `1` | `"Perspective Camera"` | Vertical `fov = 60°`, `aspect = seqW / seqH`, `near = 0.1`, `far = 5000.0` |
| `2` | `"Orthographic Camera"` | `left = -0.05 * seqW`, `right = +0.05 * seqW`, `top = +0.05 * seqH`, `bottom = -0.05 * seqH`, `near = 0.1`, `far = 5000.0` |

| Key | UI Name | Type | Animated | Default | Constraints / Notes |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | `"Perspective Camera"` / `"Orthographic Camera"` | Hidden |
| `position` | Position | `VECTOR3` (2) | Yes | `[0, 0, 80]` | Camera looks along local `-Z` |
| `rotation` | Rotation | `VECTOR3` (2) | Yes | `[0, 0, 0]` | Radians stored, UI degrees |
| `eulerOrder` | Rotation order | `LIST` (9) | No | `"XYZ"` | Standard 6 orders |
| `shakeAmplitude` | Shake amount | `NUMBER` (0) | Yes | `0.0` | `[0.0, 1.0]` |
| `shakeSpeed` | Shake speed | `NUMBER` (0) | Yes | `30.0` | `[0.0, ∞)` (frames/sec reference = 30) |
| `shake.mode` | Shake mode | `OPTION` (6) | No | `0` (`rotate`) | `0: rotate`, `1: move` |
| `shake.amplitude` | Amplitude | `VECTOR4` (3) | No | `[12, 12, 40, 20]` | Min `0`; channels `[X, Y, Z, Roll]` |
| `shake.frequency` | Frequency | `VECTOR4` (3) | No | `[0.6, 0.8, 0.7, 0.9]` | Min `0` |
| `shake.phase` | Phase | `VECTOR4` (3) | No | `[0, 11, 20, 31]` | Min `0` |
| `shake.noise` | Noise | `VECTOR4` (3) | No | `[0.9, 0.9, 0.9, 0.55]` | `[0, 4]` |
| `shake.smooth` | Smooth | `VECTOR4` (3) | No | `[1, 1, 1, 1]` | `0` = linear, `1` = cubic smoothstep |

> **ZoidCraft Camera Extensions** (`.zcraft`): Optional animatable `fov` (default `60°`), `near` (`0.1`), `far` (`5000`), `zoom` (`1.0`), and `target` / `lookAt` mode (or Zoidium `Track To` constraint).

#### Exact Procedural Camera Shake Algorithm

Given clip-local frame $f$, $A = \text{shakeAmplitude}(f)$, and $t = f \cdot (\text{shakeSpeed}(f) / 30)$:

For each channel $i \in \{0, 1, 2, 3\}$:
1. $u = t \cdot \text{freq}_i + \text{phase}_i$, $k_0 = \lfloor u \rfloor$, $k_1 = k_0 + 1$, $h = u - k_0$.
2. If $\text{smooth}_i \ne 0$, set $h \leftarrow h^2 (3 - 2h)$.
3. Define deterministic sample value for integer $k$:
   $$W(k) = 2(k \mathbin{\&} 1) - 1, \qquad N(k) = \text{fract}(10000 \cdot \sin(k)) - 0.5, \qquad V_i(k) = W(k) + \text{noise}_i \cdot N(k)$$
4. Channel output: $q_i = \text{amp}_i \cdot \bigl((1 - h) V_i(k_0) + h V_i(k_1)\bigr)$.
5. Apply sequentially to the camera's local transform after setting `position` and `rotation`:
   - If `shake.mode == 0` (`rotate`): local `rotateX(q_0 * A * 0.1)`, then `rotateY(q_1 * A * 0.1)`.
   - If `shake.mode == 1` (`move`): local `translateX(q_0 * A)`, then `translateY(q_1 * A)`.
   - Always: local `translateZ(q_2 * A)`, then `rotateZ(q_3 * A * 0.1)`.

### 3.8 Particles (`type: 4`)

CM3 particles are **100% stateless and deterministic** — every particle with integer index $\text{pid} \in \{0, \dots, N-1\}$ computes its exact position, rotation, size, and color directly from `time` without frame-to-frame simulation state.

| Key | UI Name | Type | Animated | Default | Constraints / Notes |
|---|---|---|---|---|---|
| `name` | Name | `TEXT` (7) | No | `"Particles"` | Hidden |
| `time` | Time | `NUMBER` (0) | Yes | `0.0` | Simulation clock in seconds |
| `color` | Color | `GRADIENT` (5) | No | `[{position:0, color:"rgba(255,255,255,1.0)"}]` | Baked to 1D premultiplied RGBA LUT over normalized lifetime |
| `size` | Size | `GRADIENT` (5) | No | `[{position:0, color:"rgba(255,255,255,1.0)"}]` | Red channel (`[0, 1]`) drives size multiplier over lifetime |
| `number` | Number of particles | `NUMBER` (0) | No | `0` | Integer count $N$ (UI min `1`) |
| `rate` | Emitter rate | `NUMBER` (0) | No | `0` | Particles spawned per second (UI min `1`) |
| `lifetime` | Particle lifetime | `NUMBER` (0) | No | `0` | Seconds (`[0.01, 1000]`) |
| `pdist` | Position mode | `OPTION` (6) | No | `0` (`box`) | `0: box`, `1: sphere`, `2: disk` |
| `ipos` | Position | `VECTOR3` (2) | No | `[0, 0, 0]` | Emitter origin |
| `pspread` | Position spread | `VECTOR3` (2) | No | `[1, 1, 1]` | Min `0`; box extents when `pdist == 0` |
| `iradius` | Radius | `NUMBER` (0) | No | `0` | Base radius when `pdist ∈ {1, 2}` |
| `rspread` | Radius spread | `NUMBER` (0) | No | `0` | Radial thickness when `pdist ∈ {1, 2}` |
| `vdist` | Speed mode | `OPTION` (6) | No | `0` (`box`) | `0: box`, `1: sphere` (radial along $\hat{p}$) |
| `ivel` | Speed | `VECTOR3` (2) | No | `[0, 0, 0]` | Base linear velocity (`vdist == 0`) |
| `vspread` | Speed spread | `VECTOR3` (2) | No | `[0, 0, 0]` | Min `0`; velocity spread (`vdist == 0`) |
| `irvel` | Radial speed | `NUMBER` (0) | No | `0` | Min `-1000`; radial velocity (`vdist == 1`) |
| `rvspread` | Radial speed spread | `NUMBER` (0) | No | `0` | Min `0`; radial velocity spread (`vdist == 1`) |
| `accel` | Gravity | `VECTOR3` (2) | No | `[0, 0, 0]` | Constant acceleration vector |
| `iang` | Rotation | `NUMBER` (0) | No | `0` | **Degrees** `[0, 360]`, converted to radians for shader |
| `aspread` | Rotation spread | `NUMBER` (0) | No | `0` | **Degrees** `[0, 360]`, converted to radians |
| `angvel` | Rotation speed | `NUMBER` (0) | No | `0` | **Degrees/sec**, converted to radians/sec |
| `avspread` | Rotation speed spread | `NUMBER` (0) | No | `0` | **Degrees/sec**, converted to radians/sec |
| `blending` | Blending mode | `OPTION` (6) | No | `0` (`normal`) | `0: normal`, `1: add` |
| `texture` | Texture | `ASSET` (8) | No | `null` | `AssetType.IMAGE` (36 built-in presets or custom image) |

#### Deterministic Particle Evaluation Equations

Define the deterministic 2D hash PRNG:
$$R(a, b) = \text{fract}\!\left(\sin(12.9898 a + 4.1414 b) \cdot 43758.5453\right) \in [0, 1)$$

For particle index $i = \text{pid}$ and clock $t = \text{time}$:
1. **Lifecycle**:
   - $\text{delay} = i / \text{rate}$
   - $\tau = (t - \text{delay}) \bmod \text{lifetime} \in [0, \text{lifetime})$
   - $c = \lfloor (t - \text{delay}) / \text{lifetime} \rfloor + 1$ (emission cycle seed, $k = i \cdot c$)
   - Normalized age $u = \tau / \text{lifetime}$; alive mask $m = \mathbf{1}[t \ge \text{delay}]$.
2. **Spawn Offset $\mathbf{p}$**:
   - `pdist == 0` (`box`): $\mathbf{p} = \bigl(R(k, i) - 0.5,\; R(k, 2i) - 0.5,\; R(k, 3i) - 0.5\bigr) \odot \text{pspread}$
   - `pdist == 1` (`sphere`) or `2` (`disk`):
     - $\phi = 2\pi \cdot R(k, i)$
     - $\theta = \pi \cdot z_{\text{mode}} \cdot R(k, 2i)$, where $z_{\text{mode}} = 1$ for `sphere` and $0$ for `disk`
     - $\rho = \text{iradius} + (R(k, 3i) - 0.5) \cdot \text{rspread}$
     - $\mathbf{p} = \rho \, (\sin\phi \cos\theta,\; \sin\phi \sin\theta,\; \cos\phi)$ (note: `disk` lies in the horizontal XZ plane)
3. **Initial Velocity $\mathbf{v}$**:
   - `vdist == 0` (`box`): $\mathbf{v} = \text{ivel} + \bigl(R(k, 3i) - 0.5,\; R(k, 2i) - 0.5,\; R(k, i) - 0.5\bigr) \odot \text{vspread}$
   - `vdist == 1` (`sphere`): $\mathbf{v} = \bigl(\text{irvel} + (R(k, 3i) - 0.5) \cdot \text{rvspread}\bigr) \cdot \frac{\mathbf{p}}{\|\mathbf{p}\|}$
4. **Kinematics, Sprite Angle & Point Size**:
   - $\mathbf{P}(\tau) = \frac{1}{2}\text{accel}\,\tau^2 + \mathbf{v}\tau + \mathbf{p} + \text{ipos}$
   - $\alpha_0 = \text{iang} + (R(k, i) - 0.5)\,\text{aspread}$, $\omega = \text{angvel} + (R(k, 10) - 0.5)\,\text{avspread}$, $\text{angle} = \alpha_0 + \omega \tau$
   - View-space perspective sprite size (in pixels):
     $$\text{pointSize} = \frac{\text{sizeLUT}(u)_r \cdot \text{resolution}_x \cdot 50.0}{\|\mathbf{P}_{\text{view}}\|}$$
   - Fragment UV rotates point-sprite coordinate around `(0.5, 0.5)` by `-angle` (with `y = 1 - gl_PointCoord.y`), multiplies sampled texture by $m \cdot \text{colorLUT}(u)$.

### 3.9 Custom & Preset 3D Models (`type: 2` and `Shape objectType: 99`)

- **CM3 `object3d.model` (`type: 2`)**:
  - Properties: `position`, `rotation` (radians), `scale`, plus `modelProperties`.
  - Built-in CM3 presets: `objectType: 1` (`"AK47"`, Phong-shaded preset mesh), `objectType: 2` (`"Mystery box"`, two-part Phong mesh with animated `modelProperties.lidrotation` in `[0, 100]`, mapping to `lid.rotation.x = 0.032 * lidrotation`).
- **Custom Geometry Import**:
  - In CM3 `.pz` files, user-imported 3D meshes are stored as `object3d.shape` (`type: 0`) with `objectType: 99` and `geometryProperties.customGeometry` pointing to a `GEOMETRY` asset.
  - In ZoidCraft `.zcraft`, `Model` supports **OBJ** (with optional MTL) and **glTF 2.0 / GLB** (meshes, normals, tangents, UVs, PBR metallic-roughness materials, and node hierarchy), auto-computing missing vertex normals and binding embedded or overridden ZoidCraft materials.

### 3.10 3D Constraints (Zoidium Compatibility)

Optional properties stored inside any constrained 3D object's `properties` (omitted when unused):

- **Target Identity**: `zoidiumId` (UUID string assigned only to objects referenced as targets; regenerated on duplicate with subtree-internal reference remapping).
- **Track To Constraint**:
  - `zoidiumTrackTo`: target UUID string
  - `zoidiumTrackAxis`: `OPTION` (`0: +X`, `1: +Y`, `2: +Z`, `3: -X`, `4: -Y`, `5: -Z`)
  - `zoidiumUpAxis`: `OPTION` (`0: X`, `1: Y`, `2: Z`)
  - `zoidiumInfluence`: dynamic `NUMBER` (`[0, 1]`)
- **Follow Constraint**:
  - `zoidiumFollow`: target UUID string
  - `zoidiumFollowSpace`, `zoidiumFollowPosition`, `zoidiumFollowRotation`, `zoidiumFollowScale`
  - Offsets: dynamic vectors `zoidiumFollowPositionOffset`, `zoidiumFollowRotationOffset`, `zoidiumFollowScaleOffset`
  - `zoidiumFollowFallback`: local pose JSON snapshot retained if the target is deleted.
- Missing targets or dependency cycles fall back cleanly to the object's own authored pose without mutating keyframes.

---

## 4. Materials & Shading Models

### 4.1 Native CM3 Material Types

Materials are identified by string `type` in `.pz` JSON (`{ "type": "<id>", "properties": { ... } }`).

#### 1. `"singlecolor"` — Single Color (Unlit)
- **Shading Model**: Unlit (`MeshBasicMaterial`), `side = Both (2)`, `premultipliedAlpha = true`.
- **Properties**:

| Key | UI Name | Type | Animated | Default | Range |
|---|---|---|---|---|---|
| `color` | Color | `COLOR` (4) | Yes (group) | `[1, 1, 1]` | `[0, 1]` per channel |

#### 2. `"texture"` — Image Texture (Unlit)
- **Shading Model**: Unlit (`MeshBasicMaterial`), base color white `(1, 1, 1)`.
- **Properties**:

| Key | UI Name | Type | Animated | Default | Constraints / Values |
|---|---|---|---|---|---|
| `texture` | Image | `ASSET` (8) | No | `null` | `AssetType.IMAGE` |
| `wrap` | Wrap | `OPTION` (6) | No | `1` (`tile`) | `0: none` (ClampToEdge), `1: tile` (Repeat), `2: reflect` (MirroredRepeat) |
| `repeat` | Repeat | `VECTOR2` (1) | Yes (group) | `[1, 1]` | `linkRatio: true`, `decimals: 3` |
| `offset` | Offset | `VECTOR2` (1) | Yes (group) | `[0, 0]` | `decimals: 3` |
| `center` | Center | `VECTOR2` (1) | Yes (group) | `[0, 0]` | UV rotation pivot, `decimals: 3` |
| `rotation` | Rotation | `NUMBER` (0) | Yes | `0` | **Radians stored**, UI degrees (`scaleFactor = π/180`) |
| `side` | Render side | `OPTION` (6) | No | `0` (`front`) | `0: front`, `1: back`, `2: both` |

#### 3. `"video"` — Video Texture (Unlit)
- **Shading Model**: Unlit (`MeshBasicMaterial`), modulated by `color` and `opacity`.
- **Properties**:

| Key | UI Name | Type | Animated | Default | Constraints / Values |
|---|---|---|---|---|---|
| `video` | Video | `ASSET` (8) | No | `null` | `AssetType.AV` |
| `startOffset` | Start offset (frames) | `NUMBER` (0) | No | `0` | Integer, min `0`; clip-local frame when playback starts |
| `offset` | Media offset (frames) | `NUMBER` (0) | No | `0` | Integer, min `0`; initial media frame offset |
| `color` | Color | `COLOR` (4) | Yes (group) | `[1, 1, 1]` | `[0, 1]` RGB tint |
| `transparent` | Transparency | `OPTION` (6) | No | `0` (`off`) | `0: off`, `1: on` |
| `opacity` | Opacity | `NUMBER` (0) | Yes | `1.0` | `[0, 1]` |
| `blending` | Blending | `OPTION` (6) | No | `1` (`normal`) | `0: none`, `1: normal`, `2: additive`, `3: subtractive`, `4: multiply` |

#### 4. `"custom"` — Custom Material (Standard Metallic-Roughness PBR)
- **Shading Model**: PBR Metallic-Roughness (`MeshStandardMaterial`).
- **Properties**:

| Key | UI Name | Type | Animated | Default | Constraints / Values |
|---|---|---|---|---|---|
| `color` | Color | `COLOR` (4) | Yes (group) | `[1, 1, 1]` | Base albedo `[0, 1]` |
| `emissive` | Emissive | `COLOR` (4) | Yes (group) | `[0, 0, 0]` | Self-illumination `[0, 1]` |
| `roughness` | Roughness | `NUMBER` (0) | No | `0.5` | `[0, 1]` (animatable in `.zcraft`) |
| `metalness` | Metalness | `NUMBER` (0) | No | `0.5` | `[0, 1]` (animatable in `.zcraft`) |
| `texture` | Texture | `ASSET` (8) | No | `null` | Albedo map (`AssetType.IMAGE`) |
| `transparent` | Transparency | `OPTION` (6) | No | `0` (`off`) | `0: off`, `1: on` |
| `opacity` | Opacity | `NUMBER` (0) | Yes | `1.0` | `[0, 1]` |
| `blending` | Blending | `OPTION` (6) | No | `1` (`normal`) | `0: none`, `1: normal`, `2: additive`, `3: subtractive`, `4: multiply` |
| `side` | Render side | `OPTION` (6) | No | `0` (`front`) | `0: front`, `1: back`, `2: both` |
| `normalMap` | Normal map | `ASSET` (8) | No | `null` | Tangent-space normal map (`AssetType.IMAGE`) |
| `normalScale` | Normal scale | `NUMBER` (0) | No | `1.0` | Min `0`, applied equally to `(x, y)` |
| `wrap` | Wrap | `OPTION` (6) | No | `1` (`tile`) | Applied to both `texture` and `normalMap` (`0: Clamp`, `1: Repeat`, `2: Mirror`) |
| `repeat` | Repeat | `VECTOR2` (1) | Yes (group) | `[1, 1]` | Applied to both `texture` and `normalMap`; `linkRatio: true` |
| `offset` | Offset | `VECTOR2` (1) | Yes (group) | `[0, 0]` | Applied to both `texture` and `normalMap` |
| `center` | Center | `VECTOR2` (1) | Yes (group) | `[0, 0]` | Applied to both `texture` and `normalMap` |
| `rotation` | Rotation | `NUMBER` (0) | Yes | `0` | Radians stored (UI degrees), applied to both maps |
| `reflection` | Reflection | `OPTION` (6) | No | `0` (`off`) | `0: off`, `1: on` — binds scene's dynamic `256×256` reflection cubemap (`CubeCamera` rendering layer 10 objects, `near=0.1, far=5000`) |

### 4.2 Shading Models, Render States & UV Transform

ZoidCraft's 3D renderer (`zc-render`) supports the following shading models and per-material render states:

| Shading Model | Usage | Lighting Equation |
|---|---|---|
| **Basic / Unlit** | `singlecolor`, `texture`, `video` | `color * albedoTex`, unaffected by lights/shadows |
| **Standard (PBR)** | `custom`, glTF, Material+ PBR+ | Cook-Torrance GGX specular + Lambert diffuse, metallic-roughness workflow, emissive, tangent-space normal map, optional IBL reflection cubemap |
| **Lambert** | ZoidCraft / Material+ extension | Diffuse-only Gouraud/per-pixel Lambertian + emissive |
| **Phong** | CM3 preset models (`AK47`, `Mystery box`) & extension | Blinn-Phong specular (`specular`, `specularMap`, `shininess`) + diffuse + normal map |

- **Render State Controls** (native + ZoidCraft standard flags):
  - `shading`: `smooth` (default) vs `flat` (face normals via `dFdx`/`dFdy` or flat vertex split)
  - `wireframe`: `bool` (default `false`)
  - `side`: `front` (back-face culling, default), `back` (front-face culling), `both` (no culling, two-sided normal flip)
  - `transparent`: `bool` (enables alpha blending and disables depth write for back-to-front sorted transparent pass)
  - `opacity`: `f32 ∈ [0, 1]`
  - `blending`: `None (0)`, `Normal (1)`, `Additive (2)`, `Subtractive (3)`, `Multiply (4)`
  - `fog`: `bool` (default `true` when scene fog is enabled; scene fog supports linear or exponential-squared fog in `.zcraft`)
- **UV Transform Equation** (matching Three.js r91 `Texture.setUvTransform` with `offset` $\mathbf{o}$, `repeat` $\mathbf{s}$, `rotation` $\theta$, `center` $\mathbf{c}$):
  Let $c = \cos\theta$, $s = \sin\theta$. The 2D affine UV transform $\mathbf{u}' = M_{\text{uv}} \cdot [\mathbf{u}, 1]^T$ is:
  $$u' = s_x c (u - c_x) + s_x s (v - c_y) + c_x + o_x$$
  $$v' = -s_y s (u - c_x) + s_y c (v - c_y) + c_y + o_y$$
  *(Apply wrap mode `ClampToEdge`, `Repeat`, or `MirroredRepeat` after UV transformation.)*

---

## 5. Coordinate System & Units

### 5.1 3D World Space
- **Handedness**: **Right-handed, Y-up** (`+X` right, `+Y` up, `+Z` out of screen toward viewer).
- **Default Camera**: Located at `(0, 0, +80)` with rotation `(0, 0, 0)`, looking down `-Z` with `+Y` up.
- **Rotation Convention**: Intrinsic Euler angles, default order **`XYZ`** (Six supported orders: `XYZ`, `YZX`, `ZXY`, `XZY`, `YXZ`, `ZYX`). Positive rotation follows the right-hand rule around each local axis.
- **Depth Range (wgpu Adaptation)**: Three.js/WebGL NDC depth is `[-1, +1]`, whereas `wgpu` uses `[0, +1]`. Projection matrices in `zc-geom` produce `[0, 1]` clip-space Z for `wgpu`, while `toScreen`/`fromScreen` expression helpers convert depth to/from the canonical CM3 `[-1, +1]` contract.

### 5.2 Angle Units (Storage vs. UI Display)

| Property Category | Stored / Keyframed / Expression Unit | UI Display Unit | Conversion |
|---|---|---|---|
| 3D object `rotation` (Group, Shape, Text, Model, Camera) | **Radians** | Degrees | `UI = stored * 180 / π` |
| 2D `layer.rotation` & `shape.rotation` | **Radians** | Degrees | `UI = stored * 180 / π` |
| Material UV `rotation` | **Radians** | Degrees | `UI = stored * 180 / π` |
| Sequence Control `Angle` | **Radians** | Degrees | `UI = stored * 180 / π` |
| SpotLight `angle` | **Degrees** | Degrees | `shader = stored * π / 180` |
| Particles `iang`, `aspread`, `angvel`, `avspread` | **Degrees** (and deg/s) | Degrees | `shader = stored * π / 180` |
| Camera perspective `fov` | **Degrees** (vertical) | Degrees | `proj = fov * π / 180` |

### 5.3 2D Layer, Shape & Shader Coordinate Spaces

| Space | Origin | Axes | Units |
|---|---|---|---|
| **2D Layer `position`** | Center of composition | `+X` right, `+Y` up | Composition pixels (`[-W/2, +W/2] × [-H/2, +H/2]`) |
| **2D Shape `position`** | Center of shape layer | `+X` right, `+Y` up | Composition pixels |
| **Expression `toScreen` / `fromScreen`** | Top-left of composition | `+X` right, `+Y` down | Composition pixels `[0, W] × [0, H]`, `depth ∈ [-1, +1]` |
| **Shader `vUv` / `vUvScaled`** | Bottom-left `(0, 0)` (GLSL ES 1.00 contract) | `+U` right, `+V` up | Normalized `[0, 1]` (`vUvScaled = vUv * uvScale` in compositor; `zc-shader` handles Y-flip parity for `wgpu`) |

### 5.4 Color Representation
- **Dynamic `COLOR` properties**: Normalized RGB float triples `[r, g, b]` in `[0.0, 1.0]` (sRGB-encoded primaries as authored in CM3; linear conversion occurs only inside PBR lighting or explicit shader helpers `zSrgbToLinear`/`zLinearToSrgb`).
- **`GRADIENT` stops**: Ordered array of `{ "position": f64 ∈ [0, 1], "color": "rgba(R, G, B, A)" }` with integer `R, G, B ∈ [0, 255]` and float `A ∈ [0, 1]`.

---

## 6. Expressions Specification (Vanilla CM3 & Expressions+)

ZoidCraft implements a **native, sandboxed, deterministic Rust expression compiler and bytecode VM** (`zc-expr`) — zero browser/JS runtime required — supporting both Vanilla CM3 compatibility mode and Expressions+.

### 6.1 Language Tiers & Compatibility

1. **Vanilla CM3 Mode** (expressions using only CM3 syntax and API):
   - Supports ES5 expressions, `var` declarations, arithmetic/logical/ternary operators, conditionals, loops, arrays, object literals, and implicit return of the final expression statement (plus explicit `return`).
   - Disallows `let`, `const`, arrow functions, `function`, `class`, `throw`, `debugger`, `this`, `super`, `eval`, `Function`, `with`.
   - Unknown free identifiers fail at parse/compile time (`ReferenceError`).
2. **Expressions+ Mode** (automatically activated when extended syntax or API is used):
   - Adds `let`, `const`, arrow functions, named/anonymous `function`, default/rest parameters, template literals, array/object destructuring, spread (`...`), optional chaining (`?.`), `switch`, `try`/`catch`/`throw`.
   - Still strictly rejects: `class`, `import`/`export`, `async`/`await`, generators (`function*`/`yield`), `with`, `debugger`, `this`, `super`, `eval`, `Function` constructors, `Date`, `Math.random`, and all host/OS/environment identifiers.
   - Allowed standard built-ins: `Array`, `Number`, `String`, `Boolean`, `JSON`, `Math` (excluding `Math.random`), `parseInt`, `parseFloat`, `isFinite`, `isNaN`, `Infinity`, `NaN`, `undefined`.

### 6.2 Deterministic Safety Limits & Error Fallback

| Constraint | Vanilla CM3 Legacy | Expressions+ / ZoidCraft Default |
|---|---|---|
| Execution step budget (combined loop iterations + function calls per outer property read, shared across cross-property calls) | Bounded in Rust VM (default `10,000,000` safety ceiling) | **`10,000,000` steps** |
| Call / recursion depth limit | **`256`** | **`256` nested calls** |
| Cyclic property read detection | Error | Error (tracks active evaluation stack by property address) |
| Error fallback return value | Keyed/static property value (ZoidCraft adopts Expressions+ safe fallback; optional strict-legacy test mode may inspect `undefined`) | **Property's own keyed/static value** + human-readable diagnostic message |

### 6.3 Evaluation Context & Globals

| Global | Type | Meaning |
|---|---|---|
| `frame` | `f64` | Property-local frame (`sequenceFrame - clip.start + motionBlurSubOffset`). |
| `time` | `f64` | Owning clip's `time` property evaluated at `frame` (in seconds, including time remapping); `0.0` if no owning clip. Inside a clip's own `time` expression, equals the pre-expression keyed `time` value. |
| `fps` | `f64` | Owning sequence's authored `rate` (preserved even under export frame-rate overrides). |
| `value` | `Value` | Property's own keyed/static value before evaluating the expression. |
| `properties` | `Map<String, Value>` | Evaluated `customProperties` of the owning object's parent (e.g., parent `Group` or `EffectGroup`). |
| `Math.*` | `f64` / fn | All standard ECMAScript `Math` constants (`PI`, `E`, `SQRT2`, etc.) and functions (`sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `abs`, `floor`, `ceil`, `round`, `min`, `max`, `pow`, `sqrt`, `exp`, `log`, `sign`, `trunc`, `hypot`, …) exposed as unqualified globals AND on `Math`. **`Math.random` is forbidden.** |

### 6.4 Vanilla CM3 Built-In Functions (Exact 17 Methods)

| Function | Signature | Exact Semantics |
|---|---|---|
| `add` | `add(a: Vec, b: Vec) -> Vec` | Component-wise $r_i = a_i + b_i$ for $i < \min(|a|, |b|)$ (length $\max(|a|, |b|)$). |
| `sub` | `sub(a: Vec, b: Vec) -> Vec` | Component-wise $r_i = a_i - b_i$. |
| `mul` | `mul(v: Vec, s: f64) -> Vec` | Scalar multiply $r_i = v_i \cdot s$. |
| `div` | `div(v: Vec, s: f64) -> Vec` | Scalar divide $r_i = v_i \cdot (1 / s)$. |
| `clamp` | `clamp(v: f64 \| Vec, lo: f64, hi: f64)` | Vanilla: component-wise `min(max(v[i], lo), hi)` on vector; Expressions+ also accepts scalar `f64`. |
| `dot` | `dot(a: Vec, b: Vec) -> f64` | $\sum_{i=0}^{\min(|a|,|b|)-1} a_i b_i$. |
| `cross` | `cross(a: Vec3, b: Vec3) -> Vec3` | 3D cross product $(a_1 b_2 - a_2 b_1, \; a_2 b_0 - a_0 b_2, \; a_0 b_1 - a_1 b_0)$. |
| `normalize` | `normalize(v: Vec) -> Vec` | `div(v, length(v))`. |
| `length` | `length(a: Vec [, b: Vec]) -> f64` | If `b` omitted: $\sqrt{\sum a_i^2}$. If `b` provided: `length(sub(a, b))` (Euclidean distance). |
| `wave` | `wave(n: i32) -> f64` | Square wave: `2 * (n & 1) - 1` (`-1` for even, `+1` for odd). |
| `lerp` | `lerp(a, b, t)` | Alias for `linear(a, b, t)`. |
| `linear` | `linear(a, b, t)` | $(b - a) \cdot t + a$. *(5-arg AE overload in Expressions+, see §6.5.)* |
| `catmullRom` | `catmullRom(p0, p1, p2, p3, t)` | Uniform Catmull-Rom spline interpolation at parameter $t \in [0, 1]$. |
| `ease` | `ease(t, family="Quadratic", dir="InOut")` | Evaluates CM3 easing curve `easing[family][dir](t)`. *(5-arg AE overload in Expressions+, see §6.5.)* |
| `random` | `random(seed: f64) -> f64` | Deterministic CM3 hash: $\text{fract}(10000 \cdot \sin(\text{seed})) \in [0, 1)$. |
| `gaussRandom` | `gaussRandom(seed, mean=0.5, std=0.25)` | $\text{std} \cdot \bigl(\sum_{k=0}^{4} \text{random}(\text{seed} \cdot k) - 2.5\bigr) / 2.5 + \text{mean}$. |
| `shake` | `shake(f, freq=1, phase=0, waveAmp=1, noiseAmp=0, smooth=1)` | $u = f \cdot \text{freq} + \text{phase}$, $k_0 = \lfloor u \rfloor$, $k_1 = k_0 + 1$, $h = u - k_0$; if `smooth != 0`, $h \leftarrow h^2(3 - 2h)$; $s(k) = \text{waveAmp} \cdot \text{wave}(k) + 2 \cdot \text{noiseAmp} \cdot (\text{random}(k) - 0.5)$; returns $\text{lerp}(s(k_0), s(k_1), h)$. |

### 6.5 Expressions+ Extensions

#### Scene & Property Navigation Handles (Strictly Read-Only)
- `thisProperty`:
  - `.value` — pre-expression value at current frame
  - `.valueAtTime(seconds)` — evaluates the property at sequence time `seconds` (on self, samples raw keyframes without recursing into the expression)
  - `.velocity` — symmetric finite difference $\frac{\text{valueAtTime}(t + 0.005) - \text{valueAtTime}(t - 0.005)}{0.01}$ (per component for vectors)
  - `.speed` — `abs(velocity)` for scalar or Euclidean magnitude `length(velocity)` for vector
  - `.numKeys` — integer keyframe count
  - `.key(index)` — 1-based keyframe record `{ index, time, value }`
  - `.nearestKey(seconds)` — closest keyframe record to `seconds`
- Hierarchy handles: `thisObject`, `thisLayer`, `thisClip`, `thisSequence` / `thisComp` (with `.frameDuration = 1.0 / fps`).
- Handle methods:
  - `.parent` / `.parent(n)` — nearest owner or $n$ levels up
  - `.layer(nameOrIndex)` (and global `layer(nameOrIndex)`) — active video layer by name or 1-based track index
  - `.effect(name)` (and global `effect(name)`) — callable handle: `effect("Glow")("Radius")` or `.param("Radius")`
  - `.object(name)` — direct 3D child in a Scene or Group
  - `.property(name)` — property handle matching storage key, display name, or custom property name
- Path lookup: `prop(path)` — `/Layer/Obj/Prop` (sequence root), `./Prop` (current owner), `../Prop` (parent). Cross-clip reads convert the absolute sequence frame to the target clip's local frame.
- **Sequence Controls**: `controls.<Name>`, `controls["<Name>"]`, `control(name [, stableId])`. Up to **256 controls** per sequence (`Slider`, `Angle`, `Checkbox`, `Color`, `Point 2D`, `Point 3D`). Generated links pass `stableId` so renames/reorders stay bound; deleting a referenced ID produces an error and falls back to the property's own value.

#### Extended Math, PRNG, Animation & 3D Space Helpers
- **5-Argument Clamped AE Interpolation**:
  - `linear(t, tMin, tMax, a, b)` — clamps $u = \text{clamp}((t - t_{\min}) / (t_{\max} - t_{\min}), 0, 1)$, linearly interpolates scalar or vector $a \to b$.
  - `ease(t, tMin, tMax, a, b)` — cubic smoothstep $u^2(3 - 2u)$ on clamped $u$, interpolating $a \to b$.
  - `easeIn(t, tMin, tMax, a, b)` / `easeOut(t, tMin, tMax, a, b)` — ease-in / ease-out clamped interpolation.
- **Angles & Orientation**:
  - `degreesToRadians(x)` $= x \cdot \pi / 180$, `radiansToDegrees(x)` $= x \cdot 180 / \pi$.
  - `lookAt(from: Vec3, to: Vec3) -> Vec3` — returns `XYZ` Euler angles in radians orienting local `+Z` toward `to`.
- **Repeatable PRNG Stream**:
  - `seedRandom(seed, timeless = false)` — initializes a fresh per-evaluation deterministic PRNG stream (when `timeless == false`, mixes `frame` into the seed; when `timeless == true`, independent of frame). Subsequent `random()` (0-arg), `random(maxVal)`, or `random(minVal, maxVal)` calls advance the local stream. Zero state leaks across frames.
- **Wiggle**:
  - `wiggle(frequency, amplitude, octaves=1, ampMult=0.5, frame=currentFrame)` — multi-octave procedural perturbation using frame-based frequency convention; returns scalar or vector matching `amplitude` / `value`.
- **3D World & Screen Projection Helpers** (on 3D object handles and owning 3D objects):
  - `toWorld(localPoint: Vec3) -> Vec3`, `fromWorld(worldPoint: Vec3) -> Vec3`
  - `worldPosition() -> Vec3`, `worldRotation() -> Vec3` (`XYZ` Euler radians after parent hierarchy and Track To constraints)
  - `toScreen(localPoint: Vec3) -> [x, y, depth]` — composition pixels (top-left origin, `+Y` down) + WebGL NDC depth `[-1, +1]`, including active camera shake and native Scene layer 2D transform
  - `fromScreen([x, y, depth]) -> Vec3` — exact inverse (errors on singular/zero-scale transform)

#### Preview Cache Purity Classification
An expression is **cacheable** (allowing finished-frame preview caching) unless its AST performs a property mutation, `delete`, or mutating method call on any non-fresh-local value, or accesses `__proto__` / `constructor` / `prototype`. Pure expressions (math, `wiggle`, `seedRandom`/`random`, property reads, `toScreen`, fresh local arrays) remain fully cacheable.

---

## 7. `.pz` Best-Effort Import Schema & `.zcraft` Native Format

### 7.1 `.pz` Container & Entry Layout

A `.pz` file is a **gzip-compressed POSIX TAR archive** containing:

| TAR Entry Name | Required | Content |
|---|---|---|
| `meta` | Optional (CM3) | UTF-8 JSON `{ "version": <int> }` (triggers CM3 schema upgrades; absence may indicate legacy CM2/BG4 format) |
| `project` | Yes | UTF-8 JSON `{ "assets": {...}, "media": [...], "sequence": {...} }` |
| `<32-hex-sha256>` | 0..N | Raw binary asset bytes for every non-external asset with `size <= 16 MiB` |
| `zoidium.json` | Optional | Zoidium extension metadata: `{ name, entries: [{ name, size, sha256 }], missingAssets: [...], plugins: [...] }` |

#### Security & Validation Limits on Import (Mandatory)
- Maximum uncompressed archive size: **2 GiB**
- Maximum single JSON entry (`meta`, `project`, `zoidium.json`): **32 MiB**
- Maximum TAR entry count: **10,000**
- Maximum JSON object/array nesting depth: **128**
- Reject TAR paths containing `..`, leading `/`, NUL bytes, or symlinks/hardlinks.
- If `project` is structurally valid but one or more binary asset entries are missing or unreadable, open the project with those assets marked **Missing** (never silently delete asset references).

### 7.2 `.pz` Property & Keyframe JSON Schemas

#### Property Type IDs (`PZ.property.type`)

| ID | Type | JSON Value Representation |
|---:|---|---|
| `0` | `NUMBER` | `f64` |
| `1` | `VECTOR2` | `[f64, f64]` (static) or dynamic group of 2 `NUMBER`s |
| `2` | `VECTOR3` | `[f64, f64, f64]` (static) or dynamic group of 3 `NUMBER`s |
| `3` | `VECTOR4` | `[f64, f64, f64, f64]` (static) or dynamic group of 4 `NUMBER`s |
| `4` | `COLOR` | `[r, g, b]` in `[0, 1]` (dynamic group of 3 `NUMBER`s) |
| `5` | `GRADIENT` | `[{ "position": f64, "color": "rgba(r,g,b,a)" }, ...]` |
| `6` | `OPTION` | Zero-based integer index `u32` (NOTE: Shader+ static `0` may serialize as `"0"` to dodge CM3's falsy static load check) |
| `7` | `TEXT` | UTF-8 `string` |
| `8` | `ASSET` | `AssetKey` string or `null` |
| `9` | `LIST` | Selected option value `string` (e.g., `"BLEND_SRC_OVER"`, `"XYZ"`) |
| `10` | `CURVE` | Curve control-point array / JSON |
| `11` | `SHADER` | Shader asset/source |

#### Property Serialization Forms
- **Static Property**: Bare JSON value (e.g., `30`, `[1920, 1080]`, `"XYZ"`), or `{ "type": <customDef>, "properties": { "name": "..." }, "value": <val> }` if custom.
- **Dynamic Keyframe Property**:
  ```json
  {
    "animated": true,
    "expression": "sin(time) * 10",
    "keyframes": [
      {
        "value": 0.0,
        "frame": 0,
        "tween": 1,
        "controlPoints": [[-10, 0], [10, 0]],
        "continuousTangent": false
      }
    ]
  }
  ```
  - `expression`: omitted when `null`.
  - `continuousTangent`: omitted when `true` (default `true`); explicitly serialized only when `false`.
  - `controlPoints`: `[[inDeltaFrame, inDeltaValue], [outDeltaFrame, outDeltaValue]]`, default `[[-10, 0], [10, 0]]`.
  - `tween` bitfield:
    - Low byte (`tween & 0xFF`): index into CM3's `easingList` (`0` = hold/step [`fn == null`], `1` = linear, `2..` = Quadratic/Cubic/Quartic/Quintic/Sinusoidal/Exponential/Circular/Elastic/Back/Bounce In/Out/InOut).
    - High byte (`tween >> 8`): when `== 1`, segment uses custom cubic Bezier interpolation driven by `kf[i].controlPoints[1]` (outgoing) and `kf[i+1].controlPoints[0]` (incoming) with overlap correction factor `correctCurve`.
- **Dynamic Group Property** (e.g., `position`, `rotation`, `scale`, `color`):
  ```json
  {
    "expression": "[0, sin(time)*5, 0]",
    "objects": [
      { "animated": false, "keyframes": [{ "value": 0, "frame": 0, "tween": 1, "controlPoints": [[-10,0],[10,0]] }] },
      { "animated": false, "keyframes": [{ "value": 0, "frame": 0, "tween": 1, "controlPoints": [[-10,0],[10,0]] }] },
      { "animated": false, "keyframes": [{ "value": 0, "frame": 0, "tween": 1, "controlPoints": [[-10,0],[10,0]] }] }
    ]
  }
  ```
  - *Legacy import migration*: If an older `.pz` stores `keyframes` directly on a group property (where each keyframe `value` is an array `[x, y, z]`) instead of `objects`, split each component `t` into `objects[t].keyframes` preserving `frame`, `tween`, and `animated`.
- **`clipLinks` Addressing**:
  - Serializes `{ "seed": u32, "links": { "<key>": { "clips": [ [0, 1, 0, 0], [0, 2, 0, 0] ] } } }` where each address array walks `Project.children` (`[sequence=0, videoTracks=1|audioTracks=2, trackIdx, clipIdx]`). Importers resolve these into stable UUID clip link groups.

#### Preserved Zoidium Compatibility & Extension Fields
On `.pz` import, ZoidCraft reads and preserves (even when a plugin is disabled):
- Rendering mode flags: `zoidiumUvMode` (material), `zoidiumTextScaleMode` (3D Text), `zoidiumSubdivisionMode` + `zoidiumLegacySubdivisionLevel` (Twist/Warp), `zoidiumTemporalMode` (frame samplers), `zoidiumShadowMode` (3D Scene). Unmarked imported `.pz` objects default to `"legacy"` mode unless explicitly upgraded.
- Constraints: `zoidiumId`, `zoidiumTrackTo*`, `zoidiumFollow*`.
- Sequence Controls: `sequence.properties.zoidiumControls` JSON string (`{ version: 1, nextId, controls: [...] }`).
- Shader+ / Node Effect / Magic Looks metadata: `zoidiumShader`, `zoidiumControl`, `zoidiumLabel`, `zoidiumDefault`, `// @zoidium-node-effect` trailing comment, `looksInstances`, `looksLegacyFormat`.

---

### 7.3 Recommended `.zcraft` Clean Native Schema

`.zcraft` is ZoidCraft's canonical, crash-safe project format designed for clean Rust `serde` serialization, zero index fragility, full-width content addressing, and lossless plugin preservation.

#### Container Layout
A `.zcraft` file is a **gzip-compressed POSIX TAR archive** written atomically (`write temp -> flush/fsync -> validate -> rename`):

```text
my-project.zcraft (tar.gz)
├── manifest.json          // Format version, integrity SHA-256 table, plugin deps, missing assets
├── project.json           // Clean discriminated-enum ProjectDocument
└── assets/
    └── <64-hex-sha256>    // Full SHA-256 content-addressed binary blobs (no 16 MiB cap)
```

#### `manifest.json` Schema
```json
{
  "format": "zoidcraft",
  "schema_version": 1,
  "app_version": "0.1.0",
  "project_name": "Intro",
  "created_utc": "2026-11-24T00:00:00Z",
  "modified_utc": "2026-11-24T00:00:00Z",
  "imported_from_pz_version": null,
  "required_plugins": [
    { "id": "expressions-plus", "features": ["expressions", "sequence-controls"] }
  ],
  "missing_assets": [],
  "integrity": {
    "project.json": { "size": 12480, "sha256": "..." },
    "assets/<64-hex>": { "size": 524288, "sha256": "..." }
  }
}
```

#### `project.json` Core Design Rules
1. **Explicit String-Tagged Enums (No Magic Numbers)**:
   - Layers use `"kind": "video" | "adjustment" | "composite" | "image" | "scene" | "null" | "shape" | "text_2d" | "preset_shape"`.
   - 3D Objects use `"kind": "group" | "shape" | "text_3d" | "model" | "light" | "particles" | "camera"` with typed sub-variants (e.g., `ShapeKind::Box`, `LightKind::Spot`, `CameraKind::Perspective`).
   - Materials use `"kind": "single_color" | "texture" | "video" | "standard_pbr" | "lambert" | "phong" | "plugin"`.
2. **UUID Identity Everywhere**:
   - Every Track, Clip, Layer, Effect, 3D Object, Material, Sequence Control, and Clip Link Group has a stable `Uuid` (`id`).
   - `clip_links` maps `LinkGroupId -> Vec<ClipId>` (never fragile tree index arrays).
   - 3D Constraints (`track_to`, `follow`) reference `ObjectId` directly.
3. **Unified `DynamicProp<T>` & `DynamicVec<N>`**:
   - Scalar dynamic properties: `{ animated: bool, expression: Option<String>, keyframes: Vec<Keyframe<f64>> }` with typed `InterpMode` (`Hold`, `Linear`, `Ease { family, direction }`, `Bezier { in_handle, out_handle, continuous }`).
   - Vector/Color dynamic properties: separate per-channel `DynamicProp<f64>` plus optional group-level `expression: Option<String>` (matching CM3's exact per-axis keyframing + vector-expression override model).
4. **Explicit Rendering & Compatibility Modes**:
   - Scene carries `shadow_mode: "current" | "legacy"`.
   - 3D Text carries `uv_mode: "current" | "legacy"` and `text_scale_mode: "current" | "legacy"`.
   - New objects created in ZoidCraft always default to `"current"`.
5. **Explicit Compositing Order in Schema**:
   - `sequence.video_tracks` stored explicitly bottom-to-top (`0` = bottom, `N-1` = top) — matching CM3's sequence order.
   - `composite.layers` stored in the same explicit bottom-to-top rendering order in `.zcraft` (with `.pz` importer reversing `composite.objects` on load and `.pz` exporter reversing back on save so every internal rendering loop in `zc-render` uses one consistent bottom-to-top traversal!).
6. **Lossless Plugin & Unknown Field Preservation**:
   - Every extensible node (`Sequence`, `Layer`, `Effect`, `Object3D`, `Material`) carries an optional `extensions: BTreeMap<String, serde_json::Value>` map so disabled native plugins and unrecognized `.pz`/Zoidium fields survive save/reopen, copy/paste, and undo/redo without data loss.
