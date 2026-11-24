# 04 — Effects, Custom Shaders, Audio/Export & Zoidium Plugins (Clean-Room Specification)

> **Scope:** Complete clean-room behavioral specification for ZoidCraft's post-processing effect stack (all 44 standard ClipMaker 3 effects), Custom Shader GLSL ES 1.00 → `naga` (GLSL 450 / WGSL) contract, deterministic Audio & FFmpeg Export pipeline, and toggleable native implementations of all Zoidium plugins.
>
> **Clean-Room Rule:** Every shader, pass layout, mathematical model, and plugin behavior below is specified purely from observable interfaces, parameter schemas, and mathematical definitions so Team B can author 100% original Rust/WGSL implementations.

---

## Table of Contents

1. [Post-Processing Pipeline Conventions](#1-post-processing-pipeline-conventions)
2. [All 44 Standard ClipMaker 3 Effects](#2-all-44-standard-clipmaker-3-effects)
3. [Custom Shader Effect Contract & GLSL ES 1.00 → Naga Shim](#3-custom-shader-effect-contract--glsl-es-100--naga-shim)
4. [Export & Offline Audio Mixing Specification](#4-export--offline-audio-mixing-specification)
5. [Zoidium Plugin Architecture & Toggleable Native Plugins](#5-zoidium-plugin-architecture--toggleable-native-plugins)

---

## 1. Post-Processing Pipeline Conventions

### 1.1 Color Space & Premultiplied Alpha
- All layer render targets and effect ping-pong buffers store **premultiplied RGBA** (`RGBA8Unorm` for standard passes; `RGBA16Float` where high-precision intermediate accumulation is specified, such as multi-pass grading chains or optical flow).
- For any premultiplied texel $c = (r, g, b, a)$:
  - Straight color is recovered as $\text{rgb}_{\text{straight}} = \text{rgb} / \max(a, \epsilon)$ when $a > 0$, and $(0, 0, 0)$ when $a = 0$.
  - Pointwise color adjustments (Brightness/Contrast, Color Adjustment, Color Grading, Colorize, Cube LUT, Exposure, Hue Shift, Posterize, Saturation, Technicolor) MUST operate on straight RGB and re-multiply by $a$ on output so transparent pixels remain $(0, 0, 0, 0)$ and never develop halos or hidden RGB.
  - Spatial convolutions (Box Blur, Gaussian Blur, Directional Blur, Radial Blur, Bloom, Anamorphic Lens Flare) MUST filter premultiplied RGBA directly so semi-transparent edges do not bleed black fringes.
- Standard Porter-Duff **Source-Over** (premultiplied input $B$, straight foreground $S$ with opacity $\alpha_s$):
  $$\alpha_{\text{out}} = \alpha_s + B_a (1 - \alpha_s), \qquad \text{rgb}_{\text{out}} = S_{\text{rgb}} \alpha_s + B_{\text{rgb}} (1 - \alpha_s)$$

### 1.2 Coordinate Spaces & Vertex Mapping
- Logical layer UV `vUv` spans $[0, 1]^2$ with $(0, 0)$ at the **bottom-left** and $(1, 1)$ at the **top-right** (matching OpenGL / WebGL / CM3).
- Backing render-target UV `vUvScaled` equals `vUv * uvScale`, where `uvScale = activeSize / targetSize`. When ZoidCraft allocates exact layer-sized targets, `uvScale = (1.0, 1.0)` and `vUv == vUvScaled`; both MUST still be provided to shaders for compatibility.
- Because `wgpu` texture coordinates place $v = 0$ at the top, the compositor either flips $v$ consistently at the fullscreen vertex shader (`vUv = vec2(uv.x, 1.0 - uv.y)`) or stores render targets bottom-up; all built-in WGSL effects and the Custom Shader shim MUST observe `vUv = (0, 0)` at bottom-left and `(1, 1)` at top-right.

### 1.3 Common Property & Bypass Rules
- Every standard effect declares `enabled` (`OPTION`, items `"off;on"`, values `0` / `1`, default `1`, animatable).
- **Units:** Properties marked **(deg→rad)** are stored internally in **radians** (`scaleFactor = π/180`) while the UI displays and edits **degrees**. Properties marked **(rad)** store and display radians directly.
- **Auto-Bypass:** When `enabled == 0` OR an effect's documented neutral-bypass condition holds (e.g. `delta == 0` for blurs), the compositor skips the pass completely with zero GPU copy.
- **Luma Coefficients:** Standard BT.601 luma is $L(\text{rgb}) = 0.299 R + 0.587 G + 0.114 B$.
- **Blend Modes (`PZ.layer.blendModes`):** Used by `coloroverlay`, `gradientoverlay`, and `overlay`. Supports standard Porter-Duff and Photoshop separable/non-separable modes:
  `BLEND_SRC_OVER` (Normal), `BLEND_MULTIPLY`, `BLEND_SCREEN`, `BLEND_OVERLAY`, `BLEND_DARKEN`, `BLEND_LIGHTEN`, `BLEND_COLOR_DODGE`, `BLEND_COLOR_BURN`, `BLEND_HARD_LIGHT`, `BLEND_SOFT_LIGHT`, `BLEND_DIFFERENCE`, `BLEND_EXCLUSION`, `BLEND_HUE`, `BLEND_SATURATION`, `BLEND_COLOR`, `BLEND_LUMINOSITY`, `BLEND_ADD`, `BLEND_SUBTRACT`, plus Porter-Duff alpha modes. Overlay effects use `OVERLAP_MODE = 3` (source-atop / destination-alpha clipped), meaning the overlay is masked to the layer's existing alpha silhouette.

---

## 2. All 44 Standard ClipMaker 3 Effects

### 2.1 Master Index

| # | Internal ID | UI Display Name | Category | Passes | Neutral Auto-Bypass |
|---|---|---|---|---:|---|
| 1 | `anamorphicflare` | Anamorphic Lens Flare | Glow & Light | 13 | — |
| 2 | `bloom` | Bloom | Glow & Light | 13 | — |
| 3 | `boxblur` | Box Blur | Blur & Sharpen | 2 | `delta == 0` |
| 4 | `brightnesscontrast` | Brightness + Contrast | Color & Tone | 1 | — |
| 5 | `chromakey` | Chroma Key | Keying & Masking | 1 | — |
| 6 | `colorcorrection` | Color Adjustment | Color & Tone | 1 | — |
| 7 | `colorgrading` | Color Grading | Color & Tone | 1 | — |
| 8 | `colorize` | Colorize | Color & Tone | 1 | `amount == 0` |
| 9 | `coloroverlay` | Color Overlay | Overlay & Composite | 1 | — |
| 10 | `crop` | Crop | Distort & Transform | 1 | — |
| 11 | `cubelut` | Cube LUT | Color & Tone | 1 | `lut == null` |
| 12 | `directionalblur` | Directional Blur | Blur & Sharpen | 1 | `delta == 0` |
| 13 | `displacementmap` | Displacement Map | Distort & Transform | 1 | — |
| 14 | `duplicate` | Duplicate | Distort & Transform | 1 | — |
| 15 | `edgedetection` | Edge Detection | Stylize & Glitch | 1 | `amount == 0` |
| 16 | `exposure` | Exposure | Color & Tone | 1 | — |
| 17 | `fisheye` | Fisheye | Distort & Transform | 1 | `amount == 0` |
| 18 | `flip` | Flip | Distort & Transform | 1 | — |
| 19 | `fxaa` | Antialiasing | Stylize & Glitch | 1 | — |
| 20 | `gaussianblur` | Gaussian Blur | Blur & Sharpen | 6 | `delta == 0` |
| 21 | `glitch` | Glitch | Stylize & Glitch | 1 | `amount == 0` |
| 22 | `gradientoverlay` | Gradient Overlay | Overlay & Composite | 1 | — |
| 23 | `grayscale` | Saturation | Color & Tone | 1 | `saturation == 1` |
| 24 | `hueshift` | Color Shift | Color & Tone | 1 | `amount == 0` |
| 25 | `lumakey` | Luma Key | Keying & Masking | 1 | — |
| 26 | `mask` | Mask | Keying & Masking | 1 (2 draws) | — |
| 27 | `mirror` | Mirror | Distort & Transform | 1 | — |
| 28 | `negative` | Negative | Color & Tone | 1 | — |
| 29 | `overlay` | Image Overlay | Overlay & Composite | 1 | — |
| 30 | `pixelated` | Pixelated | Distort & Transform | 1 | `size == (1, 1)` |
| 31 | `pixelsort` | Pixel Sort | Distort & Transform | 2 | — |
| 32 | `posterize` | Posterize | Color & Tone | 1 | — |
| 33 | `pulse` | Pulse | Distort & Transform | 1 | `time % 1 == 0` |
| 34 | `radialblur` | Radial Blur | Blur & Sharpen | 1 | — |
| 35 | `rgbshift` | RGB Shift | Stylize & Glitch | 1 | `amount == 0` |
| 36 | `scanlines` | Scan Lines | Stylize & Glitch | 1 | `opacity == 0` |
| 37 | `shake` | Shake | Distort & Transform | 1 | — |
| 38 | `sharpen` | Sharpen | Blur & Sharpen | 1 | — |
| 39 | `shutter` | Shutter | Stylize & Glitch | 1 | `opacity == 0` |
| 40 | `static` | Static | Stylize & Glitch | 1 | `amount == 0` |
| 41 | `swirl` | Swirl | Distort & Transform | 1 | `angle == 0` |
| 42 | `technicolor` | Technicolor | Color & Tone | 1 | — |
| 43 | `transform` | Transform | Distort & Transform | 1 | — |
| 44 | `wavy` | Wavy | Distort & Transform | 1 | `amount == 0` |

---

### 2.2 Detailed Effect Specifications

#### 1. `anamorphicflare` — Anamorphic Lens Flare
- **Category:** Glow & Light | **Passes:** 13 (1 high-pass + 5×2 separable blurs + 1 composite + 1 additive copy)
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `amount` | Amount | Number | `1` | `0` | `1` | `0.01` | `2` | Yes | Additive composite opacity |
  | `inner` | Inner glow | Color (RGB) | `[1, 0, 0]` | `0` | `1` | — | — | Yes | Tints mip levels 0, 1, 2 |
  | `outer` | Outer glow | Color (RGB) | `[1, 0, 0]` | `0` | `1` | — | — | Yes | Tints mip levels 3, 4 |
  | `strength` | Strength | Number | `0.75` | `0` | `3` | `0.01` | `2` | Yes | Overall flare gain |
  | `radius` | Radius | Number | `0.8` | `0` | `1` | `0.01` | `2` | Yes | Shifts weight toward wider mips |
  | `threshold` | Threshold | Number | `0.4` | `0` | `1` | `0.01` | `2` | Yes | Luma extraction threshold |
- **Mathematical / Visual Behavior:**
  1. **Anamorphic Pyramid Allocation:** 5 mip pairs (H/V). Mip 0 has dimensions `round(W/2) × H`; each subsequent mip $m \in \{1..4\}$ halves **width only** (`round(W / 2^(m+1)) × H`), keeping full vertical resolution so streaks stretch horizontally.
  2. **Luminosity High-Pass:** Extracts bright regions into mip-0 bright buffer using BT.601 luma $v = \text{dot}(\text{rgb}, (0.299, 0.587, 0.114))$ and soft gate $\alpha = \text{smoothstep}(\text{threshold}, \text{threshold} + 0.01, v)$, outputting $\alpha \cdot \text{texel}$.
  3. **Separable Gaussian Pyramid:** For each mip $m = 0..4$ with kernel radius and standard deviation $K_m = \sigma_m \in \{3, 5, 7, 9, 11\}$, runs a 1D normalized Gaussian blur horizontally then vertically at texel step $1 / \text{size}_m$ using weights $G(x,\sigma) = \frac{1}{\sqrt{2\pi}\sigma} e^{-x^2/(2\sigma^2)}$ for $x \in [-(K_m-1), K_m-1]$. Mip $m$ feeds from mip $m-1$'s vertical output.
  4. **Weighted Tinted Composite:** Combines all 5 mips using base factors $f = [1.0, 0.8, 0.6, 0.4, 0.2]$ and effective weights $w_m = \text{mix}(f_m, 1.2 - f_m, \text{radius})$:
     $$C_{\text{flare}} = \text{strength} \cdot \sum_{m=0}^{4} w_m \cdot (\text{tint}_m, 1.0) \odot T_m(u, v)$$
     where $\text{tint}_{0,1,2} = \text{inner}$ and $\text{tint}_{3,4} = \text{outer}$.
  5. **Additive Blend:** Adds `amount * C_flare` onto the unmodified source texture using premultiplied additive blending (`src + dst`).

#### 2. `bloom` — Bloom
- **Category:** Glow & Light | **Passes:** 13 (1 high-pass + 5×2 separable blurs + 1 composite + 1 additive copy)
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `amount` | Amount | Number | `1` | `0` | `1` | `0.01` | `2` | Yes | Additive opacity |
  | `inner` | Inner glow | Color (RGB) | `[1, 1, 1]` | `0` | `1` | — | — | Yes | Tints mips 0, 1, 2 |
  | `outer` | Outer glow | Color (RGB) | `[1, 1, 1]` | `0` | `1` | — | — | Yes | Tints mips 3, 4 |
  | `strength` | Strength | Number | `0.75` | `0` | — | `0.01` | `3` | Yes | Glow intensity |
  | `radius` | Radius | Number | `0.8` | `0` | — | `0.01` | `3` | Yes | Spread across mips |
  | `threshold` | Glow threshold | Number | `0.4` | `0` | `1` | `0.01` | `3` | Yes | High-pass luma center |
  | `soften` | Glow soften | Number | `0.01` | `0` | `1` | `0.01` | `3` | Yes | High-pass half-width |
  | `blendThreshold` | Blend threshold | Number | `0.4` | `0` | `1` | `0.01` | `3` | Yes | Highlight protection center |
  | `blendSoften` | Blend soften | Number | `0.5` | `0` | `1` | `0.01` | `3` | Yes | Highlight protection half-width |
- **Mathematical / Visual Behavior:**
  - Same 5-mip separable Gaussian architecture ($K_m = \sigma_m \in \{3, 5, 7, 9, 11\}$, $f = [1.0, 0.8, 0.6, 0.4, 0.2]$) as `anamorphicflare`, with three differences:
    1. **Isotropic Mips:** Both width and height halve each level: `round(W / 2^(m+1)) × round(H / 2^(m+1))`.
    2. **Symmetric High-Pass Knee:** $\alpha = \text{smoothstep}(\text{threshold} - \text{soften}, \text{threshold} + \text{soften}, L_{\text{src}})$.
    3. **Highlight Blend Attenuation:** The composite sum is multiplied by $1 - \text{smoothstep}(\text{blendThreshold} - \text{blendSoften}, \text{blendThreshold} + \text{blendSoften}, L_{\text{src}})$ before additive blending at `amount`, preventing already-bright cores from clipping.

#### 3. `boxblur` — Box Blur
- **Category:** Blur & Sharpen | **Passes:** 2 (Vertical → Horizontal) | **Bypass:** `delta == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `delta` | Delta | Number | `1` | `0` | `10` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - Two-pass separable uniform 1D box filter operating directly on premultiplied RGBA.
  - Pass 1 filters vertically along $(0, 1/H)$; Pass 2 filters horizontally along $(1/W, 0)$.
  - Kernel half-width scales linearly with `delta` (sampling symmetrically around `vUv` with equal weights normalized to sum to 1, clamping UVs to `[0, 1]` or layer bounds).

#### 4. `brightnesscontrast` — Brightness + Contrast
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `brightness` | Brightness | Number | `0` | `-1` | `1` | `0.1` | — | Yes |
  | `contrast` | Contrast | Number | `1` | `0` | `10` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - On straight RGB: $\text{rgb}' = (\text{rgb} - 0.5) \cdot \text{contrast} + 0.5 + \text{brightness}$.
  - Re-premultiplies by source alpha $a$ (clamping straight RGB to $[0, 1]$ before output if targeting UNORM8).

#### 5. `chromakey` — Chroma Key
- **Category:** Keying & Masking | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `backgroundColor` | Color | Color (RGB) | `[0.157, 0.776, 0.129]` | `0` | `1` | — | — | Yes | Key screen color (green) |
  | `weights` | Weights | Vector3 (H,S,V) | `[4, 1, 2]` | `0` | `100` | `0.1` | `3` | Yes | Per-channel HSV distance weights |
  | `soften` | Soften | Number | `0.5` | `0` | — | `0.01` | `3` | Yes | Key feather/transition width |
  | `invert` | Invert | Option | `0` (`off`) | `0` | `1` | — | — | No | `off;on` (compile define `KEY_INVERT`) |
  | `mask` | Mode | Option | `0` (`result`) | `0` | `1` | — | — | No | `result;mask` (compile define `KEY_MASK`) |
- **Mathematical / Visual Behavior:**
  - Converts unpremultiplied source pixel and `backgroundColor` from RGB to HSV $(H, S, V) \in [0, 1]^3$.
  - Computes circular hue difference $\Delta H = \min(|H - H_k|, 1 - |H - H_k|)$, $\Delta S = |S - S_k|$, $\Delta V = |V - V_k|$.
  - Weighted distance $d = \|(\Delta H \cdot w_H, \Delta S \cdot w_S, \Delta V \cdot w_V)\|_2$.
  - Computes foreground matte $m = \text{smoothstep}(0, \max(\text{soften}, \epsilon), d)$ (or thresholded linear ramp scaled by `soften`). If `invert == 1`, replaces $m \leftarrow 1 - m$.
  - When `mask == 0` (`result`): outputs `texel * m` (premultiplied). When `mask == 1` (`mask`): outputs `(m, m, m, 1.0)`.

#### 6. `colorcorrection` — Color Adjustment
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `powRGB` | Power | Vector3 (R,G,B) | `[1, 1, 1]` | `0` | `100` | `0.1` | `2` | Yes |
  | `mulRGB` | Multiply | Vector3 (R,G,B) | `[0.8, 0.8, 0.8]` | `0` | `100` | `0.1` | `2` | Yes |
  | `addRGB` | Add | Vector3 (R,G,B) | `[0.08, 0.08, 0.08]` | `-1` | `1` | `0.01` | `2` | Yes |
- **Mathematical / Visual Behavior:**
  - Per-channel curve on straight RGB: $\text{rgb}' = \text{mulRGB} \odot \max(\text{rgb}, 0)^{\text{powRGB}} + \text{addRGB}$.
  - Preserves source alpha and outputs premultiplied `(rgb' * a, a)`.

#### 7. `colorgrading` — Color Grading
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `shadows` | Shadows | Color (RGB) | `[1, 1, 1]` | `0` | `1` | — | — | Yes |
  | `midtones` | Midtones | Color (RGB) | `[1, 1, 1]` | `0` | `1` | — | — | Yes |
  | `highlights` | Highlights | Color (RGB) | `[1, 1, 1]` | `0` | `1` | — | — | Yes |
- **Mathematical / Visual Behavior:**
  - Three-way Lift / Gamma / Gain grading operating on straight RGB:
    - Converts neutral white `(1, 1, 1)` controls into identity offsets so default `[1, 1, 1]` on all three wheels produces an exact no-op.
    - Lift (`shadows`) shifts dark values linearly as $(1 - c)$; Gain (`highlights`) scales bright values multiplicatively; Gamma (`midtones`) applies a midtone power curve $c^{1/\gamma}$.
  - Re-premultiplies by source alpha.

#### 8. `colorize` — Colorize
- **Category:** Color & Tone | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `color` | Color | Color (RGB) | `[1, 0, 0]` | `0` | `1` | — | — | Yes |
  | `amount` | Amount | Number | `1` | `0` | `1` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - Converts straight source RGB to HSL $(H_s, S_s, L_s)$ and target `color` to HSL $(H_c, S_c, L_c)$.
  - Forms colorized RGB from $(H_c, S_c, L_s)$ (preserving source lightness while adopting target hue and saturation).
  - Linearly interpolates $\text{rgb}' = \text{mix}(\text{rgb}_{\text{src}}, \text{rgb}_{\text{colorized}}, \text{amount})$ and re-premultiplies by source alpha.

#### 9. `coloroverlay` — Color Overlay
- **Category:** Overlay & Composite | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `color` | Color | Color (RGB) | `[1, 1, 1]` | `0` | `1` | — | — | Yes | Solid overlay color |
  | `opacity` | Opacity | Number | `1` | `0` | `1` | `0.1` | — | Yes | Overlay opacity |
  | `blending` | Blending mode | List | `"BLEND_SRC_OVER"` | — | — | — | — | No | Any `PZ.layer.blendModes` entry |
- **Mathematical / Visual Behavior:**
  - Evaluates the selected blend function $B(\text{dst}, \text{color})$ over the input texture (`tBG`) at opacity `opacity`, clipped to the destination alpha (`OVERLAP_MODE = 3`, source-atop), preserving the layer's original alpha channel.

#### 10. `crop` — Crop
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `left` | Left | Number | `0` | `0` | `1` | `0.1` | `3` | Yes |
  | `top` | Top | Number | `0` | `0` | `1` | `0.1` | `3` | Yes |
  | `right` | Right | Number | `0` | `0` | `1` | `0.1` | `3` | Yes |
  | `bottom` | Bottom | Number | `0` | `0` | `1` | `0.1` | `3` | Yes |
- **Mathematical / Visual Behavior:**
  - Hard-clips the layer to the normalized rectangle $u \in [\text{left}, 1 - \text{right}]$, $v \in [\text{bottom}, 1 - \text{top}]$ (with $v = 0$ at bottom, $v = 1$ at top).
  - Pixels inside the rectangle copy `tDiffuse` verbatim (no scaling or translation); pixels outside output transparent `(0, 0, 0, 0)`. If `left + right >= 1` or `top + bottom >= 1`, the entire frame outputs transparent.

#### 11. `cubelut` — Cube LUT
- **Category:** Color & Tone | **Passes:** 1 | **Bypass:** `lut == null`
- **Parameters:**
  | Key | UI Name | Type | Default | Anim | Notes |
  |---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | Yes | `off;on` |
  | `lut` | Cube LUT file | Asset (`.cube`) | `null` | No | Adobe `.cube` 1D or 3D LUT |
- **Mathematical / Visual Behavior:**
  - Parses standard Adobe `.cube` metadata (`LUT_1D_SIZE` or `LUT_3D_SIZE` $N$, `DOMAIN_MIN`, `DOMAIN_MAX` defaulting to `[0,0,0]`..`[1,1,1]`).
  - Normalizes straight RGB: $c = \text{clamp}((\text{rgb} - \text{DOMAIN\_MIN}) / (\text{DOMAIN\_MAX} - \text{DOMAIN\_MIN}), 0, 1)$.
  - **1D LUT:** Samples a 1D $N$-entry RGB texture independently per channel $(R, G, B)$ with half-texel inset $(0.5 + c(N-1))/N$.
  - **3D LUT:** Maps $c$ into a 3D $N \times N \times N$ LUT (native `texture_3d<f32>` in `wgpu` with linear filtering and half-texel inset $(0.5 + c(N-1))/N$, which is mathematically identical to CM3's 2D-unwrapped trilinear interpolation). Re-premultiplies by source alpha.

#### 12. `directionalblur` — Directional Blur
- **Category:** Blur & Sharpen | **Passes:** 1 | **Bypass:** `delta == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `delta` | Delta | Number | `1` | `0` | `10` | `0.1` | — | Yes | Blur length |
  | `direction` | Direction | Number (rad) | `0` | `0` | `10` | `0.1` | — | Yes | Angle in radians |
- **Mathematical / Visual Behavior:**
  - Single-pass 1D linear motion blur along unit vector $\hat{d} = (\cos(\text{direction}), \sin(\text{direction}))$ divided by layer `resolution` and scaled by `delta`.
  - Averages symmetric premultiplied RGBA taps from $-L$ to $+L$ along $\hat{d}$ with uniform (or triangular) weights summing to 1.

#### 13. `displacementmap` — Displacement Map
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `texture` | Displacement map | Asset (`image/*`) | `null` | — | — | — | — | No | Repeat-wrapped linear image |
  | `displacementOffset` | Displacement offset | Vector2 | `[0, 0]` | — | — | `0.01` | `3` | Yes | Constant bias added to map |
  | `uDisplacement` | X displacement | Vector3 (R,G,B) | `[1, 0, 0]` | `-1` | `1` | — | `3` | Yes | Channel weights for horizontal shift |
  | `vDisplacement` | Y displacement | Vector3 (R,G,B) | `[1, 0, 0]` | `-1` | `1` | — | `3` | Yes | Channel weights for vertical shift |
  | `amount` | Amount | Number | `0.1` | — | — | `0.01` | `3` | Yes | Global displacement strength |
  | `offset` | Offset | Vector2 | `[0, 0]` | — | — | `0.05` | `3` | Yes | Map UV translation |
  | `scale` | Scale | Vector2 | `[1, 1]` | `0.01` | — | `0.05` | `3` | Yes | Linked aspect ratio; map UV scale |
  | `rotation` | Rotation | Number (deg→rad) | `0` | — | — | `3` | `1` | Yes | Map UV rotation around center |
  | `wrap` | Wrap | Option | `2` (`reflect`) | `0` | `2` | — | — | No | `clamp;tile;reflect` |
- **Mathematical / Visual Behavior:**
  - Computes aspect-corrected 2D inverse affine transform from `offset`, `scale`, `rotation` around `(0.5, 0.5)` and samples `tDisplacement` with repeat wrapping.
  - Evaluates displacement vector $d = (\text{dot}(m.\text{rgb}, u_{\text{disp}}) + \text{displacementOffset}.x,\ \text{dot}(m.\text{rgb}, v_{\text{disp}}) + \text{displacementOffset}.y)$, scales by `amount`, and adds to `vUv`.
  - Wraps displaced UV according to `wrap`: `0 = clamp` to $[0, 1]$, `1 = tile` ($\text{fract}(u)$), `2 = reflect` (mirror triangle wave $|2\,\text{fract}(u/2) - 1|$), then samples `tDiffuse`.

#### 14. `duplicate` — Duplicate
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `offset` | Offset | Vector2 | `[0, 0]` | — | — | `0.05` | `3` | Yes | Drag step `0.001` |
  | `multiplier` | Multiplier | Vector2 | `[3, 3]` | `0.001` | — | `0.01` | `3` | Yes | Linked aspect; tile repeat count |
  | `rotation` | Rotation | Number (deg→rad) | `0` | — | — | `3` | `1` | Yes | Rotation around center |
  | `repeat` | Repeat | Option | `0` (`tile`) | `0` | `1` | — | — | No | `tile;reflect` |
- **Mathematical / Visual Behavior:**
  - Applies an aspect-corrected 2D affine UV matrix centered at `(0.5, 0.5)` scaling by `multiplier`, rotating by `rotation`, and translating by `offset`.
  - Wraps coordinates using `tile` ($\text{fract}$) or `reflect` (mirrored ping-pong) and samples `tDiffuse`.

#### 15. `edgedetection` — Edge Detection
- **Category:** Stylize & Glitch | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `amount` | Amount | Number | `1` | `-50` | `50` | `0.2` | `1` | Yes |
- **Mathematical / Visual Behavior:**
  - Evaluates a 3×3 neighborhood edge filter (Sobel/Prewitt horizontal and vertical gradient magnitudes $G = \sqrt{G_x^2 + G_y^2}$ or Laplacian kernel across $\pm 1/\text{resolution}$) on RGB channels, scaled by `amount`, preserving source alpha.

#### 16. `exposure` — Exposure
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `exposure` | Exposure | Number | `0` | `-1` | `1` | `0.1` | — | Yes |
  | `gamma` | Gamma | Number | `0` | `-1` | `1` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - Converts straight sRGB to linear light, applies photographic exposure scaling $2^{\text{exposure} \cdot k}$ and tone-curve gamma adjustment controlled by `gamma`, converts back to sRGB, and re-premultiplies by source alpha. At `(exposure=0, gamma=0)`, the transform is the exact identity.

#### 17. `fisheye` — Fisheye
- **Category:** Distort & Transform | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `amount` | Amount | Number | `25` | `-50` | `50` | `0.2` | `1` | Yes |
- **Mathematical / Visual Behavior:**
  - Aspect-corrected spherical barrel/pincushion distortion centered at `(0.5, 0.5)`.
  - Converts `vUv - 0.5` into aspect-corrected polar coordinates $(r, \theta)$. Positive `amount` applies barrel (fisheye) warp compressing outer radii via $\tan/\arctan$ radial mapping; negative `amount` applies inverse pincushion warp. Samples `tDiffuse` at the remapped UV (returning transparent `(0,0,0,0)` for out-of-bounds UVs).

#### 18. `flip` — Flip
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Anim | Notes |
  |---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | Yes | `off;on` |
  | `horizontal` | Horizontal flip | Option | `1` (`on`) | `0` | `1` | Yes | `off;on` |
  | `vertical` | Vertical flip | Option | `0` (`off`) | `0` | `1` | Yes | `off;on` |
- **Mathematical / Visual Behavior:**
  - Mirrors UVs around `(0.5, 0.5)`: $u' = \text{horizontal} ? 1 - u : u$, $v' = \text{vertical} ? 1 - v : v$, then samples `tDiffuse`.

#### 19. `fxaa` — Antialiasing
- **Category:** Stylize & Glitch | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Anim |
  |---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | Yes |
- **Mathematical / Visual Behavior:**
  - Standard single-pass FXAA edge antialiasing using inverse resolution `(1/W, 1/H)` and constants `FXAA_REDUCE_MIN = 1/128`, `FXAA_REDUCE_MUL = 1/8`, `FXAA_SPAN_MAX = 8.0`.
  - Samples center plus 4 diagonal corner texels (`NW, NE, SW, SE`), computes local luma range `[lumaMin, lumaMax]` and edge tangent vector `dir = (-((lNW+lNE)-(lSW+lSE)), (lNW+lSW)-(lNE+lSE))`, normalizes by `min(|dir.x|, |dir.y|) + max(sum * 0.25 * REDUCE_MUL, REDUCE_MIN)`, clamps to `[-SPAN_MAX, SPAN_MAX]` pixels, and blends inner 2-tap and outer 4-tap samples along `dir`, falling back to the 2-tap average if the 4-tap luma falls outside `[lumaMin, lumaMax]`.

#### 20. `gaussianblur` — Gaussian Blur
- **Category:** Blur & Sharpen | **Passes:** 6 (3× [Horizontal + Vertical]) | **Bypass:** `delta == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `delta` | Delta | Number | `1` | `0` | `10` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - Approximates a smooth Gaussian blur by executing **3 consecutive iterations** of the separable 2-pass `boxblur` filter (H then V each iteration = 6 passes total) on premultiplied RGBA with step `delta / resolution`.

#### 21. `glitch` — Glitch
- **Category:** Stylize & Glitch | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `time` | Time | Number | expr `"time"` | — | — | `0.01` | — | Yes | Auto-animated via `"time"` expression |
  | `amount` | Amount | Number | `0.4` | `0` | `1` | `0.01` | — | Yes | Block glitch coverage/intensity |
  | `offset` | Displacement | Number | `0.1` | `0` | `1` | `0.01` | — | Yes | Max horizontal UV shift |
  | `speed` | Speed | Number | `4` | `1` | — | `0.1` | `1` | Yes | Glitch ticks per second |
- **Mathematical / Visual Behavior:**
  - Deterministic pseudo-random block-displacement and RGB channel-split glitch seeded by quantized time $t_q = \lfloor \text{time} \cdot \text{speed} \rfloor$.
  - Divides the frame into multi-scale horizontal strips/blocks; strips whose hash falls below `amount` are displaced horizontally by a seeded offset in $[-\text{offset}, +\text{offset}]$ and split R/G/B samples horizontally for digital corruption artifacts.

#### 22. `gradientoverlay` — Gradient Overlay
- **Category:** Overlay & Composite | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `gradient` | Gradient | Gradient | `[{pos:0, rgba(255,255,255,1)}]` | — | — | — | — | No | Baked to 1D RGBA LUT |
  | `gradientType` | Type | Option | `0` (`linear`) | `0` | `4` | — | — | No | `linear;radial;angular;reflected;diamond` |
  | `offset` | Offset | Vector2 | `[0, 0]` | — | — | `0.05` | `3` | Yes | Gradient center shift |
  | `scale` | Scale | Vector2 | `[1, 1]` | `0.01` | — | `0.05` | `3` | Yes | Linked aspect |
  | `rotation` | Rotation | Number (deg→rad) | `0` | — | — | `3` | `1` | Yes | Gradient rotation |
  | `opacity` | Opacity | Number | `1` | `0` | `1` | `0.1` | — | Yes | Overlay opacity |
  | `blending` | Blending mode | List | `"BLEND_SRC_OVER"` | — | — | — | — | No | Any `PZ.layer.blendModes` entry |
- **Mathematical / Visual Behavior:**
  - Applies aspect-normalized (`max(W,H)/W`, `max(W,H)/H`) inverse affine transform (`offset`, `scale`, `rotation`) around `(0.5, 0.5)` to produce local coordinate $p$.
  - Computes 1D gradient coordinate $t \in [0, 1]$:
    - `0` Linear: $t = p_x$
    - `1` Radial: $t = 2 \|p - (0.5, 0.5)\|_2$
    - `2` Angular: $t = \frac{\text{atan2}(p_y - 0.5, p_x - 0.5)}{2\pi} + 0.5$
    - `3` Reflected: $t = |2 p_x - 1|$
    - `4` Diamond: $t = 2(|p_x - 0.5| + |p_y - 0.5|)$
  - Samples gradient LUT at $\text{clamp}(t, 0, 1)$, multiplies alpha by `opacity`, and blends over `tBG` using `blending` mode with `OVERLAP_MODE = 3` (destination-alpha clipped).

#### 23. `grayscale` — Saturation
- **Category:** Color & Tone | **Passes:** 1 | **Bypass:** `saturation == 1`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `saturation` | Saturation | Number | `1` | `0` | `10` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - Computes luma $L = \text{dot}(\text{rgb}, (0.299, 0.587, 0.114))$ and interpolates $\text{rgb}' = \text{mix}((L, L, L), \text{rgb}, \text{saturation})$ (linear in premultiplied space, preserving alpha). `0` = full grayscale, `1` = unchanged, `>1` = oversaturated.

#### 24. `hueshift` — Color Shift
- **Category:** Color & Tone | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `amount` | Amount | Number | `0.5` | — | — | `0.01` | `3` | Yes |
- **Mathematical / Visual Behavior:**
  - Converts straight RGB to HSV/HSL, shifts hue cyclically $H' = \text{fract}(H + \text{amount})$ (where `1.0` = 360°), converts back to RGB, and re-premultiplies by source alpha.

#### 25. `lumakey` — Luma Key
- **Category:** Keying & Masking | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `threshold` | Threshold | Number | `0.5` | `0` | `1` | `0.01` | `3` | Yes | Luma cutoff |
  | `soften` | Soften | Number | `0.1` | `0` | `1` | `0.01` | `3` | Yes | Smoothstep half-width |
  | `invert` | Invert | Option | `0` (`off`) | `0` | `1` | — | — | No | `off;on` |
  | `mask` | Mode | Option | `0` (`result`) | `0` | `1` | — | — | No | `result;mask` |
- **Mathematical / Visual Behavior:**
  - Computes straight pixel luma $L = \text{dot}(\text{rgb}_{\text{straight}}, (0.299, 0.587, 0.114))$.
  - Evaluates matte $m = \text{smoothstep}(\text{threshold} - \text{soften}, \text{threshold} + \text{soften}, L)$; if `invert == 1`, $m \leftarrow 1 - m$.
  - In `result` mode outputs `texel * m`; in `mask` mode outputs `(m, m, m, 1.0)`.

#### 26. `mask` — Mask
- **Category:** Keying & Masking | **Passes:** 1 (2 draws or unified single fragment pass)
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `mode` | Mode | Option | `0` (`rectangle`) | `0` | `1` | — | — | No | `rectangle;ellipse` |
  | `position` | Position | Vector2 (px) | `[0, 0]` | — | — | `10` | `0` | Yes | Centered pixel coords |
  | `size` | Size | Vector2 (px) | `[250, 250]` | `0.01` | — | `10` | `0` | Yes | Linked aspect; half-extent/diameter |
  | `rotation` | Rotation | Number (rad) | `0` | — | — | `0.1` | — | Yes | Z-rotation in radians |
  | `invert` | Invert | Option | `0` (`off`) | `0` | `1` | — | — | No | `off;on` |
  | `feather` | Feather | Number | `0.1` | `0` | `1` | `0.1` | — | Yes | Edge softness fraction |
  | `opacity` | Opacity | Number | `1` | `0` | `1` | `0.1` | — | Yes | Mask strength |
- **Mathematical / Visual Behavior:**
  - In centered layer pixel coordinates $P \in [-W/2, W/2] \times [-H/2, H/2]$, transforms $P - \text{position}$ by inverse Z-rotation $-\text{rotation}$ and divides by $\text{size} \cdot (1 + \text{feather})$ to obtain local shape coordinates $q$.
  - Evaluates normalized shape distance $d$:
    - `rectangle`: $d = \max(|q_x|, |q_y|)$
    - `ellipse`: $d = \sqrt{q_x^2 + q_y^2}$
  - Computes smooth coverage $c = \left(1 - \text{smoothstep}\left(\frac{1}{1+\text{feather}}, 1.0, d\right)\right) \cdot \text{opacity}$ (clamped to $[0, 1]$, 0 outside $d > 1$).
  - Multiplies premultiplied source RGBA by $c$ when `invert == 0`, or by $(1 - c)$ when `invert == 1`.

#### 27. `mirror` — Mirror
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Anim | Notes |
  |---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | Yes | `off;on` |
  | `mode` | Mode | Option | `0` (`left`) | `0` | `7` | Yes | `left;right;top;bottom;left + top;right + top;left + bottom;right + bottom` |
- **Mathematical / Visual Behavior:**
  - Maps `mode` to active reflection half-planes:
    - `left` (0, 4, 6): if $u > 0.5$, $u \leftarrow 1 - u$.
    - `right` (1, 5, 7): if $u < 0.5$, $u \leftarrow 1 - u$.
    - `top` (2, 4, 5): if $v < 0.5$, $v \leftarrow 1 - v$ (where $v = 1$ is top).
    - `bottom` (3, 6, 7): if $v > 0.5$, $v \leftarrow 1 - v$.
  - Samples `tDiffuse` at the reflected `(u, v)`.

#### 28. `negative` — Negative
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Anim |
  |---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | Yes |
- **Mathematical / Visual Behavior:**
  - Inverts RGB in premultiplied space: $\text{out} = (\text{vec3}(a) - \text{rgb},\ a)$, preserving transparency cleanly.

#### 29. `overlay` — Image Overlay
- **Category:** Overlay & Composite | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `texture` | Image | Asset (`image/*`) | `null` | — | — | — | — | No | Overlay image |
  | `repeat` | Repeat | Option | `1` (`on`) | `0` | `1` | — | — | No | `off;on` |
  | `offset` | Offset | Vector2 | `[0, 0]` | — | — | `0.05` | `3` | Yes | UV translation |
  | `scale` | Scale | Vector2 | `[1, 1]` | — | — | `0.05` | `3` | Yes | Linked aspect |
  | `rotation` | Rotation | Number (deg→rad) | `0` | — | — | `3` | `1` | Yes | UV rotation |
  | `opacity` | Opacity | Number | `1` | `0` | `1` | `0.1` | — | Yes | Overlay opacity |
  | `blending` | Blending mode | List | `"BLEND_SRC_OVER"` | — | — | — | — | No | Any `PZ.layer.blendModes` entry |
- **Mathematical / Visual Behavior:**
  - Transforms `vUv` via aspect-corrected 2D inverse affine matrix (`offset`, `scale`, `rotation`) around `(0.5, 0.5)`.
  - If `repeat == 0` and transformed UV lies outside $[0, 1]^2$, overlay sample is transparent `(0,0,0,0)`; if `repeat == 1`, wraps via modulo.
  - Multiplies overlay sample by `opacity` and blends over `tBG` using `blending` with `OVERLAP_MODE = 3` (clipped to destination alpha). When `texture == null`, passes `tBG` through unchanged.

#### 30. `pixelated` — Pixelated
- **Category:** Distort & Transform | **Passes:** 1 | **Bypass:** `size == (1, 1)`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `size` | Pixel size | Vector2 (px) | `[16, 9]` | `1` | — | `1` | `0` | Yes | Linked aspect |
- **Mathematical / Visual Behavior:**
  - Quantizes screen pixel coordinates to block centers:
    $$\text{uv}' = \frac{(\lfloor \text{vUv} \cdot \text{resolution} / \text{size} \rfloor + 0.5) \cdot \text{size}}{\text{resolution}}$$
  - Samples `tDiffuse` at `uv'`.

#### 31. `pixelsort` — Pixel Sort
- **Category:** Distort & Transform | **Passes:** 2 (`sortPass` → `drawPass`)
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `threshold` | Threshold | Number | `1` | `0` | `1` | `0.01` | `3` | Yes | Span boundary luminance |
  | `direction` | Direction | Option | `0` (`vertical`) | `0` | `1` | — | — | No | `vertical;horizontal` |
  | `reverse` | Reverse | Option | `0` (`off`) | `0` | `1` | — | — | No | `off;on` (sort order) |
  | `affect` | Affect | Option | `0` (`shadows`) | `0` | `1` | — | — | No | `shadows;highlights` |
- **Mathematical / Visual Behavior:**
  - **Pass 1 (Span / Rank Map):** Along each 1D column (`vertical`) or row (`horizontal`), marks pixels whose luma satisfies $L < \text{threshold}$ (`affect = shadows`) or $L > \text{threshold}$ (`affect = highlights`). Within each contiguous marked span, computes the sorted index permutation by luminance (ascending, or descending when `reverse = on`) and writes span/source-offset coordinates into intermediate target `tMap`.
  - **Pass 2 (Gather Draw):** Reads `tMap` at each pixel and fetches the corresponding source texel from `tDiffuse`; unmarked pixels copy `tDiffuse` verbatim.
  - *WGSL Note:* In WebGPU, Pass 1 can execute via a 1D compute shader per row/column (fast workgroup span-scan + sort) writing directly to an output storage texture or index map for Pass 2, staying 100% deterministic with zero frame history.

#### 32. `posterize` — Posterize
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `gamma` | Gamma | Number | `0.6` | `0` | `1` | `0.01` | `2` | Yes |
  | `regions` | Colors | Number | `8` | `1` | `1000` | `0.1` | `1` | Yes |
- **Mathematical / Visual Behavior:**
  - On straight RGB (guarding $\gamma = \max(\text{gamma}, 10^{-4})$):
    $$c_1 = \max(\text{rgb}, 0)^{\gamma}, \qquad c_2 = \frac{\lfloor c_1 \cdot \text{regions} \rfloor}{\text{regions}}, \qquad \text{rgb}' = c_2^{1/\gamma}$$
  - Re-premultiplies by source alpha.

#### 33. `pulse` — Pulse
- **Category:** Distort & Transform | **Passes:** 1 | **Bypass:** `time % 1 == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `time` | Time | Number | `0` | `0` | — | `0.1` | — | Yes |
  | `amplitude` | Amplitude | Number | `300` | `0` | — | `0.1` | — | Yes |
  | `size` | Size | Number | `0.5` | `0` | `2` | `0.1` | — | Yes |
- **Mathematical / Visual Behavior:**
  - Expanding circular shockwave ripple centered at `(0.5, 0.5)` with cycle phase $\phi = \text{fract}(\text{time}) \in (0, 1)$.
  - Aspect-corrected radial distance $r = \|(\text{vUv} - 0.5) \odot (\text{aspect}, 1)\|_2$; ring center radius grows with $\phi$ and half-width scales with `size`.
  - Pixels within the ring are displaced radially outward/inward along $\hat{r}$ by a smooth sinusoidal/quadratic pulse envelope scaled by $\text{amplitude} / \max(W, H)$ and attenuated by $(1 - \phi)$ as the wave reaches the edge.

#### 34. `radialblur` — Radial Blur
- **Category:** Blur & Sharpen | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `density` | Delta | Number | `0.5` | `-1` | `1` | `0.05` | `3` | Yes | Ray step length (positive=zoom out, negative=zoom in) |
  | `dither` | Dither | Number | `0.5` | `0` | `1` | `0.05` | `3` | Yes | Per-pixel tap jitter to suppress banding |
  | `decay` | Decay | Number | `0.9` | `0` | `1` | `0.05` | `3` | Yes | Exponential per-tap falloff |
  | `center` | Center | Vector2 | `[0, 0]` | — | — | `0.05` | `3` | Yes | Offset from frame center `(0.5, 0.5)` |
  | `overbright` | Overbright | Option | `0` (`off`) | `0` | `1` | — | — | No | `off` sets `CONSTANT_BRIGHTNESS`; `on` allows additive god-ray bloom |
  | `weight` | Weight | Number | `0.2` | `0` | `1` | `0.05` | `3` | Yes | Sample accumulation weight |
- **Mathematical / Visual Behavior:**
  - Marches $N$ samples (e.g. $N = 32$) starting at `vUv` along step $\Delta = (\text{vUv} - ((0.5, 0.5) + \text{center})) \cdot \frac{\text{density}}{N}$, jittered by a deterministic screen-space hash $\in [0, 1)$ multiplied by `dither`.
  - Accumulates $S = \sum_{i=0}^{N-1} T(\text{vUv} - (i + j)\Delta) \cdot \text{decay}^i \cdot \text{weight}$ and total weight $W = \sum_{i=0}^{N-1} \text{decay}^i \cdot \text{weight}$.
  - When `overbright == 0` (`CONSTANT_BRIGHTNESS`), outputs $S / W$ (energy-preserving zoom blur); when `overbright == 1`, outputs $S$ directly (volumetric light-shaft glow).

#### 35. `rgbshift` — RGB Shift
- **Category:** Stylize & Glitch | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `amount` | Amount | Number (px) | `10` | — | — | `0.1` | — | Yes | Shift distance in pixels |
  | `angle` | Angle | Number (rad) | `0` | — | — | `0.1` | — | Yes | Shift axis angle |
- **Mathematical / Visual Behavior:**
  - Offset vector $\delta = (\text{amount} \cdot \cos(\text{angle}) / W,\ \text{amount} \cdot \sin(\text{angle}) / H)$.
  - Samples $c_R = T(\text{vUv} + \delta)$, $c_G = T(\text{vUv})$, $c_B = T(\text{vUv} - \delta)$.
  - Outputs $(c_R.r,\ c_G.g,\ c_B.b,\ \max(c_R.a, \max(c_G.a, c_B.a)))$ (or weighted alpha $\frac{c_R.a + c_G.a + c_B.a}{3}$ matching premultiplied fringe bounds so displaced channels remain visible over transparent backgrounds).

#### 36. `scanlines` — Scan Lines
- **Category:** Stylize & Glitch | **Passes:** 1 | **Bypass:** `opacity == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `color` | Color | Color (RGB) | `[0, 0, 0]` | `0` | `1` | — | — | Yes | Scanline stripe color |
  | `opacity` | Opacity | Number | `1` | `0` | `1` | `0.1` | — | Yes | Stripe opacity |
  | `thickness` | Thickness | Number (px) | `1` | `0.1` | — | — | `1` | Yes | Half-period stripe width |
  | `offset` | Offset | Number (px) | `0` | — | — | `1` | `1` | Yes | Phase shift along normal |
  | `feather` | Feather | Number | `0.25` | `0` | `1` | `0.01` | `3` | Yes | Edge softness |
  | `angle` | Angle | Number (deg→rad) | `π/2` (90°) | — | — | `1` | `1` | Yes | Stripe orientation (90° = horizontal lines) |
- **Mathematical / Visual Behavior:**
  - Projects pixel position $p = \text{vUv} \cdot \text{resolution}$ onto scanline normal $n = (\cos(\text{angle}), \sin(\text{angle}))$: $s = (\text{dot}(p, n) + \text{offset}) / (2 \cdot \text{thickness})$.
  - Evaluates periodic triangle wave $w = |2\,\text{fract}(s) - 1| \in [0, 1]$ and soft stripe mask $m = \text{smoothstep}(0.5 - 0.5\cdot\text{feather}, 0.5 + 0.5\cdot\text{feather}, w)$.
  - Composites `color` over `tDiffuse` at `m * opacity` (source-over / destination-alpha aware).

#### 37. `shake` — Shake
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `amplitude` | Amount | Number | `0.01` | `0` | `1` | `0.01` | `3` | Yes | Global shake multiplier |
  | `speed` | Speed | Number | `15` | `0` | — | `0.01` | — | Yes | Time scale (`t = frame * speed / 30`) |
  | `zoom` | Zoom | Number | `100` | `0` | — | `1` | `1` | Yes | Camera forward Z offset |
  | `wrap` | Wrap | Option | `2` (`reflect`) | `0` | `2` | — | — | No | `none;tile;reflect` |
  | `shake.*` | Shake Generator | Group (`PZ.shake`) | `mode=1`, `amp=[1000,1000,1000,10]` | — | — | — | — | Yes | Seeded 4-channel (XYZ + roll) shake generator |
- **Mathematical / Visual Behavior:**
  - Renders the layer texture onto a 3D quad placed at $z = -D$ where $D = \max(W, H)$, viewed by a perspective camera with $\text{aspect} = W/H$, $\text{fov} = 2\arctan(H / (2D))$, $\text{near} = 1$, $\text{far} = 2D$.
  - Camera starts at $(0, 0, -\text{zoom})$ and is perturbed in XYZ translation and Z roll by the deterministic `PZ.shake` curve evaluated at $t = \text{frame} \cdot (\text{speed} / 30)$ scaled by `amplitude`.
  - When `wrap == 0` (`none`), the quad has extents $[-0.5W, 0.5W] \times [-0.5H, 0.5H]$ with UV $[0, 1]$ (transparent outside). When `wrap == 1` (`tile`) or `2` (`reflect`), the quad expands 5× to $[-2.5W, 2.5W] \times [-2.5H, 2.5H]$ with UV $[-2, 3]$, wrapped by modulo (`tile`) or mirrored reflection (`reflect`).

#### 38. `sharpen` — Sharpen
- **Category:** Blur & Sharpen | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `delta` | Delta | Number | `1` | `0` | `10` | — | — | Yes |
- **Mathematical / Visual Behavior:**
  - 3×3 unsharp-mask convolution at texel offsets $\pm(1/W, 0), \pm(0, 1/H)$:
    $$\text{rgb}' = (1 + 4k)\,C - k\,(L + R + U + D)$$
    with gain $k$ proportional to `delta`, preserving source alpha and clamping premultiplied RGB to $[0, a]$.

#### 39. `shutter` — Shutter
- **Category:** Stylize & Glitch | **Passes:** 1 | **Bypass:** `opacity == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `mode` | Mode | Option | `1` (`linear doublesided`) | `0` | `2` | — | — | No | `linear;linear doublesided;radial` |
  | `color` | Color | Color (RGB) | `[0, 0, 0]` | `0` | `1` | — | — | Yes | Shutter blade color |
  | `opacity` | Opacity | Number | `1` | `0` | `1` | `0.1` | — | Yes | Shutter opacity |
  | `covered` | Cover | Number | `0.25` | `0` | `1` | `0.01` | `3` | Yes | Fraction of frame covered |
  | `angle` | Angle | Number (rad) | `π/2` | `0` | — | `1` | `3` | Yes | Wipe axis angle |
  | `fade` | Fade | Number | `0.001` | `0` | `1` | `0.005` | `3` | Yes | Edge feather width |
- **Mathematical / Visual Behavior:**
  - Evaluates shutter coverage mask $m \in [0, 1]$ with smooth edge transition width `fade`:
    - `0` Linear: single blade advancing across the frame along `angle` covering fraction `covered`.
    - `1` Linear doublesided: two symmetric blades closing inward from opposite edges along `angle` (classic cinema letterbox bars at `angle = π/2`) covering total fraction `covered`.
    - `2` Radial: circular iris closing inward from frame corners toward center as `covered` goes $0 \to 1$.
  - Composites `color` at `m * opacity` over `tDiffuse` (source-over).

#### 40. `static` — Static
- **Category:** Stylize & Glitch | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `size` | Size | Number (px) | `1` | `1` | — | `1` | `0` | Yes | Grain block size |
  | `amount` | Amount | Number | `0.5` | `0` | `1` | `0.1` | — | Yes | Noise intensity |
  | `blending` | Blending mode | Option | `1` (`add`) | `0` | `3` | — | — | No | `none;add;subtract;multiply` |
- **Mathematical / Visual Behavior:**
  - Quantizes pixel coordinates to $\lfloor (\text{vUv} \cdot \text{resolution}) / \text{size} \rfloor$ and evaluates a deterministic monochrome hash $n \in [0, 1]$ seeded by `(cell, frame)`.
  - Combines with premultiplied source $(C, a)$:
    - `0` none: $\text{mix}(C, n \cdot a, \text{amount})$
    - `1` add: $C + n \cdot \text{amount} \cdot a$
    - `2` subtract: $C - n \cdot \text{amount} \cdot a$
    - `3` multiply: $C \cdot \text{mix}(1, n, \text{amount})$

#### 41. `swirl` — Swirl
- **Category:** Distort & Transform | **Passes:** 1 | **Bypass:** `angle == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim |
  |---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes |
  | `angle` | Angle | Number | `0.2` | `-100` | `100` | `0.01` | `3` | Yes |
- **Mathematical / Visual Behavior:**
  - Computes aspect-corrected offset $p$ from center `(0.5, 0.5)` and distance $r = \|p\|_2$.
  - Within swirl radius $R$ (normalized frame radius), rotates $p$ by angle $\theta(r) = \text{angle} \cdot (1 - r / R)^2$ (zero for $r \ge R$), converts back to UV, and samples `tDiffuse`.

#### 42. `technicolor` — Technicolor
- **Category:** Color & Tone | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Anim |
  |---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | Yes |
- **Mathematical / Visual Behavior:**
  - Vintage 2-strip/3-strip Technicolor dye-transfer color matrix on straight RGB: remaps red and cyan-green separations (`R' = R`, `G' = 0.5*(G+B)`, `B' = 0.5*(G+B)` / subtractive dye-transfer matrix) emphasizing warm reds and teal cyans, then re-premultiplies by source alpha.

#### 43. `transform` — Transform
- **Category:** Distort & Transform | **Passes:** 1
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `cameraType` | Camera type | Option | `0` (`perspective`) | `0` | `1` | — | — | No | `perspective;orthographic` |
  | `cameraPosition` | Camera position | Vector3 | `[0, 0, 0]` | — | — | `1` | — | Yes | Scene-pixel units |
  | `cameraRotation` | Camera rotation | Vector3 (deg→rad) | `[0, 0, 0]` | — | — | `1` | — | Yes | XYZ Euler |
  | `imagePosition` | Image position | Vector3 | `[0, 0, 0]` | — | — | `1` | — | Yes | Relative to `-max(W,H)` |
  | `imageRotation` | Image rotation | Vector3 (deg→rad) | `[0, 0, 0]` | — | — | `1` | — | Yes | XYZ Euler |
  | `imageScale` | Image scale | Vector2 | `[1, 1]` | `0.001` | — | `0.1` | `3` | Yes | Linked aspect |
  | `uvScale` | UV scale | Vector2 | `[1, 1]` | — | — | `0.01` | `3` | Yes | Linked aspect; tiles texture inside quad |
  | `wrap` | Wrap | Option | `0` (`tile`) | `0` | `1` | — | — | No | `tile;reflect` |
- **Mathematical / Visual Behavior:**
  - Renders a textured 3D quad of size $W \times H$ centered at `(imagePosition.x, imagePosition.y, imagePosition.z - D)` with $D = \max(W, H)$, rotated by `imageRotation`, scaled by `imageScale`.
  - Camera at `cameraPosition`, rotated by `cameraRotation`:
    - `perspective`: $\text{fov} = 2\arctan(H / (2D))$, $\text{aspect} = W/H$, $\text{near} = 1$, $\text{far} = 2D$.
    - `orthographic`: $[-W/2, W/2] \times [-H/2, H/2]$, $\text{near} = 1$, $\text{far} = 2D$.
  - Fragment shader multiplies quad UV by `uvScale` around `(0.5, 0.5)`, wraps using `tile` or `reflect`, and samples `tDiffuse`. Pixels outside the projected quad remain transparent `(0,0,0,0)`.

#### 44. `wavy` — Wavy
- **Category:** Distort & Transform | **Passes:** 1 | **Bypass:** `amount == 0`
- **Parameters:**
  | Key | UI Name | Type | Default | Min | Max | Step | Decimals | Anim | Notes |
  |---|---|---|---|---|---|---|---|---|---|
  | `enabled` | Enabled | Option | `1` (`on`) | `0` | `1` | — | — | Yes | `off;on` |
  | `amount` | Amount | Number | `3` | `0` | — | `0.1` | — | Yes | Wave displacement amplitude |
  | `size` | Density | Number | `25` | `0` | `200` | `0.2` | — | Yes | Spatial wave frequency |
  | `time` | Time | Number | `0` | `0` | — | `0.2` | — | Yes | Wave phase offset |
  | `angle` | Angle | Number (rad) | `0` | — | — | `0.2` | — | Yes | Wave propagation direction |
- **Mathematical / Visual Behavior:**
  - Rotates centered UVs by `angle` (or projects along $\hat{d} = (\cos(\text{angle}), \sin(\text{angle}))$), computes sinusoidal offset $\Delta = \sin(\text{coord} \cdot \text{size} + \text{time}) \cdot (\text{amount} / 1000)$, displaces the orthogonal UV axis, and samples `tDiffuse`.

---

## 3. Custom Shader Effect Contract & GLSL ES 1.00 → Naga Shim

### 3.1 CM3 Custom Shader Effect Schema
- **Effect Type:** Native Custom Shader (`type: 1`).
- **Core Properties:** `enabled`, `name`, `fragShader` (GLSL ES 1.00 source string), and `customProperties` (array of user-defined uniform controls).
- **Custom Property Uniform Mapping:**
  - `NUMBER` → `uniform float <Name>;` (also compatible with `uniform int` / `uniform bool` via float-to-int/bool conversion).
  - `VECTOR2` → `uniform vec2 <Name>;`
  - `VECTOR3` / `COLOR` → `uniform vec3 <Name>;`
  - `VECTOR4` / `Color + alpha` (Shader+) → `uniform vec4 <Name>;` (straight RGBA).
  - `OPTION` (static dropdown) → compile-time `#define <Name> <index>` (and optional `Name_ItemLabel` constant when referenced).
  - `ASSET` (`type: 8`, `assetType: 0`, `accept: "image/*"`) → `uniform sampler2D <Name>;` (exact uniform name match).
  - **Shader+ Extended Kinds:**
    - `Gradient` → 256×1 RGBA8 straight-alpha `sampler2D <Name>` + `float <Name>_repetitions`.
    - `Curve` → 1024×1 RGBA16F (fallback RGBA8) LUT `sampler2D <Name>` + range uniform.
    - `Palette` → `uniform vec4 <Name>[16]; uniform int <Name>_count;`
    - `Text` → `uniform sampler2D <Name>; uniform float <Name>_aspect; uniform vec2 <Name>_size;` (plus `<Name>_grid`, `<Name>_count` in Atlas mode; max 2048 px side, 2000 chars, 256 glyphs; supports coverage & SDF with 0.5 edge).
    - `Path` → `uniform sampler2D <Name>Coverage;` (8-bit coverage), `uniform sampler2D <Name>Points;` (256×1 RGBA32F nearest-filtered arc-length samples in UV space), `uniform float <Name>Length;` (in `u` units), `uniform int <Name>Count;` (256 when resolved, 0 when missing).
    - `Time sample` (up to 4 per effect) → `uniform sampler2D <Name>;` sampling the layer *before* the current effect at `frame + offset` (in Frames or Seconds; offset `0` reads `tDiffuse` identically).
    - `Layer Input` (when enabled) → `uniform sampler2D <Name>;` capturing another top-level video track at `Source`, `Masks`, or `Effects & Masks`.

### 3.2 Vertex & Fragment Built-In Interface
- **Vertex Shader Outputs (Varyings):**
  - `varying vec2 vUv;` — Logical $[0, 1]^2$ layer UV (`uv / uvScale`, bottom-left origin).
  - `varying vec2 vUvScaled;` — Backing-target UV (`uv`). Preferred UV source in Panzoid Shader authoring.
- **Standard Built-In Fragment Uniforms:**
  - `uniform sampler2D tDiffuse;` — Premultiplied RGBA input from previous effect / layer draw.
  - `uniform vec2 resolution;` — Active layer width & height in pixels.
  - `uniform vec2 uvScale;` — Active viewport-to-backing-target scale (`(1.0, 1.0)` in ZoidCraft).
  - `uniform float time;` — Optional clip-local time in seconds (when declared or supplied by Shader+/Expressions+).
- **Coordinate & Authoring Conventions:**
  - Centered space: `vec2 p = vUvScaled * 2.0 - 1.0;`.
  - When `Position` (`vec2`) is present: subtract `Position` before applying `16.0 / 9.0` (or `aspect`) x-scaling.
  - `Rotation` (`float`): clockwise degrees. `Scale` (`vec2`): multiplier defaulting to `(1, 1)`.
  - **Premultiplied Alpha Source-Over Rule:** Custom shaders MUST output premultiplied RGBA to `gl_FragColor`, NEVER divide premultiplied RGB by output alpha, NEVER emit nonzero RGB where `outputAlpha == 0`, and NEVER use `discard` (return original `tDiffuse` or source-over outside the mask).

### 3.3 Supported GLSL ES 1.00 Preprocessor & Extensions
- Authored shaders MUST NOT include `#version 100` (the host prepends headers).
- Supported preprocessor directives: `#define` (object & function-like macros), `#undef`, `#if`, `#ifdef`, `#ifndef`, `#elif`, `#else`, `#endif`, with predefined macros `GL_ES = 1`, `GL_FRAGMENT_PRECISION_HIGH = 1`, `__VERSION__ = 100`.
- **Includes (Shader+ / Expressions+):** `// #include <zoidium/name>` lines outside block comments are expanded before GLSL translation (deduplicated, cycle/depth-bounded), maintaining an authored-line sourcemap.
- **Extensions:** Only `#extension GL_OES_standard_derivatives : require|enable|warn|disable` is supported (`dFdx`, `dFdy`, `fwidth`). Legacy WebGL 1 extensions (`GL_EXT_shader_texture_lod`, `GL_EXT_frag_depth`, `GL_EXT_draw_buffers`, `texture2DLodEXT`, `texture2DGradEXT`, `gl_FragDepthEXT`, `gl_FragData`) are rejected with clear diagnostics.

### 3.4 GLSL ES 1.00 → `naga` (GLSL 450) Translator Shim
Because `wgpu` compiles GLSL via `naga::front::glsl` (which targets **Vulkan-flavored GLSL 450**, not WebGL 1 / GLSL ES 1.00), ZoidCraft MUST implement a deterministic front-end shim (`zoidcraft-shader-shim`) before handing source to `naga`:

| Stage | GLSL ES 1.00 Input | Rewritten GLSL 450 Output for `naga` |
|---|---|---|
| 1. Includes & Macros | `// #include <zoidium/gradient>` + static OPTION values | Expand registered include library; inject `#define GL_ES 1`, `#define GL_FRAGMENT_PRECISION_HIGH 1`, and effect OPTION `#define`s; track line map |
| 2. Header & Extensions | `#extension GL_OES_standard_derivatives : require`, `precision highp float;`, optional stray `#version 100` | Prepend `#version 450\n`; strip `#version 100`, `GL_OES_standard_derivatives` directive, and standalone `precision ...;` lines (derivatives are core in GLSL 450) |
| 3. Varyings | `varying vec2 vUv;`, `varying vec2 vUvScaled;` | `layout(location = 0) in vec2 vUv;`, `layout(location = 1) in vec2 vUvScaled;` (canonical fixed locations matching fullscreen vertex shader) |
| 4. Fragment Output | `gl_FragColor = ...;` | Inject `layout(location = 0) out vec4 _zoid_fragColor;`; token-rewrite `gl_FragColor` → `_zoid_fragColor` (skipping comments/strings) |
| 5. Samplers | `uniform sampler2D tDiffuse;`, `uniform sampler2D MyTex;` | Assign deterministic bindings in Set 1: `layout(set = 1, binding = 0) uniform sampler2D tDiffuse;`, `binding = 1, 2, ...` for extra samplers (or separate `texture2D` + `sampler` bindings if decoupling filter modes) |
| 6. Texture Built-ins | `texture2D(s, uv)`, `texture2D(s, uv, bias)`, `textureCube(s, d)`, `texture2DProj(s, p)` | Rewrite to `texture(s, uv)`, `texture(s, uv, bias)`, `texture(s, d)`, `textureProj(s, p)` |
| 7. Non-Opaque Uniforms | Loose `uniform float Foo; uniform vec3 Color; uniform vec4 Palette[16];` | Collect all non-sampler uniforms into one `std140` UBO: `layout(set = 0, binding = 0, std140) uniform _ZoidUniforms { ... } _u;` with CPU-side std140 layout table; rewrite references to each uniform `Foo` → `_u.Foo` (respecting local variable shadowing & struct field access `.Foo`) |
| 8. Type & Semantics Guard | `uniform bool Flag;` / `uniform int Mode;` | In `std140`, store `bool` as `uint` (`_u.Flag != 0u`) and `int` as `int` (`i32`); convert float property values cleanly during UBO upload |
| 9. Compile & Fallback | Parse → Validate → Lower to WGSL/Naga | Run `naga::front::glsl::Frontend` + `naga::valid::Validator`; map diagnostic line numbers back through the Stage 1 sourcemap; on error, display inline error and retain the last valid compiled pipeline |

---

## 4. Export & Offline Audio Mixing Specification

### 4.1 Export Panel Parameters

| Parameter | Options / Type | Default | Notes |
|---|---|---|---|
| **Container / Format** | `MP4 (.mp4)`, `WebM (.webm)`, `MKV (.mkv)`, `PNG Sequence (.png)` | `MP4` | PNG sequence exports `frame_000001.png`… into chosen folder or `.zip` |
| **Video Codec** | `H.264`, `H.265 (HEVC)`, `VP9`, `VP8` (legacy MKV/WebM), `AV1`, `PNG` | `H.264` (MP4), `VP9` (WebM) | Optional hardware encoder acceleration (`h264_videotoolbox`, `h264_nvenc`, `h264_vaapi`) with software fallback (`libx264`, `libx265`, `libvpx-vp9`, `libsvtav1`) |
| **Rate Control Mode** | `Quality (CRF)`, `Bitrate (Mbps)`, `CM3 Quality Preset` | `Quality (CRF)` | CM3 presets (`Fastest`, `Fast`, `Balanced`, `Good`, `Extreme`) map to concrete CRF & encoder speed presets |
| **CRF / Quality** | Integer `0..51` (H.264/H.265) or `0..63` (VP9/AV1) | `18` (H.264), `24` (VP9) | Lower = higher quality |
| **Bitrate** | `0.5..200 Mbps` | `12 Mbps` (1080p) | Used when Bitrate mode selected; keyframe interval `-g <2 * fps>` |
| **Resolution** | `width × height` (`16..7680`) | Sequence resolution | Odd dimensions padded by 1 black pixel right/bottom for 4:2:0 codecs |
| **Frame Rate** | `1..240 fps` (presets: 24, 25, 30, 50, 60, 120) | Sequence rate | Frame stepping: `frameAdvance = seqRate / exportRate`, `frame_i = start + i * frameAdvance` |
| **Alpha Channel** | `Opaque (yuv420p)` / `Transparent (yuva420p / rgba)` | `Opaque` | Transparent supported on WebM (VP9), MKV (VP9/FFV1/PNG), and PNG Sequence |
| **Range** | `start` frame, `length` (frames) | Full sequence | `totalFrames = length`; `totalSamples = floor(length / exportRate * sampleRate)` |
| **Audio Codec** | `AAC-LC`, `Opus`, `PCM (s16le/f32le)`, `None` | `AAC-LC` (MP4), `Opus` (WebM) | Automatically omitted when no active audio track exists |
| **Audio Bitrate** | `64..320 kbps` (lossy) | `192 kbps` (AAC/Opus) | Ignored for PCM / None |
| **Audio Sample Rate** | `48000 Hz`, `44100 Hz` | `48000 Hz` | Internal offline mixer always runs at `48000 Hz`, stereo (2 ch) |
| **True Datamosh (Opt-in)** | `Off`, `Full video`, `At clip cuts` | `Off` | Only exposed when `openzoid-legacy` plugin is enabled |

### 4.2 Deterministic Offline Audio Mixer
Matches CM3's exact `PZ.export` audio pipeline:
1. **Schedule Resolution:** Collects all unmuted audio clips (including audio nested inside Precomp+ compositions with remapped time, volume, and pan) overlapping `[startSec, endSec]`. Video-only containers with no audio stream contribute silence and do not force an audio track.
2. **Source Decoding:** Decodes each required clip span into interleaved stereo 32-bit float PCM at 48 kHz via FFmpeg:
   ```text
   ffmpeg -v error -i <asset_path> -ss <offset_sec> -t <duration_sec> \
          -acodec pcm_f32le -ac 2 -f f32le -ar 48000 pipe:1
   ```
3. **Resampling & Time Mapping:** Clip playback rate is `rate = (time(clipLen) - time(0)) / clipLen * seqRate`. Each output sample $p$ maps to clip-local frame $f = ((p - \text{segOffset}) / 48000 + t_0) \cdot \text{seqRate} - \text{frameOffset}$ and source sample index $\lfloor (p - \text{segOffset}) \cdot \text{rate} \rfloor$.
4. **Volume & Stereo Pan Law (Exact CM3 Panning):**
   - Multiply stereo sample $(L, R)$ by animated `volume(f)`.
   - Let $u = \text{pan}(f) \in [-1, 1]$:
     - If $u \le 0$: $\theta = (u + 1)\frac{\pi}{2}$, then $L' = L + R \cos\theta$, $R' = R \sin\theta$.
     - If $u > 0$: $\theta = |u|\frac{\pi}{2}$, then $L' = L \cos\theta$, $R' = R + L' \sin\theta$.
5. **Summation & Clamping:** Sums all overlapping clips per sample and clamps each channel to $[-1.0, +1.0]$.

### 4.3 FFmpeg Pipe Protocol & CLI Mapping
To avoid dual-pipe OS buffer deadlocks, ZoidCraft runs a **two-stage / pre-mixed audio + raw video stdin pipe** protocol:
1. **Stage A (Audio Pre-Mix):** If the project has active audio, render the full deterministic stereo `f32le` (or `s16le`) PCM stream to a temporary file `<temp_dir>/export_audio.f32` (fast, sequential, supports progress & cancel; deleted on completion/cancel/error).
2. **Stage B (Video Frame Pipe + Mux):** Spawn `ffmpeg` reading raw RGBA frames from `stdin` (`pipe:0`) and the mixed audio file as input 1.
   - Each rendered frame is read back from `wgpu` (`COPY_SRC` → staging buffer), stripped of WebGPU 256-byte row alignment padding to tightly packed `width * height * 4` RGBA8 bytes (top-down row order, so no `vflip` needed), and written sequentially to `stdin`.
   - When `width` or `height` is odd for YUV420p codecs, append `-vf pad=ceil(iw/2)*2:ceil(ih/2)*2:0:0:black`.
   - Color metadata: `-colorspace bt709 -color_primaries bt709 -color_trc bt709`.
   - Parse FFmpeg `stderr` (`-progress pipe:2 -nostats`) for mux diagnostics; on user Cancel, close `stdin`, kill FFmpeg cleanly, and delete partial output + temp audio.

**Representative FFmpeg CLI Mappings:**
- **MP4 (H.264 + AAC-LC):**
  ```text
  ffmpeg -y -f rawvideo -pix_fmt rgba -s WxH -r FPS -i pipe:0 \
         -f f32le -ar 48000 -ac 2 -i <tmp_audio.f32> \
         -c:v libx264 -preset medium -crf 18 -g <2*FPS> -pix_fmt yuv420p \
         -colorspace bt709 -color_primaries bt709 -color_trc bt709 \
         -c:a aac -b:a 192k -movflags +faststart <output.mp4>
  ```
- **MP4 (H.265 / AV1):** Swap `-c:v libx265 -tag:v hvc1 -crf 20` or `-c:v libsvtav1 -crf 26 -preset 6`.
- **WebM (VP9 + Opus, optional alpha):**
  ```text
  ffmpeg -y -f rawvideo -pix_fmt rgba -s WxH -r FPS -i pipe:0 \
         -f f32le -ar 48000 -ac 2 -i <tmp_audio.f32> \
         -c:v libvpx-vp9 -crf 24 -b:v 0 -row-mt 1 -pix_fmt <yuv420p|yuva420p> \
         -c:a libopus -b:a 192k <output.webm>
  ```
- **MKV (H.264/VP9 + PCM/Opus):** Same video args with `-c:a pcm_s16le` or `-c:a libopus` into `<output.mkv>`.
- **PNG Sequence:** Written directly in Rust via `png` crate (`frame_%06d.png`, RGBA8, optional parallel encoder pool) without requiring FFmpeg.

---

## 5. Zoidium Plugin Architecture & Toggleable Native Plugins

ZoidCraft implements Zoidium's plugin catalog as **toggleable native Rust modules** behind the exact same UX, categories, safety locks, and compatibility guarantees as Zoidium.

### 5.1 Plugin Architecture & Contracts

| Requirement | Specification |
|---|---|
| **Categories** | 4 visible categories in Plugins panel: `utilities` (**Utilities**), `scene` (**Scene Extensions**), `adjustment` (**Adjustment Extensions**), and `experimental` (**Experimental**, collapsed by default with red/amber warning banner: *"These features are extremely experimental and may cause unexpected crashes."*). Hidden `core` (**Zoidium Core**) is always enabled, hidden from UI, and never written to project metadata. |
| **Enable / Disable Toggle** | Each visible plugin has a persistent toggle switch (respecting `defaultEnabled` on fresh install). Toggling takes effect live without restarting ZoidCraft and rebuilds open inspector panels/pickers. |
| **In-Use Lock** | A plugin **cannot be disabled** while the active project contains any object, effect, material, layer, sprite, constraint, composition, or track field owned by that plugin. Attempting to disable displays the blocking usages. |
| **Zero Impact on Vanilla CM3** | When a plugin is disabled—or when opening a vanilla CM3 project that uses no plugin features—rendering, transforms, UI pickers, and exported `.pz` output MUST be 100% identical to vanilla CM3. Plugins never mutate stock CM3 object/material/camera classes; they add separate namespaced types (`zoidium:<plugin>/<item>`). |
| **Unknown Data Preservation** | Loading a project while a required plugin is disabled keeps unknown objects, effects, materials, and namespaced layers as non-rendering **Missing Placeholders** that preserve their exact raw JSON payload (including unknown top-level fields, property channels, and track fields) on re-save. Opening a project listing disabled required plugins prompts the user to enable them and restores placeholders live. |
| **Determinism Contract** | Every plugin effect, simulation, noise, and temporal operator MUST be a pure deterministic function of `(project state, assets, seed, frame)`. Wall-clock time, unseeded RNG, live audio analysers, and previous-frame feedback rings are strictly forbidden. |
| **Frame Sampler & Cache Budgets** | Shared core temporal frame sampler supports up to **64 taps**, **8 nesting depths**, `preserveInputRGB` (no premultiplied double-darkening), and a **256 MiB** LRU retained RGBA8 target pool per root compositor. Editor output frame cache defaults to **512 MiB** (configurable `0..16384 MiB`, max `8192` entries) with idle background prefetch (up to 150 frames ahead / 8 behind after 0.8 s idle, skipped when sync draw > 50 ms or video media is present). |

---

### 5.2 Category 1: Utilities (`utilities`)

| Plugin ID | Display Name | Default | Zoidium-Only | Summary |
|---|---|---:|---:|---|
| `easing-plus` | Easing+ | Off | No | Normalized cubic Bezier curve editor with overshoot & presets |
| `expressions-plus` | Expressions+ | **On** | Yes | Deterministic JS-subset expressions, property links & Sequence Controls |
| `shader-plus` | Shader+ | **On** | No | 12+ typed Custom Shader property editors, groups, visibility & time samples |
| `player-plus` | Player+ | **On** | No | Drawing pause, single-frame render & preview resolution quality controls |
| `asset-manager` | Assets | **On** | No | AE-style Project panel: folders, labels, usages, replace, locate, consolidate, collect |
| `precomp-plus` | Precomp+ | Off | Yes | Nested compositions, Compositions window & nested audio scheduling |
| `gif-support` | Animated GIF | Off | Yes | Deterministic GIF87a/89a decoder driven by clip-local time |
| `graph-plus` | Graph Editor 2 | Off | No | Full F-curve graph editor with 33 presets, marquee, ruler scrub & follow-selection |
| `motion-path` | Motion Paths | **On** | No | AE-style editable 2D/3D viewport motion paths for Position |
| `mp4-export` | MP4 Export | **On** | No | H.264 MP4 + AAC/Opus export option in native Export panel |

#### Detailed Utilities Specs
1. **`easing-plus` (Easing+):**
   - Floating normalized $[0, 1]$ cubic Bezier curve editor for `interp_1` incoming segments (previous key → current key), committing as one undo step.
   - Converts normalized handles `(x1, y1, x2, y2)` to CM3's frame-relative / value-relative `controlPoints`; equal-value segments store zero Y offsets (stay flat).
   - Auto-rescales handles when keyframes move or change value so normalized curve shape is preserved; supports crossing X handles on `[0, 1]`.
   - Vertical overshoot unlocks independently above/below after dragging 40 px past boundary (disarms within 24 px).
   - Copy/Paste supports `cubic-bezier(x1, y1, x2, y2)`, comma/space-separated 4-tuples, and JSON arrays.
2. **`expressions-plus` (Expressions+):**
   - Deterministic sandboxed expression evaluator attached to any animatable property (Alt-click stopwatch).
   - Provides `time`, `frame`, `thisComp.frameDuration`, `thisComp.layer(name)`, `thisObject.property(name)`, `.value`, `.valueAtTime(sec)`, `.velocity` (per-second scalar/vector), `.speed` (magnitude), deterministic `wiggle(...)`, and **Sequence Controls** (`ctl_*` bindings also reachable in Custom Shaders with `zTime`/`zFrame`).
   - Compiles/validates AST deterministically with zero side effects or wall-clock reads.
3. **`shader-plus` (Shader+):**
   - Adds rich editors and baked uniform bindings to native Custom Shader properties without breaking vanilla storage:
     - **Numeric & Vector Kinds:** `Slider` (min/max/step/decimals/snap), `Angle` (rad stored / deg displayed), `Toggle` (dropdown or Checkbox display, Hold keys), `Dropdown` (integer index + optional `Name_Item` constants), `Static Dropdown` (`#define Name index`), `Seed` (randomize button), `Color + Alpha` (straight RGBA `vec4`), `Point` (legacy `vec2` load support).
     - **Rich Baked Kinds:** `Gradient` (inline stop + opacity-stop + midpoint editor, Constant/Linear/Ease/Cardinal, sRGB/Linear/OKLab, cyclic bar/wheel + repetitions → 256×1 RGBA8 texture), `Curve` (Single Graph or RGBA Color Curves with monotone or explicit cubic Bezier handles → 1024×1 RGBA16F/RGBA8 LUT + range), `Palette` (up to 16 RGBA swatches → `vec4[16]` + `_count`), `Text` (CM3 font asset rasterizer, Layout or Atlas mode, coverage or SDF, max 2048 px / 2000 chars / 256 glyphs), `Path` (references layer Mask or Shape+ path → `Coverage`, 256×1 float `Points`, `Length`, `Count`), `Time sample` (up to 4 per shader, offset in Frames or Seconds, animatable/expression-driven).
     - **Organization & Rules:** Collapsible `Group`s, `Visible when` (`=`, `!=`, `<`, `<=`, `>`, `>=`, `a..b`, `;`-OR syntax; row-only visibility, hidden controls still bind), `Rename when` (dynamic display labels), and **Reset custom properties to defaults** (clears keyframes/expressions to static frame-0 default in one undo step).
     - **Includes:** `// #include <zoidium/gradient|curve|text|...>` expanded via shared shader pipeline.
4. **`player-plus` (Player+):**
   - Adds playback-bar controls: **Pause Drawing** (suspends viewport redraws while editing heavy scenes), **Single Render** (triggers one frame draw while paused), and **Preview Quality** resolution scale selector (Full, 1/2, 1/4, 1/8) affecting editor preview only (never export).
5. **`asset-manager` (Assets):**
   - After Effects–style Project panel listing every file, image, video, audio, font, 3D model, embedded plugin data (e.g. IES), and composition, with live usage counts, reveal-in-panel navigation, and missing/failed diagnostics.
   - Supports hierarchical **Folders** and 16 AE **Label colors** (saved in root `zoidiumAssets`), `Import`, `Replace` (single undo step updating all live references & keyframes), `Locate` (SHA-256 verified recovery of missing files), `Reload`, `Export File`, `Remove Unused`, `Remove Items` (refuses items used in closed compositions), `Consolidate All Footage` (deduplicates identical bytes across file/preset/data-URL items), and `Collect Files` (exports organized ZIP + `Collect Files Report.txt`).
6. **`precomp-plus` (Precomp+):**
   - Groups selected video/audio clips into reusable nested compositions managed in a **Compositions** window (New, Open, Rename, Duplicate, Delete with in-use guard, Back to Main).
   - Deterministic nested rendering & nested audio scheduling (remapping time, volume, pan, and muted track flags into Main time). Preserves track `enabled` and Camera+ `track3d` switches.
   - Prevents direct/indirect circular composition references; blocks export while editing inside a nested comp; maintains OpenZoid/Davidium `isComp`/`compLength`/`compSource` JSON compatibility.
7. **`gif-support` (Animated GIF):**
   - Deterministic GIF87a/89a decoder (LZW, interlacing, global/local palettes, transparency, disposal modes 0–3, capped at 512 frames and 64 megapixels).
   - Drives displayed frame purely from clip-local media time (`clip.properties.time`, looping cleanly; project time fallback where unclipped) across 2D image layers, 3D Image materials, and effect image inputs. Awaited during export `prepare()`.
8. **`graph-plus` (Graph Editor 2):**
   - Replaces stock graph view with a resizable F-curve editor supporting: Linear/Step/Ease/Auto/Break tangents, all 33 CM3 easing presets with thumbnails, Pan/Marquee (default), Fit (`\`), Frame (`F`), Normalize (`N`), Snap (`S`), +Key (`K`), Delete, Bezier (`I`), Link (`T`), Flat (`R`), Copy/Paste (preserving axis names), Labels, and persistent **Follow selection**.
   - Vector/color parent + per-axis rows with Eye & Solo toggles, dashed read-only expression curves, mouse-anchored wheel/pinch zoom, ruler playhead scrubbing, and Ctrl/Shift axis-locked key dragging. Right-click property → **Show in Graph Editor** or `Alt+Shift+G`. Coexists cleanly with Easing+.
9. **`motion-path` (Motion Paths):**
   - Viewport overlay (`O` shortcut or viewport toggle, remembered locally) drawing animated 2D layer and 3D object Position trajectories (blue curve, frame dots up to 900 frames, key squares, playhead ring, yellow Bezier tangent handles).
   - Direct dragging of keys (Shift constrains to dominant axis; 3D drags project onto camera-perpendicular plane through key) and Bezier handles (Alt breaks tangent continuity; Alt-click toggles linear/Bezier).
   - Writes **only native CM3 Position keyframe fields** (`animated`, `frame`, `value`, `tween`, `controlPoints`, `continuousTangent`) — zero custom project schema, fully readable in vanilla CM3.
10. **`mp4-export` (MP4 Export):**
    - Exposes H.264 MP4 export with AAC-LC (preferred) or Opus audio in the native Export panel (in ZoidCraft's native Rust desktop app, backed directly by the Section 4 FFmpeg pipeline rather than browser WebCodecs, preserving the exact toggle/UX contract).

---

### 5.3 Category 2: Scene Extensions (`scene`)

| Plugin ID | Display Name | Default | Zoidium-Only | Items | Summary |
|---|---|---:|---:|---:|---|
| `light-plus` | Light+ | Off | No | 4 | Chooser for Spot, Point, Directional & Hemisphere lights |
| `geometry-plus` | Geometry+ | Off | Yes | 8 | Rounded box, low-poly polyhedra & procedural 3D primitives |
| `repeater` | Effector | **On** | Yes | 7 | Repeater family (4) + Twist, Warp & Voronoi Fracture deformers |
| `effector-plus` | Effector+ | Off | Yes | 0 | Legacy loader mapping Davidium numeric types 7/8/9 → `zoidium:repeater/*` |
| `material-plus` | Material+ | Off | Yes | 5 | Node Material, PBR+, Matcap, Image+, UV Custom (arc-length unwrapped UVs) |
| `text-plus` | Text+ | **On** | Yes | 7 | 2D Text 3-stack character animators + 7 3D Text parents & advanced bevels |
| `expresstext-plus` | ExpressText+ | **On** | Yes | 11 | Tracked-letter 3D text effector stack + Split-into-letters command |
| `particles-plus` | Particles+ | Off | Yes | 54 | 54 procedural monochrome 128×128 RGBA8 particle sprites |
| `trapcode-suite` | Rowbyte & Red Giant Suite | Off | Yes | 14 | Particular (10), Form (11), Plexus (12) + 8 C4D lights (incl. Area LTC & IES) |
| `optical-flares` | Optical Flares | Off | Yes | 5 | 3D projected lens flares (type 13) with depth occlusion & 12 element types |
| `camera-plus` | Camera+ | Off | Yes | 1 | Camera+ object (Film/DOF/Motion Blur/Vibrate), Camera layers (9) & 3D tracks |
| `shape-plus` | Shape+ | **On** | No | — | Animated vector layers, pen masks, SVG import & preview path editing |
| `constraints` | Constraints | **On** | Yes | — | Track To & cross-scene Follow constraints with dependency graph |

#### Detailed Scene Extensions Specs
1. **`light-plus` (Light+):**
   - Upgrades the 3D Scene Light picker to choose among 4 standard light types: **Spot**, **Point**, **Directional**, and **Hemisphere** (sky + ground colors, intensity, shadow params where applicable).
2. **`geometry-plus` (Geometry+):**
   - Registers 8 procedural 3D shape classes under `zoidium:geometry-plus/*`: rounded boxes (configurable bevel radius & segments), low-poly polyhedra (tetrahedron, octahedron, dodecahedron, icosahedron), capsule/lathe/torus-knot/superquadric procedural primitives, with standard CM3 shading & dynamic numeric properties.
3. **`repeater` (Effector) & `effector-plus` (Compatibility):**
   - Publishes 7 objects under `zoidium:repeater/*` (plus `effector-plus` mapping legacy Davidium numeric types `7 → twist`, `8 → warp`, `9 → voronoi-fracture`):
     - **4 Repeaters:** `Repeater`, `Linear Repeater`, `Random Repeater`, `Echo Repeater` — procedurally instance child objects with deterministic offsets, scales, rotations, and seeded randomness.
     - **3 Deformers:** `Twist` (axial twist), `Warp` (bend/shear warp), `Voronoi Fracture` (convex bisector-plane clipping of source triangles into 1–1000 cells with planar ear-clipped caps, open-mesh guard, optional random fragment colors, dual outer/inner material clones with polygon offset, and source-normal transport).
   - **Shared Evaluation Rules:** Rebuilds from pristine immutable source geometry every frame in effector-local space (skips singular transforms). Uniform quadrisection subdivision (`1, 2, 4, 8, 16` edge segments, capped at 400,000 tris). Voronoi limits: 160k source tris, 1.6M output tris, 4,096 cap contour verts, 80M clip ops; caches up to 64 topology stages within 64 MiB so animating motion/fields only updates vertices.
   - **Shared Fields:** Animatable falloff fields: `0: Infinite`, `1: Linear`, `2: Box`, `3: Sphere`, `4: Cylinder`, `5: 3D Value Noise` (XYZ Euler degrees rotation, percentage scale).
4. **`material-plus` (Material+):**
   - Adds 5 distinct 3D materials without altering native CM3 materials:
     - **UV Custom:** Identical properties to CM3 Custom Material, but uses **corrected arc-length UV unwrapping** on 3D Text and extruded shapes (caps use text bounds, side walls unwrap by outline arc length in U and extrusion depth in V, uniform texel density across all faces; restores original UVs when switched back).
     - **Matcap:** View-space normal lookup into a Matcap texture + dynamic `Color`, `Brightness`, `Opacity`, Transparency, Blending, Render Side (unlit by scene lights).
     - **PBR+:** Full physical material with independent texture maps (`Base Color`, `Emissive`, `Roughness`, `Metalness`, `Normal`, `Alpha`), shared UV `Wrap/Repeat/Offset/Center/Rotation`, `Normal scale`, `Reflection intensity`, scene environment reflection, plus color ramps, Fresnel, cel shading, and projection controls.
     - **Image+:** Image-mapped material with dedicated animatable opacity and corrected UVs.
     - **Node Material (`nodes`):** Project-owned JSON graph (`schemaVersion: 1`, max 200 nodes / 400 links) baking `128..2048` px mipmapped maps (`Color`, `Luminance`, `Emission`, `Tangent Normal`, `Roughness`, `Metalness`, `Bump`, `Opacity`) + shader `Fresnel` on a background worker thread. Nodes include Value Noise, Ridged Fractal, Turbulence, Gradients, Checker, Image/Texture, Fresnel, Math (arithmetic/comparison/trig/rounding, signed unclipped floats), Color Correction, Colorizer, Spectra, Blackbody, Random Color, Projection, UVW Transform. Deterministic fixed seeds; lattice capped at 1024.
5. **`text-plus` (Text+):**
   - **Native 2D Text (Type 7):** Adds up to 3 stacked `Character` / `Character 2` / `Character 3` animator categories (stored in `zoidiumTextPlus2D`, ignored cleanly when disabled). Transforms glyph paths in 2D (`Position XY`, `Scale XY`, `Rotation`, `Shake`, character selection, delay, gradient order).
   - **3D Text Parents (7 types):** `Character Transform`, `Gradient Transform`, `Character Shake`, `Selected Character Transform`, `Selected Character Shake`, `Random Scatter`, `Character Material` (separate cap/side materials). Each splits child 3D Text into per-character meshes around `Character` or `Text` pivot. Shake formula: $\theta = \text{amplitude} \cdot \sin(2\pi(\text{phase} - \text{charOffset}) / \text{period})$ around normalized local 3D `Axis`.
   - **Advanced 3D Bevels:** Adds `Horizontal spacing`, `Bevel size (thickness, width)`, `Bevel side (outside|inside)`, `Bevel subdivision (1..100)`, `Cap profile (flat|round)`, and `Bevel tension (0..1)` to native 3D Text geometry.
6. **`expresstext-plus` (ExpressText+):**
   - Container object placed above 3D Text that tracks individual letters and hosts an inline stack of **11 effectors** (added via scoped Custom Properties `+`):
     `Shake`, `Plain` (letter ranges `"1,4,5"`, `"2-6"`, invert odd), `Random` (Uniform/Gaussian/Noise/Turbulence/Sorted, Global/UV space, Synchronized/Indexed), `Formula` (safe sandboxed degree-trig expression with `u, t, id, count, falloff`), `Delay` (Even/Blend/Spring deterministic chase), `Spline` (Line/Circle/Helix path, Step/Falloff/Relative, Heading/Pitch/Bank/Tilt), `Orbit` (rigid XYZ orbital revolution), `Repeater` (Linear/Random letter ghosts), `Twist`, `Wrap`, and `Voronoi`.
   - Also adds **Split into letters** button on any 3D Text row (replaces text with a Group containing one 3D Text per visible glyph at exact layout offsets, moving parent animation onto the Group in one undo step).
7. **`particles-plus` (Particles+):**
   - Adds **54 procedural monochrome particle sprites** (128×128 8-bit RGBA PNGs, pure `#FFFFFF` RGB with supersampled coverage/glow solely in alpha, transparent safety border):
     - **31 Geometric & Pattern:** `circle`, `square`, `triangle`, `pentagon`, `hexagon`, `octagon`, `diamond`, `ring_outline`, `ring_double`, `triangle_outline`, `square_outline`, `rounded_square`, `rounded_square_outline`, `hexagon_outline`, `plus`, `plus_cut_square`, `cross`, `cross_outline`, `line_horizontal`, `line_vertical`, `grid`, `dot_grid`, `radial_lines`, `spiral`, `snowflake`, `lattice`, `concentric_rings`, `broken_ring`, `star_5`, `star_6`, `star_8`.
     - **23 Glow & Flare:** `flare_glow_ring`, `flare_halo`, `flare_double_halo`, `flare_blob`, `flare_orb_soft`, `flare_burst`, `flare_burst_soft`, `flare_needles`, `flare_starburst_fine`, `flare_starburst_dust`, `flare_eight_ray`, `flare_streak_thin`, `flare_streak_wide`, `flare_streak_glow`, `flare_vertical_streak`, `flare_crossbeam`, `flare_fourdot`, `flare_fourdot_soft`, `flare_sixdot`, `flare_ring_streak`, `flare_ripple`, `flare_spectral`, `flare_glint`.
8. **`trapcode-suite` (Rowbyte & Red Giant Suite):**
   - Clean-room deterministic 3D generative objects & C4D-style lights (14 items total):
     - **Particular (type 10):** Fixed-timestep deterministic 3D particle simulator with emitter shapes, inherited velocity, mass/air-resistance, orient-to-motion streaks, Aux parent-emission trails, Displace Spin (orbital tangent motion), Fluid vortex & buoyancy fields (core size, tilt, rotate, viscosity, time scale), gravity direction, Y=0 bounce plane, turbulence size/opacity modulation, layer maps, and offline audio reactors (analyzed from decoded audio at project time).
     - **Form (type 11):** Deterministic 3D particle lattice (`box`, `sphere`, `sphere grid`, `cylinder`, `circle`, `plane`, `3D model`, `text/mask image`) deformed by disperse, twist, spherical fields, fractal noise field, fluid motion, and kaleidospace mirrors; connecting neighbor strings; layer maps (color/alpha, displacement, size, fractal, disperse, rotate); offline audio react; post-deformation Transform with selectable 6 Euler orders (default `XYZ`).
     - **Plexus (type 12):** Point-cloud generator & procedural renderer (`points`, `lines` by distance/adjacency/shape, `facets`, `triangulation`, `beams`) with modular Geometry sources (`layers`, `paths`, `OBJ`, `primitives`, `instances`, `slicer`) and Effectors (`noise`, `spherical field`, `container`, `transform`, `color map`, `shade`, `sound`).
     - **8 C4D-Style Lights:** `Point (Omni)`, `Spot`, `Infinite (Directional)` (Kelvin temperature + shadow radius/coverage), `Area Light` (rectangular `RectAreaLight` via LTC tables + Width/Height, 9-spot fallback), `Hemisphere (Sky/Ground)`, `Photometric IES Light` (LM-63 Type C `TILT=NONE` parser + Uniform/Downlight/Wall-wash presets, 128×64 angular candela LUT, max 16 per project), `Sun (Directional)` (azimuth/elevation driving color temp, intensity, sky fill), `Portal Light` (one-sided window fill driven by scene Sun/Hemisphere).
9. **`optical-flares` (Optical Flares):**
   - 3D lens flare object (`type 13`) projected through active scene camera from `Object 3D` (default, new flares placed at `(-50.9, 28.6, 0)` with Brightness 70, Scale 75), `Light` (by index), or `Screen (2D)`.
   - Lays out flare elements along the axis from projected source through screen `Center` `(0, 0)`, fading smoothly behind camera or past `Edge Margin`, with `Distance Falloff` scaling `(refDist / dist)^falloff`.
   - **Per-Pixel Depth Occlusion:** Drawn at source depth (two-pass blend controlled by `Fade Amount` 0..100: depth-tested portion + untested remainder); excluded from velocity & env-map passes.
   - **Catalog & Editor:** 12 element types, 5 canonical flare stacks, 3 tint preset families (Cinematic/Rainbow/Gold), 3 custom image layers, per-element Hide/Solo/drag-reorder, seeded deterministic flicker/shimmer.
10. **`camera-plus` (Camera+):**
    - Separate `Camera+` object (never patches stock CM3 type-6 camera; first active Camera+ in depth-first order drives the scene) + `Camera` timeline layer (`type 9`) + video track **3D** toggle (`track3d`).
    - **Film Controls:** Position, Rotation & order, Focal Length, Sensor Size (film gate), read-only 35mm equivalent, editable H/V FOV (writes focal length), Zoom, Film Offsets, Perspective / Orthographic.
    - **Depth of Field:** Per-scene (or shared sequence depth pass for Camera layer) post-blur with Focus Distance, Aperture, Focus Area Width, Near/Far blur levels (foreground out-of-focus bleeds over sharper background; sharp foreground occludes blurred background). Focus Tools (`Link` expression `focusDistanceTo(...)`, `Set Once` keyframe, `Unlink`).
    - **Camera Motion Blur:** Deterministic centered shutter up to 2 frames, 1–8 freshly evaluated subframes averaged; plus sequence multisample `Track Frame` (`auto|previous|next`), `Motion Sensitivity`, and per-layer `Motion Blur` shutter scale.
    - **Vibrate:** Pure seeded hash noise on position/rotation from `(seed, amplitude, frequency, time)`.
    - **3D Tracks:** Flat clips on `track3d` tracks render double-sided in the shared camera's world space with `Depth`, `Rotation X`, `Rotation Y`; master camera priority = Camera layers → Camera-named Scenes → shallower nesting → lowest track index.
11. **`shape-plus` (Shape+):**
    - Animated 2D vector layer (`zoidium:shape-plus/shape-layer`, `schemaVersion = renderVersion = 1`), centered composition-pixel coordinates (+Y up).
    - **Contents Tree (AE Paint-Scope Rule):** `group` (anchor, pos, scale, rot, isolated opacity), `rectangle` (size, pos, roundness), `ellipse` (4 cubics), `polystar` (star/polygon, integer point count, inner/outer radius & roundness), `path` (`vertexIds`, `vertexModes`, static closed/open, 6 animatable scalar channels `[x, y, inX, inY, outX, outY]` per vertex with relative tangents), `fill` (RGB, opacity, nonzero/even-odd), `stroke` (RGB, opacity, width, butt/round/square cap, miter/round/bevel join, miter limit), `gradient fill`, `gradient stroke`, path modifiers (Trim Paths, etc.), `repeater`, and `merge paths`.
    - Also provides interactive **Pen Masks** on Scene & Adjustment layers, **SVG Import**, and direct viewport Bezier path editing.
12. **`constraints` (Constraints):**
    - Adds two deterministic object constraints evaluated after native/extension updates in topological dependency order (skipping & warning on cycles):
      - **Track To:** Aims local `Track axis` (default `+Z` objects, `-Z` camera) at same-scene target (`zoidiumId`), keeping `Up axis` (default `Y`) closest to world `+Y` with `Influence` `0..100%` (quaternion slerp in local space through full parent transform, handling negative/nonuniform parent scale; Spot/Directional lights blend native Target position).
      - **Follow:** Copies `Position`, `Rotation`, and/or `Scale` (with offsets/multipliers) in `World` (default) or `Local` space from a target in the same scene or another scene layer in the same sequence (sampled at requested sequence time minus clip start, clamping inactive clips to endpoints, reading muted tracks). Runs before Track To; deleting a target records a hidden local-pose fallback.

---

### 5.4 Category 3: Adjustment Extensions (`adjustment`)

| Plugin ID | Display Name | Default | Zoidium-Only | Items | Summary |
|---|---|---:|---:|---:|---|
| `native-fx` | Native FX | Off | Yes | 6+ | Echo, Time Offset, Posterize Time, Time Remap, Time Displacement, Color Curves, Radial Blur Spin, Drop Shadow, Smart Motion Blur |
| `node-effect` | Node Effect | **On** | No | 1 | Visual shader node graph compiling to native Custom Shader (`type 1`) |
| `magic-looks` | Magic Looks | Off | Yes | 29 | 29 reorderable color/light/lens tools (up to 64 instances) + presets & 3D LUTs |
| `openzoid-legacy` | OpenZoid Legacy | Off | Yes | 12 | Echo/Posterize Legacy, JPEG Damage, VHS, Datamosh, Tracery, ASCII |
| `smart-motion-blur-lite` | ZSMB (beta 1.0) | Off | Yes | 1 | Lightweight optical-flow motion blur (single adjacent frame) |
| `sglow-hq-v2-preview` | S_Glow (beta 1.0) | Off | Yes | 1 | High-precision multi-scale glow with per-channel spread & atmosphere |
| *Shader Packs* | AfterClip / Afterzoid / CCFX | Off | No | 43 / 200 / 19 | Declarative Custom Shader + Preset Group pack loader format |

#### Detailed Adjustment Extensions Specs
1. **`native-fx` (Native FX):**
   - **Temporal Operators (`defineTemporal`):**
     - `Time Offset`: Shifts layer/prefix evaluation time by animated offset (frames/seconds).
     - `Posterize Time`: Quantizes layer evaluation time to discrete buckets at target FPS ($\lfloor t \cdot \text{fps} \rfloor / \text{fps}$).
     - `Time Remap`: Maps output frame to arbitrary animated source time, with optional frame-sampler cross-fade blending between integer frames.
   - **Multi-Frame Samplers (`defineFrameSampler`):**
     - `Echo`: Deterministic multi-tap temporal echo combining explicitly sampled earlier/later frames (Add, Maximum, Minimum, Screen, Composite in Front/Back, Blend) with exponential decay; never uses feedback history.
     - `Time Displacement`: Per-pixel temporal offset driven by luminance of the layer itself, an `Image` asset, or (opt-in) another track via `Layer Input`.
     - `Smart Motion Blur` (`smartmotionblur`): Dual-direction optical-flow estimation + centered shutter integration over premultiplied input.
       - *Properties:* `Blur Amount` (def `0.5`, `0..1`, shutter angle = `value × 360°`), `Motion Sensitivity` (def `70`, `0..100`), `Vector Smoothness` (def `50`, `0..100`), `Max Displacement` (def `256`, `1..256` px/frame), `Analysis Frames` (`Both`|`Previous`|`Next`), `Quality` (`Balanced` 4-px mesh / 32 taps, `High` 2-px mesh / 64 taps), `View` (`Result`|`Motion Vectors`|`Confidence`).
       - *Behavior:* Bypasses cleanly at zero Blur Amount / Sensitivity, identical neighbors, or high motion-compensated residual (scene-cut detection). Mirrors available direction at clip boundaries.
   - **Spatial & Direct-Control Filters:**
     - `Color Curves`: Inline RGB + Composite + Alpha spline curves editor (monotone cubic interpolation, 1024-entry LUT, composite applied before per-channel R/G/B, independent Alpha).
     - `Radial Blur Spin`: Rotational/angular blur spinning around a configurable center point (polar tap integration).
     - `Drop Shadow`: Multi-pass alpha-extract → colorize → offset → separable blur → composite behind source (or shadow-only).
2. **`node-effect` (Node Effect):**
   - Visual GPU node graph authoring tool that compiles directly to standard CM3 **Custom Shader** (`type: 1`) with graph JSON appended as a `// @zoidium-node-effect` comment — **zero render-time plugin dependency** (saved shaders keep rendering if Node Effect is disabled!).
   - **Limits:** Max 128 nodes, 256 links, 32 Color Ramp stops, 192 KiB compiled shader, 256 source texture samples per pixel (reachable from Output); rejects cycles/invalid links while retaining last valid shader.
   - **Nodes:** `Output` (premultiplies straight RGB by Alpha), `Layer Color`, `Sample Layer at UV`, `UV`, `Pixel Coordinate`, `Resolution`, `Time` (clip-local seconds), `Parameter` (Float/Color/Vec2 with stable ID-based uniform names so renaming keeps keyframes), `Math` / `Mix` (matching Material+), `Color Ramp` (Constant/Linear/Ease/Cardinal), `2D Value Noise`, `Voronoi` (9-cell F1/F2, 25-cell edges), `Box Blur` & `Gaussian Blur` (up to 5×5 grid on premultiplied source), `Directional Blur` & `Radial Blur` (1–32 taps), `Edge Detect` (9-tap Sobel).
   - **Built-In Recipes:** Glow, Chromatic Aberration, Displacement by Noise, Duotone.
3. **`magic-looks` (Magic Looks):**
   - Reorderable chain of up to **64 independent tool instances** drawn from **29 color, light, and lens tools** (Subject/Matte/Lens/Camera/Post categories: Exposure, Colorista/3-Way Wheels, Curves, Film Negative/Print, Diffusion, Anamorphic Flare, Chromatic Aberration, Lens Distortion, Vignette, Spot Exposure, Hue/Saturation, Pop, Mojo, Warm/Cool, Split Tone, 3D `.cube` LUT [sizes 2..128, max 80 MB], Grain, Halation, etc.), plus look presets and floating setup window with wheels, curves, and drag/keyboard reordering.
   - **Multi-Pass Execution:** Allocates at most **2 layer-sized `RGBA16Float` targets** for spatial stages; fuses up to 8 consecutive pointwise color tools (and up to 6 LUT/curve samplers) into a single GPU pass. Intermediate stages stay straight and unclipped; first pass unpremultiplies, final pass clamps and premultiplies.
   - Preserves compatibility across `looksInstances` v1, legacy single-pass `chainOrder`, and historical Davidium grading formats.
4. **`openzoid-legacy` (OpenZoid Legacy — Clean-Room Equivalents):**
   - 7 deterministic native effects with floating Setup windows:
     - `Echo (Legacy)`: Up to 16 explicitly sampled earlier frames (smear, maximum, additive, screen).
     - `Posterize Time (Legacy)`: Deterministic frame-rate bucket quantizer.
     - `JPEG Damage`: 8×8 block DCT/quantization & chroma subsampling corruption simulation.
     - `VHS`: Analog tape signal chain (tracking jitter, YIQ chroma crawl/bleed, horizontal snow, head-switch tears, OSD timestamp overlay), 15 presets, and seeded `Glitch!` randomizer (`seed = (seed + 7919) % 10000`).
     - `Datamosh`: Deterministic segment motion-matching & 9 motion families (Identity, Horizontal, Vertical, Zoom, Rotate, Wave, Swap, Random, etc.) + 80 look modes reading a sampled source anchor (never previous output), 10 presets, seeded `Mosh!` & `Remove Frame`, plus optional post-export true I-frame removal pass.
     - `Tracery`: Motion-tracking callout overlay combining GPU color-keying (key color, threshold, blur) + CPU connected-component region detection (and/or keyframed manual markers) with configurable boxes (rect/square/ellipse/circle), markers (dot/plus/cross/polygon), connection lines (spline/PCB/smooth/step, dashes/arrows), coordinate/dimension/area/custom labels, grid, and Show Mask view.
     - `ASCII`: Character-cell resampling with per-luminance-band glyph mapping, 5 font families (Consolas, Courier, Monospace, Source Code Pro, Serif), 5 color modes (band, footage, duotone, mono, invert), 5 charsets (standard, blocks, detailed, minimal, custom), brightness/contrast shaping, scale jitter, sine-wave motion, fractal noise displacement, scroll trails, and glitch cuts.
5. **`smart-motion-blur-lite` (ZSMB):**
   - Lightweight single-neighbor optical-flow motion blur (`smart-motion-blur-lite`).
   - *Properties:* `Enabled` (`on`), `Quality` (`Full` = 1×, `1/2` = 1/8 project width, `1/4` = 1/16, `1/8` = 1/32; default `1/2`, capped by rendered frame size), `Track Frame` (`Previous`|`Next`, default `Previous`; frame 0 uses `Next`), `Blur Amount` (`0..10`, default `0.5` ≈ 180° shutter), `Motion Sensitivity` (`0..100`, default `70`; `0` bypasses).
   - Runs reduced-resolution flow estimation asynchronously on a worker thread, then reconstructs motion trails via a single GPU gather pass.
6. **`sglow-hq-v2-preview` (S_Glow):**
   - High-precision multi-scale glow (`sglow-hq-v2-preview`) supporting independent X/Y and R/G/B reach and procedural atmospheric dust.
   - *Properties:*
     - Emission: `Brightness` (def `2`, `≥ 0`), `Color` (def white), `Threshold` (def `0.5`, `≥ 0`), `Threshold Add Color` (def black), `Show` (`Result`|`Threshold`).
     - Spread: `Glow Width` (def `96` px, `≥ 0`), `Width X` / `Width Y` (def `1`, `0..5`), `Width Red` / `Green` / `Blue` (def `1.0 / 1.2 / 1.4`, `0..5`), `Subpixel` (`on`), `Edge Mode` (`Reflect`|`Transparent`), `Expand Borders` (`off`, transparent padding before blur).
     - Composite: `Combine` (`Screen` default; `Mult|Add|Screen|Difference|Overlay`), `Affect Alpha` (def `1`, `≥ 0`), `Glow From Alpha` (`0..1`, def `0`), `Glow Under Source` (`0..1`, def `0`), `Light Background` (`0..1`, def `0`), `Source Opacity` (`0..1`, def `1`), `Opacity` (`Normal`|`All Opaque`|`As Premult`).
     - Engine & Atmosphere: `Quality` (`Fast|Balanced|High|Ultra`, def `High`), `Blur Engine` (`AE Match (3-box)` default, `Performance`), `Atmosphere Amp` (def `0`, `≥ 0`), `Atmosphere Freq` (def `1`, `0.1..20`), `Atmosphere Detail` (def `0.6`, `0..1`), `Atmosphere Seed` (def `0.123`, `0..100`), `Atmosphere Speed` (def `1`), `Time Mode` (`Automatic`|`Manual` + `Time`).
7. **Declarative Shader Packs (`afterclip`, `alipfx-shader-pack-4`, `ccfx-shader-pack`):**
   - Pure-data declarative plugin format (`kind: "shader-pack"`) containing standalone GLSL ES 1.00 effects (`effects[]`) and multi-effect preset groups (`groups[]`) executed through the standard Custom Shader pipeline (Section 3) with zero custom code. ZoidCraft's clean-room core supports loading user-installed declarative shader packs without bundling third-party GLSL shaders.

---

### 5.5 Category 4: Experimental (`experimental`)

| Plugin ID | Display Name | Default | Zoidium-Only | Items | Summary |
|---|---|---:|---:|---:|---|
| `layer-input` | Layer Input | Off | Yes | 4 | Cross-track texture sampling for effects, 3D materials & Custom Shaders |

#### Detailed Experimental Specs
1. **`layer-input` (Layer Input):**
   - Enables any effect, 3D material, or Custom Shader to sample another top-level video track (even if hidden) or a fallback `Image` asset via 4 entry points:
     1. **Layer Input Effect:** Composites captured source track (or image) over current layer with animated opacity.
     2. **Layer Input Displacement Map:** Mirrors native `displacementmap` controls, reading displacement vectors from the selected track (or image).
     3. **Layer Source Material:** Uses captured track (or image) as a 3D surface texture map with animated opacity.
     4. **Custom Shader Layer Input:** Binds a selected track as a `uniform sampler2D` with selectable capture stage: `Source` (before effects/masks), `Masks` (masks only), or `Effects & Masks` (final output, default).
   - **Reference & Safety Rules:** Stores stable track references (`track:id:<stable-id>`) resilient to track reordering. UI picker disables self-references and cycle-creating targets; runtime enforces a strict recursion/in-progress guard (max 8 nesting depths) and caches captured textures per `(compositor, frame, resolution, stage)`. Missing/empty source passes input unchanged (effects) or transparent (material).
