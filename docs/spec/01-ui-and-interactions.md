# ZoidCraft Clean-Room Specification 01: UI Layout, Visual System, and Dynamic Interactions

This document specifies the exact window architecture, pixel metrics, color palette, theme derivation mathematics, typography, control anatomy, viewport interactions, keyboard shortcuts, and motion behaviors of Panzoid ClipMaker 3 (v3.0.106 / UI v1.0.72) together with Zoidium's floating-window and theming extensions. It is written as a clean-room behavioral and visual specification intended for re-implementation in ZoidCraft.

---

## 1. Overall Window Layout Grid

### 1.1 Root Frame & Recursive Split-Panel System

The application occupies 100% of the host window (`0, 0` to viewport width/height) with no outer menu bar, no top website header inside the editor canvas, and no outer chrome padding other than a `1 px` black (`#000000`) perimeter inset. Overflow is clipped at the root.

All workspace regions are organized as a binary tree of **Split Panels**. Every Split Panel contains exactly two child panes (`Pane 0` and `Pane 1`) separated by a draggable splitter bar:

| Split Panel Parameter | Exact Specification |
| --- | --- |
| Splitter bar thickness (`splitSize`) | `3 px` (visual gap and pointer hitbox) |
| Orientation `vertical` (`0`) | Top (`Pane 0`) over Bottom (`Pane 1`), cursor `ns-resize` |
| Orientation `horizontal` (`1`) | Left (`Pane 0`) beside Right (`Pane 1`), cursor `ew-resize` |
| Orientation `auto` (`-1`) | Chooses `horizontal` when `containerWidth > containerHeight`, else `vertical` |
| Split center calculation | Given container extent $E$, ratio $r \in [0, 1]$, half-splitter $h = 1.5\text{ px}$: `pane0End = E * r - h`, `pane1Start = E * r + h` |
| Minimum size propagation | Parent `minWidth = child0.minWidth + child1.minWidth`; parent `minHeight = child0.minHeight + child1.minHeight` |
| Clamping precedence | First clamp `pane0End >= child0.minSize` (shifting `pane1Start = pane0End + 3`); then clamp `E - pane1Start >= child1.minSize` (shifting `pane0End = pane1Start - 3`), so `Pane 1`'s minimum wins if the container shrinks below the combined minimum |
| Live drag math | Each frame adds pointer delta $\Delta$ to `Pane 0` extent and splitter offset, updating `ratio = (pane0Extent + 1.5) / E`; clamping and recursive child `resize()` execute on pointer release (`mouseup`/`touchend`) |
| Window resize debounce | Host window resize events are debounced by `500 ms` before triggering a full layout recalculation |
| Panel border & focus frame | Every leaf editor panel has a `1 px solid #333333` border and `#2a2a2a` background; when focused or containing a focused control, its border switches immediately to the primary focus accent (`#384668`) |

### 1.2 Exact ClipMaker 3 Workspace Tree

```
+----------------------------- Root Window (#000000, 1 px padding) -----------------------------+
| [Root Split: Horizontal, ratio = 0.30]                                                        |
| +------ Left Column (30%) ------+ | +----------------- Right Workspace (70%) ---------------+ |
| | File & History Toolbar (40px) | | |                                                       | |
| |-------------------------------| | |                                                       | |
| | [Split: Vert, ratio = 0.0]    | | |            Main Viewport Panel (65% height)           | |
| | +----+----------------------+ | | |                  (#1d1d1d background)                 | |
| | |Rail|                      | | | |                                                       | |
| | |60px|  Active Side Panel   | | | |             [Centered Aspect-Fit Canvas]              | |
| | |    |   (Inspector / Tab   | | | |                                                       | |
| | |icon|      Content)        | | | |-------------------------------------------------------| |
| | |bar |                      | | | | [Split: Vert, ratio = 0.65]                           | |
| | |    |  (Objects & Effects  | | | | Playback & Viewport Toolbar (40 px fixed, ratio 0.0)  | |
| | |    |   further split 0.40 | | | |-------------------------------------------------------| |
| | |    |   list / 0.60 props) | | | | [Split: Horz, ratio = 1.0]                            | |
| | |    |                      | | | | +-------------------------------------------+ +-----+ | |
| | |    |                      | | | | | Timeline Panel (tracks + keyframes + bar) | |Audio| | |
| | |    |                      | | | | | (left 200px headers | right scrolling)    | |45px | | |
| | +----+----------------------+ | | | +-------------------------------------------+ +-----+ | |
| +-------------------------------+ | +-------------------------------------------------------+ |
+-----------------------------------------------------------------------------------------------+
```

*Note on non-subscriber web banner*: In vanilla web ClipMaker 3 without a subscription, the Main Viewport slot is wrapped in a vertical split (`ratio = 0.0`) with an advertisement strip above it. In Zoidium and ZoidCraft desktop builds, no advertisement pane exists and the Viewport occupies 100% of the upper-right pane.

### 1.3 Top-Left File & History Toolbar

- **Dimensions**: Fixed `40 px` height (pinned via `ratio = 0.0` against `minHeight = 40 px`, `minWidth = 40 px`).
- **Surface**: Background `#242424`, `5 px` uniform padding, buttons centered horizontally.
- **Button geometry**: Inline buttons, `29 px` height, `2 px solid transparent` border (`#384668` when focused), `5 px` right margin, `0` padding; each holds a `25 × 25 px` SVG icon filled with `#acacac`.
- **Separator geometry**: Inline spacer `15 px` wide × `25 px` high.
- **Exact button sequence**:
  1. **New** (`new`, tooltip `new (ctrl-m)`)
  2. **Open** (`load`, tooltip `open (ctrl-o)`)
  3. **Save** (`save`, tooltip `save (ctrl-s)`)
  4. *Separator (`15 px`)*
  5. **Undo** (`undo`, tooltip `undo (ctrl-z)`)
  6. **Redo** (`redo`, tooltip `redo (ctrl-y)`)
- **Zoidium Toolbar Extension**: Converts the toolbar into a 3-column grid (`minmax(100px, 1fr) auto minmax(0, 1fr)`): column 1 holds an inline project-name text input (`height: 29 px`, `padding: 2 px 5 px`, value color `#7e8fb9`, ellipsis overflow), column 2 holds the centered 5 action buttons, and Save displays an optional `8 × 8 px` circular unsaved-changes reminder dot at top-right (`top: 1 px, right: 1 px`; `#9a9a9a` dirty, `#384668` overdue).

### 1.4 Left Icon Sidebar (Elevator Rail) & Side Inspector Panel

The lower-left pane (`minWidth = 200 px`, `minHeight = 100 px`) consists of a collapsible **Elevator Rail** on the left and the **Side Inspector Container** to its right.

| Metric / State | Exact Value |
| --- | --- |
| Collapsed rail width | `60 px` (`top/left/bottom: 0`, `z-index: 2`, `#000000` background) |
| Expanded drawer width | `300 px` (overlays the inspector without reflowing it; `box-shadow: 5px 0 10px rgba(0, 0, 0, 0.5)`) |
| Width transition | Collapse: `50 ms`; Expand: `200 ms` |
| Menu toggle button (top of rail) | `height: 50 px`, full rail width, `menu` icon `30 × 30 px` at `left: 13 px, top: 4 px`; focus border `#384668` |
| Tab strip viewport | `top: 50 px` to `bottom: 0`, full width; scrolls vertically with hidden scrollbar when expanded (and via wheel/touch while collapsed in Zoidium, `overscroll-behavior: none`) |
| Tab item row | `height: 50 px`, `width: 100%`, `border: 2 px solid transparent`, `font-size: 22 px`, text `#dadada`, `white-space: nowrap`, clipped horizontally when collapsed |
| Tab icon | `30 × 30 px`, fill `#dadada`, positioned at `left: 13 px, top: 9 px`, `margin-right: 30 px` before label text |
| Tab hover state | Background `#222222` |
| Tab selected state | Background `#384668` (primary accent) |
| Tab keyboard focus state | Border `2 px solid #717171` |
| Tab disabled / no-context state | Text and icon fill `#303030` |
| Drag-hover auto-switch | Dragging a payload over a tab starts a `200 ms` dwell timer; remaining over the tab activates it automatically, leaving cancels the timer |
| Keyboard navigation on rail | `ArrowUp` / `ArrowDown` moves roving focus (`tabindex="0"`) across tabs; `Enter` activates the focused tab and collapses the drawer |
| Inspector content container | Positioned at `left: 60 px`, `top/bottom/right: 0`, `#000000` outer backing, vertical scrolling; active tab panel fills `100% × 100%` |
| Scrollbars (inspector & panels) | `12 × 12 px`, no end arrows, track `#222222`, thumb `#555555` (`#6d6d6d` inside Zoidium windows/panels), transparent corner |

**Exact Order of Built-In Tabs in ClipMaker 3**:

| Order | Tab Title | Icon ID | Panel Structure & Binding |
| --- | --- | --- | --- |
| 1 | **Media** | `project` | Scrollable navigation panel: Search filter box, Missing Media list, Project Media grid (seeded with 6 built-in presets: *3D Scene*, *Adjustment*, *Text*, *Preset shape*, *Shape*, *Composite*), Community Templates grid. |
| 2 | **Sequence** | `sequence` | Property editor bound to the active Sequence (`resolution`, `rate`, etc.); linked to the timeline keyframe pane. |
| 3 | **Edit** | `edit2` | Property editor bound to the selected timeline clip(s); hides row duplicate/delete buttons; empty state `"select a clip"`; linked to the timeline keyframe pane. |
| 4 | **Objects** | `objects` | Auto-oriented Split Panel (`ratio = 0.40`): **Pane 0 (40%)** = 3D object tree of the selected Scene clip (`columnLayout = 1`, empty state `"select a clip"`); **Pane 1 (60%)** = property inspector for selected 3D object(s), linked to the timeline keyframe pane. |
| 5 | **Effects** | `fx` | Auto-oriented Split Panel (`ratio = 0.40`): **Pane 0 (40%)** = effects list of the selected clip (`columnLayout = 1`, empty state `"select a clip"`); **Pane 1 (60%)** = property inspector for selected effect(s), linked to the timeline keyframe pane. |
| 6 | **Export** | `export` | Navigation stack panel with Cloud, Device, and Frame export sub-pages. |
| 7 | **About** | `about` | Version and attribution panel. |

*(Zoidium extension side panels insert immediately above **About** by default, or at an explicit anchor such as `after:Effects` or `afterAbout`.)*

### 1.5 Main Viewport Area

- **Default proportion**: Top `65%` of the Right Workspace (`ratio = 0.65` in vertical split `Z`).
- **Background**: `#1d1d1d` surrounding letterbox area (solid dark gray, never a transparency checkerboard).
- **Canvas aspect-ratio centering**: Given sequence resolution $(W, H)$ and viewport panel client rect $(V_w, V_h)$:
  $$s = \min\!\left(\frac{V_w}{W},\; \frac{V_h}{H}\right)$$
  $$C_w = \text{round}(W \cdot s), \qquad C_h = \text{round}(H \cdot s)$$
  $$\text{left} = \frac{V_w - C_w}{2}, \qquad \text{top} = \frac{V_h - C_h}{2}$$
  - GPU backing buffer dimensions: $\text{round}(C_w \cdot \text{DPR}) \times \text{round}(C_h \cdot \text{DPR})$.
  - Canvas clears to opaque `#000000` in Editing Camera mode; compositor output renders directly into the canvas in Preview mode.
- **Overlays**: No floating HUD toolbars inside the canvas; viewport mode switches live in the Playback Toolbar directly beneath the viewport. When 2D Layer Transform Controls are enabled (`T`), a DOM bounding-box widget overlays the canvas (Section 5.4).

### 1.6 Playback & Viewport Mode Toolbar

- **Dimensions**: Fixed `40 px` height (`minHeight = 40 px`, `#242424` background, `5 px` padding, centered).
- **Icon styling**: `25 × 25 px` SVGs inside `29 px` buttons; inactive fill `#acacac`, active toggle fill `#8a2828`.
- **Exact visible button sequence** (left to right):
  1. **Editing Camera** (`camera`, `C`) — toggles 3D editor camera (`#8a2828` when active, default ON)
  2. **Layer Transform Controls** (`transform`, `T`) — toggles 2D transform overlay (`#8a2828` when active, default OFF)
  3. *Separator (`15 px`)*
  4. **Start** (`start`, `Home`) — stops playback, jumps to frame `0`
  5. **Previous Frame** (`prevframe`, `ArrowLeft`) — stops playback, decrements frame by `1` (clamped $\ge 0$)
  6. **Play / Pause** (`play` / `pause`, `Space`) — toggles `speed` between `0` and `1`; switches icon to `pause` and fill to `#8a2828` whenever `speed != 0`
  7. **Next Frame** (`nextframe`, `ArrowRight`) — stops playback, increments frame by `1` (clamped $\le \text{totalFrames} - 1$)
  8. **End** (`end`, `End`) — stops playback, jumps to `totalFrames - 1`
  9. *Separator (`15 px`)*
  10. **Loop** (`loop`, `Ctrl+L`) — toggles looping (`#8a2828` when active, default ON in CM3)
  11. **Graph Editor** (`interp_1`, `Ctrl+G`) — opens a secondary Graph Editor window (`600 × 400 px` default)

### 1.7 Bottom Timeline & Audio Meter Area

The bottom `35%` of the Right Workspace splits vertically into the `40 px` Playback Toolbar and the remaining Timeline/Meter row, which splits horizontally with `ratio = 1.0` so the right-hand **Audio Meter** stays pinned to its `45 px` minimum width.

| Timeline Region | Exact Geometry & Behavior |
| --- | --- |
| Left header column width | `200 px` (timecode box + keyframe track labels + video/audio track labels) |
| Timecode display box | `200 × 35 px` at `(0, 0)`, background `#1d1d1d`, text `#cccccc`, `line-height: 30 px`, `padding-left: 10 px`. **Right-click** cycles display format modulo 3: `0` = `HH:MM:SS.mmm`, `1` = `HH:MM:SS:FF`, `2` = integer frame number (CM3 default). |
| Time ruler (`timerule`) | `left: 200 px, top: 0, height: 35 px`, right inset = horizontal scrollbar thickness, background `#242424`. Displays dynamic interval labels (`14 px`, `#656565`, left border `1 px solid #555555`, right border `1 px solid #3a3a3a`), sequence markers, and a `2 px`-tall bottom `selectionShadow` bar (`#526183`, opacity `0.8`). |
| Keyframe lane area | Vertically positioned between `top: 35 px` and `top: 98 px` (`63 px` region above tracks), alternating track backgrounds `#343434` / `#3f3f3f` and label backgrounds `#2e2e2e` / `#3b3b3b`. |
| Track container area | `top: 98 px` to `bottom: <scrollbarHeight>`, `left: 0, right: 0`. Divided vertically into **Video Tracks** (top, default row height `30 px`) and **Audio Tracks** (bottom, default row height `50 px`). |
| Video/Audio internal divider | `4 px` tall, full width, `ns-resize` cursor, `1 px solid #212121` top and bottom borders; draggable across `[0, tracksHeight - 4]`. |
| Playhead vertical cursor | `1 px` wide, `100%` timeline height (clipped to `left >= 200 px`), color `#b71e1e`, non-interactive (`pointer-events: none`). |
| Horizontal scrollbar | `left: 200 px`, `bottom: 0`, height equal to native scrollbar thickness; drives horizontal pan of ruler, keyframes, and tracks in sync. |
| Frame-to-pixel mapping | $\text{px}(f) = \frac{f}{\text{frameRate}} \cdot 400 \cdot \text{zoom}$ and $f(\text{px}) = \frac{\text{px} \cdot \text{frameRate}}{400 \cdot \text{zoom}}$ (initial `zoom = 0.3`). |
| Zoom step & anchoring | Each zoom step multiplies/divides `zoom` by $1.03125 = \frac{33}{32}$. Scroll adjusts via $\text{scroll}' = (\text{scroll} + x_{\text{anchor}}) \cdot \frac{\text{zoom}'}{\text{zoom}} - x_{\text{anchor}}$, anchored at cursor X (wheel) or playhead X (`-`/`=` keys). |
| Ruler interval formula | Base unit list `[1, 5, 15, 300, 1800]` seconds; index $s = \text{clamp}(\lfloor\log_{10}(0.5/\text{zoom})\rfloor + 1, 0, 4)$; interval $n = U_s \cdot 2^{\lfloor\log_2((1/\text{zoom})/U_s)\rfloor}$; major tick spacing $r = 400 \cdot \text{zoom} \cdot n$. |
| Audio Level Meter | Fixed `45 px` wide (`minWidth = 45 px, minHeight = 45 px`), `#242424` background, inner bar inset `4 px` all sides. Vertical gradient bottom-to-top: `green` -> `yellow` (90%) -> `red`. Masked from top by `#111111` (95% opacity) scaled vertically to `1 - RMS` (1024-sample time-domain RMS). Peak hold bar: `1 px` tall `#cece62` at `peak * 100%` from bottom, decaying each frame by $\text{peak}_{t+1} = \max(\text{maxSample},\; \text{peak}_t - 0.01 / e^{\text{peak}_t})$. |

---

## 2. Complete Color Palette & Theme Derivation

### 2.1 Neutral Chrome & Structural Surfaces (Fixed Dark)

CM3's neutral surfaces remain fixed dark across all built-in color profiles so extension surfaces match native panels seamlessly.

| Semantic Role | Hex Color | Usage |
| --- | --- | --- |
| Root Void / Rail Base | `#000000` | Outer `.editorwindow` padding/gaps, Elevator sidebar rail, shadow pigment |
| Sunken Surface (`--zui-bg-sunken`) | `#1d1d1d` | Viewport letterbox background, timeline timecode box, listbox background, tabstrip background |
| Input Field Surface (`--zui-bg-input`) | `#202020` | Numeric edit box, text input, select dropdown, tween button, floating dropdown list |
| Header / Title Surface (`--zui-bg-title`) | `#222222` | Section titles (`proptitle`), spacer rows, top-level tree rows, elevator hover, floating window title bar (unfocused) & footer, scrollbar track, progress bar track |
| Toolbar & Divider (`--zui-border-soft`) | `#242424` | Top/playback toolbars, audio meter frame, time ruler, property row & tree row bottom dividers |
| Panel Surface (`--zui-bg`) | `#2a2a2a` | All `.editorpanel` inspectors, color picker popup, floating window bodies, filter input background, inactive stopwatch icon |
| Dropdown Hover (CM3 native) | `#2c2c2c` | Active item in native dropdowns (overridden to `--zui-accent` in Zoidium) |
| Option Hover | `#2f2f2f` | Hover state for `.pz-option` and `.pz-options` rows |
| Hover / Card / Unfocused Border (`--zui-hover`) | `#333333` | Unfocused panel border, media grid cards, timeline track labels, floating window hover buttons & unfocused border |
| Keyframe Row Alternating | `#343434` / `#3f3f3f` | Even/odd keyframe lane backgrounds (labels: `#2e2e2e` / `#3b3b3b`) |
| Active Option Row | `#3f3f3f` | Keyboard-selected row in Add Object/Effect list (with `1 px solid #676767` border) |
| Button Surface | `#434343` | `.proprow.propbutton` background (`#444444` pressed, `2 px solid #545454` hover border) |
| Back Button Surface | `#474747` | Navigation panel Back button (`#444444` pressed, `2 px solid #545454` hover border) |
| Scrollbar Thumb | `#555555` | Native CM3 scrollbar thumb (`#6d6d6d` inside Zoidium windows) |

### 2.2 Borders & Dividers

| Border Role | Hex Color | Width / Context |
| --- | --- | --- |
| Primary Control Border (`--zui-border`) | `#1b1b1b` | `1 px` around inputs, selects, tween buttons, color swatches, window title/footer separators |
| Track Splitter Border | `#212121` | `1 px` top & bottom of the 4 px video/audio divider |
| Clip Border | `#232323` | `1 px` right edge of timeline clips; left edge highlight `#838383` |
| Row Divider (`--zui-border-soft`) | `#242424` | `1 px` bottom border on property and tree rows |
| Panel Default Border | `#333333` | `1 px` around unfocused `.editorpanel` |
| Filter Input Border | `#363636` | `2 px` around `.pz-filterbox` |
| Ruler Tick Borders | `#555555` / `#3a3a3a` | `1 px` major left tick / `1 px` half-step right tick |
| Button Hover Border | `#545454` | `2 px` on hovered `.propbutton` / `.backbutton` |
| Sidebar Tab Focus Border | `#717171` | `2 px` on keyboard-focused elevator tab |

### 2.3 Text Tiers

| Text Tier | Hex Color | Usage |
| --- | --- | --- |
| Bright (`--zui-text-bright`) | `#e4e4e4` | Selected list item text, primary button label, hovered window title icons (`#eeeeee` on toast/restore titles) |
| Strong (`--zui-text-strong`) | `#dadada` | Sidebar elevator tab labels & icons, floating window titles, section headings |
| Primary (`--zui-text`) | `#cccccc` | Property labels, tree item names, timecode, dropdown text, action icons (`#cbcbcb` on buttons, `#bdbdbd` on track labels) |
| Disabled / Read-Only / Punctuation | `#999999` | Read-only numeric values, disabled selects, vector separator commas, file picker filename |
| Muted (`--zui-text-muted`) | `#9a9a9a` | Collapse carets, window title icons, helper notes (`#969696` on option descriptions) |
| Faint (`--zui-text-faint`) | `#6d6d6d` | Category section banners in add pickers, units, empty list labels (`#666666` empty panel message, `#656565` ruler labels) |
| No-Context / Inactive Tab | `#303030` | Disabled sidebar tab text & icon |
| Clip Label Auto-Contrast | `#2c323f` or `#f4f4f4` | Chosen per clip background by higher WCAG 2.x contrast ratio |

### 2.4 Semantic & Status Colors (Fixed, Unthemed)

| Role | Hex / Formula | Usage |
| --- | --- | --- |
| Toolbar Toggle Active | `#8a2828` | Active fill for Editing Camera, 2D Transform Controls, Play/Pause (while playing), Loop; error toast border |
| Toolbar Icon Inactive | `#acacac` | Default fill for all top and playback toolbar icons |
| Active Icon Alert | `#a92424` | `.iconactive` emphasis fill |
| Playhead Line | `#b71e1e` | `1 px` vertical current-frame cursor across timeline |
| Trim Handle Selected (CM3 native) | `#982b2b` | Selected clip left/right trim handle border (Zoidium remaps to `--zui-accent`) |
| Danger Text (`--zui-danger`) | `#df9b9b` | Danger buttons, warning notes, missing/failed font badges |
| Warning Surface | `rgba(120, 34, 34, 0.18)` | Warning banner fill (with `3 px solid #a94b4b` left bar) |
| Audio Meter Peak | `#cece62` | `1 px` peak-hold indicator line |
| Keyframe Diamond (Inspector) | Stroke `#cccccc`; Fill `none` (unkeyed) / `#cccccc` (keyed) | Keyframe toggle button in property row |
| Keyframe Diamond (Timeline) | Fill = property track color; Selected = `#292929` | Rotated `12 × 12 px` diamond on keyframe tracks |
| Animated Stopwatch Hue | $\text{hsl}((40 \cdot \text{stringHash}(\text{name})) \bmod 360,\; 60\%,\; 45\%)$ | Active stopwatch icon tint per property name (inactive `#2a2a2a`) |

### 2.5 Panzoid Classic 5-Color Accent Palette & Derived Tokens

| Palette Key | Classic Hex | RGB | Roles Driven |
| --- | --- | --- | --- |
| `focus` (`--zui-accent`) | `#384668` | `56, 70, 104` | Panel/control focus borders, selected sidebar tab, selected tree/media item, progress bar fill, checkbox accent, primary button fill |
| `link` (`--zui-value`) | `#7e8fb9` | `126, 143, 185` | Numeric scrub values, text/select inputs, links, easing/font labels, window subtitles, slider thumbs |
| `clip` | `#526183` | `82, 97, 131` | Unselected timeline clip fill, ruler selection shadow bar (80% opacity) |
| `clipSelected` | `#7f94c7` | `127, 148, 199` | Selected timeline clip fill |
| `clipSelection` | `#273465` | `39, 52, 101` | Box-selected / active clip selection highlight |

**Zoidium Derived Surface Tokens** (computed in sRGB):
- `--zui-accent-strong`: 70% `--zui-accent` + 30% `#ffffff`
- `--zui-value-strong`: 60% `--zui-value` + 40% `#ffffff`
- `--zui-bg-title-focused`: 22% `--zui-accent` + 78% `#222222`
- `--zui-menu-active`: 55% `--zui-accent` + 45% `#2a2a2a`

### 2.6 Theme Derivation Formulas

#### A. Main-Color Auto-Derivation (User Themes)
Given user main hex color $M$ and the Classic reference palette $P = \{p_k\}$ with Classic focus anchor $f = \text{\#384668}$, convert each color to HSL with $H \in [0, 1)$, $S \in [0, 1]$, $L \in [0, 1]$:
- `focus` $= M$ (exact input hex, unless explicitly overridden).
- For every other key $k \in \{\text{link}, \text{clip}, \text{clipSelected}, \text{clipSelection}\}$ without a manual override:
  $$H_k' = (H_M + H_{p_k} - H_f) \bmod 1 \quad (\text{wrapped into } [0, 1))$$
  $$S_k' = \text{clamp}\!\left(S_M \cdot \frac{S_{p_k}}{S_f},\; 0,\; 1\right)$$
  $$L_k' = \text{clamp}(L_M + L_{p_k} - L_f,\; 0,\; 1)$$
- Convert $(H_k', S_k', L_k')$ to sRGB and round each 8-bit channel to the nearest integer (`Math.round`). Setting $M = \text{\#384668}$ reproduces all five Classic colors byte-for-byte.

#### B. Built-In Preset Profiles (Hue/Saturation Shift)
Each built-in profile transforms every Classic palette entry $p_k$ independently via:
$$H_k' = \left(H_{p_k} + \frac{\Delta\theta}{360}\right) \bmod 1, \qquad S_k' = \text{clamp}(S_{p_k} \cdot \sigma,\; 0,\; 1), \qquad L_k' = L_{p_k}$$

| Profile ID | Label | Hue Shift $\Delta\theta$ | Saturation Scale $\sigma$ |
| --- | --- | --- | --- |
| `panzoid` | Panzoid Classic | `0°` | `1.0` |
| `signal` | Signal Blue | `-18°` | `1.0` |
| `ember` | Ember Console | `+154°` | `1.0` |
| `moss` | Moss Terminal | `+78°` | `1.0` |
| `rose` | Rose Signal | `-72°` | `1.0` |
| `gray` | Gray | `0°` | `0.0` |
| `custom` | Custom | User degrees | $\text{clamp}(\text{sat}\% / 100,\; 0,\; 2)$ |

#### C. Per-Clip Custom Color Selection & Label Contrast
- **Selected state of a custom-colored clip**: Given base clip HSL $(H, S, L)$, shift lightness 35% toward white:
  $$L_{\text{selected}} = L + (1 - L) \times 0.35$$
- **Automatic label legibility**: Compute WCAG 2.x relative luminance $Y = 0.2126 R_{\text{lin}} + 0.7152 G_{\text{lin}} + 0.0722 B_{\text{lin}}$ (linearizing each channel $c \in [0,1]$ via $c/12.92$ for $c \le 0.04045$, else $((c + 0.055)/1.055)^{2.4}$). Choose `#2c323f` if $\frac{Y + 0.05}{Y_{\text{\#2c323f}} + 0.05} \ge \frac{Y_{\text{\#f4f4f4}} + 0.05}{Y + 0.05}$, otherwise `#f4f4f4`.
- **Light extension chrome threshold**: In the Theme Editor, if the main focus color's HSL lightness exceeds `60%`, extension window chrome derives light neutral tokens while all built-in presets retain dark chrome.

---

## 3. Typography

- **Primary Editor Font**: `'Source Code Pro', monospace` across the entire editor workspace, property inspectors, timeline, dropdowns, and floating windows (user-configurable in Zoidium Settings).
- **Exact Font Size Scale**:

| Role | Size (px) | Weight / Style | Line / Row Metrics |
| --- | --- | --- | --- |
| Sidebar elevator tab labels | `22 px` | Regular | `50 px` row height, `white-space: nowrap` |
| Window titles & section headers (`--zui-font-size-title`) | `18 px` | Bold (window/section titles) | `30 px` window title bar; `28 px` Zoidium section header; `30 px` line-height on CM3 `proptitle` |
| Standard UI body, property labels, numeric values, inputs, selects, buttons, filter box, media titles (`--zui-font-size`) | `16 px` | Regular | Normal line-height; `min-height: 30 px` on `.zoidium-field` |
| Add-picker category banner (`.pz-options > span`) | `15 px` | Regular | `4 px 5 px` padding |
| Secondary text, option descriptions, timeline ruler labels, window subtitles, units, list details (`--zui-font-size-small`) | `14 px` | Regular | `line-height: 1.35–1.4` on notes/metadata |
| Context & anchored menu items (`--zui-menu-font-size`) | `13 px` | Regular | `5 px 14 px` item padding; `4 px 0` menu padding |
| Collapse carets, menu section headings (`--zui-menu-title-font-size`), compact canvas/node headers (`--zui-font-size-compact`) | `12 px` | Regular | `4 px 14 px 5 px` menu heading padding |
| Easing button label text, fallback font text | `11 px` | Regular | `25 px` line-height in font button |
| Dropdown uppercase separators & error pill badges | `10 px` | Regular (`uppercase`, `0.08em` tracking for separators) | `16 px` line-height on pill badge (`8 px` radius) |

---

## 4. UI Control Anatomy & Dynamic Interactions

### 4.1 Property Rows & Column Resizing

CM3 renders properties inside a nested tree (`ul.tree li`) as well as flat utility rows (`.proprow`).

- **Tree row geometry**: Two-column grid with `5 px` column gap, `3 px` top/bottom padding, `1 px solid #242424` bottom border, `white-space: nowrap`.
  - **Column 1 (Label)**: Overflow clipped with ellipsis. Left padding is `15 px × nestingDepth`; leaf rows without a collapse triangle add `+20 px` so all labels align vertically with expandable rows.
  - **Column 2 (Value / Actions)**: Horizontal padding `0 5 px`.
- **Column layout modes**:
  - `layout = 0` (Default property panels): `<labelWidth>px auto`. Initial `labelWidth` is `50%` of the panel's client width (`0.5 * clientWidth`). A full-height `5 px`-wide invisible splitter (`.columnadjust`, `ew-resize` cursor) sits at `left = labelWidth` and drags live, clamped to `>= 10 px`.
  - `layout = 1` (Objects & Effects lists): `minmax(0, auto) min-content` so labels expand and action icons hug the right edge.
  - `layout = 2`: `auto auto`.
- **Zoidium plugin form rows (`.zoidium-field`)**: `minmax(96px, 44%) minmax(0, 1fr)`, `8 px` column gap, `min-height: 30 px`, `3 px 8 px` padding, right-aligned controls; disabled rows render at `opacity: 0.45`.
- **Reset button**: Every `propertyList` header row displays a `17 × 17 px` `reset` button (`title = "reset"`). Clicking executes one atomic undo operation that deletes all extra keyframes, disables animation, moves keyframe 0 back to frame 0, and sets every property in the list to its default value.

### 4.2 Animation Stopwatch, Keyframe Bar, and Expression Editor

Dynamic properties separate **enabling animation** (in the label column) from **keyframe/tween editing** (in the value column):

1. **Stopwatch Button** (left of property label, `17 × 17 px` `stopwatch` icon in a `2 px`-bordered `.actionbutton`, `5 px` right margin, tooltip `"toggle animation"`):
   - **Visual state**: Unanimated = `#2a2a2a`; Animated (keyframed or expression) = `hsl((40 * stringHash(name)) % 360, 60%, 45%)`.
   - **Left-click**: Toggles `animated` at `localFrame = currentFrame - frameOffset` in one undo operation.
     - Turning **ON** (without Alt): if the property holds 1 static keyframe, moves that keyframe to `localFrame`.
     - Turning **OFF**: clears any expression, deletes all keyframes, and writes a single static keyframe at frame `0` holding the current evaluated value (or two keyframes at `0` and `clip.length` for properties with `staticBehavior = LINEAR`).
   - **Alt+Click**: Turns animation **ON** and initializes an **Expression** equal to `JSON.stringify(property.get(localFrame))`, replacing the keyframe bar with the Expression editor and making the numeric/color/option control read-only. Clicking again turns animation off and removes the expression.

2. **Keyframe Control Bar** (shown above the value control when `animated && !expression`; `height: 27 px`, `margin-bottom: 5 px`):
   - **Previous Keyframe** (`prevkf`, `17 × 17 px`, `#cccccc`): jumps `currentFrame` to `prevKeyframe.frame + frameOffset`.
   - **Keyframe Diamond Toggle** (`keyframe`, `17 × 17 px`, `#cccccc` stroke):
     - *No keyframe at `localFrame`*: hollow (`fill: none`), tooltip `"add keyframe"`. Click inserts a keyframe at `localFrame` with the evaluated value (inheriting the surrounding segment's tween if interpolated).
     - *Keyframe exists at `localFrame`*: solid (`fill: #cccccc`), tooltip `"delete keyframe"`. Click removes that keyframe (blocked if it is the last remaining keyframe unless `allowEmpty` is true).
     - *Vector group (`property.dynamic.group`)*: shows solid diamond if any child channel is keyed at `localFrame`; clicking toggles all child channels together.
   - **Next Keyframe** (`nextkf`, `17 × 17 px`, `#cccccc`): jumps `currentFrame` to `nextKeyframe.frame + frameOffset`.
   - **Interpolation Toggle** (`.pz-tweens`, `27 px` tall, `35 × 23 px` icon, `#202020` bg, `1 px solid #1b1b1b` border, `5 px` right margin): visible only when keyed at `localFrame` and property is interpolated (and for groups, only when all keyed child channels share the same tween). Click toggles high byte (`bit 8`) between `interp_0` and `interp_1`.
   - **Easing Dropdown Button** (`.pz-tweens`, `27 px` tall, `35 × 23 px` icon): visible under the same condition as Interpolation. Click opens floating `.pz-dropdown` (`250 px` wide, `max-height: 300 px`, `#202020` bg, `1 px solid #384668` border) listing all 33 easing curves (`ease_0` through `ease_32`, stored in low byte `tween & 0xFF`). Supports hover selection, `ArrowUp`/`ArrowDown` with auto-scroll, `Enter` commit, `Escape` cancel, and viewport edge flip/clamp.

3. **Expression Controls** (shown below the read-only value widget when `animated && expression != null`; `margin-top: 5 px`):
   - Inline `<textarea>` (`240 × 40 px`, no wrap, spellcheck off, `.pz-inputbox`). Pressing `Ctrl+Enter` blurs and commits; `onchange` records one undoable `setExpression`.
   - **Open Expression Editor** button (`code` icon, `17 × 17 px`): launches a dedicated Expression editor window.
   - **Error Indicator** button (`error` icon, `17 × 17 px`): hidden while valid; shown whenever `expression.error` is non-empty, displaying the diagnostic message on click.

### 4.3 Numeric Scrub Inputs (`.pzinput`)

Numeric controls act as dual-mode **scrub-by-drag** and **click-to-type** widgets.

- **Visual states**:
  - *Idle*: inline text in `#7e8fb9`, `2 px` padding, `1 px solid transparent` border, `cursor: move`, formatted to `value.toFixed(decimals)` (defaults: `decimals = 2`, `step = 1`, `min = -∞`, `max = +∞`).
  - *Editing (text entry)*: `#202020` background, `1 px solid #1b1b1b` border (`#384668` while focused), `cursor: auto`, entire value pre-selected.
  - *Read-only* (when `definition.readOnly`, expression-driven, or un-animated with >1 keyframe): `#999999` text, `cursor: default`, `pointer-events: none`.

- **Interaction State Machine**:

| State | Trigger | Next State & Exact Behavior |
| --- | --- | --- |
| **Idle** | `mousedown` (left button) | Blurs any focused element; begins undo transaction (`editStart(altKey)`, snapshotting `oldValue`, auto-inserting a keyframe at `localFrame` if animated and unkeyed, and recording proportional ratios if `linkRatio && !altKey`); sets global cursor to `move`; sets `moved = false` -> **PendingDrag**. |
| **PendingDrag** | Pointer moves such that $\|\Delta x - \Delta y\| > 2\text{ px}$ | Sets `moved = true` -> **Scrubbing**. |
| **PendingDrag** | `mouseup` (before exceeding `2 px` threshold) | Discards the drag's undo transaction (`editFinish(cancel = true)`), restores cursor, stores `noLink = altKey`, focuses the element, selects all text, starts a fresh `editStart` -> **TextEdit**. |
| **Scrubbing** | Every animation frame (`requestAnimationFrame`) | Computes combined 2D motion $\Delta = (x - x_{\text{prev}}) - (y - y_{\text{prev}})$ (right or up increases; left or down decreases). Updates internal accumulator: $v \leftarrow v + \Delta \cdot \text{step} \cdot 0.01 \cdot M_{\text{shift}} \cdot M_{\text{ctrl}}$, where $M_{\text{shift}} = 10$ if `Shift` is held (else `1`) and $M_{\text{ctrl}} = 0.1$ if `Ctrl` is held (else `1`). Clamps to `[min, max]`, rounds via $\text{round}(v \cdot 10^d) / 10^d$, displays live, and writes `v * scaleFactor` immediately to the property for live viewport preview without pushing extra history steps. |
| **Scrubbing** | `mouseup` | Restores cursor. Calls `editFinish(cancel = (value === initialValue))`: if changed, commits **one** undo command restoring `oldValue`; if unchanged, discards transaction -> **Idle**. |
| **Scrubbing** | `Escape` keydown | Ends drag cleanly; commits/discards based on final state (in Zoidium kit `number` controls, `Escape` reverts to `initialValue` and cancels). |
| **TextEdit** | `ArrowUp` / `ArrowDown` | Steps value by `±step` immediately (clamped & rounded). |
| **TextEdit** | `PageUp` / `PageDown` | Steps value by `±10 * step` immediately. |
| **TextEdit** | `ArrowLeft` / `ArrowRight` | Moves text caret normally (stops propagation to timeline). |
| **TextEdit** | `Enter` or blur | Parses float; if `NaN`, reverts to `initialValue`; clamps & rounds; commits **one** undo operation if `value !== initialValue`, else discards -> **Idle**. |
| **TextEdit** | `Escape` | Clears text buffer and blurs, causing `NaN` fallback to `initialValue` and discarding the undo operation -> **Idle**. |

*(Note: Zoidium plugin kit `controls.number` / `controls.slider` use `70 px` right-aligned numeric fields with `ew-resize` cursor, `Shift` = `×10`, `Alt` = `×0.1`, live `onInput` preview, and single-commit `onChange`.)*

### 4.4 Vector2 / Vector3 / Vector4 Inputs

- Composed of 2, 3, or 4 inline `.pzinput` scrubbers separated by `#999999` separator glyphs (`definition.spaceChar`, defaulting to `","`).
- **Proportional Linking (`linkRatio`)**: When `definition.linkRatio` is true and `Alt` is **not** held at edit start, the control records normalized ratios $r_i = v_i / v_{\text{active}}$ (provided $v_{\text{active}} \ne 0$). Scrubbing or typing into any component updates all components to $v_{\text{new}} \cdot r_i$. Holding `Alt` on click/drag temporarily unlinks components for that single edit.
- **Display Scale (`scaleFactor`)**: Displays `storedValue / scaleFactor` and writes `uiValue * scaleFactor`.
- **Expandable Channels**: Vector groups (`property.dynamic.group`) include a collapse toggle defaulted to **collapsed** (`-90°`), allowing the user to expand individual X/Y/Z/W sub-rows to keyframe or scrub axes independently.

### 4.5 Color Input, Eyedropper & HSVA Popup Picker

- **Inline row control**:
  - Color swatch button: `30 × 20 px`, `.pz-inputbox` styling, filled with the live RGB/RGBA color; disabled when read-only or expression-driven.
  - **Eyedropper button** (when editable): `10 px` to the right of the swatch, `17 × 17 px` `eyedropper` icon (`#cccccc`). Clicking sets global cursor to `crosshair` and captures the next click anywhere in the document; if the target is a viewport WebGL `<canvas>`, reads the exact pixel at DPR-scaled inverted Y via `readPixels`, updates the color, and commits one undo operation.
- **Popup HSVA Picker**:
  - Size: `470 × 250 px` (RGB) or `500 × 250 px` (RGBA), `5 px` padding, `#2a2a2a` background, `1 px solid #384668` border. Placed directly below the swatch, flipping above or shifting left to stay within the viewport.
  - **Saturation × Value square**: `250 × 250 px` at `(5, 5)`; X = Saturation $[0,1]$, Y (bottom-to-top) = Value $[0,1]$; `6 × 6 px` circular `#aaaaaa` reticle.
  - **Hue slider**: `25 × 250 px` at `(260, 5)`, vertical full-spectrum gradient (`0°` red at top through `360°` red at bottom); `100% × 6 px` `#aaaaaa` horizontal bar cursor.
  - **Alpha slider** (RGBA only): `25 × 250 px` at `(290, 5)`, top = `1.0` opaque to bottom = `0.0` transparent; `100% × 6 px` `#aaaaaa` bar cursor.
  - **Right panel** (`185 px` wide, `right: 5 px`):
    - Top `25 px` split comparison bar: left half = initial color (clicking reverts to initial value), right half = current color.
    - `185 px` recent-color grid (`min 25 px` tiles, `4 px` gap) displaying 36 swatches (pre-seeded with 12 random colors, 6 evenly spaced grayscales, and 18 full-saturation rainbow hues); clicking any swatch applies it immediately.
    - Bottom CSS color text box (`spellcheck="false"`): parses any valid CSS color string on blur; `Enter` returns focus to the picker.
  - **Commit / Cancel**: `Enter` or clicking outside closes the picker, pushes unique colors onto the 36-slot recent stack, and commits one undo operation; `Escape` reverts to `initialValue` and discards the undo operation.

### 4.6 Gradient, Curve, Dropdown, Checkbox, Text, File/Font/Texture, and Button Controls

| Control Type | Exact Visual Anatomy & Dynamic Interaction |
| --- | --- |
| **Gradient** (native) | `245 px` wide container (`5 px` padding): `30 px`-tall gradient preview bar above a `15 px` stop lane. Clicking the bar inserts a new stop interpolated at pointer X. Stops are `10 × 10 px` squares (`1 px solid #1b1b1b`, centered via `-5 px` margin): left-click opens RGBA color picker, horizontal drag moves position in `[0, 1]`, right-click or `Delete`/`Backspace` removes stop (if $>1$ stop), `ArrowUp`/`ArrowDown` moves focus across stops, `ArrowLeft`/`ArrowRight` nudges. *(Zoidium upgraded gradient editor adds segment midpoint diamonds, click-above-bar independent opacity stops, vertical drag `> 65 px` off-bar to delete a handle, `Shift` for coarse arrow nudges, and up to 24 local presets.)* |
| **Curve** | Interactive curve plot fitting the value column (Zoidium shared curve widget uses a square plot with optional RGBA channels and explicit Bezier handles; dragging, editing fields, presets, or reset each commit one undo entry; `Escape` cancels an active drag). |
| **Option / List Dropdown** | Native `<select class="pz-inputbox">` (`#202020` bg, `1 px solid #1b1b1b` border, `#7e8fb9` text, `#384668` focus border). Read-only/expression sets `disabled` (transparent background, no border, `#999999` text, native arrow hidden). `Option` stores integer `selectedIndex`; `List` stores item `.value`. Commits one undo operation on change. |
| **Checkbox / Toggle** | Native CM3 represents boolean properties as 2-item `Option` dropdowns (`Off`/`On`) and toolbar toggles (`#acacac` <-> `#8a2828`). Zoidium's UI kit provides `16 × 16 px` checkboxes tinted with `--zui-accent`. |
| **Text Input** | `200 px` wide `.pz-inputbox`; `Enter` blurs and commits via `onchange` (one undo op). For dynamic text properties with no keyframe at `localFrame`, the input hides (`visibility: hidden`). |
| **File / Asset Picker** | Hidden file input + `choose file...` button (`load`, `17 × 17 px`) + clear button (`remove`, `17 × 17 px`, hidden when value is `null`) + `#999999` filename label (`"(none)"` when unset; truncated at `20` characters with `"..."`, full filename in hover tooltip). |
| **Font Picker** | `.pz-tweens` button displaying a `200 × 25 px` font preview sprite row; clicking opens floating `.pz-dropdown` of preset fonts plus a trailing `"custom"` row. Selecting `"custom"` reveals an inline File Picker row underneath. |
| **Texture Picker** | Native `<select>` of preset texture names plus trailing `"custom"`; selecting `"custom"` reveals an inline File Picker row underneath. |
| **Shader Code Button** | `code` icon button (`17 × 17 px`) opening the Shader editor window. *(Zoidium Shader+ upgrades this with line numbers, `10–24 px` zoomable font, Tab/Shift+Tab indent, autocomplete, `400 ms` idle background WebGL compile checks, and `Apply` / `Ctrl+S` / `Ctrl+Enter` commit.)* |
| **Button & Button Rows** | `.proprow.propbutton`: full width, `#434343` bg, `#cbcbcb` text, `2 px solid transparent` border (`#545454` hover, `#384668` focus, `#444444` pressed), `5 px 7 px` padding. Zoidium adds `primary` (`--zui-accent` bg, `#e4e4e4` text) and `danger` (`#df9b9b` text) variants and flex-wrapping `.zoidium-button-row` containers (`5 px` gap). |

### 4.7 Collapsible Sections, Object/Effect/Layer Lists & Drag-to-Reorder

- **Collapse toggle (`.collapse`)**: `20 px` wide, `12 px` `⯆` chevron (`#9a9a9a`, `#cccccc` hover). Toggling rotates the glyph `-90°` over `50 ms linear` and shows/hides the child `<ul>`.
- **Add Object / Add Effect Inline Picker**: Clicking `+` (`add`) or `change` on an `objectList` / `objectSingleton` hides the tree in place and renders a 3-row grid:
  1. Full-width **Cancel** button (`Escape` or `Enter` closes and restores exact prior scroll position).
  2. Auto-focused fuzzy-search input (`.pz-filterbox`, placeholder `"type to filter"`, `16 px`, `#2a2a2a` bg, `2 px solid #363636` border, `#384668` focus border). Fuzzy matching scores `name` (weight `0.6`) and `desc` (weight `0.4`) with threshold `0.45`.
  3. Scrollable `.pz-options` list: category banners (`15 px`, `#222222` bg, `#6d6d6d` text, `4 px 5 px` padding) and selectable items (`16 px` title `#cccccc`, optional `14 px` description `#969696` `5 px` below; hover `#2f2f2f`; keyboard active `#3f3f3f` with `1 px solid #676767` border). `ArrowUp`/`ArrowDown` skips category headers and wraps; `Enter` selects active item (`Ctrl+Enter` or `Ctrl+Click` enters hidden sublists).
- **Row Commands**:
  - Regular list items (when `showListItemButtons` is enabled): `duplicate` (`clone`, omitted on singletons) and `remove` (`delete`), both `17 × 17 px` `#cccccc` icons.
  - Zoidium registered custom editors add a `settings` gear icon on matching rows and an `"Open <title>"` launch button row above their property list.
  - 3D object visibility/enablement is controlled via each object's keyframeable `Enabled` property (not a separate eyeball column in the native tree).
- **Multi-Selection**:
  - Plain click selects one item and clears prior selection (top-level container headers `.toplevel` are non-selectable).
  - `Shift+Click` toggles the clicked item in/out of the active selection, constrained to items with the same `baseTypeString` (preventing mixed-type selections).
  - Selected rows fill with `#384668`.
- **Inline Rename (Right-Click)**:
  - Right-clicking any tree item (or timeline clip) with a writable `name` property replaces its label with an inline `<input>` (`pz-inputbox rename` in trees; `pz-listitem-edit` with `#cccccc` bg, `#000000` text, `1 px solid #868686` border on clips).
  - `Enter` or blur commits one undoable rename; `Escape` restores the original text without modifying history.
- **Drag-to-Reorder & Group Nesting**:
  - Vertical drag threshold: `5 px` (`|pageY - startY| >= 5`).
  - Visual drop indicator: translucent white (`#ffffff`, `opacity: 0.25`).
  - Given hovered target row of height $H$, let $t \in \{0,1\}$ indicate whether target has a compatible child `objectList` (e.g., a 3D Group) and $i \in \{0,1\}$ indicate whether target's parent list accepts sibling insertion. Define normalized pointer offset $n \in [0, 1]$ within the row and edge threshold $s = \frac{1}{t + 2i}$:
    - **Insert Before** ($i = 1$ and $n < s$): renders a `3 px`-tall horizontal bar along the top edge of the target row.
    - **Insert After** ($i = 1$ and $1 - n < s$): renders a `3 px`-tall horizontal bar along the bottom edge of the target row.
    - **Nest Inside** ($t = 1$ and $s \le n \le 1 - s$, or $i = 0$): highlights the entire target row height ($H\text{ px}$).
  - Dropping moves the object in one undo operation; holding **`Alt`** on release **clones** the object into the drop target instead of moving it. Dropping an ancestor into its own descendant is rejected.
- **Context Menus**:
  - Native CM3 suppresses browser context menus everywhere outside `<input>` elements, using right-click directly for inline rename (tree items, clips), time-format cycling (timecode box), and stop deletion (gradient handles).
  - Zoidium anchored/context menus (`openMenu`) render at `13 px` item text (`12 px` section headers), `4 px 0` container padding, `5 px 14 px` item padding, `color-mix(in srgb, accent 55%, #2a2a2a)` hover/keyboard highlight, auto-flip and clamp to viewport bounds, support `ArrowUp`/`ArrowDown`/`Home`/`End`/`Tab` cycling and `Enter`/`Space` activation, and dismiss on `Escape`, outside `pointerdown`, host scroll/resize, or window blur. Only one menu may be open application-wide.

### 4.8 Floating Windows & Pop-Out Windows

Native CM3 spawns secondary OS/browser windows (`600 × 400 px` default) for Graph Editor, Expression Editor, Shader Editor, and `Ctrl+Shift+Backquote` panel clones. Zoidium provides an in-page **Floating Window Layer** (`position: fixed; inset: 0; z-index: 900`) so plugin dialogs and editors stay visible above the workspace without blocking live viewport preview.

| Floating Window Feature | Exact Specification |
| --- | --- |
| Surface & elevation | `#2a2a2a` body, `1 px solid #333333` border (switches to `--zui-accent` `#384668` when focused), shadow `0 8px 28px rgba(0, 0, 0, 0.55)` (`0 12px 40px rgba(0, 0, 0, 0.60)` + `1 px` border ring when maximized). Dropdowns and color pickers opened from inside a floating window stack at `z-index: 1000` (above the `900` layer). |
| Default & minimum sizes | Default size `340 × 420 px` (`380 px` wide shrink-wrapped for confirm dialogs); minimum bounds `minWidth >= 180 px` (default `240 px`), `minHeight >= 80 px` (default `120 px`). |
| Initial placement & cascade | 6-step cascade (`cascade = (cascade + 1) % 6`): `"right"` (default) places at `x = vw - width - 380 - 36 * step, y = 70 + 36 * step`; `"left"` at `x = 70 + 36 * step, y = 70 + 36 * step`; `"center"` centered with `+24 * step` offset. Saved `persistKey` geometry always overrides defaults. |
| Title bar | `30 px` height, `padding: 0 2px 0 9px`, `8 px` gap, `1 px solid #1b1b1b` bottom border, `cursor: move`. Background `#222222` (unfocused) / `color-mix(in srgb, accent 22%, #222222)` (focused). Bold `#dadada` title + dimmed `#7e8fb9` `14 px` subtitle. |
| Title bar buttons | `26 × 26 px` (`5 px` padding, `12 × 12 px` SVG in `#9a9a9a`, hover bg `#333333` & icon `#e4e4e4`, focus border `--zui-accent`): **Pop out** (hidden until window hover or focus-within), **Collapse**, **Maximize/Restore**, **Close**. |
| Dragging | Left-button pointer drag on title bar (or custom drag handle); window sets `opacity: 0.92` while dragging; disabled when maximized or popped out. Position clamps continuously so $x \in [8 - w + 80,\; V_w - 80]$ and $y \in [8,\; V_h - 32]$, keeping at least `80 px` width and `32 px` height reachable. |
| Resize grips (5 edges) | Invisible hit targets straddling borders (hidden when maximized, collapsed, or popped out; top edge reserved for move): **East / West** `10 px` wide (`-6 px` outside, `top: 30 px` to `bottom: 12 px`, `ew-resize`); **South** `10 px` tall (`-6 px` outside, `left/right: 12 px`, `ns-resize`); **South-East / South-West** `20 × 20 px` (`-7 px` offset, `nwse-resize` / `nesw-resize`). SE corner renders a `6 × 6 px` `2 px`-thick `#6d6d6d` L-bracket indicator. West/SW resizing keeps the right edge anchored while updating `x` and `width`. |
| Collapse (double-click) | Double-clicking a standard title bar (or clicking Collapse) collapses the window into a `260 px`-wide title-bar-only bar, hiding subtitle, body, footer, and grips. Double-clicking a custom-chrome drag handle toggles Maximize instead. |
| Maximize / Restore | Fills the host viewport with an `8 px` margin on all four sides (`left/top = 8 px`, `width = vw - 16`, `height = vh - 16`), preserving unmaximized bounds for Restore. |
| Focus & Z-order | Pointer-down anywhere inside a window (or reopening an existing `id`) increments a monotonic `zCounter`, brings the window to front, and assigns `.focused`. Clicking outside all floating windows blurs the active control and clears `.focused` so editor shortcuts resume immediately. Closing a window auto-focuses the highest remaining z-index window. |
| Keyboard isolation & `Escape` | Floating windows stop propagation of `keydown`, `keyup`, `keypress`, and `wheel` events so Space/Delete/arrow keys never leak to CM3. **Exceptions**: `Escape` (if unconsumed and not in IME composition) closes the window; `Ctrl/Cmd+Z`, `Ctrl/Cmd+Y`, `Ctrl/Cmd+Shift+Z`, and `Ctrl/Cmd+S` (when focus is outside text inputs/selects/contenteditable) forward to CM3 undo/redo/save. |
| Validity polling & persistence | Geometry `(x, y, width, height, collapsed, maximized)` persists under `persistKey` on drag/resize/toggle end. Target-bound windows poll `isValid()` every `500 ms` and close automatically if their target object is removed or the project changes. |

---

## 5. Viewport Dynamic Interactions

### 5.1 Editing Camera vs. Compositor Preview Camera

- **Editing Camera Mode (`edit = true`, default ON, toggled via `C` or camera toolbar button)**:
  - Engages whenever `edit === true` **and** exactly one 3D Scene clip (`PZ.layer.scene`) is selected in the timeline.
  - Renders the 3D scene through a dedicated perspective editor camera: `FOV = 60°`, aspect = canvas aspect, `near = 0.1`, `far = 5000`, initial position `(0, 30, 80)`, looking at `(0, 0, 0)`.
  - Enables WebGL layer 1 (showing the ground grid, 3D transform gizmo, and active object/camera/light helpers), `PCFSoftShadowMap`, `autoClear = true`, clear color `#000000`.
  - Enables `EditorControls` pointer/wheel navigation.
- **Compositor Preview Mode (`edit = false`, or when zero / multiple / non-3D-Scene clips are selected)**:
  - Renders the full sequence compositor at `currentFrame` through the scene's authored Camera object(s), including all video tracks, blend modes, 2D layers, and post-processing effects.
  - Disables `EditorControls` and hides all layer-1 helpers and grid lines.

### 5.2 Camera Orbit, Pan, and Zoom (`EditorControls`)

Let $\mathbf{p}$ be the editor camera world position, $\mathbf{c}$ be the orbit pivot (initialized at $(0, 0, 0)$), $d = \|\mathbf{p} - \mathbf{c}\|$, and $\mathbf{N}$ be the camera's $3 \times 3$ normal matrix (camera-local orientation basis).

| Input Gesture | Action | Exact Mathematical Formula |
| --- | --- | --- |
| **Left Mouse Drag** (`button = 0`) | **Orbit** around $\mathbf{c}$ | Convert $\mathbf{p} - \mathbf{c}$ to spherical $(r, \phi, \theta)$. Update $\theta \leftarrow \theta - 0.005\,\Delta x$ and $\phi \leftarrow \phi - 0.005\,\Delta y$ (`rotationSpeed = 0.005` rad/px), clamp $\phi$ away from polar singularities (`makeSafe`), convert back to Cartesian $\mathbf{p} = \mathbf{c} + \mathbf{s}$, and call `lookAt(c)`. |
| **Middle Mouse Drag** (`button = 1`) | **Dolly Zoom** | Compute local offset $\mathbf{v} = (0, 0, \Delta y) \cdot (d \cdot 0.001)$ (`zoomSpeed = 0.001`). If $\|\mathbf{v}\| \le d$, transform by $\mathbf{N}$ and set $\mathbf{p} \leftarrow \mathbf{p} + \mathbf{N}\mathbf{v}$ (distance clamp prevents crossing through $\mathbf{c}$). |
| **Mouse Wheel** | **Dolly Zoom** | Identical to Middle Mouse Drag using `event.deltaY` in place of $\Delta y$. |
| **Right Mouse Drag** (`button = 2`) | **Pan** | Compute local offset $\mathbf{v} = (-\Delta x, +\Delta y, 0) \cdot (d \cdot 0.001)$ (`panSpeed = 0.001`). Transform by $\mathbf{N}$ and translate both $\mathbf{p} \leftarrow \mathbf{p} + \mathbf{N}\mathbf{v}$ and $\mathbf{c} \leftarrow \mathbf{c} + \mathbf{N}\mathbf{v}$. Context menu is suppressed on the canvas. |
| **1-Finger Touch Drag** | **Orbit** | Same as Left Mouse Drag using touch delta $\times -0.005$. |
| **2-Finger Pinch & Drag** | **Zoom + Pan** | Zoom delta = $D_{\text{prev}} - D_{\text{curr}}$ (finger separation change); Pan vector = mean of the two fingers' screen deltas (with X negated). |

*(No `Shift` or `Alt` modifiers alter `EditorControls` in vanilla CM3.)*

### 5.3 Grid, Object Helpers & 3D Transform Gizmos

All 3D helpers and gizmos reside on render **Layer 1** (visible exclusively in Editing Camera mode) with `fog = false`.

- **Ground Grid**: `200 × 200` world units with `10` divisions (`20`-unit cells), center axis lines `#cccccc`, interior grid lines `#333333`.
- **Selected 3D Object Helper** (active when exactly 1 `object3d` is selected in the Objects tree):
  - Camera -> Frustum wireframe (`CameraHelper`)
  - Spot Light -> Cone wireframe (`SpotLightHelper`)
  - Point Light -> Octahedron helper (`size = 5`)
  - Directional Light -> Directional plane + ray helper (`size = 5`)
  - Hemisphere Light -> Hemisphere octahedron helper (`size = 5`)
  - All other 3D meshes/groups -> Axis-aligned wireframe bounding box (`BoxHelper`), refreshed prior to each render.
- **3D Transform Gizmo (`widget3d`)**:
  - Attaches to a single selected 3D object exposing `Position`, `Rotation`, or `Scale` properties. Rendered with `depthTest = false` and `depthWrite = false` so it stays visible through geometry.
  - **Constant screen size**: Scaled every frame by $s = \frac{\|\mathbf{p}_{\text{obj}} - \mathbf{p}_{\text{cam}}\|}{6}$.
  - **Modes** (switched via number keys `1`–`4` while viewport has focus):
    - `1` — **Translate** *(default, `world` space)*: Red (`#ff0000`) X arrow, Green (`#00ff00`) Y arrow, Blue (`#0000ff`) Z arrow, translucent white center octahedron (`XYZ`, opacity `0.25`), and three planar squares (`XY` `#ffff00`, `YZ` `#00ffff`, `XZ` `#ff00ff`, opacity `0.25`).
    - `2` — **Rotate**: Red/Green/Blue camera-facing half-rings (radius `1.0`), yellow outer view-plane ring `E` (`#cccc00`, radius `1.25`), gray inner ring `XYZE` (`#787878`, radius `1.0`).
    - `3` — **Scale** *(auto-switches to `local` space)*: Red/Green/Blue lines terminated by `0.125`-unit cubes, plus a translucent white center cube for uniform `XYZ` scaling.
    - `4` — **None**: Detaches and hides the gizmo.
  - **Hover & Drag**: Raycast hover highlights the active handle in `#ffff00` (`opacity = 1.0`). Left-click on a handle stops event propagation (preventing camera orbit), begins a history transaction and auto-keyframes if the property is animated, updates the property live during drag, and commits one undo operation on release.

### 5.4 2D Layer Transform Overlay (`widget2d`)

Active when `Layer Transform Controls` is toggled ON (`T`) and exactly one video clip is selected.

- **Overlay geometry**: DOM bounding box (`1 px solid #cccccc` border, `1 px solid #666666` outer outline, `cursor: move`) positioned at the layer's screen-projected center, sized to `|resolution * scale| * canvasScaleRatio`, and rotated by `-rotation` radians.
- **Handles**:
  - 8 resize handles (`10 × 10 px`, `#cccccc` fill, `1 px solid #555555` border, `-7 px` centered offset) at the 4 corners (`45°`, `135°`, `225°`, `315°`) and 4 edge midpoints (`0°`, `90°`, `180°`, `270°`).
  - 1 circular rotation handle (`10 × 10 px`, `border-radius: 7 px`) positioned `25 px` above the top-center edge.
- **Drag formulas**:
  - *Body drag*: translates `position` by $(\Delta x / \text{ratio},\; -\Delta y / \text{ratio})$.
  - *Resize handle drag*: projects pointer delta onto the rotated handle axis guide vector and scales the corresponding `scale` component(s).
  - *Rotation handle drag (or `Shift + drag` on any resize handle)*: sets `rotation = initialRotation + (-atan2(y - cy, x - cx) - initialAngle)`.
  - Each gesture begins one history operation (with auto-keyframe if animated), updates live, commits on release, and discards if cancelled via `Escape`.

---

## 6. Complete Keyboard Shortcuts Table

### 6.1 Global Input-Focus Guard

Global editor and toolbar shortcuts are **suppressed** whenever the focused element is an `<input>`, `<select>`, `<textarea>`, or `contentEditable` element, **or** when focus is on a `<button>` and the pressed key is `Enter` or `Space`. Inside Zoidium floating windows, all keys are isolated except `Escape` (closes window) and non-input `Ctrl/Cmd+Z`, `Ctrl/Cmd+Y`, `Ctrl/Cmd+Shift+Z`, and `Ctrl/Cmd+S` (forwarded to CM3).

### 6.2 Master Shortcut Reference

| Scope | Shortcut | Action |
| --- | --- | --- |
| **Global (Project & History)** | `Ctrl+N` or `Ctrl+M` | New project (prompts confirmation if dirty) |
| **Global (Project & History)** | `Ctrl+O` | Open `.pz` project file picker (prompts if dirty) |
| **Global (Project & History)** | `Ctrl+S` | Save project archive (`.pz`) with assets |
| **Global (Project & History)** | `Ctrl+Shift+S` | Save project archive without embedded assets |
| **Global (Project & History)** | `Ctrl+Z` | Undo last history operation |
| **Global (Project & History)** | `Ctrl+Y` | Redo (floating windows also map `Ctrl/Cmd+Shift+Z` to Redo) |
| **Global (Window Layout)** | `` ` `` (`Backquote`) | Toggle maximize / restore the currently focused `.editorpanel` within the window |
| **Global (Window Layout)** | `Ctrl+Shift+Backquote` | Spawn a duplicate of the currently focused `.editorpanel` in a new secondary window |
| **Global (Playback)** | `Space` | Toggle Play (`speed = 1`) / Pause (`speed = 0`) |
| **Global (Playback)** | `Home` | Stop playback and jump to frame `0` |
| **Global (Playback)** | `End` | Stop playback and jump to `totalFrames - 1` |
| **Global (Playback)** | `ArrowLeft` | Stop playback and step `-1` frame |
| **Global (Playback)** | `ArrowRight` | Stop playback and step `+1` frame |
| **Global (Playback)** | `Shift+ArrowLeft` | Stop playback and skip `-5` frames |
| **Global (Playback)** | `Shift+ArrowRight` | Stop playback and skip `+5` frames |
| **Global (Playback)** | `J` | Decrease playback speed by `-0.25` (reverse shuttle) |
| **Global (Playback)** | `K` | Stop playback (`speed = 0`) |
| **Global (Playback)** | `L` | Increase playback speed by `+0.25` (forward shuttle) |
| **Global (Playback)** | `Ctrl+L` | Toggle Loop playback |
| **Global (Markers & Graph)** | `M` | Toggle sequence marker at current frame |
| **Global (Markers & Graph)** | `Ctrl+ArrowLeft` | Jump to previous sequence marker |
| **Global (Markers & Graph)** | `Ctrl+ArrowRight` | Jump to next sequence marker |
| **Global (Markers & Graph)** | `Ctrl+G` | Open Graph Editor window |
| **Global / Viewport** | `C` | Toggle 3D Editing Camera vs. Compositor Preview *(note: when Timeline Tracks panel has focus, `C` splits selected clips instead)* |
| **Global (Viewport)** | `T` | Toggle 2D Layer Transform Controls overlay |
| **Viewport (focused)** | `1` | 3D Transform Gizmo -> Translate (Position) |
| **Viewport (focused)** | `2` | 3D Transform Gizmo -> Rotate |
| **Viewport (focused)** | `3` | 3D Transform Gizmo -> Scale |
| **Viewport (focused)** | `4` | 3D Transform Gizmo -> Hide (None) |
| **Timeline (focused)** | `\` | Fit entire sequence duration into visible timeline width |
| **Timeline (focused)** | `=` | Zoom timeline in (`× 1.03125`) anchored at playhead |
| **Timeline (focused)** | `-` | Zoom timeline out (`/ 1.03125`) anchored at playhead |
| **Timeline (focused)** | `Ctrl+Wheel` | Zoom timeline in/out (`× 1.03125` / `/ 1.03125`) anchored at cursor X |
| **Timeline (focused)** | `Shift+Wheel` | Horizontal timeline scroll using vertical wheel delta |
| **Timeline Tracks (focused)** | `C` | Split selected clip(s) at current playhead frame |
| **Timeline Tracks (focused)** | `L` | Link selected clips |
| **Timeline Tracks (focused)** | `U` | Unlink selected clips |
| **Timeline Tracks (focused)** | `Ctrl+C` / `Ctrl+X` / `Ctrl+V` | Copy / Cut / Paste selected clip(s) at current frame |
| **Timeline Tracks (focused)** | `Delete` / `Backspace` | Delete selected clip(s) |
| **Timeline Keyframes (focused)** | `Ctrl+C` / `Ctrl+X` / `Ctrl+V` | Copy / Cut / Paste selected keyframe(s) aligned so earliest copied keyframe lands at current frame |
| **Timeline Keyframes (focused)** | `Delete` / `Backspace` | Delete selected keyframe(s) (preserving sole keyframe unless `allowEmpty`) |
| **Objects / Effects Tree (focused)** | `Ctrl+C` / `Ctrl+X` / `Ctrl+V` | Copy / Cut / Paste selected object(s) or effect(s) into matching list |
| **Objects / Effects Tree (focused)** | `Delete` / `Backspace` | Delete selected object(s) or effect(s) |
| **Objects / Effects Tree (drag)** | `Alt+Drop` | Duplicate dragged object/effect at drop location instead of moving |
| **Elevator Rail (focused)** | `ArrowUp` / `ArrowDown` / `Enter` | Move focus between sidebar tabs / activate tab |
| **Numeric Scrub Input (editing)** | `Enter` / `Escape` | Commit edit / Revert to initial value |
| **Numeric Scrub Input (editing)** | `ArrowUp` / `ArrowDown` | Step `±1 × step` |
| **Numeric Scrub Input (editing)** | `PageUp` / `PageDown` | Step `±10 × step` |
| **Numeric Scrub Input (scrub)** | `Shift` / `Ctrl` / `Alt` | `×10` coarse / `×0.1` fine / temporarily break `linkRatio` |
| **Expression Textarea** | `Ctrl+Enter` | Commit expression (blur) |
| **Color Picker / Dropdown / Add Filter** | `Enter` / `Escape` | Commit selection / Cancel & revert (Add Filter & Dropdowns also use `ArrowUp`/`ArrowDown`) |
| **Floating Window / Active Drag** | `Escape` | Cancel active drag gesture, or close focused floating window |
| **Inline Rename (Tree / Clip)** | `Enter` / `Escape` | Commit rename / Cancel rename |

---

## 7. Animations, Transitions & Motion Timing

| Element / Interaction | Property & Timing | Easing / Curve | Notes |
| --- | --- | --- | --- |
| Elevator sidebar collapse | `width` over `50 ms` | Default CSS ease | Snappy return to `60 px` rail |
| Elevator sidebar expand (`.open`) | `width` over `200 ms` | Default CSS ease | Smoothly reveals `300 px` drawer + `5px 0 10px rgba(0,0,0,0.5)` shadow |
| Drag-over elevator tab switch | `200 ms` dwell timeout | Discrete trigger | Cancelled immediately on `dragleave` |
| Tree & section collapse chevron | `transform` (`0°` <-> `-90°`) over `50 ms` | `linear` | Crisp micro-rotation |
| Progress bar fill (`.pz-progress span`) | `width` over `1000 ms` | `linear` | Smooths discrete export progress updates |
| Toast notification (Zoidium) | `opacity` & `transform` (`translateX(100% + 20px)` -> `0`) over `180 ms` | `ease-out` | Disabled (`transition: none`) when `prefers-reduced-motion: reduce` is active |
| Floating window drag opacity | Instant `opacity: 0.92` | Immediate | Restores to `1.0` on pointer release |
| Panel & control focus rings | Border color -> `#384668` | Immediate (`0 ms`) | Zero browser outline; `1 px` on panels/inputs, `2 px` on buttons/tabs |
| Button hover borders | `transparent` -> `#545454` (`2 px`) | Immediate (`0 ms`) | `.propbutton`, `.backbutton` |
| Timeline playback auto-scroll | Page-snap jump | Discrete | When `speed != 0` and playhead leaves `[0, viewportWidth]`, snaps `scrollLeft = floor(playheadPx / viewportWidth) * viewportWidth` |
| Timeline ruler scrub auto-scroll (`keepInView`) | Per-frame proportional pan | Exponential approach ($\Delta / 5$ per frame) | Activates within `50 px` of left or right ruler edge |
| Audio meter peak hold decay | Per-frame nonlinear decay | $\text{peak}_{t+1} = \max(\text{cur},\; \text{peak}_t - 0.01 e^{-\text{peak}_t})$ | Slower initial falloff near 100%, faster falloff near 0% |
| Host window resize recalculation | `500 ms` debounce | Trailing timer | Prevents excessive WebGL backing-buffer reallocations during window resize |
| Floating window target validity check | `500 ms` interval | Periodic poll | Auto-closes orphaned editor windows after target deletion |
