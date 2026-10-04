---
name: Synthetica Research Console
colors:
  surface: '#f8f9ff'
  surface-dim: '#cbdbf5'
  surface-bright: '#f8f9ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#eff4ff'
  surface-container: '#e5eeff'
  surface-container-high: '#dce9ff'
  surface-container-highest: '#d3e4fe'
  on-surface: '#0b1c30'
  on-surface-variant: '#3d4a42'
  inverse-surface: '#213145'
  inverse-on-surface: '#eaf1ff'
  outline: '#6d7a72'
  outline-variant: '#bccac0'
  surface-tint: '#006c4a'
  primary: '#006948'
  on-primary: '#ffffff'
  primary-container: '#00855d'
  on-primary-container: '#f5fff7'
  inverse-primary: '#68dba9'
  secondary: '#006c49'
  on-secondary: '#ffffff'
  secondary-container: '#6cf8bb'
  on-secondary-container: '#00714d'
  tertiary: '#545c72'
  on-tertiary: '#ffffff'
  tertiary-container: '#6c748b'
  on-tertiary-container: '#fefcff'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#85f8c4'
  primary-fixed-dim: '#68dba9'
  on-primary-fixed: '#002114'
  on-primary-fixed-variant: '#005137'
  secondary-fixed: '#6ffbbe'
  secondary-fixed-dim: '#4edea3'
  on-secondary-fixed: '#002113'
  on-secondary-fixed-variant: '#005236'
  tertiary-fixed: '#dae2fd'
  tertiary-fixed-dim: '#bec6e0'
  on-tertiary-fixed: '#131b2e'
  on-tertiary-fixed-variant: '#3f465c'
  background: '#f8f9ff'
  on-background: '#0b1c30'
  surface-variant: '#d3e4fe'
typography:
  headline-xl:
    fontFamily: Space Grotesk
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  headline-xl-mobile:
    fontFamily: Space Grotesk
    fontSize: 28px
    fontWeight: '700'
    lineHeight: 36px
    letterSpacing: -0.015em
  headline-lg:
    fontFamily: Space Grotesk
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.01em
  headline-md:
    fontFamily: Space Grotesk
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
  headline-sm:
    fontFamily: Space Grotesk
    fontSize: 16px
    fontWeight: '600'
    lineHeight: 24px
  body-lg:
    fontFamily: Inter
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  body-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '400'
    lineHeight: 20px
  body-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '400'
    lineHeight: 16px
  label-lg:
    fontFamily: JetBrains Mono
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
  label-md:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.02em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 11px
    fontWeight: '400'
    lineHeight: 14px
    letterSpacing: 0.03em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  gutter: 1rem
  gutter-desktop: 1.5rem
  margin: 1rem
  margin-desktop: 2rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2rem
---

## Brand & Style

This design system embodies an authoritative, scientific, and experimental research environment engineered for computer vision and generative deep learning practitioners. Designed specifically for academic benchmarking, evaluation of latent diffusion, GAN-based restoration pipelines, and cross-domain edge-to-image synthesis, the aesthetic relies on **Technical Modernism**—a synthesis of high-density computational tools (such as PyTorch Profiler or Weights & Biases) with the disciplined restraint of classic academic publishing.

The interface prioritizes analytical objectivity, high reproducibility, and cognitive clarity:
- **Audience:** Post-graduate researchers, lab fellows, computer science evaluators, and engineers testing model weights, inference runs, ablation studies, and structural fidelity.
- **Atmosphere:** Rigorous, unembellished, and transparent. The interface recedes to let visual artifacts, fidelity graphs, and reconstruction side-by-side analyses dominate.
- **Visual Balance:** Clean light workspaces, structural division via hairline slate dividers, high-legibility dark slate typography, and decisive emerald operational signals representing validation, active processes, and model convergence.

## Colors

The color architecture is built around an analytical, low-distraction spectrum configured strictly for light-mode scientific evaluation.

### Palette Architecture
- **Primary Surface & Canvas:** Base canvas is set to `#f8fafc` (Slate 50), providing soft separation against pure `#ffffff` structural cards and viewport panels.
- **Interactive Primary:** `#059669` (Emerald 600) drives execution vectors, primary triggers (e.g., "Run Inference", "Generate Latent Matrix"), and positive parameter states. State shifts transition to `#047857` (Emerald 700) on hover and `#065f46` (Emerald 800) under active press.
- **Subtle Mint/Emerald Accents:** `#ecfdf5` (Emerald 50) and `#d1fae5` (Emerald 100) are reserved for active tab indicators, confidence tags, PSNR/SSIM positive metric delta pills, and active slider fills.
- **Secondary & Auxiliary State:** `#10b981` (Emerald 500) serves as live execution feedback, progress ticks, and benchmark highlights.
- **Borders & Dividers:** Rigid structural segmentation uses `#e2e8f0` (Slate 200) for standard card and viewport borders, transitioning to `#cbd5e1` (Slate 300) for active inputs or segmented dividers.
- **Typography & Neutrals:** Primary readouts use `#0f172a` (Slate 900) for max-contrast semantic clarity; secondary descriptions and structural labels use `#334155` (Slate 700) and `#64748b` (Slate 500).
- **Diagnostics & Error Warnings:** Real-time MSE spike metrics, perceptual loss warnings, out-of-bounds parameter errors, and GPU VRAM saturation warnings utilize `#f59e0b` (Amber 500) on `#fffbeb` (Amber 50), and `#e11d48` (Rose 600) on `#fff1f2` (Rose 50).

## Typography

The typographic hierarchy implements three distinct operational roles:
1. **Space Grotesk (Display & Structural Anchors):** Anchors module names, panel headings, and experiment session identifiers with technical geometry, distinct aperture cuts, and confident proportions.
2. **Inter (Analytical Body & System Instructions):** Delivers clean readability for documentation, methodology summaries, tooltips, parameter descriptions, and prompt configuration.
3. **JetBrains Mono (Quantitative Data & Telemetry):** Applied strictly to floating-point metrics (SSIM, PSNR, LPIPS), compute execution durations (`ms`), tensor dimensions (`[B, C, H, W]`), epoch cycles, seed values, and parameter slider readouts. All tabular numerical figures must rely on tabular lining (`tnum`) to eliminate spatial jitter during iterative inference runs.

## Layout & Spacing

The layout is built upon an 8pt modular grid system arranged as a **Fixed-Fluid Hybrid Console**:
- **Application Shell:** Operates inside a full-height viewport frame (`100vh`) with a fixed high-density lateral parameter sidebar (360px wide on desktop), and a fluid primary analytics canvas that scales to available workspace width.
- **Card-Level Structural Alignment:** Component internal padding adheres to strict multiples of `space-sm` (8px) and `space-md` (16px). High-density metric trays use `space-xs` (4px) gaps between visual chips.
- **Breakpoints & Adaptation:**
  - **Desktop (>= 1280px):** Multi-pane split view with side-by-side comparative inspection tiles (Input, Ground Truth, Synthesized Output, Difference Mask), docked floating execution telemetry, and locked parameter inspector.
  - **Tablet (768px - 1279px):** Split-view cascades to a 2x2 comparison matrix; lateral controls collapse into a collapsable top panel with a horizontal parameter ribbon.
  - **Mobile (< 768px):** Linear stacked reflow. Comparison view shifts from side-by-side layout to an interactive swipe-slider or tabbed image toggle.

## Elevation & Depth

To maintain academic rigor and prevent visual fatigue during long analysis sessions, this design system minimizes heavy drop shadows in favor of **Crisp Structural Layering and Low-Contrast Hairline Outlines**:

- **Ground Plane (Level 0):** Background foundation set to `#f8fafc`. Flat, non-elevated.
- **Structural Modules (Level 1):** Main functional cards, inspector modules, and output viewports sit on `#ffffff`, framed by a consistent 1px solid border of `#e2e8f0`. No drop shadow is used on base structural cards to enforce crisp demarcation.
- **Floating Overlays & Tooltips (Level 2):** Floating metadata panels, magnified visual inspectors, zoom viewports, and context dropdowns use a sharp, minimal ambient shadow: `0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.03)` with a solid `#cbd5e1` hairline border.
- **Interactive State Elevation:** Active parameter sliders, draggable split-screen divider nodes, and focused inputs introduce an elevated emerald focus ring (`0 0 0 2px #ecfdf5, 0 0 0 4px #10b981`) rather than artificial physical elevation.

## Shapes

The interface embraces a restrained **Soft (`1`)** shape language that highlights machine-like utility over playful consumer software cues:

- **Base Radius (0.25rem / 4px):** Standard interactive elements, including primary action buttons, segmented toggle items, metric pills, parameter input fields, and slider tracks.
- **Container Radius (0.5rem / 8px):** Structural cards, comparison viewports, modal panes, and lateral controls panels.
- **Divider Lines:** 1px hairline cuts using `#e2e8f0`. Corners within segmented button groups join without overlapping border radii to maintain sharp mechanical precision.

## Components

### Buttons & Interactive Controls
- **Primary Operational Button:** Solid `#059669` fill with `#ffffff` text, font `JetBrains Mono` medium 12px, letter-spacing `0.02em`. Micro-transition to `#047857` on hover. Padding is `space-sm` vertically, `space-md` horizontally.
- **Secondary / Action Ghost:** Flat transparent surface, `#334155` text, 1px border `#e2e8f0`. On hover, background shifts to `#f1f5f9` with border `#cbd5e1`.
- **Destructive / Reset:** Border `#e2e8f0`, text `#e11d48`. On hover, tint background with `#fff1f2`.

### Segmented Controls & Tabs
- **Segmented Pipeline Selector:** Container encased in `#f1f5f9` with a 1px border `#e2e8f0`. Active option sits as a discrete white card (`#ffffff`) with 1px border `#cbd5e1` and font weight `600` in `#0f172a`.
- **System Tabs:** Flat horizontal baseline indicator. Active state highlights with a 2px `#059669` bottom bar and `#0f172a` text weight; inactive tabs render `#64748b` with no line.

### Parameter Sliders
- **Track & Fill:** Inactive track has a 4px height in `#e2e8f0`. Active fill uses `#059669`.
- **Thumb Controller:** 14px circular thumb in `#ffffff`, surrounded by a 2px solid `#059669` rim. Shows a `#ecfdf5` halo on grab.
- **Integrated Digital Readout:** Sits adjacent to the slider label as a mono badge (`JetBrains Mono`, 12px) in `#0f172a`, wrapped in `#f8fafc` with a 1px `#e2e8f0` stroke.

### Metric Chips & Status Indicators
- **Standard Metric Pill:** Surface `#f8fafc`, 1px border `#e2e8f0`, label in `Inter` (11px, `#64748b`), numeric score in `JetBrains Mono` (12px bold, `#0f172a`).
- **Optimal State Badge:** Surface `#ecfdf5`, border `#a7f3d0`, text `#047857`.
- **Degradation / Error Badge:** Surface `#fff1f2`, border `#fecdd3`, text `#e11d48`.

### Inspection & Comparison Cards
- **Dual/Quad Canvas Cards:** Crisp `#ffffff` container, 1px `#e2e8f0` outline. Top metadata header contains artifact title, resolution scale, and color space tag in `#64748b`.
- **Interactive Split Slider Divider:** Vertical 2px line in `#ffffff` with a drop-shadowed centered circular handle containing horizontal glyph indicators (`↔`), permitting real-time before/after raster scrubbing.

### Input Fields & Terminal Seeds
- **Text & Numeric Inputs:** Solid `#ffffff` background, 1px border `#e2e8f0`, inner padding `space-sm` by `space-md`. Typography uses `JetBrains Mono` for precise character alignment. Focused field switches border to `#059669` with an outward 3px soft tint ring (`#ecfdf5`).