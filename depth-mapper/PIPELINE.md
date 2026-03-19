# Parallax Effect Pipeline

## Overall Flow

```
                         ┌─────────────────┐
                         │   Input Image    │
                         │   (josh.jpg)     │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  Step 1: Depth Estimation   │
                    │  depth_map_api.py           │
                    │                             │
                    │  HuggingFace API            │
                    │  (Depth Anything V2)        │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Step 2: Depth Cleaning     │
                    │  depth_processing.py        │
                    │                             │
                    │  Bilateral Filter (L5)      │
                    │  + Edge-Guided Sharpen (L8) │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Step 3: Layer Segmentation  │
                    │  layer_segmentation.py       │
                    │                              │
                    │  K-Means Clustering (L9)     │
                    │  + Morphological Cleanup     │
                    └─────────────┬────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  N RGBA Layers   │
                         │  (layer_00.png   │
                         │   ...            │
                         │   layer_04.png)  │
                         └────────┬─────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Step 4: Compositing         │
                    │  compositing.py              │
                    │                              │
                    │  Premultiplied Alpha (L3/L4) │
                    │  + Affine Transform (L11)    │
                    │  + Real-time Gap Fill        │
                    └─────────────┬────────────────┘
                                  │
                         ┌────────▼────────┐
                         │  Interactive     │
                         │  Parallax Viewer │
                         └─────────────────┘
```

---

## Step 1: Depth Estimation

```
┌──────────────┐          ┌──────────────────────┐          ┌──────────────┐
│              │  upload  │   HuggingFace Space   │  return  │              │
│  Input RGB   │ ───────► │   Depth Anything V2   │ ───────► │  Depth Map   │
│  (H x W x 3)│          │   (Transformer Model) │          │  (H x W)     │
│              │          │                       │          │              │
│  ██████████  │          │  Monocular Depth      │          │  ░░▒▒▓▓████  │
│  ██████████  │          │  Estimation           │          │  ░░▒▒▓▓████  │
│  ██████████  │          │                       │          │  ░░▒▒▓▓████  │
└──────────────┘          └───────────────────────┘          └──────────────┘
                                                              dark = far
                                                              bright = near
```

- **Input**: RGB image (e.g. `images/josh.jpg`)
- **Output**: Grayscale depth map saved to `depth_maps/depth_josh_run1.png`
- **Caching**: Skips API call if depth map already exists

---

## Step 2: Depth Cleaning

```
                    Bilateral Filter (L5)          Edge-Guided Sharpen (L8)
                    ─────────────────────          ────────────────────────

  Raw Depth         Smoothed Depth                 Final Clean Depth
 ┌──────────┐      ┌──────────┐                   ┌──────────┐
 │░░▒▓█ ▓▒░ │      │░░▒▓██▓▒░ │                   │░░▒▓██▓▒░ │
 │░▒▓█▓ █▓▒ │ ───► │░▒▓██ █▓▒ │ ───────────────►  │░▒▓██ █▓▒ │
 │░▒▓ ▓█▓▒░ │      │░▒▓ ▓██▒░ │                   │░▒▓ ▓██▒░ │
 └──────────┘      └──────────┘                   └──────────┘
  noisy edges       smooth within                  sharp edges aligned
  depth jumps       objects, but                   to actual RGB object
                    edges blurred                  boundaries (Canny)
```

| Operation | Course | What It Does |
|-----------|--------|--------------|
| `cv2.bilateralFilter` | L5 Spatial Filtering | Smooths depth noise while preserving large depth edges |
| Canny + Unsharp Mask | L8 Feature Extraction | Detects RGB edges, uses them to sharpen depth boundaries |

---

## Step 3: Layer Segmentation

```
  Clean Depth Map          K-Means (L9)              Label Map              RGBA Layers
 ┌──────────────┐        ┌──────────────┐        ┌──────────────┐        ┌──────────────┐
 │              │        │  Cluster     │        │ 00000001111  │        │ Layer 0 (far)│
 │  Continuous  │        │  depth values│        │ 00000011111  │        │ ████████     │
 │  grayscale   │ ─────► │  into N      │ ─────► │ 00222233333  │ ─────► │              │
 │  0-255       │        │  groups      │        │ 22222234444  │        │ Layer 1      │
 │              │        │              │        │ 22233344444  │        │     ████     │
 └──────────────┘        └──────────────┘        └──────────────┘        │              │
                                                                         │ Layer 2      │
                          + Smoothing (L5)         + Connected            │       ████   │
                            cv2.blur               Components (L6-7)     │              │
                            removes noise          removes tiny           │ Layer 3      │
                            in labels              fragments              │         ████ │
                                                                         │ (near)       │
                                                                         └──────────────┘
```

**Layer ordering**:
```
Layer 0:  ████████░░░░░░░░  background (farthest, moves least)
Layer 1:  ░░░░████████░░░░
Layer 2:  ░░░░░░░░████████
Layer 3:  ░░░░░░░░░░░░████  foreground (nearest, moves most)

█ = opaque (alpha = 255)
░ = transparent (alpha = 0)
```

---

## Step 4: Compositing (Ken's Implementation)

### 4a. Premultiplied Alpha Conversion (L3 Colour Spaces + L4 Intensity Transforms)

```
  RGBA uint8 Layer                         Premultiplied float32
 ┌─────────────────┐                      ┌─────────────────────────┐
 │ R  G  B  │  A   │                      │ premul_rgb    │  alpha  │
 │ 200 50 30│ 255  │   alpha /= 255.0     │ 200  50  30  │  1.0   │
 │ 200 50 30│ 255  │ ──────────────────►   │ 200  50  30  │  1.0   │
 │   0  0  0│   0  │   rgb *= alpha       │   0   0   0  │  0.0   │
 │   0  0  0│   0  │                      │   0   0   0  │  0.0   │
 └─────────────────┘                      └─────────────────────────┘

  Channel separation: layer[:,:,:3] and layer[:,:,3:4]  ← L3
  Normalisation: alpha / 255.0  (0-255 → 0.0-1.0)      ← L4
```

### 4b. Parallax Shift via Affine Transform (L11 Geometric Transforms)

```
  Mouse position → offset_x ∈ [-1, +1]

  For each layer i:
    depth_weight = i / (num_layers - 1)      0.0 for background, 1.0 for foreground
    dx = -offset_x × PARALLAX_STRENGTH × depth_weight

  ┌─────────────────────────────────────────────────────────────┐
  │                                                             │
  │  Layer 0 (bg):   dx = -offset × 20 × 0.00 =  0px  ──────  │
  │  Layer 1:        dx = -offset × 20 × 0.25 =  5px  ─────   │
  │  Layer 2:        dx = -offset × 20 × 0.50 = 10px  ────    │
  │  Layer 3:        dx = -offset × 20 × 0.75 = 15px  ───     │
  │  Layer 4 (fg):   dx = -offset × 20 × 1.00 = 20px  ──      │
  │                                                             │
  └─────────────────────────────────────────────────────────────┘
                  Background stays still, foreground moves most
                  → creates depth illusion (motion parallax)

  Affine matrix:  M = | 1  0  dx |     x' = x + dx
                      | 0  1  dy |     y' = y + dy

  cv2.warpAffine(layer, M, (w, h))      ← L11
```

### 4c. Porter-Duff Alpha Compositing

```
  Blending order: back-to-front (Layer 0 first, Layer N-1 last)

  canvas = black (all zeros)

  For each layer:
    canvas = canvas × (1 - src_alpha) + src_premul_rgb
             ─────────────────────────   ──────────────
             fade existing content        add new layer
             where new layer is opaque    (already multiplied by alpha)

  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ Layer 0  │ + │ Layer 1  │ + │ Layer 2  │ + │ Layer 3  │
  │ ████░░░░ │   │ ░░████░░ │   │ ░░░░████ │   │ ░░░░░░██ │
  └──────────┘   └──────────┘   └──────────┘   └──────────┘
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                              │
                    ┌─────────▼─────────┐
                    │ ████████████████  │
                    │  Complete Frame   │
                    └──────────────────┘
```

### 4d. Real-time Gap Detection & Fill

```
  When mouse moves far → layers shift apart → gaps appear

  WITHOUT gap-fill:
  ┌──────────────────────────────────┐
  │ ████ ■ ████████████ ■ ████████  │   ■ = black gaps (coverage = 0)
  └──────────────────────────────────┘

  Gap detection (every frame):
  ┌───────────────────────────────────────────────────────────────┐
  │                                                               │
  │  coverage += shifted_alpha × (1 - coverage)   per layer       │
  │                                                               │
  │  After all layers:                                            │
  │    coverage ≈ 1.0  → pixel is covered      → do nothing      │
  │    coverage < 0.99 → pixel is a gap        → mark for fill   │
  │                                                               │
  │  gap_mask = (coverage < 0.99) × 255                           │
  │                                                               │
  └───────────────────────────────────────────────────────────────┘

  cv2.inpaint(bgr, gap_mask, radius=5, INPAINT_TELEA):

  ┌──────────────────────────────────┐
  │ ████ ■ ████████████ ■ ████████  │   Before: black gaps
  └──────────────────────────────────┘
                    │
                    ▼  Telea: fill from gap edges inward
  ┌──────────────────────────────────┐
  │ ████████████████████████████████ │   After: gaps filled with
  └──────────────────────────────────┘   interpolated colours
```

---

## Course Lecture Mapping

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE STAGE                              │
│                                                                     │
│  Depth Estimation ──► Depth Cleaning ──► Segmentation ──► Composit │
│                                                                     │
│       L13               L5    L8           L9   L6-7      L3  L4   │
│  Classification    Spatial  Feature    Segment  Morpho   Colour    │
│  (CNN/Transform)   Filter  Extract    (K-Means) Ops     Spaces    │
│                                                                     │
│                                                           L5  L11  │
│                                                         Gauss  Geo │
│                                                         Blur  Trans│
│                                                                     │
│  Ken's part: ─────────────────────────────────────── ▶  Step 4     │
│  Stefan's part: Steps 1-3                                          │
└─────────────────────────────────────────────────────────────────────┘

  L2  Anatomy of Vision ──── Motion parallax = depth cue from movement
  L3  Colour Spaces ──────── BGR↔RGB, RGBA channel separation
  L4  Intensity Transforms ─ Depth normalisation, alpha 0-255 → 0.0-1.0
  L5  Spatial Filtering ──── Bilateral filter, Gaussian blur (feathering)
  L6-7 Morphological Ops ── cv2.dilate (mask expansion), connected components
  L8  Feature Extraction ─── Canny edge detection (depth boundary alignment)
  L9  Segmentation ────────── K-Means clustering (depth → layers)
  L11 Geometric Transforms ─ cv2.warpAffine (parallax layer shifting)
  L13 Classification ──────── Depth Anything V2 (neural network)
```

---

## Interactive Controls

```
┌─────────────────────────────────┐
│         Parallax Viewer         │
│                                 │
│  Mouse move = parallax shift    │
│  D key     = toggle gap-fill    │
│  Q key     = quit / next image  │
│                                 │
│  python compositing.py --all    │
│  → cycles through all 9 images  │
└─────────────────────────────────┘
```
