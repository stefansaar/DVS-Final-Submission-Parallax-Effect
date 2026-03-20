"""
compositing.py

Enhanced parallax compositor with mask dilation, inpaint-based pixel fill,
and feathered alpha edges.

Implements:
    T1.3 — Mask dilation: expand each layer mask outward so overlapping buffers
            cover the gaps that appear when layers shift apart.
    T1.4 — Pixel fill: fill the newly dilated region with smooth colour via
            cv2.inpaint (Telea algorithm).
    T1.6 — Premultiplied-alpha compositor with feathered edges: fewer per-pixel
            multiplies than straight-alpha blending, and soft alpha transitions
            eliminate hard seams between layers.

Drop-in replacement for the precompute_layers() and composite() functions
in test_parallax.py.

Usage:
    python compositing.py <image_path> <depth_map_path> [num_layers]
"""

import os
import sys
import cv2
import numpy as np
from layer_segmentation import segment_layers


PARALLAX_STRENGTH = 20
DISPLAY_MAX_WIDTH = 900


# ---------------------------------------------------------------------------
# T1.3 — Mask dilation
# ---------------------------------------------------------------------------

def dilate_mask(alpha: np.ndarray, dilation_px: int) -> np.ndarray:
    """
    Expand a binary alpha mask outward by *dilation_px* pixels.

    Args:
        alpha:        (H, W) uint8 mask, values in {0, 255}.
        dilation_px:  Radius of the dilation kernel in pixels.

    Returns:
        Dilated alpha mask, same shape and dtype.
    """
    k = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(alpha, kernel, iterations=1)


# ---------------------------------------------------------------------------
# T1.4 — Inpaint-based pixel fill
# ---------------------------------------------------------------------------

def fill_inpaint(rgb: np.ndarray, original_alpha: np.ndarray,
                 radius: int = 5) -> np.ndarray:
    """
    Fill ALL transparent pixels with smooth colour using cv2.inpaint (Telea).

    The mask covers every pixel where original_alpha == 0, so the algorithm
    only draws from truly opaque source pixels — no black contamination from
    the empty regions outside the dilated zone.

    Pixels far from any opaque boundary get interpolated colours too, but
    they will have alpha == 0 in the final layer so they are invisible.

    Args:
        rgb:            (H, W, 3) uint8 colour data.
        original_alpha: (H, W) uint8 mask (opaque pixels are the source).
        radius:         Inpaint neighbourhood radius (pixels).

    Returns:
        (H, W, 3) uint8 with all transparent regions filled.
    """
    inpaint_mask = (original_alpha == 0).astype(np.uint8) * 255
    if not inpaint_mask.any():
        return rgb
    return cv2.inpaint(rgb, inpaint_mask, radius, cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Alpha feathering
# ---------------------------------------------------------------------------

def feather_alpha(dilated_alpha: np.ndarray, original_alpha: np.ndarray,
                  sigma: float = 4.0) -> np.ndarray:
    """
    Soften the hard edges of a dilated mask with a Gaussian blur.

    The original opaque region stays at 255; only the dilated fringe gets a
    smooth falloff, so layer transitions look natural instead of aliased.

    Args:
        dilated_alpha:  (H, W) uint8 — the expanded mask.
        original_alpha: (H, W) uint8 — the mask before dilation.
        sigma:          Gaussian sigma controlling the falloff width.

    Returns:
        (H, W) uint8 feathered alpha.
    """
    blurred = cv2.GaussianBlur(dilated_alpha.astype(np.float32), (0, 0), sigmaX=sigma)
    # Keep original opaque pixels at full 255
    feathered = np.maximum(blurred, original_alpha.astype(np.float32))
    return feathered.clip(0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# T1.3 + T1.4 combined
# ---------------------------------------------------------------------------

def dilate_and_fill(layers: list, dilation_px: int = 20) -> list:
    """
    Dilate each layer's mask, inpaint the new region, and feather the edges.

    Args:
        layers:       List of RGBA uint8 arrays (H, W, 4) from segment_layers().
        dilation_px:  How many pixels to expand each mask outward.

    Returns:
        List of RGBA uint8 arrays, same format — can be passed directly to
        precompute_layers().
    """
    out = []
    for layer in layers:
        rgb = layer[:, :, :3]
        alpha = layer[:, :, 3]

        dilated_alpha = dilate_mask(alpha, dilation_px)
        filled_rgb = fill_inpaint(rgb, alpha, radius=5)
        soft_alpha = feather_alpha(dilated_alpha, alpha, sigma=dilation_px / 5)

        out.append(np.dstack([filled_rgb, soft_alpha]))
    return out


# ---------------------------------------------------------------------------
# T1.6 — Premultiplied-alpha compositor
# ---------------------------------------------------------------------------

def precompute_layers(layers: list) -> list:
    """
    Convert RGBA uint8 layers to premultiplied-alpha float32.

    Drop-in replacement for test_parallax.precompute_layers().

    Returns:
        List of (premul_rgb (H,W,3) float32, alpha (H,W,1) float32) tuples.
    """
    out = []
    for layer in layers:
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        premul_rgb = layer[:, :, :3].astype(np.float32) * alpha
        out.append((premul_rgb, alpha))
    return out


def composite(float_layers: list, shifts: list, h: int, w: int,
              fill_gaps: bool = False) -> np.ndarray:
    """
    Shift each layer and alpha-blend back-to-front.  Returns BGR uint8.

    Drop-in replacement for test_parallax.composite().

    When fill_gaps=True, tracks total alpha coverage and uses cv2.inpaint
    to fill only the gap pixels (where no layer covers).
    """
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    coverage = np.zeros((h, w, 1), dtype=np.float32)

    for (premul_rgb, alpha), shift in zip(float_layers, shifts):
        if len(shift) == 3:
            dx, dy, scale = shift
        else:
            dx, dy = shift
            scale = 1.0

        if scale != 1.0:
            cx, cy = w / 2, h / 2
            M = np.float32([
                [scale, 0, dx + cx * (1 - scale)],
                [0, scale, dy + cy * (1 - scale)]
            ])
        else:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_rgb = cv2.warpAffine(premul_rgb, M, (w, h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
        shifted_alpha = cv2.warpAffine(alpha, M, (w, h),
                                       flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT,
                                       borderValue=0)
        if shifted_alpha.ndim == 2:
            shifted_alpha = shifted_alpha[:, :, np.newaxis]

        canvas = canvas * (1.0 - shifted_alpha) + shifted_rgb
        coverage = coverage + shifted_alpha * (1.0 - coverage)

    bgr = canvas.clip(0, 255).astype(np.uint8)[:, :, ::-1].copy()

    if fill_gaps:
        gap_mask = (coverage[:, :, 0] < 0.99).astype(np.uint8) * 255
        if gap_mask.any():
            bgr = cv2.inpaint(bgr, gap_mask, 5, cv2.INPAINT_TELEA)

    return bgr


# ---------------------------------------------------------------------------
# Standalone viewer
# ---------------------------------------------------------------------------

def run_viewer(image_bgr, depth_map, num_layers, title="Parallax"):
    """Run the interactive parallax viewer. Returns when user presses Q."""
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_w > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / orig_w
        disp_w = DISPLAY_MAX_WIDTH
        disp_h = int(orig_h * scale)
        image_bgr = cv2.resize(image_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        depth_map = cv2.resize(depth_map, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    h, w = image_bgr.shape[:2]

    layers = segment_layers(image_bgr, depth_map, num_layers)
    float_layers = precompute_layers(layers)

    fill_gaps = False
    mouse = {"x": w // 2, "y": h // 2}

    def on_mouse(_event, x, y, _flags, _param):
        mouse["x"] = x
        mouse["y"] = y

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, on_mouse)

    while True:
        offset_x = (mouse["x"] - w / 2) / (w / 2)
        offset_y = (mouse["y"] - h / 2) / (h / 2)

        shifts = []
        for i in range(num_layers):
            depth_weight = i / max(num_layers - 1, 1)
            dx = -offset_x * PARALLAX_STRENGTH * depth_weight
            dy = -offset_y * PARALLAX_STRENGTH * depth_weight
            shifts.append((dx, dy))

        frame = composite(float_layers, shifts, h, w, fill_gaps=fill_gaps)

        label = "GAP-FILL ON" if fill_gaps else "GAP-FILL OFF"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.putText(frame, title, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 0), 2)
        cv2.imshow(title, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            fill_gaps = not fill_gaps
            print(f"Gap-fill: {'ON' if fill_gaps else 'OFF'}")

    cv2.destroyAllWindows()


def find_all_cases(images_dir="./images", depth_dir="./depth_maps"):
    """Find all image+depth pairs available for demo."""
    import glob as g
    cases = []
    for img_path in sorted(g.glob(os.path.join(images_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(img_path))[0]
        depth_pattern = os.path.join(depth_dir, f"depth_{stem}_run*.png")
        matches = sorted(g.glob(depth_pattern))
        if matches:
            cases.append((stem, img_path, matches[-1]))
    return cases


def main():
    import os as _os

    # --- Mode 1: --all flag → cycle through all cases ---
    if "--all" in sys.argv:
        num_layers = 5
        for arg in sys.argv[1:]:
            if arg.isdigit():
                num_layers = int(arg)

        cases = find_all_cases()
        if not cases:
            print("No image+depth pairs found."); sys.exit(1)

        total = len(cases)
        for idx, (stem, img_path, depth_path) in enumerate(cases):
            print(f"\n[{idx+1}/{total}] {stem}")
            image_bgr = cv2.imread(img_path)
            depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            if image_bgr is None or depth_map is None:
                print(f"  Skipping (could not load)"); continue
            print(f"  Press Q to go to next image. Press D to toggle gap-fill.")
            run_viewer(image_bgr, depth_map, num_layers, title=f"Parallax — {stem} [{idx+1}/{total}]")

        print("\nAll done!")
        return

    # --- Mode 2: single image (original behaviour) ---
    if len(sys.argv) < 3:
        print("Usage: python compositing.py <image_path> <depth_map_path> [num_layers]")
        print("       python compositing.py --all [num_layers]")
        sys.exit(1)

    image_path = sys.argv[1]
    depth_path = sys.argv[2]
    num_layers = int(sys.argv[3]) if len(sys.argv) >= 4 else 5

    print("Loading image and depth map...")
    image_bgr = cv2.imread(image_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        print(f"Error: could not read image: {image_path}"); sys.exit(1)
    if depth_map is None:
        print(f"Error: could not read depth map: {depth_path}"); sys.exit(1)

    run_viewer(image_bgr, depth_map, num_layers)
    print("Done.")


if __name__ == "__main__":
    main()
