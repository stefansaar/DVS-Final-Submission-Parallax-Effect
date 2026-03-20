"""
run.py  —  End-to-end parallax pipeline

Give it an image filename (inside ./0_source_images/) and it will:
    1. Estimate the depth map via the HuggingFace API  (skipped if one already exists)
    2. Segment into RGBA layers  (k-means)

Usage:
    python run.py <image_filename> [num_layers]

Examples:
    python run.py josh.jpg
    python run.py josh.jpg 8
    python run.py josh.jpg 5 --rerun-depth   # force re-run depth estimation
"""

import os
import sys
import glob
import cv2
import numpy as np
from shutil import copy2
from layer_segmentation import segment_layers, save_layers, save_depth_coloured, save_depth_histogram, estimate_num_layers
from compositing import precompute_layers, composite

IMAGES_DIR        = "./0_source_images"
DEPTH_DIR         = "./1_depth_maps"
DISPLAY_MAX_WIDTH = 900
PARALLAX_STRENGTH = 20


# ---------------------------------------------------------------------------
# Step 1 — Depth estimation
# ---------------------------------------------------------------------------

def find_existing_depth(stem: str) -> str | None:
    """Return the most recent depth map for this image stem, or None."""
    pattern = os.path.join(DEPTH_DIR, f"depth_{stem}_run*.png")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def run_depth_estimation(image_path: str, stem: str) -> str:
    """Call the HuggingFace API and save the depth map. Returns the saved path."""
    from gradio_client import Client, handle_file

    os.makedirs(DEPTH_DIR, exist_ok=True)

    n = 1
    while True:
        out_path = os.path.join(DEPTH_DIR, f"depth_{stem}_run{n}.png")
        if not os.path.exists(out_path):
            break
        n += 1

    print("Connecting to HuggingFace Space for depth estimation...")
    client = Client("saarstefan/depth-mapping-test")
    print(f"Estimating depth for {image_path} ...")
    result = client.predict(handle_file(image_path), api_name="/estimate_depth")
    copy2(result, out_path)
    print(f"Depth map saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Step 2-3 — Segmentation (clean + split into layers)
# ---------------------------------------------------------------------------

def build_layers(image_bgr: np.ndarray, depth_map: np.ndarray, num_layers: int | None, stem: str):
    if num_layers is None:
        num_layers = estimate_num_layers(depth_map)
        print(f"Auto-detected {num_layers} depth layers from histogram.")

    out_dir = os.path.join("./2_layers", stem)
    os.makedirs(out_dir, exist_ok=True)

    # Remove stale layer files from any previous run with a different layer count
    for old_file in glob.glob(os.path.join(out_dir, "layer_*.png")):
        os.remove(old_file)

    save_depth_coloured(depth_map, os.path.join(out_dir, "depth_original.png"))
    save_depth_histogram(depth_map, num_layers, os.path.join(out_dir, "depth_histogram.png"))

    print(f"Segmenting into {num_layers} layers (k-means)...")
    layers = segment_layers(image_bgr, depth_map, num_layers=num_layers)
    save_layers(layers, out_dir)
    print(f"Layers saved to {out_dir}/")
    return layers, num_layers


# ---------------------------------------------------------------------------
# Step 4 — Parallax viewer
# ---------------------------------------------------------------------------

def run_viewer(image_bgr: np.ndarray, layers: list, num_layers: int):
    h, w = image_bgr.shape[:2]
    float_layers = precompute_layers(layers)
    mouse = {"x": w // 2, "y": h // 2}

    def on_mouse(_event, x, y, _flags, _param):
        mouse["x"] = x
        mouse["y"] = y

    print("Opening viewer — move your mouse to test the effect. Press Q to quit.")
    cv2.namedWindow("Parallax")
    cv2.setMouseCallback("Parallax", on_mouse)

    while True:
        offset_x = (mouse["x"] - w / 2) / (w / 2)
        offset_y = (mouse["y"] - h / 2) / (h / 2)

        shifts = []
        for i in range(num_layers):
            depth_weight = i / max(num_layers - 1, 1)
            dx = -offset_x * PARALLAX_STRENGTH * depth_weight
            dy = -offset_y * PARALLAX_STRENGTH * depth_weight
            shifts.append((dx, dy))

        frame = composite(float_layers, shifts, h, w)
        cv2.imshow("Parallax", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py <image_filename> [num_layers] [--rerun-depth]")
        sys.exit(1)

    filename   = sys.argv[1]
    num_layers = None  # None = auto-detect from depth histogram
    rerun      = "--rerun-depth" in sys.argv
    no_viewer  = "--no-viewer" in sys.argv

    for arg in sys.argv[2:]:
        if arg.isdigit():
            num_layers = int(arg)

    image_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.isfile(image_path):
        print(f"Error: image not found at {image_path}")
        sys.exit(1)

    stem = os.path.splitext(filename)[0]

    # --- Step 1: depth map ---
    existing = find_existing_depth(stem)
    if existing and not rerun:
        print(f"Found existing depth map: {existing}  (use --rerun-depth to regenerate)")
        depth_path = existing
    else:
        depth_path = run_depth_estimation(image_path, stem)

    # --- Load both ---
    print("Loading image and depth map...")
    image_bgr = cv2.imread(image_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        print(f"Error: could not read image: {image_path}"); sys.exit(1)
    if depth_map is None:
        print(f"Error: could not read depth map: {depth_path}"); sys.exit(1)

    # Resize for performance
    orig_h, orig_w = image_bgr.shape[:2]
    if orig_w > DISPLAY_MAX_WIDTH:
        scale     = DISPLAY_MAX_WIDTH / orig_w
        image_bgr = cv2.resize(image_bgr, (DISPLAY_MAX_WIDTH, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
        depth_map = cv2.resize(depth_map, (DISPLAY_MAX_WIDTH, int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)
        print(f"Resized to {DISPLAY_MAX_WIDTH}x{int(orig_h * scale)} for display")

    # --- Step 2-3: segment ---
    layers, num_layers = build_layers(image_bgr, depth_map, num_layers, stem)

    # --- Step 4: viewer ---
    if not no_viewer:
        run_viewer(image_bgr, layers, num_layers)

    print("\nDone.")
    print(f"  layers/{stem}/depth_original.png   — raw depth map from model")
    print(f"  layers/{stem}/depth_histogram.png  — depth histogram with peaks and layer boundaries")
    print(f"  layers/{stem}/layer_00.png        — background (farthest)")
    print(f"  layers/{stem}/layer_{num_layers - 1:02d}.png        — foreground (nearest)")


if __name__ == "__main__":
    main()
