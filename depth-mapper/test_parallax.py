"""
test_parallax.py

Parallax viewer — combines the RGBA layers produced by layer_segmentation.py
into a real-time interactive display.

TODO (Ken):
    T1.3 — Mask dilation: expand each layer mask outward by K pixels so that
            when layers shift apart there is an overlap buffer, not a gap.
            Use cv2.dilate with MORPH_ELLIPSE kernel.

    T1.4 — Nearest-neighbour pixel fill: fill the newly dilated region (which
            has no colour data) by copying from the nearest pixel that does.
            Use scipy.ndimage.distance_transform_edt for efficiency.

    T1.6 — Replace the stub compositor below with a proper implementation that
            uses the dilated + filled layers from T1.3/T1.4. The alpha blending
            formula is the same; the difference is that each layer's mask has
            already been expanded so gaps don't appear when layers shift apart.

Usage:
    python test_parallax.py <image_path> <depth_map_path> [num_layers]

Example:
    python test_parallax.py images/josh.jpg depth_maps/depth_josh_run1.png
"""

import sys
import cv2
import numpy as np
from layer_segmentation import segment_layers

PARALLAX_STRENGTH = 20   # max pixel shift for the nearest layer
DISPLAY_MAX_WIDTH  = 900  # resize window to this width for performance


# ---------------------------------------------------------------------------
# Stub compositor (no dilation / no fill — Ken will replace this with T1.6)
# ---------------------------------------------------------------------------

def precompute_layers(layers: list) -> list:
    """Convert RGBA uint8 layers to float32 RGB + float32 alpha once."""
    out = []
    for layer in layers:
        rgb   = layer[:, :, :3].astype(np.float32)
        alpha = layer[:, :, 3:4].astype(np.float32) / 255.0
        out.append((rgb, alpha))
    return out


def composite(float_layers: list, shifts: list, h: int, w: int) -> np.ndarray:
    """
    Shift each layer and alpha-blend back-to-front. Returns BGR uint8.

    Stub implementation — no dilation or fill, so gaps will appear at edges
    when layers shift. Ken's T1.3/T1.4/T1.6 will fix this.
    """
    canvas = np.zeros((h, w, 3), dtype=np.float32)

    for (rgb, alpha), (dx, dy) in zip(float_layers, shifts):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted_rgb   = cv2.warpAffine(rgb,   M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        shifted_alpha = cv2.warpAffine(alpha, M, (w, h), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if shifted_alpha.ndim == 2:
            shifted_alpha = shifted_alpha[:, :, np.newaxis]

        canvas = canvas * (1.0 - shifted_alpha) + shifted_rgb * shifted_alpha

    bgr = canvas.clip(0, 255).astype(np.uint8)[:, :, ::-1]
    return bgr


# ---------------------------------------------------------------------------
# Viewer
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python test_parallax.py <image_path> <depth_map_path> [num_layers]")
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

    orig_h, orig_w = image_bgr.shape[:2]
    if orig_w > DISPLAY_MAX_WIDTH:
        scale = DISPLAY_MAX_WIDTH / orig_w
        disp_w = DISPLAY_MAX_WIDTH
        disp_h = int(orig_h * scale)
        image_bgr = cv2.resize(image_bgr, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        depth_map = cv2.resize(depth_map, (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)
        print(f"Resized to {disp_w}x{disp_h} for display")

    h, w = image_bgr.shape[:2]

    print(f"Segmenting into {num_layers} layers...")
    layers = segment_layers(image_bgr, depth_map, num_layers)
    float_layers = precompute_layers(layers)
    print("Done. Move your mouse over the window. Press Q to quit.")

    mouse = {"x": w // 2, "y": h // 2}

    def on_mouse(_event, x, y, _flags, _param):
        mouse["x"] = x
        mouse["y"] = y

    cv2.namedWindow("Parallax Test")
    cv2.setMouseCallback("Parallax Test", on_mouse)

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
        cv2.imshow("Parallax Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
