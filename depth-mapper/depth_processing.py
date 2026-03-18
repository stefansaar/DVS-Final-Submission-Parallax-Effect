"""
depth_processing.py

Post-processing functions to clean up a depth map before layer segmentation.

Two steps:
    1. bilateral_filter    — smooth noise within objects, preserve edges
    2. edge_guided_sharpen — snap depth boundaries to real object edges from RGB

Main function to call:
    clean_depth_map(depth_map, image_bgr) -> cleaned depth map

Usage (CLI, saves a side-by-side comparison):
    python depth_processing.py <image_path> <depth_map_path>
"""

import os
import sys
import cv2
import numpy as np


def bilateral_filter(
    depth_map: np.ndarray,
    d: int = 9,
    sigma_color: float = 50,
    sigma_space: float = 75,
) -> np.ndarray:
    """
    Smooth noise within depth regions while preserving sharp edges between them.

    Unlike Gaussian blur, bilateral filter only averages pixels that are both
    spatially close AND have similar depth values. Pixels across a depth boundary
    are not averaged together, so object edges stay sharp.

    Args:
        depth_map:   Grayscale uint8 array (H, W).
        d:           Neighbourhood diameter in pixels.
        sigma_color: How different two depth values can be before the filter
                     stops averaging them. Higher = more smoothing across edges.
        sigma_space: Spatial spread (like sigma in a Gaussian).

    Returns:
        Filtered depth map, same shape and dtype as input.
    """
    return cv2.bilateralFilter(depth_map, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def edge_guided_sharpen(
    depth_map: np.ndarray,
    image_bgr: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    edge_dilate: int = 2,
    sharpen_strength: float = 3.0,
) -> np.ndarray:
    """
    Sharpen depth boundaries by snapping them to real object edges from the RGB image.

    The depth model produces soft transitions at object boundaries. This function
    uses Canny edge detection on the original photo to find where real edges are,
    then applies unsharp masking specifically at those locations to make the depth
    jump more abruptly — giving cleaner layer cuts.

    Args:
        depth_map:        Grayscale uint8 array (H, W).
        image_bgr:        Original BGR image (H, W, 3).
        canny_low:        Lower threshold for Canny hysteresis.
        canny_high:       Upper threshold for Canny hysteresis.
        edge_dilate:      How many pixels to dilate detected edges outward.
                          Creates a narrow zone around each edge where sharpening applies.
        sharpen_strength: How aggressively to push depth values apart at edges.
                          1.0 = subtle, 2.0-3.0 = noticeable, >4 = may clip.

    Returns:
        Sharpened depth map, same shape and dtype as input.
    """
    # Detect edges in the RGB image
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Dilate edges to create a narrow boundary zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_mask = cv2.dilate(edges, kernel, iterations=edge_dilate)

    # Unsharp mask: amplify high-frequency depth detail at edge locations
    # sharpened = original + strength * (original - blurred)
    depth_f = depth_map.astype(np.float32)
    blurred = cv2.GaussianBlur(depth_f, (0, 0), sigmaX=3)
    sharpened = depth_f + sharpen_strength * (depth_f - blurred)
    sharpened = sharpened.clip(0, 255).astype(np.uint8)

    # Only apply sharpening where edges were detected
    result = depth_map.copy()
    result[edge_mask > 0] = sharpened[edge_mask > 0]

    return result


def clean_depth_map(
    depth_map: np.ndarray,
    image_bgr: np.ndarray,
) -> np.ndarray:
    """
    Full depth map cleanup pipeline: bilateral filter then edge-guided sharpening.

    Args:
        depth_map:  Grayscale uint8 array (H, W).
        image_bgr:  Original BGR image (H, W, 3), used for edge detection.

    Returns:
        Cleaned depth map, same shape and dtype as input.
    """
    # Ensure image matches depth map size before edge detection
    dh, dw = depth_map.shape[:2]
    ih, iw = image_bgr.shape[:2]
    if (ih, iw) != (dh, dw):
        image_bgr = cv2.resize(image_bgr, (dw, dh), interpolation=cv2.INTER_AREA)

    depth = bilateral_filter(depth_map)
    depth = edge_guided_sharpen(depth, image_bgr)
    return depth


# ---------------------------------------------------------------------------
# CLI — saves a before/after comparison image
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python depth_processing.py <image_path> <depth_map_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    depth_path = sys.argv[2]

    image_bgr = cv2.imread(image_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        print(f"Error: could not read image: {image_path}"); sys.exit(1)
    if depth_map is None:
        print(f"Error: could not read depth map: {depth_path}"); sys.exit(1)

    # Resize depth to match image if needed
    h, w = image_bgr.shape[:2]
    if depth_map.shape[:2] != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    print("Applying bilateral filter...")
    after_bilateral = bilateral_filter(depth_map)

    print("Applying edge-guided sharpening...")
    after_sharpen = edge_guided_sharpen(after_bilateral, image_bgr)

    # Save a side-by-side: original | after bilateral | after sharpening
    # Convert all to colour for easier visual comparison
    col_original  = cv2.applyColorMap(depth_map,       cv2.COLORMAP_TURBO)
    col_bilateral = cv2.applyColorMap(after_bilateral,  cv2.COLORMAP_TURBO)
    col_final     = cv2.applyColorMap(after_sharpen,    cv2.COLORMAP_TURBO)

    def label(img, text):
        out = img.copy()
        cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return out

    comparison = np.hstack([
        label(col_original,  "original"),
        label(col_bilateral, "bilateral"),
        label(col_final,     "edge sharpen"),
    ])

    stem = os.path.splitext(os.path.basename(depth_path))[0]
    out_dir = os.path.join("./depth_processing_output")
    os.makedirs(out_dir, exist_ok=True)

    comparison_path = os.path.join(out_dir, f"{stem}_comparison.png")
    cleaned_path    = os.path.join(out_dir, f"{stem}_cleaned.png")

    cv2.imwrite(comparison_path, comparison)
    cv2.imwrite(cleaned_path, after_sharpen)

    print(f"Saved comparison: {comparison_path}")
    print(f"Saved cleaned:    {cleaned_path}")


if __name__ == "__main__":
    main()
