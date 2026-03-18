"""
layer_segmentation.py

Takes a depth map and original image, produces N depth-ordered RGBA layers.

Depth convention (Depth Anything V2):
    bright pixel (high value) = near = foreground
    dark pixel  (low value)   = far  = background

Layer ordering in returned list:
    index 0            = farthest background  (lowest parallax shift)
    index num_layers-1 = nearest foreground   (highest parallax shift)

Usage (CLI):
    python layer_segmentation.py <image_path> <depth_map_path> [num_layers]

    Saves each RGBA layer as a PNG in ./layers/<stem>/layer_00.png ... layer_N.png
    Also saves a visualisation of the depth boundaries as depth_bands.png.

Usage (library):
    from layer_segmentation import segment_layers

    layers = segment_layers(image_bgr, depth_map_gray)
    # layers[0] is the background RGBA array, layers[-1] is the foreground RGBA array
"""

import os
import sys
import numpy as np
import cv2
from sklearn.cluster import KMeans
from depth_processing import clean_depth_map


# ---------------------------------------------------------------------------
# Core segmentation
# ---------------------------------------------------------------------------

def depth_to_label_map(depth_map: np.ndarray, num_layers: int) -> np.ndarray:
    """
    Cluster a depth map into N layers using k-means, returning a label map.

    Every pixel is assigned exactly one integer label 0..N-1.
    Label 0 = farthest background, label N-1 = nearest foreground.

    Returns:
        (H, W) int32 array with values in 0..num_layers-1.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2D, got shape {depth_map.shape}")

    h, w = depth_map.shape
    pixels = depth_map.astype(np.float32).reshape(-1, 1)

    kmeans = KMeans(n_clusters=num_layers, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_.flatten()

    sorted_ids = np.argsort(centers)
    rank = np.empty_like(sorted_ids)
    rank[sorted_ids] = np.arange(num_layers)

    return rank[labels].reshape(h, w).astype(np.int32)


def smooth_label_map(label_map: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """
    Smooth a label map by assigning each pixel to the dominant label in its neighbourhood.

    Works like morphological closing but operates on the full label map, so the result
    is still complete (every pixel has a label) and non-overlapping (each pixel has
    exactly one label).

    Uses a blur-voting scheme: for each label, blur its binary presence map, then
    assign each pixel to whichever label scored highest in its neighbourhood.

    Args:
        label_map:   (H, W) int array with values 0..k-1.
        kernel_size: Neighbourhood size. Larger = fills bigger gaps.

    Returns:
        Smoothed label map, same shape and dtype.
    """
    h, w = label_map.shape
    num_labels = int(label_map.max()) + 1

    votes = np.zeros((h, w, num_labels), dtype=np.float32)
    for label_id in range(num_labels):
        binary = (label_map == label_id).astype(np.float32)
        votes[:, :, label_id] = cv2.blur(binary, (kernel_size, kernel_size))

    return votes.argmax(axis=2).astype(label_map.dtype)


def clean_label_map(
    label_map: np.ndarray,
    smooth_kernel: int = 7,
    min_area_fraction: float = 0.001,
) -> np.ndarray:
    """
    Clean a label map while maintaining complete, non-overlapping coverage.

    Two steps:
        1. Smooth: assign each pixel to the dominant label in its neighbourhood,
           filling minor gaps without creating any overlap.
        2. Reassign small isolated components to their dominant neighbour label,
           so scattered specks don't form their own isolated layer regions.

    Args:
        label_map:         (H, W) int array with values 0..k-1.
        smooth_kernel:     Neighbourhood size for the smoothing step.
        min_area_fraction: Components smaller than this fraction of the image
                           are reassigned to their dominant neighbour.

    Returns:
        Cleaned label map, same shape and dtype, still complete and non-overlapping.
    """
    num_labels = int(label_map.max()) + 1
    h, w = label_map.shape
    min_area = max(1, int(h * w * min_area_fraction))

    result = smooth_label_map(label_map, smooth_kernel)

    for label_id in range(num_labels):
        binary = (result == label_id).astype(np.uint8) * 255
        n_comps, comp_map, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        for comp_id in range(1, n_comps):
            if stats[comp_id, cv2.CC_STAT_AREA] < min_area:
                comp_pixels = comp_map == comp_id
                dilated = cv2.dilate(
                    comp_pixels.astype(np.uint8),
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                )
                ring = dilated.astype(bool) & ~comp_pixels
                neighbour_labels = result[ring]
                if len(neighbour_labels) > 0:
                    counts = np.bincount(neighbour_labels.astype(np.int32), minlength=num_labels)
                    counts[label_id] = 0
                    result[comp_pixels] = counts.argmax()

    return result


def label_map_to_masks(label_map: np.ndarray, num_labels: int) -> list[np.ndarray]:
    """Convert an integer label map to a list of binary masks (one per label)."""
    return [
        np.where(label_map == i, np.uint8(255), np.uint8(0))
        for i in range(num_labels)
    ]


def normalise_depth_range(depth_map: np.ndarray) -> np.ndarray:
    """
    Stretch the occupied depth range to fill 0-255.

    Clips to the 1st-99th percentile so extreme outlier pixels don't compress
    the useful depth range into a narrow band, wasting k-means clusters.
    """
    lo = float(np.percentile(depth_map, 1))
    hi = float(np.percentile(depth_map, 99))
    if hi <= lo:
        return depth_map
    clipped = np.clip(depth_map.astype(np.float32), lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def estimate_num_layers(depth_map: np.ndarray, min_k: int = 3, max_k: int = 7) -> int:
    """
    Estimate the natural number of depth layers from the image's depth histogram.

    Builds a histogram of depth values, smooths it heavily, and counts dominant peaks.
    Each peak corresponds to a cluster of pixels at a similar depth.
    Result is clamped to [min_k, max_k].
    """
    lo = float(np.percentile(depth_map, 1))
    hi = float(np.percentile(depth_map, 99))

    hist, _ = np.histogram(depth_map, bins=64, range=(lo, hi))
    hist = hist.astype(np.float32)

    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32)
    kernel /= kernel.sum()
    smoothed = hist.copy()
    for _ in range(3):
        smoothed = np.convolve(smoothed, kernel, mode='same')

    threshold = 0.10 * smoothed.max()
    peaks = [
        i for i in range(1, len(smoothed) - 1)
        if smoothed[i] > smoothed[i - 1]
        and smoothed[i] > smoothed[i + 1]
        and smoothed[i] > threshold
    ]

    k = len(peaks) if peaks else min_k
    return max(min_k, min(max_k, k))


def masks_to_rgba_layers(
    image_bgr: np.ndarray,
    masks: list[np.ndarray],
) -> list[np.ndarray]:
    """
    Extract an RGBA layer for each binary mask with hard edges.

    Pixels inside the mask are fully opaque (255), outside are fully transparent (0).
    """
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"image_bgr must be (H, W, 3), got {image_bgr.shape}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    layers = []

    for mask in masks:
        rgba = np.zeros((*image_rgb.shape[:2], 4), dtype=np.uint8)
        visible = mask > 0
        rgba[visible, :3] = image_rgb[visible]
        rgba[:, :, 3] = mask
        layers.append(rgba)

    return layers


def segment_layers(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    num_layers: int | None = None,
) -> list[np.ndarray]:
    """
    Full pipeline: depth map + image → list of RGBA layers ordered back-to-front.

    Depth map should already be cleaned (bilateral filter + edge sharpening) before
    calling this. Call clean_depth_map() from depth_processing.py first.

    Args:
        image_bgr:  BGR uint8 array (H, W, 3).
        depth_map:  Grayscale uint8 or float array (H, W).
                    If float, values are normalised to [0, 255] automatically.
        num_layers: How many depth layers to produce.
                    None (default) = auto-detect from the depth histogram.

    Returns:
        List of N RGBA uint8 arrays (H, W, 4).
        layers[0]   = farthest background (moves least in parallax)
        layers[-1]  = nearest foreground  (moves most  in parallax)
    """
    # Normalise depth to uint8 if needed
    if depth_map.dtype != np.uint8:
        d = depth_map.astype(np.float32)
        d_min, d_max = d.min(), d.max()
        if d_max > d_min:
            d = (d - d_min) / (d_max - d_min) * 255.0
        depth_u8 = d.clip(0, 255).astype(np.uint8)
    else:
        depth_u8 = depth_map

    # Resize depth to match image if necessary
    h, w = image_bgr.shape[:2]
    if depth_u8.shape[:2] != (h, w):
        depth_u8 = cv2.resize(depth_u8, (w, h), interpolation=cv2.INTER_LINEAR)

    # Stretch occupied depth range so k-means clusters aren't wasted on empty bands
    depth_u8 = normalise_depth_range(depth_u8)

    if num_layers is None:
        num_layers = estimate_num_layers(depth_u8)
        print(f"  Auto-detected {num_layers} depth layers from histogram.")

    print(f"  Segmenting with k-means (k={num_layers})...")
    label_map = depth_to_label_map(depth_u8, num_layers)

    print("  Smoothing and reassigning small fragments...")
    label_map = clean_label_map(label_map)

    masks = label_map_to_masks(label_map, num_layers)
    return masks_to_rgba_layers(image_bgr, masks)


# ---------------------------------------------------------------------------
# Debug / visualisation helpers
# ---------------------------------------------------------------------------

def save_layers(layers: list[np.ndarray], out_dir: str) -> None:
    """Save each RGBA layer as a PNG file."""
    os.makedirs(out_dir, exist_ok=True)
    for i, layer in enumerate(layers):
        path = os.path.join(out_dir, f"layer_{i:02d}.png")
        try:
            from PIL import Image
            Image.fromarray(layer, "RGBA").save(path)
        except ImportError:
            bgra = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA)
            cv2.imwrite(path, bgra)
        print(f"  Saved {path}")


def save_depth_coloured(depth_map: np.ndarray, out_path: str) -> None:
    """Save the depth map as a false-colour image."""
    depth_u8 = depth_map if depth_map.dtype == np.uint8 else (
        ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255)
        .clip(0, 255).astype(np.uint8)
    )
    coloured = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    cv2.imwrite(out_path, coloured)
    print(f"  Saved coloured depth map: {out_path}")


def save_depth_bands(depth_map: np.ndarray, num_layers: int, out_path: str) -> None:
    """Save a false-colour visualisation with depth band boundary lines overlaid."""
    depth_u8 = depth_map if depth_map.dtype == np.uint8 else (
        ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8) * 255)
        .clip(0, 255).astype(np.uint8)
    )
    coloured = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    h, w = coloured.shape[:2]
    thresholds = np.linspace(0, 255, num_layers + 1)[1:-1]
    for t in thresholds:
        x = int(t / 255 * (w - 1))
        cv2.line(coloured, (x, 0), (x, h - 1), (255, 255, 255), 1)

    cv2.imwrite(out_path, coloured)
    print(f"  Saved depth bands visualisation: {out_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python layer_segmentation.py <image_path> <depth_map_path> [num_layers]")
        sys.exit(1)

    image_path = sys.argv[1]
    depth_path = sys.argv[2]
    num_layers = int(sys.argv[3]) if len(sys.argv) >= 4 else None

    if not os.path.isfile(image_path):
        print(f"Error: image not found: {image_path}"); sys.exit(1)
    if not os.path.isfile(depth_path):
        print(f"Error: depth map not found: {depth_path}"); sys.exit(1)

    image_bgr = cv2.imread(image_path)
    depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        print("Error: could not read image."); sys.exit(1)
    if depth_map is None:
        print("Error: could not read depth map."); sys.exit(1)

    print(f"Image:     {image_bgr.shape[1]}x{image_bgr.shape[0]}")
    print(f"Depth map: {depth_map.shape[1]}x{depth_map.shape[0]}")

    stem    = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join("./layers", stem)

    print("Cleaning depth map...")
    cleaned = clean_depth_map(depth_map, image_bgr)

    layers = segment_layers(image_bgr, cleaned, num_layers)
    n = len(layers)

    print(f"\nSaving to {out_dir}/")
    save_depth_coloured(depth_map, os.path.join(out_dir, "depth_original.png"))
    save_depth_coloured(cleaned,   os.path.join(out_dir, "depth_cleaned.png"))
    save_depth_bands(normalise_depth_range(cleaned), n, os.path.join(out_dir, "depth_bands.png"))
    save_layers(layers, out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
