import sys
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, DepthAnythingForDepthEstimation


def main():
    if len(sys.argv) != 2:
        print("Usage: python depth_map.py <image_filename>")
        sys.exit(1)

    filename = sys.argv[1]
    images_dir = "./0_source_images"
    depth_maps_dir = "./1_depth_maps"
    input_path = os.path.join(images_dir, filename)

    if not os.path.isdir(images_dir):
        print(f"Error: images/ directory not found at {images_dir}")
        sys.exit(1)

    if not os.path.isfile(input_path):
        print(f"Error: image file not found at {input_path}")
        sys.exit(1)

    os.makedirs(depth_maps_dir, exist_ok=True)

    stem = os.path.splitext(filename)[0]
    output_filename = f"depth_{stem}2.png"
    output_path = os.path.join(depth_maps_dir, output_filename)

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load and resize image (longest side = 518)
    image = Image.open(input_path).convert("RGB")
    w, h = image.size
    scale = 518 / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    image_resized = image.resize((new_w, new_h), Image.LANCZOS)
    print(f"Input:  {input_path} ({w}x{h})")
    print(f"Resized to: {new_w}x{new_h}")

    # Load model
    model_name = "depth-anything/Depth-Anything-V2-Large-hf"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = DepthAnythingForDepthEstimation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    # Inference
    inputs = processor(images=image_resized, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
    elapsed = time.time() - start

    # Post-process
    depth = outputs.predicted_depth  # shape: (1, H, W)
    depth = depth.unsqueeze(1).float()  # shape: (1, 1, H, W)
    depth = F.interpolate(depth, size=(new_h, new_w), mode="bicubic", align_corners=False)
    depth = depth.squeeze()  # shape: (H, W)

    depth_min = depth.min()
    depth_max = depth.max()
    depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)
    depth_uint8 = (depth_normalized * 255).clamp(0, 255).byte().cpu().numpy().astype(np.uint8)

    # Save
    depth_image = Image.fromarray(depth_uint8, mode="L")
    depth_image.save(output_path)

    print(f"Output: {output_path}")
    print(f"Inference time: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
