import gradio as gr
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# Load model and processor once at startup
MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
print(f"Loading model: {MODEL_ID}...")
image_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded.")


def estimate_depth(input_image):
    """
    Takes a PIL image, runs depth estimation, returns the depth map as a grayscale PIL image.
    """
    if input_image is None:
        raise gr.Error("Please upload an image.")

    # Resize so longest side is 518px (model's native resolution)
    w, h = input_image.size
    scale = 518 / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)
    resized = input_image.resize((new_w, new_h), Image.LANCZOS)

    # Preprocess
    inputs = image_processor(images=resized, return_tensors="pt")

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process: interpolate to resized image dimensions
    predicted_depth = outputs.predicted_depth.unsqueeze(1)
    predicted_depth = torch.nn.functional.interpolate(
        predicted_depth,
        size=(new_h, new_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    # Normalize to 0-255
    depth = predicted_depth.cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_image = Image.fromarray(depth.astype(np.uint8))

    return depth_image


# Build the Gradio interface
demo = gr.Interface(
    fn=estimate_depth,
    inputs=gr.Image(type="pil", label="Upload Image"),
    outputs=gr.Image(type="pil", label="Depth Map"),
    title="Depth Anything V2 — Depth Estimation",
    description="Upload an image to generate a depth map. Lighter areas are closer, darker areas are farther.",
    examples=[],
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
