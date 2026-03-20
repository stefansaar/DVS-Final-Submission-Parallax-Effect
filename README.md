# Parallax Effect — DVS Final Submission

Turn any 2D image into an interactive parallax scene driven by head tracking (webcam) or mouse movement.

## Setup

```bash
pip install flask flask-sock numpy opencv-python-headless scikit-learn Pillow gradio_client
```

## Run

```bash
cd depth-mapper
python web_viewer.py
```

Open http://localhost:5002 in your browser (Chrome recommended for webcam access).

## Usage

- **Head tracking**: Allow camera access — move your head to control the parallax. Move closer/further for zoom effect.
- **Mouse fallback**: If no camera, hover over the parallax canvas to control X/Y. Scroll wheel controls zoom.
- **Upload**: Click "Upload" or "Selfie" in the bottom bar to process your own image (requires internet for HuggingFace depth API).
- **Controls**: Switch between sample images with the dropdown. Toggle "Gap Fill" and adjust "Intensity" slider.
