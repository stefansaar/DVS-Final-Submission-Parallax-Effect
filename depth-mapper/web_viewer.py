"""
web_viewer.py

Browser-based parallax viewer with head tracking via WebSocket.
Reuses compositing pipeline from compositing.py and serves a MediaPipe
Face Mesh frontend that sends head position over WebSocket.

Usage:
    cd depth-mapper
    python web_viewer.py
    # Open http://localhost:5002
"""

import os
import sys
import json
import threading
import uuid

import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

# WebSocket support
try:
    from flask_sock import Sock
except ImportError:
    print("flask-sock is required: pip install flask-sock")
    sys.exit(1)

import glob as g

from compositing import (
    precompute_layers, composite,
    PARALLAX_STRENGTH,
)
from layer_segmentation import segment_layers, save_layers

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB upload limit
sock = Sock(app)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
JPEG_QUALITY = 70
ZOOM_STRENGTH = 0.15  # max scale factor per unit Z

state = {
    "float_layers": None,
    "num_layers": 0,
    "h": 0,
    "w": 0,
    "max_w": 640,              # max width, updated by client
    "max_h": 480,              # max height, updated by client
    "fill_gaps": False,
    "cases": [],              # list of stem names
    "current_stem": None,
    "raw_layers_cache": {},   # stem → list of RGBA uint8 arrays (full res)
    "lock": threading.Lock(),
    "pipeline_lock": threading.Lock(),
    "jobs": {},               # job_id → {status, progress, stem, error}
}


def find_layer_cases(layers_dir="./layers"):
    """Find all stems that have pre-processed layer PNGs."""
    cases = []
    for d in sorted(g.glob(os.path.join(layers_dir, "*"))):
        if not os.path.isdir(d):
            continue
        pngs = sorted(g.glob(os.path.join(d, "layer_*.png")))
        if pngs:
            cases.append(os.path.basename(d))
    return cases


def _load_raw_layers(stem, layers_dir="./layers"):
    """Read layer PNGs from disk and cache at full resolution."""
    if stem in state["raw_layers_cache"]:
        return state["raw_layers_cache"][stem]

    layer_dir = os.path.join(layers_dir, stem)
    png_paths = sorted(g.glob(os.path.join(layer_dir, "layer_*.png")))
    if not png_paths:
        return None

    layers = []
    for p in png_paths:
        bgra = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if bgra is None:
            return None
        rgba = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA)
        layers.append(rgba)

    state["raw_layers_cache"][stem] = layers
    return layers


def load_case(stem, max_w=None, max_h=None):
    """Load layers for stem, downscaled to fit within max_w x max_h."""
    raw = _load_raw_layers(stem)
    if raw is None:
        return False

    if max_w is not None:
        state["max_w"] = max_w
    if max_h is not None:
        state["max_h"] = max_h

    mw = state["max_w"]
    mh = state["max_h"]
    layers = raw
    h, w = layers[0].shape[:2]

    # Scale to fit within both constraints
    scale = min(mw / w, mh / h, 1.0)
    if scale < 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        layers = [cv2.resize(l, (new_w, new_h), interpolation=cv2.INTER_AREA)
                  for l in layers]
        h, w = new_h, new_w

    float_layers = precompute_layers(layers)

    state["float_layers"] = float_layers
    state["num_layers"] = len(float_layers)
    state["h"] = h
    state["w"] = w
    state["current_stem"] = stem
    return True


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("viewer.html")


@app.route("/api/cases")
def api_cases():
    return jsonify(state["cases"])


@app.route("/api/raw/<stem>")
def api_raw(stem):
    """Serve the original source image."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join("./images", stem + ext)
        if os.path.isfile(path):
            return send_from_directory("./images", stem + ext)
    return jsonify({"error": "not found"}), 404


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Accept an image upload, run the depth pipeline in background."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    safe = secure_filename(f.filename)
    stem = os.path.splitext(safe)[0].lower()
    save_path = os.path.join("./images", stem + ".jpg")
    f.save(save_path)

    job_id = str(uuid.uuid4())[:8]
    state["jobs"][job_id] = {
        "status": "queued",
        "progress": "Queued",
        "stem": None,
        "error": None,
    }

    t = threading.Thread(target=_run_pipeline, args=(job_id, stem), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route("/api/upload/<job_id>")
def api_upload_status(job_id):
    """Poll pipeline job status."""
    job = state["jobs"].get(job_id)
    if not job:
        return jsonify({"error": "Unknown job"}), 404
    return jsonify(job)


def _run_pipeline(job_id, stem):
    """Background thread: run full depth pipeline for an uploaded image."""
    job = state["jobs"][job_id]

    try:
        job["progress"] = "Waiting for pipeline lock..."
        with state["pipeline_lock"]:
            image_path = os.path.join("./images", stem + ".jpg")
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise RuntimeError(f"Failed to read {image_path}")

            # 1. Depth estimation via HuggingFace API
            job["progress"] = "Estimating depth (HuggingFace API)..."
            from gradio_client import Client, handle_file
            from shutil import copy2

            client = Client("saarstefan/depth-mapping-test")
            result = client.predict(handle_file(image_path), api_name="/estimate_depth")

            os.makedirs("./depth_maps", exist_ok=True)
            depth_path = os.path.join("./depth_maps", f"depth_{stem}_run1.png")
            copy2(result, depth_path)

            # 2. Load depth map
            depth_map = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

            # 3. Segment layers
            job["progress"] = "Segmenting layers..."
            layers = segment_layers(image_bgr, depth_map, num_layers=None)

            # 4. Save layers
            job["progress"] = "Saving layers..."
            layer_dir = os.path.join("./layers", stem)
            save_layers(layers, layer_dir)

            # 5. Update state
            if stem not in state["cases"]:
                state["cases"].append(stem)
            state["raw_layers_cache"].pop(stem, None)

            job["status"] = "done"
            job["progress"] = "Complete"
            job["stem"] = stem

    except Exception as e:
        job["status"] = "error"
        job["progress"] = "Failed"
        job["error"] = str(e)
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------
@sock.route("/ws")
def ws_handler(ws):
    last_x, last_y, last_z, last_intensity = None, None, None, None
    last_cached_jpg = None

    while True:
        try:
            raw = ws.receive(timeout=10)
        except Exception:
            break
        if raw is None:
            break

        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        msg_type = msg.get("type")

        if msg_type == "head":
            x = float(msg.get("x", 0))
            y = float(msg.get("y", 0))
            z = float(msg.get("z", 0))
            intensity = float(msg.get("intensity", 1.0))

            # Skip if head hasn't moved enough (< 0.5px of shift)
            threshold = 0.5 / PARALLAX_STRENGTH
            if (last_x is not None
                    and abs(x - last_x) < threshold
                    and abs(y - last_y) < threshold
                    and abs(z - last_z) < 0.005
                    and abs(intensity - last_intensity) < 0.01
                    and last_cached_jpg is not None):
                try:
                    ws.send(last_cached_jpg)
                except Exception:
                    break
                continue

            last_x, last_y, last_z, last_intensity = x, y, z, intensity

            with state["lock"]:
                fl = state["float_layers"]
                nl = state["num_layers"]
                h = state["h"]
                w = state["w"]
                fill = state["fill_gaps"]

            if fl is None:
                continue

            shifts = []
            for i in range(nl):
                depth_weight = i / max(nl - 1, 1)
                dx = -x * PARALLAX_STRENGTH * depth_weight * intensity
                dy = -y * PARALLAX_STRENGTH * depth_weight * intensity
                scale = 1.0 + z * ZOOM_STRENGTH * depth_weight * intensity
                shifts.append((dx, dy, scale))

            frame = composite(fl, shifts, h, w, fill_gaps=fill)

            _, buf = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            jpg_bytes = buf.tobytes()
            last_cached_jpg = jpg_bytes
            try:
                ws.send(jpg_bytes)
            except Exception:
                break

        elif msg_type == "toggle_fill":
            with state["lock"]:
                state["fill_gaps"] = not state["fill_gaps"]
                status = state["fill_gaps"]
            last_cached_jpg = None  # force re-render
            ws.send(json.dumps({"type": "fill_status", "fill": status}))

        elif msg_type == "switch":
            stem = msg.get("stem", "")
            if stem not in state["cases"]:
                ws.send(json.dumps({"type": "error", "msg": f"Unknown: {stem}"}))
                continue

            with state["lock"]:
                ok = load_case(stem)
            last_cached_jpg = None
            last_x = None
            if ok:
                ws.send(json.dumps({
                    "type": "switched",
                    "stem": stem,
                    "w": state["w"],
                    "h": state["h"],
                    "num_layers": state["num_layers"],
                }))
            else:
                ws.send(json.dumps({"type": "error", "msg": f"Failed to load {stem}"}))

        elif msg_type == "resize":
            new_mw = max(200, min(int(msg.get("max_w", 640)), 2560))
            new_mh = max(200, min(int(msg.get("max_h", 480)), 2560))
            current = state["current_stem"]
            if current and (new_mw != state["max_w"] or new_mh != state["max_h"]):
                with state["lock"]:
                    load_case(current, max_w=new_mw, max_h=new_mh)
                last_cached_jpg = None
                last_x = None
                ws.send(json.dumps({
                    "type": "resized",
                    "w": state["w"],
                    "h": state["h"],
                }))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cases = find_layer_cases()
    if not cases:
        print("No pre-processed layers found in ./layers/")
        print("Run from inside depth-mapper/")
        sys.exit(1)

    state["cases"] = cases
    print(f"Found {len(cases)} image(s): {cases}")

    # Load first case
    first = cases[0]
    print(f"Preloading '{first}'...")
    load_case(first)
    print(f"Ready: {state['w']}x{state['h']}, {state['num_layers']} layers")

    port = int(os.environ.get("PORT", 5002))
    print(f"\nOpen http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
