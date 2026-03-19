# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a single-file, no-build web application that uses MediaPipe Face Mesh to detect head position and orientation in real-time, driving a layered CSS parallax scene.

## Running the Project

No build step required. Open directly or serve via a local server (required for webcam access in some browsers):

```bash
python3 -m http.server 8000
# Visit: http://localhost:8000/head-tracking.html
```

The app requires webcam permissions and either `localhost` or HTTPS (WebRTC security restriction).

## Architecture

All application code lives in `head-tracking.html` as a single self-contained file with inline CSS and JavaScript. All dependencies (MediaPipe Face Mesh, Camera Utils) are loaded from CDN — there is no `package.json` or install step.

### Three core systems in `head-tracking.html`:

**1. Face Tracking** — MediaPipe Face Mesh detects 468 landmarks per frame. The app extracts:
- Head position (X/Y/size) from the landmark bounding box
- Yaw/pitch/roll from specific landmark indices: nose (1), forehead (10), chin (152), right eye (33), left eye (263)
- A canvas overlay renders the mesh visualization on top of the video feed

**2. Parallax Animation** — Ten PNG layers in `assets/` (named `01_Mist.png` through `10_Sky.png`) are stacked at varying depth values (0–80). Head position drives per-layer offsets with 0.18 smoothing factor, and a 3D CSS perspective transform is applied to the scene container at 60fps via `requestAnimationFrame`.

**3. UI Panel** — Displays head coordinates, yaw/pitch/roll angles, FPS, and an intensity slider (0–200%) that scales the parallax effect strength.

### Key functions:
- `start()` — async entry point, initializes webcam and MediaPipe
- `animateParallax()` — continuous `requestAnimationFrame` loop driving layer transforms
- The `onResults` callback receives each Face Mesh frame and updates tracked head state
