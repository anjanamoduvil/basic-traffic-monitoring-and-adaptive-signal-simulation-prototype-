# Traffic Vision AI: Premium Adaptive Intersection Management & Traffic Analytics

[![Project Demo](https://img.shields.io/badge/Demo-Watch%20Project%20Video-red?style=for-the-badge&logo=google-drive)](https://drive.google.com/file/d/1oiK3GH-wvTonvcvAOKf4cY-raWHhCw0c/view?usp=sharing)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Object%20Detection-YOLOv8-green?style=for-the-badge&logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)

A state-of-the-art, real-time traffic flow intelligence engine and adaptive intersection controller. Powered by **YOLOv8** and state-of-the-art multi-object tracking, this system calculates live vehicle queue density using speed-based stationary detection, manages signal cycles using an adaptive traffic engineering state-machine, and visualizes live intersection metrics in a beautifully responsive dark-mode glassmorphic dashboard.

---

## 📸 Dashboard Overview & Responsive Design

The front-end is engineered using modern CSS grid layouts, modern typography, glassmorphism, neon accent states, and micro-animations. It provides:
1. **Live Camera Feed**: Streaming annotated video frames with colored bounding boxes indicating movement states.
2. **Traffic Density Panel**: Displays **ROI Density**, **ROI Stuck (Stopped)**, and **Total Screen** metrics cleanly in a cache-resistant 2-column grid to prevent squeezed layouts.
3. **Adaptive Signal Controller UI**: Provides visual cues (flashing Red, Yellow, and Green lights) reflecting the state machine's decision process.
4. **Total Tracked Panel**: Category breakdown with live counts and unique cumulative counts of vehicles.
5. **Real-time Density Timeline**: Interactive line chart powered by **Chart.js** displaying queue accumulation against warning limits.

---

## ⚡ Core Technical Features & Architecture

### 1. Speed-Based Queue Stop Detection (Solving the Green Light Paradox)
Traditional traffic density models rely solely on **vehicle counts (volume)** inside a Region of Interest (ROI). This creates the **"Green Light Paradox"**: high-speed vehicles flowing smoothly through a green light are flagged as "congestion" because the volume remains high.

**Traffic Vision AI** resolves this by measuring **actual physical stoppage (velocity)**:
* **Centroid History Rolling Buffer**: Tracks the displacement of every vehicle's centroid `(cx, cy)` across a rolling window of the last **30 frames** (1.0 second of playback time).
* **Frame-Rate Independent Velocity Calculations**: Speed is calculated as displacement divided by the video frame playback delta ($\Delta t = \text{frames} / \text{FPS}$), making the calculations absolute and immune to server-side CPU processing latency.
* **YOLO Bounding Box Jitter Mitigation**: Bounding boxes for stationary vehicles often slide by 1–3 pixels from frame to frame due to model confidence fluctuations. A calibrated threshold of `< 18.0` pixels per video-second combined with a minimum tracking requirement of `5 frames` (`0.15` seconds) filters out this noise, marking vehicles as **`[STUCK]`** only when they are truly stopped or crawling, and **`[MOVING]`** otherwise.

### 2. High-Accuracy Multi-Object Tracking & Classification Correction
* **State-of-the-art Tracking**: Integrated with **YOLOv8** and its default **BoT-SORT** / **ByteTrack** multi-object tracking engines, maximizing ID persistence and preventing track switches.
* **Stand-Alone Pedestrian Class Correction**: Motorcyclists on highways are sometimes misclassified as individual pedestrians by detection models. To prevent traffic metric skewing, a custom off-road spatial heuristic is implemented:
  * Any `person` detected inside active traffic lanes or the ROI is automatically corrected to a **`motorcycle`** vehicle class.
  * Overlapping pedestrian bounding boxes (rider + bike overlap > 15%) are merged and discarded as duplicate noise.
  * Pedestrians are only kept if they are walking on off-road segments with high confidence.

### 3. Density-Driven Adaptive Signal State Machine
An intelligent logic controller that acts as a real-time signal controller:
* **Dynamic Green Extensions**: During a `GREEN` phase, the controller continuously monitors stuck vehicles inside the ROI. If the queue grows beyond a threshold, green light time extends dynamically:
  $$\text{Duration} = \min(\text{Max Green (20s)}, \text{Base Green (10s)} + (\text{Stuck Count} - \text{Threshold}) \times 2.0\text{s})$$
* **Early Green Triggers**: During a `RED` phase, if the queue build-up exceeds severe levels ($\text{Stuck Count} \geq \text{Threshold} + 3$), the remaining red light duration is immediately reduced to a 5.0-second safety minimum, flushing the congested lane early.

---

## 🛠️ Codebase Structure & Main Components

The codebase consists of two primary operational entry points maintaining absolute parity in coordinate mathematics and queue speed parameters:

### 1. 🖥️ Web Dashboard Server (`app.py`)
Provides the main web service utilizing **FastAPI** for high performance:
* **Async Lifespan Management**: Pre-loads the YOLO model asynchronously on startup to guarantee zero-latency request handling.
* **Multipart Video Streaming**: Streams custom JPEG-encoded frame buffers directly to the `<img>` tag in the browser.
* **REST API Endpoint (`/api/metrics`)**: Serves a global JSON state object containing live counts, vehicle breakdown categories, signal timer status, and queue states.

### 2. 💻 Offline Simulator & Visualizer (`traffic_simulation.py`)
Processes videos offline to export high-performance visual logs for analysis:
* **Hard Frame Limit**: Processes a set video segment (up to 400 frames) and exports a high-bitrate encoded visual simulation.
* **Event-Driven Screenshot Logging**: Detects peak congestion points or adaptive signal events (dynamic green/red phase changes) and auto-saves snapshot logs inside the `screenshots/` directory.

### 3. 🎨 Front-End Dashboard (`templates/index.html` & `static/app.js`)
* **Neon-Cyan Density Timeline**: A line chart visualizing vehicle queue history.
* **Zero-Delay Polling**: Fetches REST metric streams every `500ms`, dynamically updating CSS warning banners (`⚠️ ROI Stuck Queue`), badge values, category charts, and the active traffic light state.
* **Asset Cache Busting**: The assets in `index.html` utilize version-locking (`?v=12`) to prevent browsers from serving stale CSS grids or cached javascript logic.

---

## 📚 Libraries & Technologies Used

### Backend & Vision
* **`ultralytics` (YOLOv8)**: The core engine for high-speed object detection and BoT-SORT / ByteTrack vehicle tracking.
* **`opencv-python` (OpenCV)**: Handles high-performance frame capture, color conversions, video drawing operations (drawing bounding boxes, labels, and polygon lines), and encoding visual output.
* **`FastAPI` & `uvicorn`**: An asynchronous Python framework and lightning-fast ASGI server for low-latency metrics polling and server-sent video streaming.
* **`numpy`**: Powering point-in-polygon coordinates calculations (`cv2.pointPolygonTest`) and centroid distance math.

### Frontend
* **`Chart.js`**: Interactive rendering of the density timeline with custom linear gradients.
* **`Inter (Google Fonts)`**: Premium modern typography.
* **`HTML5 / Vanilla CSS Grid`**: Formats the layout responsive viewport scales and glassmorphic transparency states.

---

## 🚀 Installation & Getting Started

### 📋 Prerequisites
Ensure you have the required packages installed in your Python environment:
```bash
pip install opencv-python ultralytics fastapi uvicorn jinja2
```

### 🏃 Running the Live Web Dashboard
To start the live dashboard on your local machine:
1. Start the server:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to:
   ```text
   http://127.0.0.1:8000
   ```

### 📁 Running the Offline CLI Tool
To process the video offline and generate the annotated demonstration video:
1. Run the script:
   ```bash
   python traffic_simulation.py
   ```
2. Verify the output files:
   - Exported Video: `output_heavy_congestion_simulation.mp4`
   - Peak Event Logs: `screenshots/`
