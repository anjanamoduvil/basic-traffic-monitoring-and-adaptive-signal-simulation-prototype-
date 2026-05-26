# video of the project:[Watch Project Demo](https://drive.google.com/file/d/1oiK3GH-wvTonvcvAOKf4cY-raWHhCw0c/view?usp=sharing)

# Traffic Vision AI: Adaptive Intersection Management Prototype

A real-time traffic monitoring and adaptive traffic signal simulation system that uses **YOLOv8** for high-accuracy vehicle detection and advanced multi-object tracking, calculates live traffic queue density within a Region of Interest (ROI), and simulates intelligent, density-driven traffic signal timings.

---

## ✨ Features Implemented

### 1. Load Traffic Intersection Video
* The system utilizes **OpenCV's** standard `cv2.VideoCapture()` to load intersection videos (`sample_traffic.mp4` and `complex_traffic.mp4`).
* Includes robust error-handling for end-of-file video frames (handles looping seamlessly in the web dashboard).

### 2. Detect and Count Vehicles (YOLO & SORT/DeepSORT Tracking)
* Integrates **YOLOv8** (`yolov8n.pt` / `yolov8s.pt`) which represents the state-of-the-art in real-time object detection.
* Specifically filters classes for target vehicles: **Cars (2), Motorcycles (3), Buses (5), and Trucks (7)**.
* **Tracking Algorithm:** The codebase leverages YOLOv8's built-in tracking module (`model.track(persist=True)`), which implements **BoT-SORT** (default) or **ByteTrack**. 
  > **Note on SORT/DeepSORT:** BoT-SORT and ByteTrack are modern, state-of-the-art direct descendants of **SORT** (Simple Online and Realtime Tracking) and **DeepSORT**. They are highly optimized versions that achieve much higher tracking accuracy, fewer identity switches, and significantly faster inference times without needing massive external feature extraction networks.

### 3. Estimate Traffic Density
* Defines a custom polygon **Region of Interest (ROI)** representing the incoming queue lane of the intersection.
* For each tracked vehicle, the bottom-center coordinate is calculated.
* Uses OpenCV's point-in-polygon algorithm `cv2.pointPolygonTest()` to determine whether the vehicle is inside the active queue area.
* Estimating **traffic density** dynamically based on the exact count of vehicles residing inside this ROI.

### 4. Simulate Adaptive Traffic Signal Timing
* Features a custom state-machine **`TrafficSignalSimulator`** supporting **RED 🔴, YELLOW 🟡, and GREEN 🟢** phases.
* **Adaptive Green Extension:** When the light is GREEN, if traffic density inside the ROI exceeds the congestion threshold, the green light duration is dynamically extended:
  $$\text{Duration} = \min(\text{Max Green}, \text{Base Green} + (\text{Count} - \text{Threshold}) \times 2.0\text{s})$$
* **Early Green (Red Reduction):** When the light is RED, if the incoming queue builds up heavily (density exceeds threshold + 3), the red timer is immediately cut down to a minimum safety margin (5.0 seconds) to switch to Green early and clear the backup.

### 5. Display Congestion Warning
* When the vehicle count inside the ROI exceeds the designated threshold, the system immediately flags a **CONGESTION WARNING!**:
  * **Video Output:** Overlays a prominent red, flashing congestion warning card on top of the frame.
  * **Web Dashboard:** Shows a bright, neon-bordered warning banner saying `⚠️ High Congestion Detected` in real-time.
  * **Screenshots:** Automates event-driven logging by saving screenshots of congestion points directly to `screenshots/congestion_warning.jpg`.

### 6. Conceptual Comparison: Fixed-Time vs. Adaptive Signal Allocation
Below is the structural comparison of how adaptive signal allocation compares to traditional fixed-time controllers:

| Feature | Fixed-Time Signal Allocation | Adaptive Signal Allocation (YOLO + OpenCV) |
| :--- | :--- | :--- |
| **Logic Basis** | Pre-programmed cycle times based on historical average traffic volumes. | Real-time vehicle detection and queue length estimation. |
| **Congestion Response** | Static. If traffic builds up unexpectedly, the light remains red, causing long queues. | Dynamic. Extends green lights for heavy lanes or triggers early green to clear queues. |
| **Off-Peak Performance** | Inefficient. Drivers wait at red lights even when the opposing intersection is empty. | Highly efficient. Instantly switches phases when no vehicles are detected on active lanes. |
| **Hardware Required** | Basic industrial PLC (Programmable Logic Controller) or timer chips. | Camera/Sensor + Processing unit (Edge GPU, e.g., NVIDIA Jetson or Server). |
| **Setup & Maintenance Cost** | Low upfront, but high long-term cost due to manual retiming surveys. | Higher initial setup, but minimal manual adjustments; adapts to urban changes. |
| **Environmental Impact** | High idle emissions and fuel consumption due to unnecessary stop-and-go delays. | Lower vehicle idling times, leading to reduced carbon emissions and fuel saving. |
| **Emergency Vehicles** | Cannot react unless integrated with separate optical/siren override systems. | Can be trained to detect emergency vehicle classes (e.g. ambulance) and prioritize them. |

---

## 🛠️ Installation & Getting Started

### 📋 Prerequisites
Ensure you have the required packages installed in your Python environment:
```bash
pip install opencv-python ultralytics fastapi uvicorn jinja2
```

### 🚀 Running the Web Dashboard (Recommended)
This runs the high-fidelity real-time monitoring web app.
1. Start the server:
   ```bash
   python app.py
   ```
2. Navigate to your browser:
   ```text
   http://127.0.0.1:8000
   ```

### 💻 Running the Offline CLI Tool
This processes the video directly and outputs a fully annotated simulation video file.
1. Run the script:
   ```bash
   python traffic_simulation.py
   ```
2. Check the output files:
   - Annotated Video: `output_traffic_simulation.mp4`
   - Key screenshots: `screenshots/`
