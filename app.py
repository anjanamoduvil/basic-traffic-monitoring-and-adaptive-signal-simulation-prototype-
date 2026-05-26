import cv2
import numpy as np
import time
import os

os.environ["YOLO_OFFLINE"] = "True"
os.environ["YOLO_VERBOSE"] = "False"
from ultralytics import YOLO

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager

MODEL_NAME = "yolov8s.pt"
VIDEO_PATH = "new_congested_traffic.mp4"

# Global variables
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print("Loading YOLO model...")
    model = YOLO(MODEL_NAME)
    print("YOLO model loaded.")
    yield

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

RED = "RED"
GREEN = "GREEN"
YELLOW = "YELLOW"

class TrafficSignalSimulator:
    def __init__(self):
        self.state = RED
        self.timer = 0.0
        self.base_red_time = 10.0
        self.base_green_time = 10.0
        self.yellow_time = 3.0
        
        self.max_green_time = 20.0
        self.min_red_time = 5.0
        
        self.current_duration = self.base_red_time
        self.congestion_threshold = 6

    def update(self, dt, vehicle_count):
        self.timer += dt
        
        if self.state == GREEN:
            adaptive_duration = self.base_green_time
            if vehicle_count > self.congestion_threshold:
                adaptive_duration = min(self.max_green_time, self.base_green_time + (vehicle_count - self.congestion_threshold) * 2.0)
            self.current_duration = adaptive_duration
            if self.timer >= self.current_duration:
                self.state = YELLOW
                self.timer = 0.0
                self.current_duration = self.yellow_time
                
        elif self.state == YELLOW:
            if self.timer >= self.current_duration:
                self.state = RED
                self.timer = 0.0
                self.current_duration = self.base_red_time
                
        elif self.state == RED:
            adaptive_duration = self.base_red_time
            if vehicle_count > self.congestion_threshold + 3:
                adaptive_duration = self.min_red_time
            self.current_duration = adaptive_duration
            if self.timer >= self.current_duration:
                self.state = GREEN
                self.timer = 0.0
                self.current_duration = self.base_green_time

simulator = TrafficSignalSimulator()

# Global metrics state
global_metrics = {
    "vehicles_in_roi": 0,
    "state": RED,
    "time_left": 10.0,
    "is_congested": False,
    "congestion_threshold": 6,
    "live_counts": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    "total_counts": {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
}

def generate_frames():
    global global_metrics
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video {VIDEO_PATH}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or np.isnan(fps) or fps == 0: fps = 30.0
    dt = 1.0 / fps

    roi_points = np.array([
        [int(width * 0.05), int(height * 0.95)],
        [int(width * 0.95), int(height * 0.95)],
        [int(width * 0.75), int(height * 0.45)],
        [int(width * 0.25), int(height * 0.45)]
    ], np.int32)

    class_names = {
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    unique_seen = {
        "car": set(),
        "motorcycle": set(),
        "bus": set(),
        "truck": set()
    }

    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Reset unique trackers on loop
            for k in unique_seen:
                unique_seen[k].clear()
            continue
            
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], conf=0.15, iou=0.5, verbose=False)
        vehicles_in_roi = 0
        live_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None] * len(boxes)
            
            for box, cls_idx, track_id in zip(boxes, clss, ids):
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                cls_idx = int(cls_idx)
                v_type = class_names.get(cls_idx, "car")
                
                # Update live count
                live_counts[v_type] += 1
                
                # Update unique count
                if track_id is not None:
                    unique_seen[v_type].add(int(track_id))
                
                # Highly accurate road-contact check (bottom-center of bounding box)
                cy_check = y2
                dist = cv2.pointPolygonTest(roi_points, (cx, cy_check), False)
                in_roi = (dist >= 0)
                
                if in_roi:
                    vehicles_in_roi += 1
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                    
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_text = f"{v_type.capitalize()} ID:{int(track_id)}" if track_id is not None else f"{v_type.capitalize()}"
                cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.circle(frame, (cx, cy_check), 4, color, -1)

        # We compute real dt for smooth simulation, or just use 1/fps
        # For a web stream it's often better to use real time elapsed if processing takes long
        current_time = time.time()
        real_dt = current_time - last_time
        last_time = current_time
        
        # update simulator
        simulator.update(real_dt, vehicles_in_roi)
        
        # Draw ROI
        cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=3)

        # Update metrics global state
        global_metrics["vehicles_in_roi"] = vehicles_in_roi
        global_metrics["state"] = simulator.state
        global_metrics["time_left"] = max(0.0, simulator.current_duration - simulator.timer)
        global_metrics["is_congested"] = vehicles_in_roi > simulator.congestion_threshold
        global_metrics["congestion_threshold"] = simulator.congestion_threshold
        global_metrics["live_counts"] = live_counts
        
        totals = {k: len(v) for k, v in unique_seen.items()}
        global_metrics["total_counts"] = totals
        global_metrics["total_vehicles"] = sum(totals.values())

        # Draw Information Panel natively on video to match sample
        panel_h = 220
        panel_w = 350
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        
        cv2.putText(frame, f"Traffic Density: {vehicles_in_roi} vehicles", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        if vehicles_in_roi > simulator.congestion_threshold:
            cv2.putText(frame, "CONGESTION WARNING!", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                        
        cv2.putText(frame, f"Signal: {simulator.state}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time left: {max(0, simulator.current_duration - simulator.timer):.1f}s", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        if simulator.state == GREEN and simulator.current_duration > simulator.base_green_time:
            cv2.putText(frame, "ADAPTIVE: GREEN EXTENDED", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if simulator.state == RED and simulator.current_duration < simulator.base_red_time:
            cv2.putText(frame, "ADAPTIVE: EARLY GREEN PENDING", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/metrics")
def get_metrics():
    return JSONResponse(content=global_metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
