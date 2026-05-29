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
VIDEO_PATH = "C:/Users/VICTUS/NIT/traffic_prototype/new_congested_traffic.mp4"

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

# Coordinated Signal States
LANE1_GREEN = "LANE1_GREEN"
LANE1_YELLOW = "LANE1_YELLOW"
LANE2_GREEN = "LANE2_GREEN"
LANE2_YELLOW = "LANE2_YELLOW"

class CoordinatedSignalSimulator:
    def __init__(self):
        # Adaptive system state
        self.state = LANE1_GREEN
        self.timer = 0.0
        self.base_green_time = 10.0
        self.yellow_time = 3.0
        self.max_green_time = 20.0
        self.alpha = 2.0
        
        self.current_duration = self.base_green_time
        self.congestion_threshold = 6
        
        # Parallel Fixed-Time system state
        self.fixed_state = LANE1_GREEN
        self.fixed_timer = 0.0
        self.fixed_green_duration = 15.0
        self.fixed_yellow_duration = 3.0
        
        # Performance Metrics Accumulators
        self.adaptive_total_wait = 0.0
        self.fixed_total_wait = 0.0
        self.vehicles_cleared_adaptive = 0
        self.vehicles_cleared_fixed = 0
        
        # Decision Log Feed (keeps latest 5 logs)
        self.decision_logs = ["Coordinated dual-lane swarm control active."]

    def log_decision(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.decision_logs.append(log_entry)
        if len(self.decision_logs) > 5:
            self.decision_logs.pop(0)

    def update(self, dt, lane1_stuck, lane2_stuck):
        # 1. Update Adaptive State Machine
        self.timer += dt
        
        if self.state == LANE1_GREEN:
            active_stuck = lane1_stuck
            waiting_stuck = lane2_stuck
            
            # Dynamic Green Allocation Formula
            adaptive_duration = min(self.max_green_time, self.base_green_time + self.alpha * active_stuck)
            self.current_duration = adaptive_duration
            
            # Swarm Optimization: Early switch if waiting lane is highly congested and active lane base green is done
            if waiting_stuck >= self.congestion_threshold and self.timer >= self.base_green_time and active_stuck < waiting_stuck:
                self.log_decision(f"Swarm Priority: Switching to Lane 2 early due to heavy queue ({waiting_stuck} vehicles).")
                self.state = LANE1_YELLOW
                self.timer = 0.0
                self.current_duration = self.yellow_time
            elif self.timer >= self.current_duration:
                self.log_decision(f"Adaptive timer expired for Lane 1 ({self.current_duration:.1f}s). Switching to Lane 2.")
                self.state = LANE1_YELLOW
                self.timer = 0.0
                self.current_duration = self.yellow_time
                
        elif self.state == LANE1_YELLOW:
            if self.timer >= self.current_duration:
                self.state = LANE2_GREEN
                self.timer = 0.0
                self.current_duration = self.base_green_time
                
        elif self.state == LANE2_GREEN:
            active_stuck = lane2_stuck
            waiting_stuck = lane1_stuck
            
            # Dynamic Green Allocation Formula
            adaptive_duration = min(self.max_green_time, self.base_green_time + self.alpha * active_stuck)
            self.current_duration = adaptive_duration
            
            # Swarm Optimization: Early switch
            if waiting_stuck >= self.congestion_threshold and self.timer >= self.base_green_time and active_stuck < waiting_stuck:
                self.log_decision(f"Swarm Priority: Switching to Lane 1 early due to heavy queue ({waiting_stuck} vehicles).")
                self.state = LANE2_YELLOW
                self.timer = 0.0
                self.current_duration = self.yellow_time
            elif self.timer >= self.current_duration:
                self.log_decision(f"Adaptive timer expired for Lane 2 ({self.current_duration:.1f}s). Switching to Lane 1.")
                self.state = LANE2_YELLOW
                self.timer = 0.0
                self.current_duration = self.yellow_time
                
        elif self.state == LANE2_YELLOW:
            if self.timer >= self.current_duration:
                self.state = LANE1_GREEN
                self.timer = 0.0
                self.current_duration = self.base_green_time

        # 2. Update Fixed-Time State Machine (Shadow simulation)
        self.fixed_timer += dt
        if self.fixed_state == LANE1_GREEN:
            if self.fixed_timer >= self.fixed_green_duration:
                self.fixed_state = LANE1_YELLOW
                self.fixed_timer = 0.0
        elif self.fixed_state == LANE1_YELLOW:
            if self.fixed_timer >= self.fixed_yellow_duration:
                self.fixed_state = LANE2_GREEN
                self.fixed_timer = 0.0
        elif self.fixed_state == LANE2_GREEN:
            if self.fixed_timer >= self.fixed_green_duration:
                self.fixed_state = LANE2_YELLOW
                self.fixed_timer = 0.0
        elif self.fixed_state == LANE2_YELLOW:
            if self.fixed_timer >= self.fixed_yellow_duration:
                self.fixed_state = LANE1_GREEN
                self.fixed_timer = 0.0

        # 3. Accumulate Waiting Times for Comparison
        if self.state in [LANE2_GREEN, LANE2_YELLOW]:
            self.adaptive_total_wait += lane1_stuck * dt
        if self.state in [LANE1_GREEN, LANE1_YELLOW]:
            self.adaptive_total_wait += lane2_stuck * dt
            
        if self.fixed_state in [LANE2_GREEN, LANE2_YELLOW]:
            self.fixed_total_wait += lane1_stuck * dt
        if self.fixed_state in [LANE1_GREEN, LANE1_YELLOW]:
            self.fixed_total_wait += lane2_stuck * dt

simulator = CoordinatedSignalSimulator()

# Expanded Coordinated Global Metrics State
global_metrics = {
    "lane1_density": 0,
    "lane2_density": 0,
    "lane1_stuck": 0,
    "lane2_stuck": 0,
    "lane1_congestion": "LOW",
    "lane2_congestion": "LOW",
    "total_live_vehicles": 0,
    "state": LANE1_GREEN,
    "fixed_state": LANE1_GREEN,
    "time_left": 10.0,
    "live_counts": {"person": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    "total_counts": {"person": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0},
    "total_vehicles": 0,
    "adaptive_total_wait": 0.0,
    "fixed_total_wait": 0.0,
    "efficiency_gain": 0.0,
    "vehicles_cleared_adaptive": 0,
    "vehicles_cleared_fixed": 0,
    "decision_logs": []
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

    # Split road down the middle into two lanes
    roi_points_lane1 = np.array([
        [int(width * 0.02), int(height * 0.95)],
        [int(width * 0.50), int(height * 0.95)],
        [int(width * 0.50), int(height * 0.45)],
        [int(width * 0.15), int(height * 0.45)]
    ], np.int32)

    roi_points_lane2 = np.array([
        [int(width * 0.50), int(height * 0.95)],
        [int(width * 0.98), int(height * 0.95)],
        [int(width * 0.85), int(height * 0.45)],
        [int(width * 0.50), int(height * 0.45)]
    ], np.int32)

    class_names = {
        0: "person",
        1: "motorcycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck"
    }

    unique_seen = {
        "person": set(),
        "car": set(),
        "motorcycle": set(),
        "bus": set(),
        "truck": set()
    }

    centroid_history = {}
    frame_count = 0
    last_time = time.time()

    # Trackers for cleared vehicles
    lane1_prev_ids = set()
    lane2_prev_ids = set()

    cached_boxes = None
    cached_clss = None
    cached_confs = None
    cached_ids = None

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            # Reset unique trackers and centroid history on loop
            for k in unique_seen:
                unique_seen[k].clear()
            centroid_history.clear()
            frame_count = 0
            lane1_prev_ids.clear()
            lane2_prev_ids.clear()
            cached_boxes = None
            continue
            
        frame_count += 1
            
        # Run YOLO tracking every 2nd frame to double the processing speed
        if frame_count % 2 == 1 or cached_boxes is None:
            results = model.track(frame, persist=True, classes=[0, 1, 2, 3, 5, 7], conf=0.10, iou=0.5, verbose=False)
            if results[0].boxes is not None:
                cached_boxes = results[0].boxes.xyxy.cpu().numpy()
                cached_clss = results[0].boxes.cls.cpu().numpy()
                cached_confs = results[0].boxes.conf.cpu().numpy()
                cached_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None] * len(cached_boxes)
            else:
                cached_boxes = []
                cached_clss = []
                cached_confs = []
                cached_ids = []
                
        boxes = cached_boxes
        clss = cached_clss
        confs = cached_confs
        ids = cached_ids
        
        persons_detected = []
        vehicles_detected = []
        
        for box, cls_idx, conf, track_id in zip(boxes, clss, confs, ids):
            x1, y1, x2, y2 = map(int, box)
            cls_idx = int(cls_idx)
            v_type = class_names.get(cls_idx, "car")
            
            if v_type == "person":
                if conf < 0.30:
                    continue
                persons_detected.append((x1, y1, x2, y2, track_id, conf))
            else:
                vehicles_detected.append((x1, y1, x2, y2, track_id, v_type))
                    
        # Filter persons to exclude riders
        def box_intersection_fraction(box_person, box_vehicle):
            px1, py1, px2, py2 = box_person[:4]
            vx1, vy1, vx2, vy2 = box_vehicle[:4]
            ix1 = max(px1, vx1)
            iy1 = max(py1, vy1)
            ix2 = min(px2, vx2)
            iy2 = min(py2, vy2)
            if ix1 < ix2 and iy1 < iy2:
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                person_area = (px2 - px1) * (py2 - py1)
                if person_area > 0:
                    return inter_area / person_area
            return 0.0

        actual_pedestrians = []
        for p_box in persons_detected:
            px1, py1, px2, py2, p_id, p_conf = p_box
            is_rider = False
            for v_box in vehicles_detected:
                if box_intersection_fraction((px1, py1, px2, py2), v_box[:4]) > 0.15:
                    is_rider = True
                    break
            
            if is_rider:
                continue
                
            cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
            in_roi_or_lanes = (cv2.pointPolygonTest(roi_points_lane1, (cx, py2), False) >= 0 or 
                               cv2.pointPolygonTest(roi_points_lane2, (cx, py2), False) >= 0 or 
                               (px2 < width * 0.9))
            
            if in_roi_or_lanes:
                vehicles_detected.append((px1, py1, px2, py2, p_id, "motorcycle"))
            else:
                if p_conf >= 0.40:
                    actual_pedestrians.append((px1, py1, px2, py2, p_id))

        # Process Vehicles
        lane1_density = 0
        lane2_density = 0
        lane1_stuck = 0
        lane2_stuck = 0
        live_counts = {"person": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        
        lane1_curr_ids = set()
        lane2_curr_ids = set()
        
        for x1, y1, x2, y2, track_id, v_type in vehicles_detected:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            tid = int(track_id) if track_id is not None else None
            
            live_counts[v_type] += 1
            if tid is not None:
                unique_seen[v_type].add(tid)
                
            points_to_check = [
                (cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, y1), (cx, y2)
            ]
            
            in_roi_lane1 = False
            in_roi_lane2 = False
            for pt in points_to_check:
                if cv2.pointPolygonTest(roi_points_lane1, pt, False) >= 0:
                    in_roi_lane1 = True
                    break
                if cv2.pointPolygonTest(roi_points_lane2, pt, False) >= 0:
                    in_roi_lane2 = True
                    break
            
            if in_roi_lane1:
                lane1_density += 1
                if tid is not None:
                    lane1_curr_ids.add(tid)
            elif in_roi_lane2:
                lane2_density += 1
                if tid is not None:
                    lane2_curr_ids.add(tid)
                
            # Speed Estimation using Centroid History
            is_stationary = False
            estimated_speed = None
            if tid is not None:
                if tid not in centroid_history:
                    centroid_history[tid] = []
                centroid_history[tid].append((cx, cy, frame_count))
                
                centroid_history[tid] = [entry for entry in centroid_history[tid] if entry[2] >= frame_count - 30]
                
                history = centroid_history[tid]
                if len(history) > 2:
                    df = history[-1][2] - history[0][2]
                    dt_play = df / fps
                    if dt_play >= 0.15:
                        dx = history[-1][0] - history[0][0]
                        dy = history[-1][1] - history[0][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        estimated_speed = dist / dt_play
                        
                        if estimated_speed < 18.0:
                            is_stationary = True
            
            # Color assignment:
            # - Stationary: Red (0, 0, 255)
            # - Moving Lane 1: Cyan (255, 255, 0)
            # - Moving Lane 2: Purple/Magenta (255, 0, 255)
            # - Moving Outside: Blue (255, 0, 0)
            if is_stationary:
                color = (0, 0, 255) # Red for stuck
                if in_roi_lane1:
                    lane1_stuck += 1
                elif in_roi_lane2:
                    lane2_stuck += 1
            elif in_roi_lane1:
                color = (255, 255, 0) # Cyan for moving in Lane 1
            elif in_roi_lane2:
                color = (255, 0, 255) # Purple/Magenta for moving in Lane 2
            else:
                color = (255, 0, 0) # Blue for moving outside
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status_text = " [STUCK]" if is_stationary else " [MOVING]"
            id_text = f"{v_type.capitalize()} ID:{tid}{status_text}" if tid is not None else f"{v_type.capitalize()}"
            cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.circle(frame, (cx, y2), 4, color, -1)

        # Process Actual Pedestrians
        for x1, y1, x2, y2, track_id in actual_pedestrians:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            v_type = "person"
            
            live_counts[v_type] += 1
            if track_id is not None:
                unique_seen[v_type].add(int(track_id))
                
            color = (255, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            id_text = f"Pedestrian ID:{int(track_id)}" if track_id is not None else "Pedestrian"
            cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Check which vehicles have cleared (exited active green lane ROI)
        for tid in lane1_prev_ids:
            if tid not in lane1_curr_ids:
                if simulator.state in [LANE1_GREEN, LANE1_YELLOW]:
                    simulator.vehicles_cleared_adaptive += 1
                if simulator.fixed_state in [LANE1_GREEN, LANE1_YELLOW]:
                    simulator.vehicles_cleared_fixed += 1
                    
        for tid in lane2_prev_ids:
            if tid not in lane2_curr_ids:
                if simulator.state in [LANE2_GREEN, LANE2_YELLOW]:
                    simulator.vehicles_cleared_adaptive += 1
                if simulator.fixed_state in [LANE2_GREEN, LANE2_YELLOW]:
                    simulator.vehicles_cleared_fixed += 1
                    
        lane1_prev_ids = lane1_curr_ids
        lane2_prev_ids = lane2_curr_ids

        current_time = time.time()
        real_dt = current_time - last_time
        last_time = current_time
        
        # Update simulator based on both lane stuck vehicles
        simulator.update(real_dt, lane1_stuck, lane2_stuck)
        
        # Draw ROI Polygons
        cv2.polylines(frame, [roi_points_lane1], isClosed=True, color=(255, 255, 0), thickness=3) # Cyan
        cv2.polylines(frame, [roi_points_lane2], isClosed=True, color=(255, 0, 255), thickness=3) # Purple
        
        # Helper to draw visual traffic light
        def draw_traffic_light(img, x, y, active_color):
            cv2.rectangle(img, (x - 20, y - 60), (x + 20, y + 60), (30, 30, 30), -1)
            cv2.rectangle(img, (x - 20, y - 60), (x + 20, y + 60), (60, 60, 60), 2)
            r_color = (0, 0, 255) if active_color == "RED" else (0, 0, 40)
            cv2.circle(img, (x, y - 40), 12, r_color, -1)
            y_color = (0, 255, 255) if active_color == "YELLOW" else (0, 40, 40)
            cv2.circle(img, (x, y), 12, y_color, -1)
            g_color = (0, 255, 0) if active_color == "GREEN" else (0, 40, 0)
            cv2.circle(img, (x, y + 40), 12, g_color, -1)
            
        # Draw both traffic lights
        l1_color = "GREEN" if simulator.state == LANE1_GREEN else ("YELLOW" if simulator.state == LANE1_YELLOW else "RED")
        draw_traffic_light(frame, 60, 380, l1_color)
        cv2.putText(frame, "LANE 1", (35, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        l2_color = "GREEN" if simulator.state == LANE2_GREEN else ("YELLOW" if simulator.state == LANE2_YELLOW else "RED")
        draw_traffic_light(frame, width - 60, 380, l2_color)
        cv2.putText(frame, "LANE 2", (width - 85, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw Swarm Decision Logs box on-stream
        log_h = 130
        log_w = 480
        log_y_start = height - 145
        cv2.rectangle(frame, (10, log_y_start), (10 + log_w, log_y_start + log_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, log_y_start), (10 + log_w, log_y_start + log_h), (0, 255, 255), 1)
        cv2.putText(frame, "SWARM OPTIMIZER DECISIONS:", (20, log_y_start + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        for idx, log in enumerate(simulator.decision_logs[-3:]):
            cv2.putText(frame, log[:65], (20, log_y_start + 55 + idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # Congestion classification mapping
        def get_congestion_level(stuck_count):
            if stuck_count >= 6:
                return "HIGH"
            elif stuck_count >= 3:
                return "MEDIUM"
            else:
                return "LOW"
                
        lane1_congestion = get_congestion_level(lane1_stuck)
        lane2_congestion = get_congestion_level(lane2_stuck)
        total_live_vehicles = sum(v for k, v in live_counts.items() if k != "person")

        # Update metrics global state
        global_metrics["lane1_density"] = lane1_density
        global_metrics["lane2_density"] = lane2_density
        global_metrics["lane1_stuck"] = lane1_stuck
        global_metrics["lane2_stuck"] = lane2_stuck
        global_metrics["lane1_congestion"] = lane1_congestion
        global_metrics["lane2_congestion"] = lane2_congestion
        global_metrics["total_live_vehicles"] = total_live_vehicles
        global_metrics["state"] = simulator.state
        global_metrics["fixed_state"] = simulator.fixed_state
        global_metrics["time_left"] = max(0.0, simulator.current_duration - simulator.timer)
        global_metrics["live_counts"] = live_counts
        
        totals = {k: len(v) for k, v in unique_seen.items()}
        global_metrics["total_counts"] = totals
        global_metrics["total_vehicles"] = sum(v for k, v in totals.items() if k != "person")
        
        global_metrics["adaptive_total_wait"] = simulator.adaptive_total_wait
        global_metrics["fixed_total_wait"] = simulator.fixed_total_wait
        
        if simulator.fixed_total_wait > 0:
            gain = ((simulator.fixed_total_wait - simulator.adaptive_total_wait) / simulator.fixed_total_wait) * 100.0
            global_metrics["efficiency_gain"] = max(0.0, gain)
        else:
            global_metrics["efficiency_gain"] = 0.0
            
        global_metrics["vehicles_cleared_adaptive"] = simulator.vehicles_cleared_adaptive
        global_metrics["vehicles_cleared_fixed"] = simulator.vehicles_cleared_fixed
        global_metrics["decision_logs"] = list(simulator.decision_logs)

        # Draw Information Panel natively on video to match sample
        panel_h = 240
        panel_w = 355
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        
        cv2.putText(frame, f"L1 Density: {lane1_density} | Stuck: {lane1_stuck}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"L2 Density: {lane2_density} | Stuck: {lane2_stuck}", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Screen Density: {total_live_vehicles} vehicles", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        cv2.putText(frame, f"L1 Congestion: {lane1_congestion}", (20, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if lane1_congestion == "HIGH" else ((0, 165, 255) if lane1_congestion == "MEDIUM" else (0, 255, 0)), 2)
        cv2.putText(frame, f"L2 Congestion: {lane2_congestion}", (20, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if lane2_congestion == "HIGH" else ((0, 165, 255) if lane2_congestion == "MEDIUM" else (0, 255, 0)), 2)
                        
        cv2.putText(frame, f"Signal State: {simulator.state}", (20, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time left: {max(0, simulator.current_duration - simulator.timer):.1f}s", (20, 215), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, private",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/api/metrics")
def get_metrics():
    return JSONResponse(content=global_metrics)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
