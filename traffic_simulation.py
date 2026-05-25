import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Configuration
VIDEO_PATH = "new_congested_traffic.mp4"
OUTPUT_VIDEO_PATH = "output_heavy_congestion_simulation.mp4"
SCREENSHOT_DIR = "screenshots"
MODEL_NAME = "yolov8n.pt"

# Create screenshot directory
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Define ROI polygon coordinates
# These will be dynamically adjusted based on frame size, 
# but we will set default relative coordinates.
# E.g., a trapezoid in the lower half.

# Traffic Light States
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
        
        self.congestion_threshold = 10

    def update(self, dt, vehicle_count):
        self.timer += dt
        
        if self.state == GREEN:
            # Adaptive logic: Extend green time if there are many vehicles
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
            # Adaptive logic: Shorten red time if congestion is severe
            adaptive_duration = self.base_red_time
            if vehicle_count > self.congestion_threshold + 3:
                # Force early green if waiting too long (but enforce minimum red time)
                adaptive_duration = self.min_red_time
                
            self.current_duration = adaptive_duration
            
            if self.timer >= self.current_duration:
                self.state = GREEN
                self.timer = 0.0
                self.current_duration = self.base_green_time

def main():
    print("Initializing YOLO Model...")
    # Load YOLOv8 model
    model = YOLO(MODEL_NAME)
    
    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None or np.isnan(fps):
        fps = 30.0
    
    # Define ROI based on frame dimensions
    # A trapezoid covering the lower middle part of the frame
    roi_points = np.array([
        [int(width * 0.18), int(height * 0.9)],
        [int(width * 0.70), int(height * 0.9)],
        [int(width * 0.52), int(height * 0.60)],
        [int(width * 0.38), int(height * 0.60)]
    ], np.int32)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    simulator = TrafficSignalSimulator()
    dt = 1.0 / fps
    
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

    frame_count = 0
    screenshot_flags = {
        "congestion": False,
        "adaptive_green": False,
        "adaptive_red": False
    }

    print("Starting processing loop...")
    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= 400:
            break
            
        frame_count += 1
        
        # Run YOLO detection and tracking
        # COCO Classes: 2=car, 3=motorcycle, 5=bus, 7=truck
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
        
        vehicles_in_roi = 0
        live_counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        
        # Process detections
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
                    color = (0, 255, 0) # Green for in ROI
                else:
                    color = (255, 0, 0) # Blue for outside
                    
                # Draw bounding box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                id_text = f"{v_type.capitalize()} ID:{int(track_id)}" if track_id is not None else f"{v_type.capitalize()}"
                cv2.putText(frame, id_text, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                # Draw contact point
                cv2.circle(frame, (cx, cy_check), 4, color, -1)

        # Update Traffic Simulator
        simulator.update(dt, vehicles_in_roi)
        
        # Draw ROI Polygon
        cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=3)
        
        # Draw Information Panel
        panel_h = 220
        panel_w = 350
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        
        # Display Texts
        cv2.putText(frame, f"Traffic Density: {vehicles_in_roi} vehicles", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        # Congestion Warning
        if vehicles_in_roi > simulator.congestion_threshold:
            cv2.putText(frame, "CONGESTION WARNING!", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            if not screenshot_flags["congestion"]:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "congestion_warning.jpg"), frame)
                screenshot_flags["congestion"] = True
                
        # Signal Status
        cv2.putText(frame, f"Signal: {simulator.state}", (20, 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time left: {max(0, simulator.current_duration - simulator.timer):.1f}s", (20, 160), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        if simulator.state == GREEN and simulator.current_duration > simulator.base_green_time:
            cv2.putText(frame, "ADAPTIVE: GREEN EXTENDED", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if not screenshot_flags["adaptive_green"]:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "adaptive_green.jpg"), frame)
                screenshot_flags["adaptive_green"] = True
                
        if simulator.state == RED and simulator.current_duration < simulator.base_red_time:
            cv2.putText(frame, "ADAPTIVE: EARLY GREEN PENDING", (20, 200), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if not screenshot_flags["adaptive_red"]:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "adaptive_red.jpg"), frame)
                screenshot_flags["adaptive_red"] = True

        # Draw Vehicle Breakdown Panel
        breakdown_h = 170
        breakdown_w = 350
        cv2.rectangle(frame, (10, 240), (10 + breakdown_w, 240 + breakdown_h), (0, 0, 0), -1)
        cv2.putText(frame, "Vehicle Breakdown:", (20, 270), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Cars: {live_counts['car']} (Total: {len(unique_seen['car'])})", (20, 305), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Bikes: {live_counts['motorcycle']} (Total: {len(unique_seen['motorcycle'])})", (20, 335), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Buses: {live_counts['bus']} (Total: {len(unique_seen['bus'])})", (20, 365), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Trucks: {live_counts['truck']} (Total: {len(unique_seen['truck'])})", (20, 395), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
        # Draw Visual Traffic Light
        light_x, light_y = width - 100, 150
        cv2.rectangle(frame, (light_x - 30, light_y - 90), (light_x + 30, light_y + 90), (50, 50, 50), -1)
        
        # Red
        color = (0, 0, 255) if simulator.state == RED else (0, 0, 50)
        cv2.circle(frame, (light_x, light_y - 50), 20, color, -1)
        # Yellow
        color = (0, 255, 255) if simulator.state == YELLOW else (0, 50, 50)
        cv2.circle(frame, (light_x, light_y), 20, color, -1)
        # Green
        color = (0, 255, 0) if simulator.state == GREEN else (0, 50, 0)
        cv2.circle(frame, (light_x, light_y + 50), 20, color, -1)

        out.write(frame)
        
        if frame_count % 30 == 0:
            print(f"Processed frame {frame_count}")
            # Take a generic sample screenshot
            if frame_count == 60:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "tracking_sample.jpg"), frame)

    cap.release()
    out.release()
    print("Processing complete!")

if __name__ == "__main__":
    main()
