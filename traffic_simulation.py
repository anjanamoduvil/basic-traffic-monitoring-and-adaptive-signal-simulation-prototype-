import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Configuration
VIDEO_PATH = "C:/Users/VICTUS/NIT/traffic_prototype/new_congested_traffic.mp4"
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
            if vehicle_count >= self.congestion_threshold:
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
            if vehicle_count >= self.congestion_threshold + 3:
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
        [int(width * 0.02), int(height * 0.95)],
        [int(width * 0.98), int(height * 0.95)],
        [int(width * 0.85), int(height * 0.45)],
        [int(width * 0.15), int(height * 0.45)]
    ], np.int32)
    
    # Video writer setup
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
    
    simulator = TrafficSignalSimulator()
    dt = 1.0 / fps
    
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
        # COCO Classes: 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
        results = model.track(frame, persist=True, classes=[0, 1, 2, 3, 5, 7], conf=0.10, iou=0.5, verbose=False)
        
        vehicles_in_roi = 0
        live_counts = {"person": 0, "car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
        
        persons_detected = []
        vehicles_detected = []
        
        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else [None] * len(boxes)
            
            for box, cls_idx, conf, track_id in zip(boxes, clss, confs, ids):
                x1, y1, x2, y2 = map(int, box)
                cls_idx = int(cls_idx)
                v_type = class_names.get(cls_idx, "car")
                
                if v_type == "person":
                    # Filter out weak standalone pedestrian detections to avoid noise
                    if conf < 0.30:
                        continue
                    persons_detected.append((x1, y1, x2, y2, track_id, conf))
                else:
                    vehicles_detected.append((x1, y1, x2, y2, track_id, v_type))
                    
        # Filter persons to exclude riders (persons overlapping with motorcycles/vehicles)
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
                # Lower overlap threshold to 0.15 to handle high-perspective offset
                if box_intersection_fraction((px1, py1, px2, py2), v_box[:4]) > 0.15:
                    is_rider = True
                    break
            
            if is_rider:
                # Discard duplicate person bounding box overlapping with a vehicle
                continue
                
            # Heuristic: Since this is a high-speed motorway with no pedestrian walking lanes,
            # any standalone person detection inside the active lanes or ROI is a motorcycle rider
            # whose bike was not detected separately by YOLO. Let's auto-correct them to motorcycle!
            cx, cy = (px1 + px2) // 2, (py1 + py2) // 2
            in_roi_or_lanes = cv2.pointPolygonTest(roi_points, (cx, py2), False) >= 0 or (px2 < width * 0.9)
            
            if in_roi_or_lanes:
                vehicles_detected.append((px1, py1, px2, py2, p_id, "motorcycle"))
            else:
                # Only keep as actual pedestrian if high confidence and completely off-road
                if p_conf >= 0.40:
                    actual_pedestrians.append((px1, py1, px2, py2, p_id))

        # Process Vehicles
        stationary_in_roi = 0
        for x1, y1, x2, y2, track_id, v_type in vehicles_detected:
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            
            live_counts[v_type] += 1
            if track_id is not None:
                unique_seen[v_type].add(int(track_id))
                
            points_to_check = [
                (cx, cy), (x1, y1), (x2, y1), (x1, y2), (x2, y2), (cx, y1), (cx, y2)
            ]
            in_roi = False
            for pt in points_to_check:
                if cv2.pointPolygonTest(roi_points, pt, False) >= 0:
                    in_roi = True
                    break
            
            if in_roi:
                vehicles_in_roi += 1
                
            # Speed Estimation using Centroid History (Frame-based time to bypass wall-clock lags)
            is_stationary = False
            estimated_speed = None
            if track_id is not None:
                tid = int(track_id)
                if tid not in centroid_history:
                    centroid_history[tid] = []
                centroid_history[tid].append((cx, cy, frame_count))
                
                # Keep rolling history up to 30 frames (1.0 second of video playback time)
                centroid_history[tid] = [entry for entry in centroid_history[tid] if entry[2] >= frame_count - 30]
                
                # Calculate speed if we have history covering at least 5 frames (0.15 seconds of video playback)
                history = centroid_history[tid]
                if len(history) > 2:
                    df = history[-1][2] - history[0][2]
                    dt_play = df / fps
                    if dt_play >= 0.15:
                        dx = history[-1][0] - history[0][0]
                        dy = history[-1][1] - history[0][1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        estimated_speed = dist / dt_play  # pixels per video second
                        
                        # An optimized threshold of 18.0 pixels/sec accounts for YOLO bounding box jitter and perspective compression
                        if estimated_speed < 18.0:
                            is_stationary = True

            # Color assignment:
            # - Stationary vehicles are colored Red (0, 0, 255)
            # - Moving vehicles inside the ROI are colored Green (0, 255, 0)
            # - Moving vehicles outside the ROI are colored Blue (255, 0, 0)
            if is_stationary:
                color = (0, 0, 255) # Red for stuck
                if in_roi:
                    stationary_in_roi += 1
            elif in_roi:
                color = (0, 255, 0) # Green for moving in ROI
            else:
                color = (255, 0, 0) # Blue for moving outside ROI
                
            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            status_text = " [STUCK]" if is_stationary else " [MOVING]"
            id_text = f"{v_type.capitalize()} ID:{int(track_id)}{status_text}" if track_id is not None else f"{v_type.capitalize()}"
            cv2.putText(frame, id_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            # Draw contact point
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

        # Update Traffic Simulator based on stationary count to solve the green-light paradox
        simulator.update(dt, stationary_in_roi)
        
        # Draw ROI Polygon
        cv2.polylines(frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=3)
        
        # Draw Information Panel
        panel_h = 240
        panel_w = 355
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        
        # Display Texts
        total_live_vehicles = sum(v for k, v in live_counts.items() if k != "person")
        cv2.putText(frame, f"ROI Density: {vehicles_in_roi} vehicles", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Screen Density: {total_live_vehicles} vehicles", (20, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(frame, f"ROI Stuck: {stationary_in_roi} vehicles", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if stationary_in_roi > 0 else (200, 200, 200), 2)
                    
        is_congested = (stationary_in_roi >= simulator.congestion_threshold) or (total_live_vehicles >= 10)
        if is_congested:
            warning_text = "CONGESTION WARNING!"
            if stationary_in_roi >= simulator.congestion_threshold and total_live_vehicles >= 10:
                warning_text = "HIGH CONGESTION (BOTH)"
            elif stationary_in_roi >= simulator.congestion_threshold:
                warning_text = "ROI CONGESTION"
            else:
                warning_text = "SCREEN CONGESTION"
            cv2.putText(frame, warning_text, (20, 125), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not screenshot_flags["congestion"]:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "congestion_warning.jpg"), frame)
                screenshot_flags["congestion"] = True
                
        # Signal Status
        cv2.putText(frame, f"Signal: {simulator.state}", (20, 155), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Time left: {max(0, simulator.current_duration - simulator.timer):.1f}s", (20, 185), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
        if simulator.state == GREEN and simulator.current_duration > simulator.base_green_time:
            cv2.putText(frame, "ADAPTIVE: GREEN EXTENDED", (20, 215), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if not screenshot_flags["adaptive_green"]:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "adaptive_green.jpg"), frame)
                screenshot_flags["adaptive_green"] = True
                
        if simulator.state == RED and simulator.current_duration < simulator.base_red_time:
            cv2.putText(frame, "ADAPTIVE: EARLY GREEN PENDING", (20, 215), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if not screenshot_flags["adaptive_red"]:
                cv2.imwrite(os.path.join(SCREENSHOT_DIR, "adaptive_red.jpg"), frame)
                screenshot_flags["adaptive_red"] = True

        # Draw Vehicle Breakdown Panel
        breakdown_h = 200
        breakdown_w = 350
        cv2.rectangle(frame, (10, 260), (10 + breakdown_w, 260 + breakdown_h), (0, 0, 0), -1)
        cv2.putText(frame, "Breakdown & Pedestrians:", (20, 290), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Cars: {live_counts['car']} (Total: {len(unique_seen['car'])})", (20, 320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Bikes: {live_counts['motorcycle']} (Total: {len(unique_seen['motorcycle'])})", (20, 350), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Buses: {live_counts['bus']} (Total: {len(unique_seen['bus'])})", (20, 380), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Trucks: {live_counts['truck']} (Total: {len(unique_seen['truck'])})", (20, 410), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Pedestrians: {live_counts['person']} (Total: {len(unique_seen['person'])})", (20, 440), 
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
