import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("new_congested_traffic.mp4")

if not cap.isOpened():
    print("Could not open new_congested_traffic.mp4")
    exit()

print("Analyzing first 100 frames...")
max_vehicles = 0
for i in range(100):
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    if boxes is not None:
        vehicle_count = 0
        for box in boxes:
            cls = int(box.cls[0])
            if cls in [2, 3, 5, 7]: # Car, Motorcycle, Bus, Truck
                vehicle_count += 1
        if vehicle_count > max_vehicles:
            max_vehicles = vehicle_count

print(f"Analysis complete! Max vehicles detected in a single frame: {max_vehicles}")
cap.release()
