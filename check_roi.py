import cv2
from ultralytics import YOLO

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("sample_traffic.mp4")

if not cap.isOpened():
    print("Could not open sample_traffic.mp4")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video Resolution: {width}x{height}")

# Process 150 frames
for frame_idx in range(150):
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, verbose=False)
    boxes = results[0].boxes
    if len(boxes) > 0:
        found_vehicles = []
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            if cls in [2, 3, 5, 7]:
                found_vehicles.append(f"Class {cls} ({model.names[cls]}, conf={conf:.2f}) at center ({cx}, {cy})")
        if found_vehicles:
            print(f"Frame {frame_idx}: Detected {len(found_vehicles)} vehicles:")
            for item in found_vehicles:
                print("  " + item)

cap.release()
