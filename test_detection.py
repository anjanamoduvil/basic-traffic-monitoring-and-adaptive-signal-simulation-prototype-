import cv2
import os
from ultralytics import YOLO

# Create a directory to store test frames
os.makedirs("test_frames", exist_ok=True)

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture("complex_traffic.mp4")

if not cap.isOpened():
    print("Could not open complex_traffic.mp4")
    exit()

print("Processing 60 frames for visual verification...")

for frame_idx in range(60):
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run raw prediction (without tracker first)
    results = model(frame, verbose=False)
    
    # Annotate frame
    annotated_frame = results[0].plot()
    
    # Save frame every 10 frames
    if frame_idx % 10 == 0:
        output_path = f"test_frames/frame_{frame_idx:03d}.jpg"
        cv2.imwrite(output_path, annotated_frame)
        print(f"Saved {output_path}")

cap.release()
print("Verification complete! Check the 'test_frames' folder.")
