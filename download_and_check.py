import urllib.request
import cv2
import os

url = "https://github.com/imkevinabraham/traffic_analysis/raw/master/traffic.mp4"
dest = "new_congested_traffic.mp4"

print("Downloading video from:", url)
try:
    urllib.request.urlretrieve(url, dest)
    print("Download completed successfully!")
except Exception as e:
    print("Download failed:", e)
    exit()

if os.path.exists(dest):
    print(f"File size: {os.path.getsize(dest)} bytes")
    cap = cv2.VideoCapture(dest)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Success! Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")
        cap.release()
    else:
        print("Error: Could not open the downloaded file as a video.")
else:
    print("Error: File does not exist.")
