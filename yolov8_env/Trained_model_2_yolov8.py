import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO  

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 model on CUDA
model = YOLO("best.pt").to(device)

# Configure RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start RealSense camera
pipeline.start(config)

try:
    while True:
        # Get RealSense frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to NumPy array
        frame = np.asanyarray(color_frame.get_data())

        # Perform YOLO detection on GPU
        results = model(frame, device=device)  

        # Draw detection results on the frame
        frame_with_detections = results[0].plot()  

        # Display the frame
        cv2.imshow('Basketball Hoop Detection', frame_with_detections)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
