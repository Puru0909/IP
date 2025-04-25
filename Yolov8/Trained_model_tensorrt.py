import cv2
import numpy as np
import pyrealsense2 as rs
import torch
import time  # Import time module
from ultralytics import YOLO  

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load YOLOv8 TensorRT model (no .to(device) needed)
model = YOLO("best.engine")

# Configure RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start RealSense camera
pipeline.start(config)

# Initialize FPS calculation
prev_time = time.time()

try:
    while True:
        # Get RealSense frames
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to NumPy array
        frame = np.asanyarray(color_frame.get_data())

        # Start timer for FPS calculation
        start_time = time.time()

        # Perform YOLO detection on TensorRT
        results = model.predict(frame, device=device)  

        # Draw detection results on the frame
        frame_with_detections = results[0].plot()  

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - prev_time)  # FPS formula
        prev_time = end_time  # Update previous time

        # Display FPS on the frame
        cv2.putText(frame_with_detections, f"FPS: {fps:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Basketball Hoop Detection', frame_with_detections)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
