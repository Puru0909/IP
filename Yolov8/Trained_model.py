import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO  # Correct way to load YOLOv8

# Load YOLOv8 model
model = YOLO("best.pt")  # Ensure "best.pt" is in the same directory as your project (py file)

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

        # Perform YOLO detection
        results = model(frame)  

        # Draw detection results on the frame
        frame_with_detections = results[0].plot()  # Proper way to render YOLOv8 results

        # Display the frame
        cv2.imshow('Basketball Hoop Detection', frame_with_detections)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    
