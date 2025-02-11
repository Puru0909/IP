import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Set Matplotlib to interactive mode
plt.ion()  # Enables real-time updating of the plot

# Load YOLOv8 model
MODEL_PATH = r"C:\Users\Dell\Downloads\best (1).pt"  # Ensure correct model path

try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

try:
    pipeline.start(config)
    print("RealSense camera started successfully.")
except Exception as e:
    print(f"Error starting RealSense camera: {e}")
    exit()

# Create Matplotlib figure
fig, ax = plt.subplots()
img_display = ax.imshow(np.zeros((720, 1280, 3), dtype=np.uint8))  # Placeholder for real-time update
plt.axis("off")  # Hide axes

try:
    while True:
        # Get frames from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 detection
        results = model(color_image)

        # Draw detection results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {confidence:.2f}"

                # Draw bounding box and label
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Update Matplotlib plot in real-time
        img_display.set_data(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB
        plt.draw()
        plt.pause(0.01)  # Pause for smooth updating

except Exception as e:
    print(f"Runtime error: {e}")

finally:
    print("Stopping camera and closing windows.")
    pipeline.stop()
    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the last captured frame