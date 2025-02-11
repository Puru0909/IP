import pyrealsense2 as rs  # Intel RealSense SDK for accessing camera streams
import numpy as np  # NumPy for numerical operations
from ultralytics import YOLO  # YOLO for object detection
import matplotlib.pyplot as plt  # Matplotlib for displaying images

# Load YOLOv8-small model
model = YOLO("yolov8s.pt")  # Load the pre-trained YOLOv8 model

# Configure depth and color streams for RealSense camera
pipeline = rs.pipeline()  # Create a pipeline to manage camera streamingQ
config = rs.config()

# Enable the color stream (640x480 resolution, 30 FPS)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming from the RealSense camera
pipeline.start(config)

# Create a figure for real-time display using Matplotlib
fig, ax = plt.subplots()

try:
    while True:
        # Wait for a coherent pair of frames: color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue  # Skip loop iteration if no color frame is available

        # Convert the RealSense color frame to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 inference on the captured frame
        results = model(color_image)  # Perform object detection

        # Loop through results and annotate the frame
        for result in results:
            annotated_img = result.plot()  # Get an annotated image with detections

        # Convert BGR to RGB for correct colors in Matplotlib
        annotated_img_rgb = annotated_img[..., ::-1]

        # Display the resulting frame using Matplotlib
        ax.imshow(annotated_img_rgb)
        ax.axis("off")  # Remove axes for better visualization
        plt.draw()  # Update the plot in real-time
        plt.pause(0.001)  # Short pause to allow real-time display
        ax.clear()  # Clear the axis for the next frame

finally:
    # Stop streaming from the RealSense camera
    pipeline.stop()
    plt.close(fig)  # Close the Matplotlib window