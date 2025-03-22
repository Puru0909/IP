import numpy as np
import cv2 as cv
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLO model with device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)
print(f"Using device: {device}")

def get_valid_depth(depth_frame, x, y, depth_scale):
    """ Extracts a valid depth value from a small region around (x, y) to improve accuracy. """
    kernel_size = 7  # Increase sampling area for better accuracy
    half_k = kernel_size // 2

    # Ensure coordinates are within image bounds
    x_min, x_max = max(0, x - half_k), min(depth_frame.shape[1] - 1, x + half_k)
    y_min, y_max = max(0, y - half_k), min(depth_frame.shape[0] - 1, y + half_k)

    # Extract small depth region
    depth_region = depth_frame[y_min:y_max, x_min:x_max].flatten()

    # Remove invalid depth values (0 means no depth)
    valid_depths = depth_region[(depth_region > 0) & (depth_region < 10000)]  # Allow farther objects

    if len(valid_depths) > 0:
        median_depth = np.median(valid_depths) * depth_scale  # Apply depth scale
        return median_depth
    return -1  # Invalid depth

def detect_objects_with_depth():
    """ Detect objects using YOLO and determine their distance using Intel RealSense depth data. """

    # Configure RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    align = rs.align(rs.stream.color)  # Align depth to color
    pipeline.start(config)

    try:
        # Get RealSense depth scale
        profile = pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()  # Convert depth units to meters

        # Get RealSense Camera Intrinsics
        depth_stream = profile.get_stream(rs.stream.depth)
        intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

        print(f"[INFO] Depth scale: {depth_scale}")  # Debugging depth scale

        while True:
            # Capture frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Run YOLO detection
            results = model(color_image)

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
                    conf = float(box.conf[0])  # Confidence score
                    cls = int(box.cls[0])  # Class ID

                    # Find center of detected object
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

                    # Get improved depth estimate
                    depth = get_valid_depth(depth_image, center_x, center_y, depth_scale)

                    if depth > 0:
                        # Convert pixel coordinates to real-world 3D coordinates
                        Xtarget = depth * (center_x - intrinsics.ppx) / intrinsics.fx
                        Ytarget = depth * (center_y - intrinsics.ppy) / intrinsics.fy
                        Ztarget = depth  # Depth is already in meters

                        # Calculate Euclidean distance
                        distance = np.sqrt(Xtarget ** 2 + Ytarget ** 2 + Ztarget ** 2)

                        # Debugging: Print depth values to check correctness
                        print(f"[DEBUG] Distance: {distance:.2f}m")

                        # Draw bounding box
                        cv.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"Object {cls} ({conf:.2f}) | Dist: {distance:.2f}m"
                        cv.putText(color_image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Display result
            cv.imshow("YOLO Object Detect5ion with Depth", color_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv.destroyAllWindows()

if __name__ == "__main__":
    detect_objects_with_depth()