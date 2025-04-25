import pyrealsense2 as rs
import numpy as np
import cv2
import torch
import time
from ultralytics import YOLO

# Load YOLO model with device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.engine")
print(f"Using device: {device}")

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30) 
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Get camera intrinsics (needed for depth-to-world conversion)
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()  # Depth scale factor
intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

# Print depth scale for debugging
print(f"Depth scale: {depth_scale}")

# Depth filtering
decimation = rs.decimation_filter(2)  # Downsample
spatial = rs.spatial_filter()  # Edge-preserving smoothing
spatial.set_option(rs.option.filter_magnitude, 2)
temporal = rs.temporal_filter()  # Reduce flickering
temporal.set_option(rs.option.filter_smooth_alpha, 0.4)
hole_filling = rs.hole_filling_filter(1)  # Fill missing depth pixels

# FPS init
prev_time = time.time()

def pixel_to_world(x, y, depth):
    """ Convert depth pixel to real-world coordinates """
    if depth <= 0:  # Prevent incorrect calculations for invalid depth
        return 0, 0, 0
    X = (x - intrinsics.ppx) * depth / intrinsics.fx
    Y = (y - intrinsics.ppy) * depth / intrinsics.fy
    Z = depth
    return X, Y, Z

def get_median_depth(depth_image, x1, y1, x2, y2, roi_size=7):
    """ Extract median depth from a small center region of the bounding box """
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # Define a small ROI around the center
    x_start = max(center_x - roi_size // 2, 0)
    x_end = min(center_x + roi_size // 2, depth_image.shape[1])
    y_start = max(center_y - roi_size // 2, 0)
    y_end = min(center_y + roi_size // 2, depth_image.shape[0])

    # Extract depth values and remove invalid (zero) values
    depth_roi = depth_image[y_start:y_end, x_start:x_end].flatten()
    depth_roi = depth_roi[depth_roi > 0]  # Remove zero-depth pixels

    if len(depth_roi) == 0:
        return 0  # No valid depth values
    return np.median(depth_roi)

try:
    while True:

        # Start FPS timer
        start_time = time.time()

        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Apply filters to depth frame
        depth_frame = decimation.process(depth_frame)
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)
        depth_frame = hole_filling.process(depth_frame)

        # Convert frames to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLO object detection
        results = model.predict(source=color_image, device=0)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                
                # Get median depth from center region
                depth_value = get_median_depth(depth_image, x1, y1, x2, y2) * depth_scale  # Convert to meters
                # depth_value = depth_frame.get_distance((x1 + x2) // 2, (y1 + y2) // 2)


                # Debugging: Print raw depth value
                print(f"Raw depth for {model.names[int(box.cls[0])]}: {depth_value:.2f}m")

                # Ignore invalid depth values
                if depth_value < 0.1 or depth_value > 6.0:  # Set min/max limits
                    print(f"Invalid depth detected: {depth_value:.2f}m, skipping...")
                    continue  

                # Convert pixel coordinates to world coordinates
                world_x, world_y, world_z = pixel_to_world((x1 + x2) // 2, (y1 + y2) // 2, depth_value)

                # Calculate Correct Distances
                total_distance = np.sqrt(world_x**2 + world_y**2 + world_z**2)  # True 3D distance
                # horizontal_distance = np.sqrt(world_x**2 + world_z**2)  # Ignores height
                # vertical_distance = abs(world_y)  # Height component only

                # Draw bounding box and distance label
                label = f"{model.names[int(box.cls[0])]} {total_distance:.2f}m"
                cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(color_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Debugging: Print calculated distance
                print(f"Detected {model.names[int(box.cls[0])]} at {total_distance:.2f} meters")


        # Calculate FPS
        end_time = time.time()
        fps = 1.0 / (end_time - prev_time)
        prev_time = end_time
        cv2.putText(color_image, f"FPS: {fps:.2f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)        

        # Display the results
        cv2.imshow("RealSense YOLO Detection", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
