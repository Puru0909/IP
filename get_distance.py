import cv2
import numpy as np
import pyrealsense2 as rs

# Configure RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable color and depth streams
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)

# Start the pipeline
pipeline.start(config)

# Create an align object to align depth frames to color frames
align_to = rs.stream.color
align = rs.align(align_to)

# Create windows
cv2.namedWindow("Basketball Detection")

# Create trackbars for tuning circle detection parameters
cv2.createTrackbar("Param1", "Basketball Detection", 200, 500, lambda x: None)
cv2.createTrackbar("Param2", "Basketball Detection", 20, 100, lambda x: None)
cv2.createTrackbar("Min Radius", "Basketball Detection", 10, 100, lambda x: None)
cv2.createTrackbar("Max Radius", "Basketball Detection", 150, 500, lambda x: None)

# Define the range for red color in HSV
lower_red1 = np.array([0, 120, 70])    # Lower range for red
upper_red1 = np.array([10, 255, 255])  # Upper range for red
lower_red2 = np.array([170, 120, 70])  # Lower range for red (wrap-around)
upper_red2 = np.array([180, 255, 255]) # Upper range for red (wrap-around)

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()

        # Align the depth frame to the color frame
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # Convert frames to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Mask for the first range of red
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Mask for the second range of red
        red_mask = cv2.bitwise_or(mask1, mask2)           # Combine both masks

        # Apply the red mask to the original frame
        masked_frame = cv2.bitwise_and(color_image, color_image, mask=red_mask)

        # Convert the masked frame to grayscale
        gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)

        # Apply morphological operations to remove small noise
        kernel = np.ones((5, 5), np.uint8)
        morphed = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)

        # Get trackbar values for circle detection
        param1 = cv2.getTrackbarPos("Param1", "Basketball Detection")
        param2 = cv2.getTrackbarPos("Param2", "Basketball Detection")
        minRadius = cv2.getTrackbarPos("Min Radius", "Basketball Detection")
        maxRadius = cv2.getTrackbarPos("Max Radius", "Basketball Detection")

        # Detect circles using Hough Transform (only in the red-masked region)
        circles = cv2.HoughCircles(
            morphed, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
            param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
        )

        if circles is not None:
            detected_circles = np.uint16(np.around(circles))
            for (x, y, r) in detected_circles[0, :]:
                # Draw a bounding box around the detected circle
                cv2.rectangle(color_image, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)

                # Draw the circle
                cv2.circle(color_image, (x, y), r, (0, 255, 0), 3)

                # Draw the center of the circle
                cv2.circle(color_image, (x, y), 2, (0, 255, 0), 3)

                # Check if (x, y) is within bounds
                if 0 <= x < 848 and 0 <= y < 480:
                    # Calculate distance to the center of the circle
                    distance = depth_frame.get_distance(x, y)
                    if distance > 0:  # Check for valid depth data
                        # Display distance on the frame
                        cv2.putText(color_image, f"Distance: {distance:.2f} m", (x - 50, y + r + 20), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    else:
                        print(f"No valid depth data at ({x}, {y})")
                else:
                    print(f"Invalid coordinates: ({x}, {y})")

        # Show the final output
        cv2.imshow('Basketball Detection', color_image)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()