import pickle
import cv2
import pyrealsense2 as rs
import numpy as np

# Load calibration data
try:
    with open("calibration_realsense.pkl", "rb") as f:
        cameraMatrix, dist = pickle.load(f)
    print("Calibration data loaded successfully.")
except Exception as e:
    print("Error loading calibration file:", e)
    exit()  # Stop execution if calibration data fails to load

def detect_basketball_hoop_and_backboard_from_realsense_with_calibration():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)
    profile = pipeline.get_active_profile()
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    # 🔹 Ensure proper exposure & laser power
    depth_sensor.set_option(rs.option.laser_power, 360)  # Maximize laser power
    depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    depth_sensor.set_option(rs.option.exposure, 500)  # Adjust exposure for better long-distance performance
    # 🔹 Post-processing filters
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()
    hole_filling = rs.hole_filling_filter()

    align = rs.align(rs.stream.color)

    def get_depth_average(depth_frame, center_x, center_y, kernel_size=10):
        depth_image = np.asanyarray(depth_frame.get_data())
        x_start = max(0, center_x - kernel_size // 2)
        y_start = max(0, center_y - kernel_size // 2)
        x_end = min(depth_image.shape[1], center_x + kernel_size // 2)
        y_end = min(depth_image.shape[0], center_y + kernel_size // 2)

        depth_values = []
        for y in range(y_start, y_end):
            for x in range(x_start, x_end):
                depth = depth_image[y, x] * depth_scale
                if 0.2 < depth < 6.0:
                    depth_values.append(depth)

        if depth_values:
            avg_depth = np.median(depth_values)
            print(f"Raw depth values: {depth_values}")  # 🔹 Debugging
            return avg_depth
        return None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # Apply undistortion to the color image
            color_image = np.asanyarray(color_frame.get_data())
            undistorted_image = cv2.undistort(color_image, cameraMatrix, dist)

            depth_frame = spatial.process(depth_frame)  # 🔹 Edge-preserving smoothing
            depth_frame = temporal.process(depth_frame)  # 🔹 NOISE REDUCTION
            depth_frame = hole_filling.process(depth_frame)  # 🔹 Fill MISSING PIXELS

            depth_image = np.asanyarray(depth_frame.get_data())
            hsv_image = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2HSV)

            lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
            lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
            mask_red = cv2.bitwise_or(cv2.inRange(hsv_image, lower_red1, upper_red1),
                                      cv2.inRange(hsv_image, lower_red2, upper_red2))

            # White mask for the backboard
            lower_white, upper_white = np.array([0, 0, 200]), np.array([180, 55, 255])
            mask_white = cv2.inRange(hsv_image, lower_white, upper_white)

            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_red:
                if cv2.contourArea(contour) > 500:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if 10 < radius < 100:
                        center = (int(x), int(y))
                        cv2.circle(undistorted_image, center, int(radius), (0, 0, 255), 3)
                        depth = get_depth_average(depth_frame, center[0], center[1])
                        if depth is not None:
                            cv2.putText(undistorted_image, f"Hoop: {depth:.2f}m", (center[0] - 10, center[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            contours_white, _ = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours_white:
                if cv2.contourArea(contour) > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    if 0.9 < aspect_ratio < 2.1:
                        cv2.rectangle(undistorted_image, (x, y), (x + w, y + h), (255, 255, 255), 3)

                        # Depth
                        center_x, center_y = x + w // 2, y + h // 2
                        depth = depth_image[center_y, center_x] * depth_scale
                        if depth > 0:
                            cv2.putText(undistorted_image, f"Backboard: {depth:.2f}m", (center_x, center_y - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Basketball Hoop and Backboard Detection with Calibration", undistorted_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


detect_basketball_hoop_and_backboard_from_realsense_with_calibration()