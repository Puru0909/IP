import numpy as np
import cv2 as cv
import pyrealsense2 as rs
import pickle

# Chessboard configuration
chessboardSize = (9, 6)
size_of_chessboard_squares_mm = 20

# Prepare object points
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)
objp *= size_of_chessboard_squares_mm

# Termination criteria for corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)
print("Press 's' to capture frames for calibration. Press 'q' to quit.")

try:
    while True:
        # Get frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        gray = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(color_image, chessboardSize, corners2, ret)

        # Display the live feed
        cv.imshow('RealSense Camera', color_image)

        # Capture frames for calibration
        key = cv.waitKey(1) & 0xFF
        if key == ord('s') and ret:
            print("Frame captured for calibration.")
            objpoints.append(objp)
            imgpoints.append(corners2)
        elif key == ord('q'):
            break

finally:
    pipeline.stop()
    cv.destroyAllWindows()

if len(objpoints) > 0:
    # Perform camera calibration
    print("Calibrating camera...")
    ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if ret:
        print("Camera calibration successful!")
        print("Camera matrix:\n", cameraMatrix)
        print("Distortion coefficients:\n", dist)

        # Save calibration results
        with open("calibration_realSense.pkl", "wb") as f:
            pickle.dump((cameraMatrix, dist), f)
        print("Calibration data saved.")
    else:
        print("Calibration failed.")
else:
    print("No frames captured for calibration.")