
import pickle

try:
    with open("calibration_realsense.pkl", "rb") as f:
        cameraMatrix, dist = pickle.load(f)
    print("Loaded Camera Matrix:\n", cameraMatrix)
    print("Loaded Distortion Coefficients:\n", dist)
except Exception as e:
    print("Error loading calibration file:", e)
    exit()  # Stop execution if loading fails





















        
    