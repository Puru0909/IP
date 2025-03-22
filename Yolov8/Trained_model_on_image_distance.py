import numpy as np
import cv2 as cv
import torch
from ultralytics import YOLO

# Load YOLO model with device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)
print(f"Using device: {device}")

def get_valid_depth(depth_image, x, y, depth_scale):
    """ Extracts a valid depth value from a small region around (x, y) to improve accuracy. """
    kernel_size = 7  # Increase sampling area for better accuracy
    half_k = kernel_size // 2

    x_min, x_max = max(0, x - half_k), min(depth_image.shape[1] - 1, x + half_k)
    y_min, y_max = max(0, y - half_k), min(depth_image.shape[0] - 1, y + half_k)

    depth_region = depth_image[y_min:y_max, x_min:x_max].flatten()
    valid_depths = depth_region[(depth_region > 0) & (depth_region < 10000)]

    if len(valid_depths) > 0:
        median_depth = np.median(valid_depths) * depth_scale
        return median_depth
    return -1  # Invalid depth

def detect_objects_in_image(image_path, depth_image_path, depth_scale=0.001):
    """ Detect objects using YOLO on a single image with depth data."""
    # Read images
    image = cv.imread(image_path)
    depth_image = cv.imread(depth_image_path, cv.IMREAD_UNCHANGED)
    
    if image is None or depth_image is None:
        print("[ERROR] Unable to load image or depth data. Check the paths.")
        return
    
    # Run YOLO detection
    results = model(image)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            depth = get_valid_depth(depth_image, center_x, center_y, depth_scale)

            if depth > 0:
                Xtarget = depth * (center_x - image.shape[1] // 2) / 1000  # Approximate intrinsics
                Ytarget = depth * (center_y - image.shape[0] // 2) / 1000
                Ztarget = depth
                distance = np.sqrt(Xtarget ** 2 + Ytarget ** 2 + Ztarget ** 2)
                
                label = f"Object {cls} ({conf:.2f}) | Dist: {distance:.2f}m"
            else:
                label = f"Object {cls} ({conf:.2f}) | Dist: N/A"
            
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display result
    cv.imshow("YOLO Object Detection with Depth", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Update with your image path
    depth_image_path = "path/to/your/depth_image.png"  # Update with depth image path
    detect_objects_in_image(image_path, depth_image_path)
