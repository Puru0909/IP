import numpy as np
import cv2 as cv
import torch
from ultralytics import YOLO

# Load YOLO model with device selection
device = "cuda" if torch.cuda.is_available() else "cpu"
model = YOLO("best.pt").to(device)
print(f"Using device: {device}")

def detect_objects_in_image(image_path):
    """ Detect objects using YOLO on a single image."""
    # Read image
    image = cv.imread(image_path)
    if image is None:
        print("[ERROR] Unable to load image. Check the path.")
        return
    
    # Run YOLO detection
    results = model(image)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            # Draw bounding box
            cv.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Object {cls} ({conf:.2f})"
            cv.putText(image, label, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display result
    cv.imshow("YOLO Object Detection", image)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"  # Update with your image path
    detect_objects_in_image(image_path)
