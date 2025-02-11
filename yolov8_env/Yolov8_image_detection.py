# Import required libraries
from ultralytics import YOLO  # YOLO class for loading the model
import matplotlib.pyplot as plt  # Library for displaying the image
import cv2  # OpenCV for image processing

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Load the pre-trained YOLOv8-small model

# Load the image
image_path = "C:\\Users\\Dell\\Downloads\\img_6.webp"  # Specify the path to the input image
image = cv2.imread(image_path)  # Read the image using OpenCV

# Check if the image is loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: The image at {image_path} could not be loaded. Check the file path.")

# Run inference on the image
results = model(image)  # Perform object detection using YOLOv8

# Process and display results
for result in results:
    annotated_img = result.plot()  # Generate an image with detected objects

# Convert BGR to RGB for correct color representation in Matplotlib
plt.imshow(annotated_img[..., ::-1])  # OpenCV loads images in BGR format; converting to RGB
plt.axis("off")  # Hide axes for better visualization
plt.show()  # Display the annotated image