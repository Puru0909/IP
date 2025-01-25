import cv2

# Open the camera (0 is the default camera, you can change it if needed)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create a window
cv2.namedWindow("Edge Detection")

# Create trackbars for tuning Canny edge detection parameters
cv2.createTrackbar("Threshold1", "Edge Detection", 150, 500, lambda x: None)
cv2.createTrackbar("Threshold2", "Edge Detection", 400, 500, lambda x: None)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame reading is successful
    if ret:
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Get the current values of the thresholds from the trackbars
        threshold1 = cv2.getTrackbarPos("Threshold1", "Edge Detection")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Edge Detection")

        # Apply Canny edge detection with the selected thresholds
        canny_edges = cv2.Canny(gray_frame, threshold1, threshold2)

        # Display the original frame
        cv2.imshow('Original Camera Feed', frame)

        # Display the edge-detected frame
        cv2.imshow('Canny Edges', canny_edges)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
