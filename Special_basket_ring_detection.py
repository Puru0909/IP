import cv2
import numpy as np

# Open the camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Create windows
cv2.namedWindow("Basketball Detection")

# Create trackbars for tuning circle detection parameters
cv2.createTrackbar("Param1", "Basketball Detection", 200, 500, lambda x: None)
cv2.createTrackbar("Param2", "Basketball Detection", 20, 100, lambda x: None)
cv2.createTrackbar("Min Radius", "Basketball Detection", 10, 100, lambda x: None)
cv2.createTrackbar("Max Radius", "Basketball Detection", 150, 500, lambda x: None)

# Create trackbars for edge detection
cv2.createTrackbar("Threshold1", "Basketball Detection", 150, 500, lambda x: None)
cv2.createTrackbar("Threshold2", "Basketball Detection", 400, 500, lambda x: None)

# Define the range for red color in HSV
# Red is tricky because it spans both ends of the HSV spectrum
lower_red1 = np.array([0, 120, 70])    # Lower range for red
upper_red1 = np.array([10, 255, 255])  # Upper range for red
lower_red2 = np.array([170, 120, 70])  # Lower range for red (wrap-around)  
upper_red2 = np.array([180, 255, 255]) # Upper range for red (wrap-around)


while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create masks for red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)  # Mask for the first range of red
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)  # Mask for the second range of red
    red_mask = cv2.bitwise_or(mask1, mask2)           # Combine both masks

    # Apply the red mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=red_mask)

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
            # Draw circle
            cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
            # Draw center
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 3)
            # Display radius and center coordinates
            cv2.putText(frame, f"Red Circle: ({x}, {y}) R: {r}", (x - 50, y - r - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Get trackbar values for edge detection
    threshold1 = cv2.getTrackbarPos("Threshold1", "Basketball Detection")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Basketball Detection")

    # Apply Canny edge detection
    edges = cv2.Canny(morphed, threshold1, threshold2)
    cv2.imshow("Canny Edges", edges)

    # Apply contour detection on the original frame (for backboard detection)
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_original = cv2.GaussianBlur(gray_original, (9, 9), 2)
    morphed_original = cv2.morphologyEx(blurred_original, cv2.MORPH_OPEN, kernel)
    ret, binary = cv2.threshold(morphed_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Loop through contours to find the backboard
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Check if the contour has 4 vertices (likely a rectangle)
        if len(approx) == 4:
            # Get the bounding box of the rectangle
            x, y, w, h = cv2.boundingRect(approx)
            # Filter out small bounding boxes (noise)
            if w * h > 1000:  # Adjust this threshold as needed
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                # Display the coordinates of the bounding box
                cv2.putText(frame, f"Backboard: ({x}, {y}) to ({x + w}, {y + h})", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show the final output
    cv2.imshow('Basketball Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

