import numpy as np
import cv2

# Start video capture (0 for default camera, change to 1 or 2 for external cameras)
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow("Circle Detection")

# Create trackbars for tuning parameters
cv2.createTrackbar("Param1", "Circle Detection", 200, 500, lambda x: None)
cv2.createTrackbar("Param2", "Circle Detection", 20, 100, lambda x: None)
cv2.createTrackbar("Min Radius", "Circle Detection", 10, 100, lambda x: None)
cv2.createTrackbar("Max Radius", "Circle Detection", 150, 500, lambda x: None)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    output = frame.copy()

    # Convert to grayscale and apply Gaussian blur for noise reduction
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Get current values of trackbars
    param1 = cv2.getTrackbarPos("Param1", "Circle Detection")
    param2 = cv2.getTrackbarPos("Param2", "Circle Detection")
    minRadius = cv2.getTrackbarPos("Min Radius", "Circle Detection")
    maxRadius = cv2.getTrackbarPos("Max Radius", "Circle Detection")

    # Detect circles using HoughCircles with the current parameter values
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2,
        param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius
    )

    # Draw detected circles
    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        for (x, y, r) in detected_circles[0, :]:
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)  # Outer circle
            cv2.circle(output, (x, y), 2, (0, 255, 0), 3)  # Center dot


            # Display the radius value near the detected circle
            text = f"Radius: {r}"
            cv2.putText(output, text, (x - 50, y - r - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
             # Print the detected circle radius in the terminal
            print(f"Circle detected at (X: {x}, Y: {y}) with Radius: {r}")
    # Show output frame
    cv2.imshow('Circle Detection', output)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
