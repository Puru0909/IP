import numpy as np
import cv2

# Start video capture
cap = cv2.VideoCapture(0)

# Create a window
cv2.namedWindow("HSV Tuning")

# Create trackbars for tuning HSV values
cv2.createTrackbar("Lower H", "HSV Tuning", 90, 180, lambda x: None)  # Hue range [0, 180]
cv2.createTrackbar("Lower S", "HSV Tuning", 50, 255, lambda x: None)  # Saturation range [0, 255]
cv2.createTrackbar("Lower V", "HSV Tuning", 50, 255, lambda x: None)  # Value range [0, 255]
cv2.createTrackbar("Upper H", "HSV Tuning", 130, 180, lambda x: None)
cv2.createTrackbar("Upper S", "HSV Tuning", 255, 255, lambda x: None)
cv2.createTrackbar("Upper V", "HSV Tuning", 255, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Get the width and height of the frame
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Get current values from trackbars
    lower_hue = cv2.getTrackbarPos("Lower H", "HSV Tuning")
    lower_saturation = cv2.getTrackbarPos("Lower S", "HSV Tuning")
    lower_value = cv2.getTrackbarPos("Lower V", "HSV Tuning")
    upper_hue = cv2.getTrackbarPos("Upper H", "HSV Tuning")
    upper_saturation = cv2.getTrackbarPos("Upper S", "HSV Tuning")
    upper_value = cv2.getTrackbarPos("Upper V", "HSV Tuning")

    # Define the lower and upper HSV range based on trackbar values
    lower_blue = np.array([lower_hue, lower_saturation, lower_value])
    upper_blue = np.array([upper_hue, upper_saturation, upper_value])

    # Convert the frame to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create the mask based on the selected range
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND the mask with the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the result
    cv2.imshow('HSV Tuning', result)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
