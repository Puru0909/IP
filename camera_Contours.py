import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame")
        break
    
    # Resize frame (optional)
    frame = cv2.resize(frame, None, fx=0.9, fy=0.9)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply threshold to get binary image
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find contours
    contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Print number of contours found
    print("Number of contours: ", len(contours))

    # Draw contours on a copy of the original frame
    frame_copy = frame.copy()
    cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), thickness=2)

    # Display the results
    cv2.imshow('Original Frame', frame)
    cv2.imshow('Grayscale Image', gray)
    cv2.imshow('Binary Image', binary)
    cv2.imshow('Contours', frame_copy)

    # Wait for the 'Esc' key to break the loop
    if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value of the Esc key
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
# import cv2
# import numpy as np

# # Initialize the camera
# cap = cv2.VideoCapture(0)  # 0 is the default camera, change if you have multiple cameras

# # Create a window to hold the trackbars for tuning
# cv2.namedWindow("Contours")

# # Create trackbars for tuning threshold and contour thickness
# cv2.createTrackbar("Threshold", "Contours", 0, 255, lambda x: None)  # Range for threshold [0, 255]
# cv2.createTrackbar("Thickness", "Contours", 2, 10, lambda x: None)  # Range for contour thickness [1, 10]

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
    
#     if not ret:
#         print("Failed to grab frame")
#         break
    
#     # Resize frame (optional)
#     frame = cv2.resize(frame, None, fx=0.9, fy=0.9)

#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Get the current threshold and contour thickness from the trackbars
#     threshold_value = cv2.getTrackbarPos("Threshold", "Contours")
#     contour_thickness = cv2.getTrackbarPos("Thickness", "Contours")

#     # Apply threshold to get binary image
#     ret, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

#     # Find contours
#     contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

#     # Print number of contours found
#     print("Number of contours: ", len(contours))

#     # Draw contours on a copy of the original frame
#     frame_copy = frame.copy()
#     cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), thickness=contour_thickness)

#     # Display the contour-detected frame
#     cv2.imshow('Contours', frame_copy)

#     # Wait for the 'Esc' key to break the loop
#     if cv2.waitKey(1) & 0xFF == 27:  # 27 is the ASCII value of the Esc key
#         break

# # Release the camera and close windows
# cap.release()
# cv2.destroyAllWindows()
