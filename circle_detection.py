# import cv2
# import numpy as np

# # Load the image
# img = cv2.imread('C:\\Users\\Dell\\Pictures\\balls.jpg')
# output = img.copy()

# # Convert to grayscale and apply Gaussian blur for noise reduction
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# gray = cv2.GaussianBlur(gray, (9, 9), 2)

# # Function to update parameters dynamically
# def update_hough(_):
#     # Get trackbar positions
#     dp = cv2.getTrackbarPos('dp', 'Hough Circle Tuner') / 10.0  # Divide by 10 to get decimal values
#     min_dist = cv2.getTrackbarPos('minDist', 'Hough Circle Tuner')
#     param1 = cv2.getTrackbarPos('param1', 'Hough Circle Tuner')
#     param2 = cv2.getTrackbarPos('param2', 'Hough Circle Tuner')
#     min_radius = cv2.getTrackbarPos('minRadius', 'Hough Circle Tuner')
#     max_radius = cv2.getTrackbarPos('maxRadius', 'Hough Circle Tuner')

#     # Clone original image for drawing
#     output = img.copy()

#     # Detect circles using updated parameters
#     circles = cv2.HoughCircles(
#         gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
#         param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius
#     )

#     # Draw detected circles
#     if circles is not None:
#         detected_circles = np.uint16(np.around(circles))
#         for (x, y, r) in detected_circles[0, :]:
#             cv2.circle(output, (x, y), r, (0, 255, 0), 3)  # Outer circle
#             cv2.circle(output, (x, y), 3, (0, 255, 0), 3)  # Center dot
#     # Display radius as text on the image
#             text = f"Radius: {r}"
#             cv2.putText(output, text, (x - 50, y - r - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

#             # Print radius in the terminal
#             print(f"Circle detected at (X: {x}, Y: {y}) with Radius: {r}")


#     # Display updated output
#     cv2.imshow('output', output)

# # Create a window
# cv2.namedWindow('Hough Circle Tuner')

# # Create trackbars for tuning parameters
# cv2.createTrackbar('dp', 'Hough Circle Tuner', 10, 20, update_hough)  # Multiplied by 10 to allow decimal values
# cv2.createTrackbar('minDist', 'Hough Circle Tuner', 20, 200, update_hough)
# cv2.createTrackbar('param1', 'Hough Circle Tuner', 317, 500, update_hough)
# cv2.createTrackbar('param2', 'Hough Circle Tuner', 30, 100, update_hough)
# cv2.createTrackbar('minRadius', 'Hough Circle Tuner', 30, 200, update_hough)
# cv2.createTrackbar('maxRadius', 'Hough Circle Tuner', 150, 300, update_hough)

# # Show initial output
# update_hough(0)

# # Wait until user closes the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
import cv2
import numpy as np

# Load the image
img = cv2.imread('C:\\Users\\Dell\\Downloads\\img_6.webp')
output = img.copy()

# Convert to grayscale and apply Gaussian blur for noise reduction
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# Function to update parameters dynamically
def update_hough(_):
    # Get trackbar positions
    canny_threshold1 = cv2.getTrackbarPos('Canny Threshold 1', 'Basketball Ring Detection')
    canny_threshold2 = cv2.getTrackbarPos('Canny Threshold 2', 'Basketball Ring Detection')
    min_radius = cv2.getTrackbarPos('Min Radius', 'Basketball Ring Detection')
    max_radius = cv2.getTrackbarPos('Max Radius', 'Basketball Ring Detection')
    aspect_ratio_min = cv2.getTrackbarPos('Aspect Ratio Min', 'Basketball Ring Detection') / 100.0
    aspect_ratio_max = cv2.getTrackbarPos('Aspect Ratio Max', 'Basketball Ring Detection') / 100.0

    # Use Canny edge detection with updated thresholds
    edges = cv2.Canny(gray, canny_threshold1, canny_threshold2)

    # Clear the output image
    output = img.copy()

    # Detect circles using Hough Circle Transform
    circles = cv2.HoughCircles(
        edges, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
        param1=50, param2=30, minRadius=min_radius, maxRadius=max_radius
    )
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            # Draw the circle and its center
            cv2.circle(output, (x, y), r, (0, 255, 0), 3)
            cv2.circle(output, (x, y), 2, (0, 255, 0), 3)
            text = f"Circle: Radius={r}"
            cv2.putText(output, text, (x - 50, y - r - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            print(f"Circle detected at (X: {x}, Y: {y}) with Radius: {r}")
            
    # Detect ellipses using contour-based ellipse fitting
    _, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if len(contour) < 5:
            continue
        ellipse = cv2.fitEllipse(contour)
        (x, y), (major_axis, minor_axis), angle = ellipse
        aspect_ratio = major_axis / minor_axis
        if aspect_ratio_min <= aspect_ratio <= aspect_ratio_max and min_radius <= major_axis <= max_radius:
            cv2.ellipse(output, ellipse, (255, 0, 0), 2)
            text = f"Ellipse: Major={major_axis:.1f}, Minor={minor_axis:.1f}, Angle={angle:.1f}"
            cv2.putText(output, text, (int(x) - 100, int(y) - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            print(f"Ellipse detected at (X: {x:.1f}, Y: {y:.1f}) with Major Axis: {major_axis:.1f}, "
                  f"Minor Axis: {minor_axis:.1f}, Angle: {angle:.1f}")

    # Display the output image
    cv2.imshow('Basketball Ring Detection', output)

# Create a window for the UI
cv2.namedWindow('Basketball Ring Detection')

# Create trackbars for parameter tuning
cv2.createTrackbar('Canny Threshold 1', 'Basketball Ring Detection', 50, 500, update_hough)
cv2.createTrackbar('Canny Threshold 2', 'Basketball Ring Detection', 150, 500, update_hough)
cv2.createTrackbar('Min Radius', 'Basketball Ring Detection', 50, 200, update_hough)
cv2.createTrackbar('Max Radius', 'Basketball Ring Detection', 150, 300, update_hough)
cv2.createTrackbar('Aspect Ratio Min', 'Basketball Ring Detection', 70, 100, update_hough)
cv2.createTrackbar('Aspect Ratio Max', 'Basketball Ring Detection', 130, 200, update_hough)

# Initialize the detection with default parameters
update_hough(0)

# Wait until user closes the window
cv2.waitKey(0)
cv2.destroyAllWindows()
