import math
import cv2
import cvzone
from cvzone.ColorModule import ColorFinder
import numpy as np

# Initialize the Camera
cap = cv2.VideoCapture(0)  # Use webcam (change to 1 or 2 if using an external camera)

if not cap.isOpened():
    print("Error: Could not access the camera")
    exit()

# Create the color Finder object
myColorFinder = ColorFinder(False)

# Default HSV values for tuning
hsvVals = {'hmin': 8, 'smin': 96, 'vmin': 115, 'hmax': 14, 'smax': 255, 'vmax': 255}

# Trackbar callback function
def nothing(x):
    pass

# Create a UI window with trackbars for parameter tuning
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 400, 300)

cv2.createTrackbar("H Min", "Trackbars", hsvVals['hmin'], 179, nothing)
cv2.createTrackbar("S Min", "Trackbars", hsvVals['smin'], 255, nothing)
cv2.createTrackbar("V Min", "Trackbars", hsvVals['vmin'], 255, nothing)
cv2.createTrackbar("H Max", "Trackbars", hsvVals['hmax'], 179, nothing)
cv2.createTrackbar("S Max", "Trackbars", hsvVals['smax'], 255, nothing)
cv2.createTrackbar("V Max", "Trackbars", hsvVals['vmax'], 255, nothing)

# Variables
posListX, posListY = [], []
xList = [item for item in range(0, 1300)]
prediction = False

while True:
    # Grab the image
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Resize for better performance (optional)
    img = cv2.resize(img, (640, 480))
    
    # Update HSV values from trackbars
    hsvVals['hmin'] = cv2.getTrackbarPos("H Min", "Trackbars")
    hsvVals['smin'] = cv2.getTrackbarPos("S Min", "Trackbars")
    hsvVals['vmin'] = cv2.getTrackbarPos("V Min", "Trackbars")
    hsvVals['hmax'] = cv2.getTrackbarPos("H Max", "Trackbars")
    hsvVals['smax'] = cv2.getTrackbarPos("S Max", "Trackbars")
    hsvVals['vmax'] = cv2.getTrackbarPos("V Max", "Trackbars")

    # Process the image
    imgColor, mask = myColorFinder.update(img, hsvVals)
    imgContours, contours = cvzone.findContours(img, mask, minArea=500)

    if contours:
        posListX.append(contours[0]['center'][0])
        posListY.append(contours[0]['center'][1])

    if posListX:
        # Polynomial Regression y = Ax^2 + Bx + C
        A, B, C = np.polyfit(posListX, posListY, 2)

        for i, (posX, posY) in enumerate(zip(posListX, posListY)):
            cv2.circle(imgContours, (posX, posY), 10, (0, 255, 0), cv2.FILLED)
            if i > 0:
                cv2.line(imgContours, (posX, posY), (posListX[i - 1], posListY[i - 1]), (0, 255, 0), 5)

        for x in xList:
            y = int(A * x ** 2 + B * x + C)
            if 0 <= y < imgContours.shape[0]:  # Ensure points are within frame
                cv2.circle(imgContours, (x, y), 2, (255, 0, 255), cv2.FILLED)

        if len(posListX) < 10:
            # Prediction if the ball will go into the basket (X values between 330-430 at Y=590)
            a, b, c = A, B, C - 590
            discriminant = b ** 2 - (4 * a * c)
            if discriminant >= 0:
                x_pred = int((-b - math.sqrt(discriminant)) / (2 * a))
                prediction = 330 < x_pred < 430

        if prediction:
            cvzone.putTextRect(imgContours, "Basket", (50, 150), scale=5, thickness=5, colorR=(0, 200, 0), offset=20)
        else:
            cvzone.putTextRect(imgContours, "No Basket", (50, 150), scale=5, thickness=5, colorR=(0, 0, 200), offset=20)

    # Show the images
    cv2.imshow("ImageColor", imgContours)
    cv2.imshow("Mask", mask)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
