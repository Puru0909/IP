import cv2
import numpy

image = cv2.imread('C:\\Users\\Dell\\Pictures\\balls.jpg')
image = cv2.resize(image, None, fx=0.9, fy=0.9)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

contour , hierarchy =cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

print("Length of contours{}".format(len(contour)))
print(contour)

image_copy = image.copy()
image_copy = cv2.drawContours(image_copy, contour, -1, (0, 255, 0), thickness=2)

cv2.imshow('Grayscale Image', gray)
cv2.imshow('Drawn contour', image_copy)
cv2.imshow('Binary Image', binary)

cv2.waitKey(0)
cv2.destroyAllWindows()