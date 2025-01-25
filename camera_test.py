import numpy as np # type: ignore
import cv2 # type: ignore

cap = cv2.VideoCapture(0)

while True:
   ret, frame = cap.read()

   cv2.imshow('frame', frame)

   if cv2.waitKey(1) == ord('q'):
     break
cap.release()
cv2.destroyAllWindows()
