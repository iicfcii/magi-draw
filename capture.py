import cv2
import numpy as np
import ar
import triangulation
import animation
import snake

import time

cam = cv2.VideoCapture(0)


while True:
    ret, img = cam.read()

    if not ret: continue

    M = ar.findHomography(img)
    print(M)
    cv2.imshow("Snake Game", img)
    cv2.waitKey(10)


cam.release()

cv2.destroyAllWindows()
