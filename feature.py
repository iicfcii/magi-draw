import numpy as np
import cv2

img_color = cv2.imread('snake.jpg', 1)
img_gray = cv2.imread('snake.jpg',0)

fast = cv2.FastFeatureDetector_create(threshold=10)
kps = fast.detect(img_gray,None)

# kps_best = [kp for kp in kps if kp.response>150]
for kp in kps:
    print(kp.size, kp.response, kp.angle)

img_features = cv2.drawKeypoints(img_gray, kps, None, color=(255,0,0))

cv2.imshow('Snake Features', img_features)
cv2.waitKey(0)
cv2.destroyAllWindows()
