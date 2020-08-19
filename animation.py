import numpy as np
import cv2
import triangulation
import snake

img_color = cv2.imread('snake.jpg', 1)
triangles = triangulation.triangulate(img_color)
bones_frames = snake.bones_frames

for triangle in triangles:
    cv2.polylines(img_color, [triangle.astype(np.int32)], True, (0,0,255))
cv2.imshow('color',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
