import numpy as np
import cv2

image = cv2.imread('snake.jpg',0)
image_color = cv2.imread('snake.jpg',1)

# Fast line detection
fld = cv2.ximgproc.createFastLineDetector()
lines = fld.detect(image)
image_lines = fld.drawSegments(image, lines)

pts = lines[:,0,:2].astype(np.int32) # First points of every line
for pt in pts:
    cv2.circle(image_color,(pt[0],pt[1]),5,(255, 0, 0))

rect = cv2.boundingRect(pts)
print(rect)

subdiv = cv2.Subdiv2D(rect)
subdiv.insert(pts.tolist())
triangles = subdiv.getTriangleList().astype(np.int32)

for triangle in triangles:
    cv2.polylines(image_color, [triangle.reshape((3,2))], True, (255,0,0))

cv2.imshow('Snake Lines',image_lines)
cv2.imshow('Snake',image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
