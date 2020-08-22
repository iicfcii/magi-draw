import numpy as np
import cv2
import triangulation
import animation
import snake_test

# Prepare source image, triangulation, and default bones
img_src = snake_test.img_src.copy() # Source image
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

img_tmp = img_src.copy()
contour = triangulation.contour(img_gray)
cv2.polylines(img_tmp, [contour.astype(np.int32)], True, (0,0,255))
cv2.imshow('Contour',img_tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()

keypoints = triangulation.keypoints(img_gray, contour)
triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)
img_tmp = img_src.copy()
for triangle in triangles_unconstrained:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
cv2.imshow('Triangulation',img_tmp)
cv2.waitKey(0)

triangles = triangulation.constrain(contour, triangles_unconstrained, edges)
img_tmp = img_src.copy()
for triangle in triangles:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
bones_c = snake_test.bones_frames[0]
for bone in bones_c:
    cv2.polylines(img_tmp, [bone.reshape((2,2)).astype(np.int32)], True, (255,0,0), 2)
cv2.imshow('Constrained Triangulation with Bones',img_tmp)
cv2.waitKey(0)

for i in range(len(snake_test.bones_frames)):
    bones_n = snake_test.bones_frames[i]
    triangles_next = animation.animate(bones_c,bones_n,triangles)
    img_n = animation.warp(img_src, triangles, triangles_next)

    cv2.imshow('Frame',img_n)
    cv2.waitKey(0)

cv2.destroyAllWindows()
