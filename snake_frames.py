import numpy as np
import cv2
import triangulation
import animation
import snake

# Prepare source image, triangulation, and default bones
img_src = snake.img.copy() # Source image
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

img_tmp = img_src.copy()
contour = triangulation.contour(img_gray)
# keypoints = triangulation.keypoints(img_gray, contour)
keypoints = triangulation.keypoints_uniform(img_gray, contour)
# img_tmp = img_src.copy()
# for point in contour:
#     cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (255,0,0), thickness=-1)
# for point in keypoints:
#     cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (0,0,0), thickness=-1)
# cv2.imshow('Key points',img_tmp)
# cv2.waitKey(0)

triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)
img_tmp = img_src.copy()
for triangle in triangles_unconstrained:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
for point in contour:
    cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (255,0,0), thickness=-1)
cv2.imshow('Triangulation',img_tmp)
cv2.waitKey(0)

triangles = triangulation.constrain(contour, triangles_unconstrained, edges, img_src)
img_tmp = img_src.copy()
for triangle in triangles:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
bones_default = snake.bones_default
for bone in bones_default:
    cv2.polylines(img_tmp, [bone.reshape((2,2)).astype(np.int32)], True, (255,0,0), 2)
cv2.imshow('Constrained Triangulation with Bones',img_tmp)
cv2.waitKey(0)

for i in range(len(snake.bones_frames)):
    bones_n = snake.bones_frames[i]
    triangles_next = animation.animate(bones_default,bones_n,triangles)

    # img_tmp = img_src.copy()
    # for triangle in triangles_next:
    #     cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
    # cv2.imshow('Frame',img_tmp)
    # cv2.waitKey(0)

    img_n = animation.warp(img_src, triangles, triangles_next)
    cv2.imshow('Frame',img_n)
    cv2.waitKey(0)

cv2.destroyAllWindows()
