import numpy as np
import cv2
import triangulation
import animation
import man
import time

# Prepare source image, triangulation, and default bones
img_src = man.img.copy() # Source image
bones_default = man.bones_default

img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
contour = triangulation.contour(img_gray)
keypoints = triangulation.keypoints_uniform(img_gray, contour)
triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)

img_tmp = img_src.copy()
for triangle in triangles_unconstrained:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
for point in contour:
    cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (255,0,0), thickness=-1)
for point in keypoints:
    cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (0,255,0), thickness=-1)
cv2.imshow('Triangulation',img_tmp)
cv2.waitKey(0)

start_time = time.time()
triangles = triangulation.constrain(contour, triangles_unconstrained, edges, img_src)
constrain_time = time.time()
print(constrain_time - start_time)
weights = animation.calcWeights(bones_default,triangles)
weights_time = time.time()
print(weights_time - constrain_time)

for i in range(len(bones_default)):
    img_tmp = img_src.copy()
    for triangle in triangles:
        cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
    for bone in bones_default:
        bone = bone.astype(np.int32)
        cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
        cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
        cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
    for point_key in weights.keys():
        val = 255*weights[point_key]['weight'][i]
        cv2.circle(img_tmp, point_key, 2, (0,val,0), thickness=-1)
    cv2.imshow('Weights' + str(i),img_tmp)

cv2.waitKey(0)
cv2.destroyAllWindows()
