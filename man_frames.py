import numpy as np
import cv2
import triangulation
import animation
import man

# Prepare source image, triangulation, and default bones
img_src = man.img.copy() # Source image
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
bones_default = man.bones_default

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

triangles = triangulation.constrain(contour, triangles_unconstrained, edges, img_src)
weights = animation.calcWeights(bones_default,triangles)

for i in range(len(man.bones_frames)):
    bones_n = man.bones_frames[i]
    triangles_next = animation.animate(bones_default,bones_n,triangles,weights)
    img_n, anchor = animation.warp(img_src, triangles, triangles_next,bones_n[0])

    # for bone in bones_n:
    #     bone_offset = (bone-np.tile(bones_n[0,0:2],2)+np.tile(anchor, 2)).astype(np.int32)
    #     cv2.polylines(img_n, [bone_offset.reshape((2,2))], True, (255,0,0), 2)
    #     cv2.circle(img_n, tuple(bone_offset[0:2]), 5, (0,0,0), thickness=-1)
    #     cv2.circle(img_n, tuple(bone_offset[2:4]), 5, (0,0,0), thickness=-1)
    # for triangle in triangles_next:
    #     triangle = triangle-np.tile(np.array([-anchor[0]+bones_n[0,0],-anchor[1]+bones_n[0,1]]),(3,1))
    #     cv2.polylines(img_n, [triangle.astype(np.int32)], True, (0,0,255))
    #     cv2.circle(img_n, tuple(anchor), 5, (0,0,0), thickness=-1)
    cv2.imshow('Frame',img_n)
    cv2.waitKey(0)
    cv2.destroyWindow('Frame')

cv2.destroyAllWindows()
