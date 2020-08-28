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
keypoints = triangulation.keypoints_uniform(img_gray, contour)

triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)
triangles = triangulation.constrain(contour, triangles_unconstrained, edges, img_src)
bones_default = snake.bones_default

# for point_key in weights.keys():
#     val = 255*weights[point_key]['weight'][4]
#     cv2.circle(img_tmp, point_key, 2, (0,val,0), thickness=-1)

# for triangle in triangles:
#     path = animation.findPath(triangle[0], bones_default[0], triangles)
#     img_tmp = img_src.copy()
#     for triangle in triangles:
#         cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
#     for bone in bones_default:
#         bone = bone.astype(np.int32)
#         cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
#         cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
#         cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
#     cv2.polylines(img_tmp, [path.astype(np.int32)], False, (0,255,0), 2)
#     cv2.imshow('Path',img_tmp)
#     cv2.waitKey(0)
