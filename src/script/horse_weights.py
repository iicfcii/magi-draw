import sys
sys.path.append(sys.path[0] + "/..")
import numpy as np
import cv2

import animator.ar as ar

from horse.horse_bones import *
from horse.horse_animator import *

# Photo of scene
img = cv2.imread('img/horse_game_2.jpg')
# cv2.imshow('Source', img)
# cv2.waitKey(0)

# Get drawing
mat = ar.homography(img, CORNERS_REF)
img_drawing = ar.drawing(img, mat, HORSE_DRAW_REF)
# cv2.imshow('Drawing', img_drawing)
# cv2.waitKey(0)

# img_tmp = img_drawing.copy()
# for bone in params2bones(DEFAULT_PARAMS):
#     bone = bone.astype(np.int32)
#     cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
#     cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
#     cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
# cv2.imshow('Default bones',img_tmp)
# cv2.waitKey(0)

animator = HorseAnimator(img_drawing, None, params2bones(DEFAULT_PARAMS))

# img_tmp = animator.drawing.copy()
# for triangle in animator.triangles:
#     cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
# cv2.imshow('Triangulation', img_tmp)
# cv2.waitKey(0)

for i in range(len(animator.bones)):
    img_tmp = animator.drawing.copy()
    for triangle in animator.triangles:
        cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
    for bone in animator.bones:
        bone = bone.astype(np.int32)
        cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
        cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
        cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
    for point_key in animator.weights.keys():
        if np.argmax(animator.weights[point_key]['weight']) == i:
            size = 3
        else:
            size = 1
        val = 255*animator.weights[point_key]['weight'][i]
        cv2.circle(img_tmp, point_key, size, (0,val,0), thickness=-1)
    cv2.imshow('Weights' + str(i),img_tmp)
cv2.waitKey(0)

while True:
    for frame in animator.test.frames:
        img_frame, anchor_frame, mask_frame = frame
        # Make anchor point fixed
        position = (int(BOARD_REF[0,0]),int(BOARD_REF[0,1]))

        frame_tmp = ar.render(img.copy(), img_frame, mask_frame, position, mat)
        cv2.imshow('Frame',frame_tmp)
        cv2.waitKey(0)

cv2.destroyAllWindows()
