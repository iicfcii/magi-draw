import sys
sys.path.append(sys.path[0] + "/..")
import numpy as np
import cv2

import animator.ar as ar
import animator.animation as animation

from ball.ball_bones import *
from ball.ball_animator import *
from ball.ball_model import *

# Photo of scene
img = cv2.imread('img/ball_game_1.jpg')
# cv2.imshow('Source', img)
# cv2.waitKey(0)

# Get drawing
mat = ar.homography(img, CORNERS_REF)
img_drawing = ar.drawing(img, mat, BALL_DRAW_REF)
# cv2.imshow('Drawing', img_drawing)
# cv2.waitKey(0)

img_tmp = img_drawing.copy()
for bone in params2bones(DEFAULT_PARAMS):
    bone = bone.astype(np.int32)
    cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
    cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
    cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
cv2.imshow('Default bones',img_tmp)
cv2.waitKey(0)

animator = BallAnimator(img_drawing, BallModel(), params2bones(DEFAULT_PARAMS))

# img_tmp = animator.drawing.copy()
# for triangle in animator.triangles:
#     cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
#     cv2.imshow('Triangulation', img_tmp)
#     cv2.waitKey(10)

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
        val = 255*animator.weights[point_key]['weight'][i]
        cv2.circle(img_tmp, point_key, 2, (0,val,0), thickness=-1)
    cv2.imshow('Weights' + str(i),img_tmp)
cv2.waitKey(0)

# animator.model.look='happy'

while True:
    animator.update()
    img_frame, anchor_frame, mask_frame = animator.current_frame
    # Make anchor point fixed
    position = (int(BOARD_REF[0,0]-anchor_frame[0]+200),int(BOARD_REF[0,1]-anchor_frame[1]+100))

    frame_tmp = ar.render(img.copy(), img_frame, mask_frame, position, mat)
    cv2.imshow('Frame',frame_tmp)
    cv2.waitKey(0)

cv2.destroyAllWindows()
