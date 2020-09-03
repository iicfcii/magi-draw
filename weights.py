import numpy as np
import cv2
import triangulation
import animation
import snake
import ar
import time

# Photo of scene
img = cv2.imread('img/snake_game_1.jpg')
cv2.imshow('Source', img)
cv2.waitKey(0)

# Get drawing
M = ar.findHomography(img)
img_drawing = ar.getDrawing(img, M)
snake_animator = snake.SnakeAnimator(img_drawing)

img_tmp = snake_animator.drawing.copy()
for triangle in snake_animator.triangles:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
cv2.imshow('Triangulation', img_tmp)
cv2.waitKey(0)

for i in range(len(snake_animator.bones_default)):
    img_tmp = snake_animator.drawing.copy()
    for triangle in snake_animator.triangles:
        cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
    for bone in snake_animator.bones_default:
        bone = bone.astype(np.int32)
        cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
        cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
        cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
    for point_key in snake_animator.weights.keys():
        val = 255*snake_animator.weights[point_key]['weight'][i]
        cv2.circle(img_tmp, point_key, 2, (0,val,0), thickness=-1)
    cv2.imshow('Weights' + str(i),img_tmp)

cv2.waitKey(0)
cv2.destroyAllWindows()
