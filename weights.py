import numpy as np
import cv2
import triangulation
import animation
import snake
import ar

# Photo of scene
img = cv2.imread('img/snake_game_3.jpg')
# cv2.imshow('Source', img)
# cv2.waitKey(0)

# Get drawing
mat = ar.findHomography(img, snake.CORNERS_REF)
img_drawing = ar.getDrawing(img, mat, snake.DRAW_REF)
img_drawing = cv2.rotate(img_drawing, cv2.ROTATE_90_CLOCKWISE)
# cv2.imshow('Drawing', img_drawing)
# cv2.waitKey(0)

snake_animator = snake.SnakeAnimator(img_drawing, snake.SnakeModel())

img_tmp = snake_animator.drawing.copy()
for triangle in snake_animator.triangles:
    cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
# cv2.imshow('Triangulation', img_tmp)
# cv2.waitKey(0)

for i in range(len(snake_animator.default)):
    img_tmp = snake_animator.drawing.copy()
    for triangle in snake_animator.triangles:
        cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
    for bone in snake_animator.default:
        bone = bone.astype(np.int32)
        cv2.polylines(img_tmp, [bone.reshape((2,2))], True, (255,0,0), 2)
        cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
        cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
    for point_key in snake_animator.weights.keys():
        val = 255*snake_animator.weights[point_key]['weight'][i]
        cv2.circle(img_tmp, point_key, 2, (0,val,0), thickness=-1)
    cv2.imshow('Weights' + str(i),img_tmp)
cv2.waitKey(0)

for frame in snake_animator.turn_right.frames:
    img_frame, anchor_frame, mask_frame = frame
    # Make anchor point fixed
    position = (int(snake.BOARD_REF[0,0]),int(snake.BOARD_REF[0,1]))

    frame_tmp = ar.render(img.copy(), img_frame, mask_frame, position, mat)
    cv2.imshow('Frame',frame_tmp)
    cv2.waitKey(0)

cv2.destroyAllWindows()
