import cv2
import numpy as np
import ar
import triangulation
import animation
import snake

import time

# Get drawing
MAT_PREV_COUNT = 2
mat_prev_ptr = 0
mat_prev = [None] * MAT_PREV_COUNT

def increment_ptr(mat_prev_ptr):
    mat_prev_ptr = mat_prev_ptr + 1
    if mat_prev_ptr >=  MAT_PREV_COUNT:
        mat_prev_ptr = 0

    return mat_prev_ptr

def decrement_ptr(mat_prev_ptr):
    mat_prev_ptr = mat_prev_ptr - 1
    if mat_prev_ptr <  0:
        mat_prev_ptr = MAT_PREV_COUNT-1

    return mat_prev_ptr

vid = cv2.VideoCapture('img/snake_game_video_1.MOV')

while True:
    ret, img = vid.read()
    if not ret: break

    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    mat = ar.findHomography(img)
    if mat is not None:
        drawing_box = cv2.perspectiveTransform(ar.DRAW_REF.reshape((-1,1,2)), mat)
        cv2.polylines(img, [drawing_box.astype(np.int32)], True, (0,0,255),2)
    else:
        mat_1 = mat_prev[mat_prev_ptr]
        mat_2 = mat_prev[decrement_ptr(mat_prev_ptr)]

        # interpolate the missing mat 
        if mat_1 is not None and mat_2 is not None:
            mat = mat_1+mat_1-mat_2
            drawing_box = cv2.perspectiveTransform(ar.DRAW_REF.reshape((-1,1,2)), mat)
            cv2.polylines(img, [drawing_box.astype(np.int32)], True, (0,0,255),2)

    # Record mat
    mat_prev_ptr = increment_ptr(mat_prev_ptr)
    mat_prev[mat_prev_ptr] = mat

    cv2.imshow("Snake Game", img)
    cv2.waitKey(10)

vid.release()

cv2.destroyAllWindows()
