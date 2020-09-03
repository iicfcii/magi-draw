import cv2
import numpy as np
import ar
import triangulation
import animation
import snake

import time


MAT_PREV_COUNT = 2
mat_prev_ptr = 0
mats_prev = [(None, False)] * MAT_PREV_COUNT

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

vid = cv2.VideoCapture(0)

interpolation_count = 0
while True:
    ret, img = vid.read()
    if not ret: break

    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    mat = ar.findHomography(img)
    interpolated = False
    if mat is None and interpolation_count < 2:
        # interpolate the missing mat
        mat_1 = mats_prev[mat_prev_ptr][0]
        interpolated_prev = mats_prev[mat_prev_ptr][1]
        mat_2 = mats_prev[decrement_ptr(mat_prev_ptr)][0]
        # Avoid initial Nones
        if mat_1 is not None and mat_2 is not None:
            mat = (mat_1+mat_1-mat_2)*0.5+mat_1*0.5
            interpolated = True
            if interpolated_prev:
                interpolation_count += 1
    else:
        interpolation_count = 0

    # if interpolation_count > 0: print(interpolation_count)
    # Draw drawing bounding box
    if mat is not None:
        if interpolated:
            color = (0,255,255)
        else:
            color = (0,0,255)
        drawing_box = cv2.perspectiveTransform(ar.DRAW_REF.reshape((-1,1,2)), mat)
        cv2.polylines(img, [drawing_box.astype(np.int32)], True, color, 2)

    # Record mat
    mat_prev_ptr = increment_ptr(mat_prev_ptr)
    mats_prev[mat_prev_ptr] = (mat, interpolated)

    cv2.imshow("Snake Game", img)
    cv2.waitKey(1)

vid.release()

cv2.destroyAllWindows()
