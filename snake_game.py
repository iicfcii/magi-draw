import cv2
import numpy as np
import ar
import triangulation
import animation
import snake

import time

# Load video
vid = cv2.VideoCapture('img/snake_game_video_1.MOV')
# vid = cv2.VideoCapture(0)
ratio = 0.75

# Pick drawing frame
while True:
    ret, img = vid.read()
    if not ret: break
    img = cv2.resize(img, None, fx=ratio, fy=ratio)
    img_tmp = img.copy()
    mat = ar.findHomography(img)
    # Draw drawing bounding box
    if mat is not None:
        drawing_box = cv2.perspectiveTransform(ar.DRAW_REF.reshape((-1,1,2)), mat)
        img_tmp = cv2.polylines(img_tmp, [drawing_box.astype(np.int32)], True, (0,0,255), 2)

    cv2.imshow("Drawing Frame", img_tmp)
    key = cv2.waitKey(20)
    if key == 32: # Space
        img_drawing = ar.getDrawing(img, mat)
        break

# Generate animation
snake_animator = snake.SnakeAnimator(img_drawing)

key = -1
while True:
    time_start = time.time()

    ret, img = vid.read()
    if not ret: break
    img = cv2.resize(img, None, fx=ratio, fy=ratio)

    frame_snake = snake_animator.move(key,(0,0,ar.BOARD_SIZE,ar.BOARD_SIZE))
    key = -1

    # Render
    M = ar.findHomography(img)
    if M is not None:
        img_snake = frame_snake[0]
        anchor_snake = frame_snake[1]
        position_snake = (int(snake_animator.x-anchor_snake[0]), int(snake_animator.y-anchor_snake[1]))
        img_render = ar.render(img, img_snake, position_snake, M)
        cv2.imshow('AR', img_render)

    # Busy waiting and polling key event
    key_tmp = cv2.waitKey(1)
    if key_tmp != -1: key = key_tmp # Record key if pressed
    time_elapsed = time.time()-time_start
    while time_elapsed < 0.1: # Less than 10 fps
        key_tmp = cv2.waitKey(1)
        if key_tmp != -1: key = key_tmp
        time_elapsed = time.time()-time_start
    print('fps', int(1/time_elapsed))

vid.release()
cv2.destroyAllWindows()
