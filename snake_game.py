import cv2
import numpy as np
import ar
import triangulation
import animation
import snake

import time

# Photo of scene
img = cv2.imread('img/snake_game_2.jpg')

# Get drawing
M = ar.findHomography(img)
img_drawing = ar.getDrawing(img, M)
# img_drawing = cv2.imread('img/snake.jpg')
cv2.imshow('Drawing', img_drawing)
cv2.waitKey(0)

snake_animator = snake.SnakeAnimator(img_drawing)

w_scene = int(ar.BOARD_SIZE+ar.MARKER_SIZE*2)
h_scene = int(ar.BOARD_SIZE)
mask_scene = np.zeros((h_scene, w_scene), np.uint8)
cv2.fillConvexPoly(mask_scene, ar.BOARD_REF.astype(np.int32), 255)

key = -1

while True:
    time_start = time.time()
    # M = ar.findHomography(img)

    frame_snake = snake_animator.move(key,(ar.MARKER_SIZE,0,ar.BOARD_SIZE,ar.BOARD_SIZE))
    key = -1

    # Render
    img_scene = np.zeros((h_scene,w_scene,3),np.uint8)
    img_scene[:,:] = (255,255,255)
    # cv2.circle(img_scene, (int(snake_animator.p[0]), int(snake_animator.p[1])), 50, (0,0,0), thickness=-1)

    img_snake = frame_snake[0]
    anchor_snake = frame_snake[1]
    x_snake = int(snake_animator.p[0]-anchor_snake[0])
    y_snake = int(snake_animator.p[1]-anchor_snake[1])
    w_snake = img_snake.shape[1]
    h_snake = img_snake.shape[0]
    # TODO: make sure render is within scene
    img_scene[y_snake:y_snake+h_snake,x_snake:x_snake+w_snake] = img_snake

    # cv2.imshow('Scene', img_scene)
    # cv2.imshow('Scene Mask', mask_scene)
    # cv2.waitKey(0)

    img_scene_warpped = cv2.warpPerspective(img_scene, M, (1200,900), flags=cv2.INTER_LINEAR)
    mask_scene_warpped = cv2.warpPerspective(mask_scene, M, (1200,900), flags=cv2.INTER_LINEAR)
    img_render = img.copy()
    img_render[mask_scene_warpped>0] = img_scene_warpped[mask_scene_warpped>0]

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


cv2.destroyAllWindows()
