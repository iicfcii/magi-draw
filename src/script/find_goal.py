import sys
sys.path.append(sys.path[0] + "/..")
import numpy as np
import cv2

import animator.ar as ar
from dog.dog_model import *
from dog.dog_bones import *

# img = cv2.imread('img/dog_game_4_sd.jpg')

vid = cv2.VideoCapture('img/dog_game_video_4.MOV')

while vid.isOpened():
    ret, img = vid.read()

    if ret:
        mat = ar.homography(img, CORNERS_REF)

        model = DogModel()
        model.find_goal(img, mat)
