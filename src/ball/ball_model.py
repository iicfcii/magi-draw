import cv2
import numpy as np

from ball.ball_bones import *

class BallModel:
    def __init__(self):
        self.x = (BOARD_REF[0,0]+BOARD_REF[1,0])/2
        self.y = BALL_DRAW_SIZE/2

    def update(self):
        pass
