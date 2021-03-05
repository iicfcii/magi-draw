import cv2
import numpy as np

from animator.animation import calcPointLineDistance
from animator.bone import *
from animator.ar import *
from ball.ball_bones import *

RADIUS= BALL_DRAW_SIZE/2
MARKER_ID = 31
CONTACT_C = 0.6
CONTACT_MARGIN = 10
DAMPING_C = 0.2
GRAVITY = 2
CENTER_X = (BOARD_REF[0,0]+BOARD_REF[1,0])/2

class BallModel:
    def __init__(self):
        self.x = CENTER_X
        self.y = 0
        self.vx = 0
        self.vy = 0
        self.ax = 0
        self.ay = GRAVITY
        self.vz = 0

    def update(self):
        self.vx += self.ax
        self.x += self.vx

        self.vy += self.ay
        self.y += self.vy

        if self.y > BOARD_HEIGHT-RADIUS or self.y < 0:
            self.ay = GRAVITY
            self.ax = 0
            self.vy = 0
            self.vx = 0
            self.y = RADIUS

        if self.x > BOARD_REF[1,0]-RADIUS:
            self.ay = GRAVITY
            self.ax = 0
            self.vy = 0
            self.vx = 0
            self.x = BOARD_REF[1,0]-RADIUS
            # self.y = 0

        if self.x < BOARD_REF[0,0]+RADIUS:
            self.ay = GRAVITY
            self.ax = 0
            self.vy = 0
            self.vx = 0
            self.x = BOARD_REF[0,0]+RADIUS
            # self.y = 0

    def move(self, img, mat):
        if img is None or mat is None: return

        size = (int(CORNERS_REF[4][2,0]),int(CORNERS_REF[4][2,1]))
        img = cv2.warpPerspective(img, mat, size, flags=cv2.WARP_INVERSE_MAP)
        # cv2.imshow('Board',img)
        # cv2.waitKey(0)
        corners, ids, rejects = cv2.aruco.detectMarkers(img, DICT, parameters=PARA) # Default parameters
        if ids is None or MARKER_ID not in ids: return

        marker_corners = (corners[list(ids).index(MARKER_ID)]).reshape((-1,2))
        center = (np.sum(marker_corners, axis=0)/4)
        angle = np.arctan2(
            np.sum(marker_corners[1:3,1])-np.sum(marker_corners[[0,3],1]),
            np.sum(marker_corners[1:3,0])-np.sum(marker_corners[[0,3],0])
        )

        beam_left_local = np.array([-BEAM_WIDTH/2,-MARKER_SIZE/2-BEAM_HEIGHT,1])
        beam_right_local = np.array([BEAM_WIDTH/2,-MARKER_SIZE/2-BEAM_HEIGHT,1])

        T = t(center[0],center[1],angle)
        beam_left = (T @ beam_left_local)[0:2].reshape(2)
        beam_right = (T @ beam_right_local)[0:2].reshape(2)
        # print(beam_left, beam_right)

        d = calcPointLineDistance(np.array([self.x, self.y]), beam_left, beam_right)

        if d < RADIUS-CONTACT_MARGIN:
            self.vy = 0
            self.ay = -(RADIUS-d)*CONTACT_C*np.cos(angle)
            self.ax = (RADIUS-d)*CONTACT_C*2*np.sin(angle) # Increase horizontal force a bit
        else:
            self.ay = GRAVITY
            self.ax = -self.vx*DAMPING_C
