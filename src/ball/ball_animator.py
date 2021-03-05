import cv2
import numpy as np
import time

from animator.animator import Animator
from ball.ball_bones import *
from animator.animation import *

class BallAnimator(Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.model = model

        # Generate custom animation
        t_start = time.time()
        self.rest = self.generate_animation(params2frames([DEFAULT_PARAMS]),delay=1)
        self.ccw = self.generate_animation(params2frames(CCW_PARAMS),delay=1)
        self.cw = self.generate_animation(params2frames(CW_PARAMS),delay=1)

        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        vmin = 1
        if np.abs(self.model.vx) < vmin:
            self.current_frame = self.rest.frame()
        else:
            if self.model.vx < -vmin:
                self.current_frame = self.cw.frame()
                self.cw.update()
            else:
                self.current_frame = self.ccw.frame()
                self.ccw.update()
