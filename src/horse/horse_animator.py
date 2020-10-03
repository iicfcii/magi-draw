import cv2
import time

from animator.animator import Animator
from horse.horse_bones import *
from animator.animation import *

class HorseAnimator(Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.model = model

        # Generate custom animation
        t_start = time.time()
        self.run = self.generate_animation(params2frames(RUN_PARAMS))
        self.run_alt = self.generate_animation(params2frames(RUN_ALT_PARAMS), hide=[0,1,2,5])
        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        frame = self.run.frame()
        frame_alt = self.run_alt.frame()

        frame_merged = merge_frames(frame, frame_alt)

        self.current_frame = frame_merged
        self.run.update()
        self.run_alt.update()
