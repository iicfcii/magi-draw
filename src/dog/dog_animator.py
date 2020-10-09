import cv2
import time

from animator.animator import Animator
from dog.dog_bones import *
from animator.animation import *

class DogAnimator(Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.model = model

        # Generate custom animation
        t_start = time.time()
        self.walk_front = self.generate_animation(params2frames(WALK_FRONT_PARAMS))
        self.walk_back = self.generate_animation(params2frames(WALK_BACK_PARAMS), hide=[0,1,2,5])
        self.run_front = self.generate_animation(params2frames(RUN_FRONT_PARAMS))
        self.run_back = self.generate_animation(params2frames(RUN_BACK_PARAMS), hide=[0,1,2,5])

        self.rest = self.generate_animation(params2frames(REST_PARAMS))

        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        if self.model.vx == 0:
            if self.model.head_right:
                self.current_frame = self.rest.frame()
            else:
                self.current_frame = flip_frame(self.rest.frame())
            self.rest.update()
        else:
            frame = merge_frames(self.walk_front.frame(), self.walk_back.frame())
            if self.model.head_right:
                self.current_frame = frame
            else:
                self.current_frame = flip_frame(frame)
            self.walk_front.update()
            self.walk_back.update()

        # self.current_frame = merge_frames(self.run_front.frame(), self.run_back.frame())
        # self.run_front.update()
        # self.run_back.update()
