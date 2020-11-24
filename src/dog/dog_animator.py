import cv2
import numpy as np
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
        self.walk_front = self.generate_animation(params2frames(WALK_FRONT_PARAMS),delay=1)
        self.walk_back = self.generate_animation(params2frames(WALK_BACK_PARAMS),hide=[0,1,2,5],delay=1)
        self.run_front = self.generate_animation(params2frames(RUN_FRONT_PARAMS))
        self.run_back = self.generate_animation(params2frames(RUN_BACK_PARAMS),hide=[0,1,2,5])

        self.rest = self.generate_animation(params2frames(REST_PARAMS),delay=1)
        self.look_up = self.generate_animation(params2frames(LOOK_UP_PARAMS),delay=1)
        self.look_down = self.generate_animation(params2frames(LOOK_DOWN_PARAMS),delay=1)
        self.happy = self.generate_animation(params2frames(HAPPY_PARAMS))


        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        if self.model.vx == 0:
            if self.model.look is None:
                if self.model.head_right:
                    self.current_frame = self.rest.frame()
                else:
                    self.current_frame = flip_frame(self.rest.frame())
                self.rest.update()
            else:
                if self.model.look == 'up':
                    if self.model.head_right:
                        self.current_frame = self.look_up.frame()
                    else:
                        self.current_frame = flip_frame(self.look_up.frame())
                    self.look_up.update()
                if self.model.look == 'down':
                    if self.model.head_right:
                        self.current_frame = self.look_down.frame()
                    else:
                        self.current_frame = flip_frame(self.look_down.frame())
                    self.look_down.update()
                if self.model.look == 'happy':
                    if self.model.head_right:
                        self.current_frame = self.happy.frame()
                    else:
                        self.current_frame = flip_frame(self.happy.frame())
                    self.happy.update()


        elif np.abs(self.model.vx) <= self.model.WALK_VX:
            frame = merge_frames(self.walk_front.frame(), self.walk_back.frame())
            if self.model.head_right:
                self.current_frame = frame
            else:
                self.current_frame = flip_frame(frame)
            self.walk_front.update()
            self.walk_back.update()
        else:
            frame = merge_frames(self.run_front.frame(), self.run_back.frame())
            if self.model.head_right:
                self.current_frame = frame
            else:
                self.current_frame = flip_frame(frame)
            self.run_front.update()
            self.run_back.update()

        # self.current_frame = merge_frames(self.run_front.frame(), self.run_back.frame())
        # self.run_front.update()
        # self.run_back.update()
