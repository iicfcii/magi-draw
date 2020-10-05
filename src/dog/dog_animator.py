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
        self.walk_front = self.generate_animation(params2frames(RUN_FRONT_PARAMS))
        self.walk_back = self.generate_animation(params2frames(RUN_BACK_PARAMS), hide=[0,1,2,5])
        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        self.current_frame = merge_frames(self.walk_front.frame(), self.walk_back.frame())
        self.walk_front.update()
        self.walk_back.update()
