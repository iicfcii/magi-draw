import time

from animator.animator import Animator
from horse.horse_bones import *

class HorseAnimator(Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.model = model

        # Generate custom animation
        t_start = time.time()
        self.test = self.generate_animation(params2frames(TEST_PARAMS))
        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        self.current_frame = self.test.frame()
        self.test.update()
