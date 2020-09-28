import time

from animator.animator import Animator
from snake.snake_bones import *

class FoodAnimator(Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.food_models = model

        # Generate custom animation
        t_start = time.time()
        self.rotate = self.generate_animation(food_bones_frames(FOOD_ROTATE_PARAMS))
        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        self.current_frame = self.rotate.frame()
        self.rotate.update()
