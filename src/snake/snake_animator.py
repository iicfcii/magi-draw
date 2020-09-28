import time

from animator.animator import Animator
from snake.snake_bones import *

class SnakeAnimator(Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.snake_model = model

        # Generate custom animation
        t_start = time.time()
        self.slither = self.generate_animation(bones_frames(SLITHER_PARAMS))
        self.turn_left = self.generate_animation(bones_frames(TURN_LEFT_PARAMS))
        self.turn_right = self.generate_animation(bones_frames(TURN_RIGHT_PARAMS))
        self.eat = self.generate_animation(bones_frames(EAT_PARAMS))
        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        if self.snake_model.v > 0:
            self.current_frame = self.turn_right.frame()
            self.turn_right.update()
            self.slither.reset()
            self.eat.reset()
            self.turn_left.reset()
        elif self.snake_model.v < 0:
            self.current_frame = self.turn_left.frame()
            self.turn_left.update()
            self.slither.reset()
            self.eat.reset()
            self.turn_right.reset()
        elif self.snake_model.eat_counter < 3:
            self.current_frame = self.eat.frame()
            self.eat.update()
            self.slither.reset()
            self.turn_right.reset()
            self.turn_left.reset()
        else:
            self.current_frame = self.slither.frame()
            self.slither.update()
            self.eat.reset()
            self.turn_right.reset()
            self.turn_left.reset()
