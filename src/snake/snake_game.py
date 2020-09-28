import threading
import cv2
import traceback

import animator.ar as ar

from snake.snake_bones import *
from snake.snake_model import *
from snake.food_models import *
from snake.snake_animator import *
from snake.food_animator import *

class SnakeGame:
    def __init__(self):
        self.snake_model = None
        self.food_models = None

        self.snake_animator = None
        self.food_animator = None

        # SCAN, PROCESS, GAME
        self.state = 'SCAN'

    def reset(self):
        self.state = 'SCAN'

    def init_game(self, img):
        # if self.state != 'SCAN': return False

        if img is None: return False

        mat = ar.homography(img, CORNERS_REF)
        if mat is None: return False
        img_snake_drawing = ar.drawing(img, mat, SNAKE_DRAW_REF)
        # Rotate depending on layout
        img_snake_drawing = cv2.rotate(img_snake_drawing, cv2.ROTATE_90_CLOCKWISE)

        img_food_drawing = ar.drawing(img, mat, FOOD_DRAW_REF)

        def init():
            try:
                self.snake_model = SnakeModel()
                self.food_models = FoodModels()

                self.snake_animator = SnakeAnimator(img_snake_drawing, self.snake_model, bones(DEFAULT_PARAMS))
                self.food_animator = FoodAnimator(img_food_drawing, self.snake_model, food_bones(FOOD_DEFAULT_PARAMS))
                self.state = 'GAME'
            except Exception:
                # traceback.print_exc()
                self.state = 'RETRY'


        self.state = 'PROCESS'
        t = threading.Thread(target=init)
        t.start()

        return True

    def render_scan(self, img, retry=False):
        if img is None: return None

        mat = ar.homography(img, CORNERS_REF)
        if mat is not None:
            img = ar.render_lines(img, SNAKE_DRAW_REF.reshape((-1,1,2)), mat, color=PINK_COLOR, thickness=2)
            img = ar.render_lines(img, FOOD_DRAW_REF.reshape((-1,1,2)), mat, color=PINK_COLOR, thickness=2)

            if not retry:
                str = 'Press any key to start.'
            else:
                str = 'Press any key to retry.'
            img = ar.render_text(img, str, INFO_REF, mat, fontScale=2, thickness=3, color=PINK_COLOR)

            return img

        return img

    def render_process(self, img):
        if img is None: return None

        mat = ar.homography(img, CORNERS_REF)
        if mat is not None:
            return ar.render_text(img, 'Processing...', INFO_REF, mat, fontScale=2, thickness=3, color=PINK_COLOR)

        return img

    def render_game(self, img):
        if img is None: return None

        mat = ar.homography(img, CORNERS_REF)
        if mat is None: return img

        snake_frame = self.snake_animator.current_frame
        if snake_frame is None: return img

        food_frame = self.food_animator.current_frame
        if food_frame is None: return food_frame

        img_render = ar.render_text(img, 'Score: ' + str(self.food_models.eat_counter), INFO_REF, mat, fontScale=2, thickness=3, color=PINK_COLOR)

        for food_model in self.food_models.models:
            food_img = food_frame[0]
            food_anchor = food_frame[1]
            food_mask = food_frame[2]
            food_position = (int(food_model.x-food_anchor[0]), int(food_model.y-food_anchor[1]))
            img_render = ar.render(img_render, food_img, food_mask, food_position, mat)

        snake_img = snake_frame[0]
        snake_anchor = snake_frame[1]
        snake_mask = snake_frame[2]
        snake_position = (int(self.snake_model.x-snake_anchor[0]), int(self.snake_model.y-snake_anchor[1]))
        img_render = ar.render(img_render, snake_img, snake_mask, snake_position, mat)

        return img_render

    def update(self, img, key):
        if self.state == 'SCAN' or self.state == 'RETRY':
            if key is not None:
                self.init_game(img)

            return self.render_scan(img, retry=self.state == 'RETRY')

        if self.state == 'PROCESS':
            return self.render_process(img)

        if self.state == 'GAME':
            self.snake_model.update()
            self.snake_model.move(key) # Orders matter
            self.snake_animator.update()

            self.food_models.update(self.snake_model)
            self.food_animator.update()

            return self.render_game(img)
