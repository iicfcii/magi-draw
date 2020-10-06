import threading
import cv2
import traceback

import animator.ar as ar

from dog.dog_bones import *
from dog.dog_animator import *

class DogGame:
    def __init__(self):
        self.dog_animator = None

        # SCAN, PROCESS, GAME
        self.state = 'SCAN'

    def reset(self):
        self.state = 'SCAN'

    def init_game(self, img):
        if img is None: return False

        mat = ar.homography(img, CORNERS_REF)
        if mat is None: return False
        img_dog_drawing = ar.drawing(img, mat, DOG_DRAW_REF)

        def init():
            try:
                self.dog_animator = DogAnimator(img_dog_drawing, None, params2bones(DEFAULT_PARAMS))
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
            img = ar.render_lines(img, DOG_DRAW_REF.reshape((-1,1,2)), mat, color=PINK_COLOR, thickness=2)

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

        dog_frame = self.dog_animator.current_frame
        if dog_frame is None: return img

        dog_img = dog_frame[0]
        dog_anchor = dog_frame[1]
        dog_mask = dog_frame[2]
        dog_position = (int(BOARD_REF[3,0]+100-dog_anchor[0]), int(BOARD_REF[3,1]-100-dog_anchor[1]))
        img_render = ar.render(img, dog_img, dog_mask, dog_position, mat)

        return img_render

    def update(self, img, key):
        if self.state == 'SCAN' or self.state == 'RETRY':
            if key is not None:
                self.init_game(img)

            return self.render_scan(img, retry=self.state == 'RETRY')

        if self.state == 'PROCESS':
            return self.render_process(img)

        if self.state == 'GAME':
            self.dog_animator.update()
            return self.render_game(img)
