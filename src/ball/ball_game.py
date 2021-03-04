import threading
import cv2
import traceback

import animator.ar as ar

from ball.ball_bones import *
from ball.ball_model import *
from ball.ball_animator import *

class BallGame:
    def __init__(self):
        self.model = None
        self.animator = None

        # SCAN, PROCESS, GAME
        self.state = 'SCAN'

    def reset(self):
        self.state = 'SCAN'

    def init_game(self, img):
        if img is None: return False

        mat = ar.homography(img, CORNERS_REF)
        if mat is None: return False
        img_drawing = ar.drawing(img, mat, BALL_DRAW_REF)

        def init():
            pass
            try:
                self.model = BallModel()
                self.animator = BallAnimator(img_drawing, self.model, params2bones(DEFAULT_PARAMS))
                self.state = 'GAME'
            except Exception:
                traceback.print_exc()
                self.state = 'RETRY'

        self.state = 'PROCESS'
        t = threading.Thread(target=init)
        t.start()

        return True

    def render_scan(self, img, retry=False):
        if img is None: return None

        mat = ar.homography(img, CORNERS_REF)
        if mat is not None:
            img = ar.render_lines(img, BALL_DRAW_REF.reshape((-1,1,2)), mat, color=PINK_COLOR, thickness=2)

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

    def render_game(self, img, mat):
        if img is None: return None
        if mat is None: return img

        frame = self.animator.current_frame
        if frame is None: return img

        imgf = frame[0]
        anchor = frame[1]
        mask = frame[2]
        position = (int(self.model.x-anchor[0]), int(self.model.y-anchor[1]))
        img_render = ar.render(img, imgf, mask, position, mat)

        return img_render

    def update(self, img, key):
        if self.state == 'SCAN' or self.state == 'RETRY':
            if key is not None:
                self.init_game(img)

            return self.render_scan(img, retry=self.state == 'RETRY')

        if self.state == 'PROCESS':
            return self.render_process(img)

        if self.state == 'GAME':
            mat = ar.homography(img, CORNERS_REF)

            # self.dog_model.move(img, mat)
            self.model.update()
            self.animator.update()
            return self.render_game(img, mat)
