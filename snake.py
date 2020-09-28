import numpy as np
import cv2
import threading
import time

import triangulation
import animation
import ar

# Ratio between design pixel size and desired size
# Camera needs to be high res otherwise image is scaled during warpPerspective
# Adjust ratio to get desired drawing size for animation
RATIO = 2.0

MARKER_SIZE = 120*RATIO
BOARD_WIDTH = 400*RATIO
BOARD_HEIGHT = 500*RATIO
SNAKE_DRAW_WIDTH = 100*RATIO
SNAKE_DRAW_HEIGHT = 250*RATIO
FOOD_DRAW_WIDTH = 100*RATIO
FOOD_DRAW_HEIGHT = 100*RATIO

CORNERS_REF = {
    7: np.array([[0,0],
                 [MARKER_SIZE,0],
                 [MARKER_SIZE,MARKER_SIZE],
                 [0,MARKER_SIZE]]),
    23: np.array([[BOARD_WIDTH+MARKER_SIZE,0],
                  [BOARD_WIDTH+2*MARKER_SIZE,0],
                  [BOARD_WIDTH+2*MARKER_SIZE,MARKER_SIZE],
                  [BOARD_WIDTH+MARKER_SIZE,MARKER_SIZE]]),
    27: np.array([[BOARD_WIDTH+MARKER_SIZE,BOARD_HEIGHT-MARKER_SIZE],
                  [BOARD_WIDTH+2*MARKER_SIZE,BOARD_HEIGHT-MARKER_SIZE],
                  [BOARD_WIDTH+2*MARKER_SIZE,BOARD_HEIGHT],
                  [BOARD_WIDTH+MARKER_SIZE,BOARD_HEIGHT]]),
    42: np.array([[0,BOARD_HEIGHT-MARKER_SIZE],
                  [MARKER_SIZE,BOARD_HEIGHT-MARKER_SIZE],
                  [MARKER_SIZE,BOARD_HEIGHT],
                  [0,BOARD_HEIGHT]]),
}
BOARD_REF = np.array([[MARKER_SIZE,0],
                      [BOARD_WIDTH+MARKER_SIZE,0],
                      [BOARD_WIDTH+MARKER_SIZE,BOARD_HEIGHT],
                      [0,BOARD_HEIGHT]])
SNAKE_DRAW_REF = np.array([[BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE-SNAKE_DRAW_WIDTH)/2, (BOARD_HEIGHT-SNAKE_DRAW_HEIGHT)/2],
                           [BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE+SNAKE_DRAW_WIDTH)/2, (BOARD_HEIGHT-SNAKE_DRAW_HEIGHT)/2],
                           [BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE+SNAKE_DRAW_WIDTH)/2, (BOARD_HEIGHT+SNAKE_DRAW_HEIGHT)/2],
                           [BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE-SNAKE_DRAW_WIDTH)/2, (BOARD_HEIGHT+SNAKE_DRAW_HEIGHT)/2]])
FOOD_DRAW_REF = np.array([[(MARKER_SIZE-FOOD_DRAW_WIDTH)/2, (BOARD_HEIGHT+SNAKE_DRAW_HEIGHT)/2-FOOD_DRAW_HEIGHT],
                          [(MARKER_SIZE+FOOD_DRAW_WIDTH)/2, (BOARD_HEIGHT+SNAKE_DRAW_HEIGHT)/2-FOOD_DRAW_HEIGHT],
                          [(MARKER_SIZE+FOOD_DRAW_WIDTH)/2, (BOARD_HEIGHT+SNAKE_DRAW_HEIGHT)/2],
                          [(MARKER_SIZE-FOOD_DRAW_WIDTH)/2, (BOARD_HEIGHT+SNAKE_DRAW_HEIGHT)/2]])
INFO_REF = np.array([BOARD_REF[0,0]+20,BOARD_REF[0,1]+20])

PINK_COLOR = (131, 124, 237)
GREEN_COLOR = (88, 206, 169)

DEFAULT_PARAMS = {
    'tail': {'x': 70, 'y': 100, 'theta': 0},
    'tail_a': {'theta': 0, 'l': 80},
    'a_b': {'theta': 0, 'l': 100},
    'b_c': {'theta': 0, 'l': 100},
    'c_head': {'theta': 0, 'l': 80},
}

SLITHER_PARAMS = [
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta':-90},
        'tail_a': {'theta': 10, 'l': 80},
        'a_b': {'theta': -30, 'l': 100},
        'b_c': {'theta': 40, 'l': 100},
        'c_head': {'theta': -30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta':-90},
        'tail_a': {'theta': 15, 'l': 80},
        'a_b': {'theta': -45, 'l': 100},
        'b_c': {'theta': 60, 'l': 100},
        'c_head': {'theta': -45, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta':-90},
        'tail_a': {'theta': 10, 'l': 80},
        'a_b': {'theta': -30, 'l': 100},
        'b_c': {'theta': 40, 'l': 100},
        'c_head': {'theta': -30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': -10, 'l': 80},
        'a_b': {'theta': 30, 'l': 100},
        'b_c': {'theta': -40, 'l': 100},
        'c_head': {'theta': 30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': -15, 'l': 80},
        'a_b': {'theta': 45, 'l': 100},
        'b_c': {'theta': -60, 'l': 100},
        'c_head': {'theta': 45, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': -10, 'l': 80},
        'a_b': {'theta': 30, 'l': 100},
        'b_c': {'theta': -40, 'l': 100},
        'c_head': {'theta': 30, 'l': 80},
    },
]

TURN_LEFT_PARAMS = [
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 20, 'l': 80},
        'a_b': {'theta': -20, 'l': 100},
        'b_c': {'theta': -30, 'l': 100},
        'c_head': {'theta': 30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 10, 'l': 80},
        'a_b': {'theta': -10, 'l': 100},
        'b_c': {'theta': -20, 'l': 100},
        'c_head': {'theta': 20, 'l': 80},
    },
]

TURN_RIGHT_PARAMS = [
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': -20, 'l': 80},
        'a_b': {'theta': 20, 'l': 100},
        'b_c': {'theta': 30, 'l': 100},
        'c_head': {'theta': -30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': -10, 'l': 80},
        'a_b': {'theta': 10, 'l': 100},
        'b_c': {'theta': 20, 'l': 100},
        'c_head': {'theta': -20, 'l': 80},
    },
]

EAT_PARAMS = [
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 140},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 100},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 80},
    },
]

FOOD_DEFAULT_PARAMS = {
    'bottom': {'x': 100, 'y': 150, 'theta': -90},
    'bottom_top': {'theta': 0, 'l': 100},
}

FOOD_ROTATE_PARAMS = [
    {
        'bottom': {'x': 100, 'y': 150, 'theta': -90},
        'bottom_top': {'theta': 0, 'l': 100},
    },
    {
        'bottom': {'x': 100, 'y': 150, 'theta': -120},
        'bottom_top': {'theta': 0, 'l': 100},
    },
    {
        'bottom': {'x': 100, 'y': 150, 'theta': -90},
        'bottom_top': {'theta': 0, 'l': 100},
    },
    {
        'bottom': {'x': 100, 'y': 150, 'theta': -60},
        'bottom_top': {'theta': 0, 'l': 100},
    }
]

# Width 540 is multiple of each turn 90 calculated from snake speed and acc
GAME_X = 370
GAME_Y = 100
GAME_WIDTH = 540
GAME_HEIGHT = 850
GAME_STEP = 90
GAME_SNAKE_BODY_LENGTH = 300
GAME_SNAKE_LENGTH = 380

def deg2rad(deg):
    return deg/180*np.pi

# Snake bones
# Bones: tail2a, a2b, b2c, c2head
# Length: l1 l2 l3 l4
# Angles: theta1 theta2 theta3 theta4
# tail is origin and first link rest pose is positive x
# world coordinate system represents image
def t_tail2world(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_a2tail(theta1,l1):
    return np.array([[np.cos(theta1),-np.sin(theta1),l1*np.cos(theta1)],
                     [np.sin(theta1),np.cos(theta1),l1*np.sin(theta1)],
                     [0,0,1]])

def t_b2a(theta2,l2):
    return np.array([[np.cos(theta2),-np.sin(theta2),l2*np.cos(theta2)],
                     [np.sin(theta2),np.cos(theta2),l2*np.sin(theta2)],
                     [0,0,1]])

def t_c2b(theta3,l3):
    return np.array([[np.cos(theta3),-np.sin(theta3),l3*np.cos(theta3)],
                     [np.sin(theta3),np.cos(theta3),l3*np.sin(theta3)],
                     [0,0,1]])

def t_head2c(theta4,l4):
    return np.array([[np.cos(theta4),-np.sin(theta4),l4*np.cos(theta4)],
                     [np.sin(theta4),np.cos(theta4),l4*np.sin(theta4)],
                     [0,0,1]])

def bones(params):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    T_tail2w = t_tail2world(params['tail']['x'],params['tail']['y'], deg2rad(params['tail']['theta']))
    T_a2tail = t_a2tail(deg2rad(params['tail_a']['theta']), params['tail_a']['l'])
    T_b2a = t_a2tail(deg2rad(params['a_b']['theta']), params['a_b']['l'])
    T_c2b = t_a2tail(deg2rad(params['b_c']['theta']), params['b_c']['l'])
    T_head2c = t_a2tail(deg2rad(params['c_head']['theta']), params['c_head']['l'])

    p_tail_world = (T_tail2w @ origin)[0:2].reshape(2)
    p_a_world = (T_tail2w @ T_a2tail @ origin)[0:2].reshape(2)
    p_b_world = (T_tail2w @ T_a2tail @ T_b2a @ origin)[0:2].reshape(2)
    p_c_world = (T_tail2w @ T_a2tail @ T_b2a @ T_c2b @ origin)[0:2].reshape(2)
    p_head_world = (T_tail2w @ T_a2tail @ T_b2a @ T_c2b @ T_head2c @ origin)[0:2].reshape(2)

    bone_tail2a = np.array([p_tail_world,p_a_world])
    bone_a2b = np.array([p_a_world,p_b_world])
    bone_b2c = np.array([p_b_world,p_c_world])
    bone_c2head = np.array([p_c_world,p_head_world])
    bones = np.array([bone_tail2a,bone_a2b,bone_b2c,bone_c2head]).reshape((-1,4))

    return bones

def bones_frames(params):
    bones_frames = []

    for p in params:
        bones_frames.append(bones(p))

    return bones_frames

def t_bottom2world(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_top2bottom(theta1,l1):
    return np.array([[np.cos(theta1),-np.sin(theta1),l1*np.cos(theta1)],
                     [np.sin(theta1),np.cos(theta1),l1*np.sin(theta1)],
                     [0,0,1]])

def food_bones(params):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    T_bottom2w = t_bottom2world(params['bottom']['x'],params['bottom']['y'], deg2rad(params['bottom']['theta']))
    T_top2bottom = t_top2bottom(deg2rad(params['bottom_top']['theta']), params['bottom_top']['l'])

    p_bottom_world = (T_bottom2w @ origin)[0:2].reshape(2)
    p_top_world = (T_bottom2w @ T_top2bottom @ origin)[0:2].reshape(2)

    bone_bottom2top = np.array([p_bottom_world,p_top_world])
    bones = np.array([bone_bottom2top]).reshape((-1,4))

    return bones

def food_bones_frames(params):
    bones_frames = []

    for p in params:
        bones_frames.append(food_bones(p))

    return bones_frames

class FoodAnimator(animation.Animator):
    def __init__(self, drawing, model, bones):
        super().__init__(drawing, bones)
        self.snake_model = model

        # Generate custom animation
        t_start = time.time()
        self.rotate = self.generate_animation(food_bones_frames(FOOD_ROTATE_PARAMS))
        t_generate = time.time()-t_start
        # print('Animation', t_generate)

    def update(self):
        self.current_frame = self.rotate.frame()
        self.rotate.update()

class SnakeAnimator(animation.Animator):
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

class SnakeModel:
    def __init__(self):
        # Make sure snake is at bottom
        self.RECT = (GAME_X,GAME_Y,GAME_WIDTH,GAME_HEIGHT)
        self.X_DEFAULT = self.RECT[0]+self.RECT[2]/2
        self.Y_DEFAULT = self.RECT[1]+self.RECT[3]
        self.SPEED = 45
        self.ACC = self.SPEED/len(TURN_LEFT_PARAMS) # Sync with number of animation frames

        self.x = self.X_DEFAULT
        self.y = self.Y_DEFAULT
        self.v = 0.0 # Horizontal speed
        self.eat_counter = 3 # Number of frames of eat animation

    # Physics
    def update(self):
        self.x = self.x + self.v # Velocity
        self.v = self.v - np.sign(self.v)*self.ACC # Acceleration
        if np.abs(self.v) < 1: self.v = 0

        self.eat_counter = self.eat_counter + 1
        if self.eat_counter > 3: self.eat_counter = 3

        # Simple constrain
        if self.x > self.RECT[0]+self.RECT[2]:
            self.x = self.RECT[0]+self.RECT[2]
        if self.x < self.RECT[0]:
            self.x = self.RECT[0]

    def move(self, key):
        if self.v == 0:
            if key == 65: # a
                self.v = -self.SPEED
            if key == 68: # d
                self.v = self.SPEED

    def constrain(self, frame):
        img, anchor, mask = frame
        # Two rectangles
        x,y,w,h = self.RECT
        x_snake = self.x-anchor[0]
        y_snake = self.y-anchor[1]
        w_snake = img.shape[1]
        h_snake = img.shape[0]

        rect_u = animation.union_rects((x,y,w,h), (x_snake,y_snake,w_snake,h_snake))
        if rect_u is not None:
            x_u,y_u,w_u,h_u = rect_u
            dw = w_snake-w_u
            dh = h_snake-h_u
            if dw > 0:
                if x_u == x:
                    self.x += dw
                else:
                    self.x -= dw
            if dh > 0:
                if y_u == y:
                    self.y += dh
                else:
                    self.y -= dh
        else:
            self.x = self.X_DEFAULT
            self.y = self.Y_DEFAULT

class FoodModels:
    def __init__(self):
        self.RECT = (GAME_X,GAME_Y,GAME_WIDTH,GAME_HEIGHT)
        self.X_RANGE = tuple(np.arange(GAME_X,GAME_X+GAME_WIDTH+1,GAME_STEP))

        self.models = []
        self.models.append(FoodModel(self.random_x(), self.RECT))

        self.eat_counter = 0
        self.frame_counter = 0

    def random_x(self):
        return self.X_RANGE[np.random.randint(0, len(self.X_RANGE))]

    def update(self, snake_model):
        self.frame_counter += 1

        # Remove bottom ones and eaten ones
        for i in range(len(self.models)-1,-1,-1):
            model = self.models[i]
            model.update()

            if model.v == 0:
                self.models.pop(i)
            else:
                if model.x == snake_model.x and model.y > snake_model.y-GAME_SNAKE_LENGTH and model.y < snake_model.y-GAME_SNAKE_BODY_LENGTH:
                    self.models.pop(i)
                    self.eat_counter += 1
                    snake_model.eat_counter = -1 # Start eat animation

        # Generate at even speed
        if self.frame_counter % 20 == 0:
            self.models.append(FoodModel(self.random_x(), self.RECT))

class FoodModel:
    def __init__(self, x, rect):
        self.RECT = rect

        self.x = x
        self.y = self.RECT[1]
        self.v = 20 # Vertical speed

    # Physics
    def update(self):
        self.y = self.y + self.v # Velocity
        if self.y > self.RECT[1]+self.RECT[3]:
            self.v = 0

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
            except:
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
