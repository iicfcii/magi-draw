import numpy as np
import cv2
import triangulation
import animation
import ar
import threading
import time

# Ratio between design pixel size and desired size
# Camera needs to be high res otherwise image is scaled during warpPerspective
# Adjust ratio to get desired drawing size for animation
RATIO = 2.0
MARKER_SIZE = 120*RATIO
BOARD_WIDTH = 400*RATIO
BOARD_HEIGHT = 500*RATIO
DRAW_WIDTH = 100*RATIO
DRAW_HEIGHT = 250*RATIO
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
DRAW_REF = np.array([[BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE-DRAW_WIDTH)/2, (BOARD_HEIGHT-DRAW_HEIGHT)/2],
                     [BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE+DRAW_WIDTH)/2, (BOARD_HEIGHT-DRAW_HEIGHT)/2],
                     [BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE+DRAW_WIDTH)/2, (BOARD_HEIGHT+DRAW_HEIGHT)/2],
                     [BOARD_WIDTH+MARKER_SIZE+(MARKER_SIZE-DRAW_WIDTH)/2, (BOARD_HEIGHT+DRAW_HEIGHT)/2]])


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

def deg2rad(deg):
    return deg/180*np.pi

def bones(para):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    T_tail2w = t_tail2world(para['tail']['x'],para['tail']['y'], deg2rad(para['tail']['theta']))
    T_a2tail = t_a2tail(deg2rad(para['tail_a']['theta']), para['tail_a']['l'])
    T_b2a = t_a2tail(deg2rad(para['a_b']['theta']), para['a_b']['l'])
    T_c2b = t_a2tail(deg2rad(para['b_c']['theta']), para['b_c']['l'])
    T_head2c = t_a2tail(deg2rad(para['c_head']['theta']), para['c_head']['l'])

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

bones_default_parameters = {
    'tail': {'x': 70, 'y': 100, 'theta': 0},
    'tail_a': {'theta': 0, 'l': 80},
    'a_b': {'theta': 0, 'l': 100},
    'b_c': {'theta': 0, 'l': 100},
    'c_head': {'theta': 0, 'l': 80},
}

slither_frames_parameters = [
    {
        'tail': {'x': 70, 'y': 100, 'theta': -90},
        'tail_a': {'theta': 0, 'l': 80},
        'a_b': {'theta': 0, 'l': 100},
        'b_c': {'theta': 0, 'l': 100},
        'c_head': {'theta': 0, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta':-90},
        'tail_a': {'theta': 15, 'l': 80},
        'a_b': {'theta': -45, 'l': 100},
        'b_c': {'theta': 60, 'l': 100},
        'c_head': {'theta': -45, 'l': 80},
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
        'tail_a': {'theta': -15, 'l': 80},
        'a_b': {'theta': 45, 'l': 100},
        'b_c': {'theta': -60, 'l': 100},
        'c_head': {'theta': 45, 'l': 80},
    },
]

class SnakeAnimator:
    def __init__(self, drawing, model):
        self.drawing = drawing
        self.model = model

        self.bones_default = bones(bones_default_parameters)

        # Skinning(triangulation and weight calculation)
        t_start = time.time()

        img_gray = cv2.cvtColor(self.drawing, cv2.COLOR_BGR2GRAY)
        contour = triangulation.contour(img_gray)
        keypoints = triangulation.keypoints_uniform(img_gray, contour)
        triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)

        t_tri = time.time()-t_start

        self.triangles = triangulation.constrain(contour, triangles_unconstrained, edges)

        t_constrain = time.time()-t_start-t_tri

        self.weights = animation.calcWeights(self.bones_default, self.triangles)

        t_weights = time.time()-t_start-t_constrain

        self.slither_frames_ptr = 0
        self.slither_frames = []

        self.generate_move(1.0)

        t_generate = time.time()-t_start-t_weights

        print('Triangulation', t_tri)
        print('Constrained Triangulation', t_constrain)
        print('Weights', t_weights)
        print('Animation', t_generate)

        self.current_frame = None

    def update(self):
        assert len(self.slither_frames) != 0

        if self.model.v != 0:
            if self.slither_frames_ptr == len(self.slither_frames)-1:
                self.slither_frames_ptr = 0
            else:
                self.slither_frames_ptr += 1
        else:
            self.slither_frames_ptr = 0

        self.current_frame = self.slither_frames[self.slither_frames_ptr]

    def generate_move(self, ratio):
        for para in slither_frames_parameters:
            bones_n = bones(para)

            triangles_next = animation.animate(self.bones_default,bones_n,self.triangles,self.weights)
            img_n, anchor, mask_img_n = animation.warp(self.drawing, self.triangles, triangles_next, bones_n[0])

            img_n = cv2.resize(img_n, None, fx=ratio, fy=ratio)
            anchor = anchor*ratio
            mask_img_n = cv2.resize(mask_img_n, None, fx=ratio, fy=ratio)

            self.slither_frames.append((img_n, anchor, mask_img_n))

class SnakeModel:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 1.0
        self.theta = 0
        self.rect = (MARKER_SIZE,0,BOARD_WIDTH,BOARD_HEIGHT) # Bounding box

    def move(self, key):
        # if key == 87: # w
        #     # Speed up
        #     self.v += SPEED_STEP
        #     if self.v > MAX_SPEED:
        #         self.v = MAX_SPEED
        # if key == 83: # s
        #     # Slow down
        #     self.v -= SPEED_STEP
        #     if self.v < 0:
        #         self.v = 0
        # if key == 65: # a
        #     # Turn left
        #     self.theta = (self.theta-THETA_STEP)%360
        # if key == 68: # d
        #     # Turn right
        #     self.theta = (self.theta+THETA_STEP)%360

        self.x = self.x+self.v*np.cos(deg2rad(self.theta))
        self.y = self.y+self.v*np.sin(deg2rad(self.theta))

    def constrain(self, frame):
        img, anchor, mask = frame
        # Two rectangles
        x,y,w,h = self.rect
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
            self.x = anchor[0] + x
            self.y = anchor[1] + y

class SnakeGame:
    def __init__(self):
        self.model = SnakeModel()
        self.animator = None

        # SCAN, PROCESS, GAME
        self.state = 'SCAN'

    def set_animator(self, img):
        if self.state != 'SCAN': return False

        if img is None: return False

        mat = ar.findHomography(img, CORNERS_REF)
        if mat is None: return False
        img_drawing = ar.getDrawing(img, mat, DRAW_REF)
        # Rotate depending on layout
        img_drawing = cv2.rotate(img_drawing, cv2.ROTATE_90_CLOCKWISE)

        def init_animator():
            self.animator = SnakeAnimator(img_drawing, self.model)
            self.state = 'GAME'

        self.state = 'PROCESS'
        t = threading.Thread(target=init_animator)
        t.start()

        return True

    def render_scan(self, img):
        if img is None: return None

        mat = ar.findHomography(img, CORNERS_REF)
        if mat is not None:
            return ar.render_lines(img, DRAW_REF.reshape((-1,1,2)), mat, color=(0,0,255), thickness=2)

        return img

    def render_process(self, img):
        if img is None: return None

        mat = ar.findHomography(img, CORNERS_REF)
        if mat is not None:
            return ar.render_text(img, 'Processing', (BOARD_REF[0,0],BOARD_REF[0,1]), mat, fontScale=2, thickness=3, color=(255,0,0))

        return img

    def render_game(self, img):
        mat = ar.findHomography(img, CORNERS_REF)
        frame_snake = self.animator.current_frame

        if mat is not None and frame_snake is not None:
            img_snake = frame_snake[0]
            anchor_snake = frame_snake[1]
            mask_snake = frame_snake[2]
            position_snake = (int(self.model.x-anchor_snake[0]), int(self.model.y-anchor_snake[1]))
            img_render = ar.render(img, img_snake, mask_snake, position_snake, mat)
            return img_render

        return img

    def update(self, img, key):
        if self.state == 'SCAN':
            if key is not None:
                self.set_animator(img)

            return self.render_scan(img)

        if self.state == 'PROCESS':
            return self.render_process(img)

        if self.state == 'GAME':
            self.model.move(key)
            self.animator.update()
            self.model.constrain(self.animator.current_frame)

            return self.render_game(img)
