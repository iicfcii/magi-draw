import numpy as np
import cv2
import triangulation
import animation
import ar
import time

MAX_SPEED = 15
SPEED_STEP = 15
THETA_STEP = 90

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

move_frames_parameters = [
    bones_default_parameters,
    {
        'tail': {'x': 70, 'y': 100, 'theta': 0},
        'tail_a': {'theta': 10, 'l': 80},
        'a_b': {'theta': -30, 'l': 100},
        'b_c': {'theta': 40, 'l': 100},
        'c_head': {'theta': -30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': 0},
        'tail_a': {'theta': 15, 'l': 80},
        'a_b': {'theta': -45, 'l': 100},
        'b_c': {'theta': 60, 'l': 100},
        'c_head': {'theta': -45, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': 0},
        'tail_a': {'theta': 10, 'l': 80},
        'a_b': {'theta': -30, 'l': 100},
        'b_c': {'theta': 40, 'l': 100},
        'c_head': {'theta': -30, 'l': 80},
    },
    bones_default_parameters,
    {
        'tail': {'x': 70, 'y': 100, 'theta': 0},
        'tail_a': {'theta': -10, 'l': 80},
        'a_b': {'theta': 30, 'l': 100},
        'b_c': {'theta': -40, 'l': 100},
        'c_head': {'theta': 30, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': 0},
        'tail_a': {'theta': -15, 'l': 80},
        'a_b': {'theta': 45, 'l': 100},
        'b_c': {'theta': -60, 'l': 100},
        'c_head': {'theta': 45, 'l': 80},
    },
    {
        'tail': {'x': 70, 'y': 100, 'theta': 0},
        'tail_a': {'theta': -10, 'l': 80},
        'a_b': {'theta': 30, 'l': 100},
        'b_c': {'theta': -40, 'l': 100},
        'c_head': {'theta': 30, 'l': 80},
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

        self.weights = animation.calcWeights(self.bones_default,self.triangles)

        t_weights = time.time()-t_start-t_constrain

        self.move_frames_ptr = 0
        self.move_frames = {}

        self.generate_move(0.5)

        t_generate = time.time()-t_start-t_weights

        print(t_tri, t_constrain, t_weights, t_generate)

        self.current_frame = None

    def update(self):
        assert len(self.move_frames) != 0
        assert self.model.theta in self.move_frames

        if self.model.v != 0:
            if self.move_frames_ptr == len(self.move_frames[self.model.theta])-1:
                self.move_frames_ptr = 0
            else:
                self.move_frames_ptr += 1
        else:
            self.move_frames_ptr = 0

        self.current_frame = self.move_frames[self.model.theta][self.move_frames_ptr]

    def generate_move(self, ratio):
        for theta in range(0,360,THETA_STEP):
            # Move frames for every angles
            frames = []
            for para in move_frames_parameters:
                para['tail']['theta'] = theta
                bones_n = bones(para)

                triangles_next = animation.animate(self.bones_default,bones_n,self.triangles,self.weights)
                img_n, anchor = animation.warp(self.drawing, self.triangles, triangles_next, bones_n[0])

                img_n = cv2.resize(img_n, None, fx=ratio, fy=ratio)
                anchor = anchor*ratio

                frames.append((img_n, anchor))
            self.move_frames[theta] = frames

class SnakeModel:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.theta = 0
        self.rect = (0,0,ar.BOARD_SIZE,ar.BOARD_SIZE) # Bounding box

    def move(self, key):
        if key == 87: # w
            # Speed up
            self.v += SPEED_STEP
            if self.v > MAX_SPEED:
                self.v = MAX_SPEED
        if key == 83: # s
            # Slow down
            self.v -= SPEED_STEP
            if self.v < 0:
                self.v = 0
        if key == 65: # a
            # Turn left
            self.theta = (self.theta-THETA_STEP)%360
        if key == 68: # d
            # Turn right
            self.theta = (self.theta+THETA_STEP)%360

        self.x = self.x+self.v*np.cos(deg2rad(self.theta))
        self.y = self.y+self.v*np.sin(deg2rad(self.theta))

    def constrain(self, frame):
        img, anchor = frame
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

    def set_animator(self, img):
        if self.animator is not None: return False

        if img is None: return False

        mat = ar.findHomography(img)
        if mat is None: return False
        img_drawing = ar.getDrawing(img, mat)
        self.animator = SnakeAnimator(img_drawing, self.model)
        return True

    def render_drawing(self, img):
        if img is None: return None

        mat = ar.findHomography(img)
        img_tmp = img.copy()
        if mat is not None:
            # Draw drawing bounding box
            drawing_box = cv2.perspectiveTransform(ar.DRAW_REF.reshape((-1,1,2)), mat)
            img_tmp = cv2.polylines(img_tmp, [drawing_box.astype(np.int32)], True, (0,0,255), 2)

        return img_tmp

    def render_game(self, img):
        mat = ar.findHomography(img)
        frame_snake = self.animator.current_frame

        if mat is not None and frame_snake is not None:
            img_snake = frame_snake[0]
            anchor_snake = frame_snake[1]
            position_snake = (int(self.model.x-anchor_snake[0]), int(self.model.y-anchor_snake[1]))
            img_render = ar.render(img, img_snake, position_snake, mat)
            return img_render

        return img

    def update(self, img, key):
        if self.animator is None:
            if key is not None:
                print('Preparing games')
                self.set_animator(img)
            return self.render_drawing(img)
        else:
            self.model.move(key)
            self.animator.update()
            self.model.constrain(self.animator.current_frame)

            return self.render_game(img)
