import numpy as np
import cv2
import triangulation
import animation

MAX_SPEED = 15
SPEED_STEP = 15
THETA_STEP = 45
BOUNDARY_OFFSET = 100

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
    def __init__(self, drawing):
        self.drawing = drawing
        self.bones_default = bones(bones_default_parameters)

        # Skinning(triangulation and weight calculation)
        img_gray = cv2.cvtColor(self.drawing, cv2.COLOR_BGR2GRAY)
        contour = triangulation.contour(img_gray)
        keypoints = triangulation.keypoints_uniform(img_gray, contour)
        triangles_unconstrained, edges = triangulation.triangulate(contour, keypoints)
        self.triangles = triangulation.constrain(contour, triangles_unconstrained, edges)
        self.weights = animation.calcWeights(self.bones_default,self.triangles)
        # img_tmp = drawing.copy()
        # for triangle in self.triangles:
        #     cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
        # for point in contour:
        #     cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (255,0,0), thickness=-1)
        # for point in keypoints:
        #     cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (0,255,0), thickness=-1)
        # cv2.imshow('Triangulation',img_tmp)
        # cv2.waitKey(0)

        self.move_frames_ptr = 0
        self.move_frames = {}

        self.generate_move()

        self.x = 0.0
        self.y = 0.0
        self.v = 0.0
        self.theta = 0

    # Move the snake within a rectangle according to key event
    def move(self, key, rect):
        if key == ord('w'):
            # Speed up
            self.v += SPEED_STEP
            if self.v > MAX_SPEED:
                self.v = MAX_SPEED
        if key == ord('s'):
            # Slow down
            self.v -= SPEED_STEP
            if self.v < 0:
                self.v = 0
        if key == ord('a'):
            # Turn left
            self.theta = (self.theta-THETA_STEP)%360
        if key == ord('d'):
            # Turn right
            self.theta = (self.theta+THETA_STEP)%360

        self.x = self.x+self.v*np.cos(deg2rad(self.theta))
        self.y = self.y+self.v*np.sin(deg2rad(self.theta))

        # Detect boudary
        # Figure out the frame to get bounding box
        assert len(self.move_frames) != 0
        assert self.theta in self.move_frames

        if self.v != 0:
            if self.move_frames_ptr == len(self.move_frames[self.theta])-1:
                self.move_frames_ptr = 0
            else:
                self.move_frames_ptr += 1
        else:
            self.move_frames_ptr = 0
        frame = self.move_frames[self.theta][self.move_frames_ptr]
        img, anchor = frame

        # Two rectangles
        x,y,w,h = rect
        x_snake = self.x-anchor[0]
        y_snake = self.y-anchor[1]
        w_snake = img.shape[1]
        h_snake = img.shape[0]

        # x direction
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

        return frame

    def generate_move(self):
        for theta in range(0,360,THETA_STEP):
            # Move frames for every angles
            frames = []
            for para in move_frames_parameters:
                para['tail']['theta'] = theta
                bones_n = bones(para)
                triangles_next = animation.animate(self.bones_default,bones_n,self.triangles,self.weights)
                img_n, anchor = animation.warp(self.drawing, self.triangles, triangles_next, bones_n[0])
                frames.append((img_n, anchor))
            self.move_frames[theta] = frames
