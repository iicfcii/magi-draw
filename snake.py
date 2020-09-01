import numpy as np
import cv2
import triangulation
import animation

MAX_SPEED = 15
SPEED_STEP = 5
THETA_STEP = 20
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

        self.move_frames_ptr = 0
        self.move_frames = []

        self.generate_move()

        self.p = [0.0, 0.0]
        self.v = 0.0
        self.theta = 0

    def move(self, key, rect):
        assert len(self.move_frames) != 0

        if self.move_frames_ptr == len(self.move_frames)-1:
            self.move_frames_ptr = 0
        else:
            self.move_frames_ptr += 1

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

        self.p[0] = self.p[0]+self.v*np.cos(deg2rad(self.theta))
        self.p[1] = self.p[1]+self.v*np.sin(deg2rad(self.theta))

        # Detect boudary
        x,y,w,h = rect

        w_snake = self.move_frames[self.move_frames_ptr][0].shape[1]
        h_snake = self.move_frames[self.move_frames_ptr][0].shape[0]
        anchor = self.move_frames[self.move_frames_ptr][1]
        x_center = self.p[0]-anchor[0]+w_snake/2
        y_center = self.p[1]-anchor[1]+h_snake/2

        if x_center < x+BOUNDARY_OFFSET: self.p[0] = x+BOUNDARY_OFFSET+anchor[0]-w_snake/2
        if x_center > x+w-BOUNDARY_OFFSET: self.p[0] = x+w-BOUNDARY_OFFSET+anchor[0]-w_snake/2
        if y_center < y+BOUNDARY_OFFSET: self.p[1] = y+BOUNDARY_OFFSET+anchor[1]-h_snake/2
        if y_center > y+h-BOUNDARY_OFFSET: self.p[1] = y+h-BOUNDARY_OFFSET+anchor[1]-h_snake/2

        return self.move_frames[self.move_frames_ptr]

    def generate_move(self):
        for para in move_frames_parameters:
            bones_n = bones(para)
            triangles_next = animation.animate(self.bones_default,bones_n,self.triangles,self.weights)
            img_n, anchor = animation.warp(self.drawing, self.triangles, triangles_next, bones_n[0])
            self.move_frames.append((img_n, anchor))
