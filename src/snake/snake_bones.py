import numpy as np

# Width 540 is multiple of each turn 90 calculated from snake speed and acc
GAME_X = 370
GAME_Y = 100
GAME_WIDTH = 540
GAME_HEIGHT = 850
GAME_STEP = 90
GAME_SNAKE_BODY_LENGTH = 300
GAME_SNAKE_LENGTH = 380

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
