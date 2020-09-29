import numpy as np

from animator.bone import *

# Adjust ratio to use full resolution of the drawing
RATIO = 2.0

MARKER_SIZE = 140*RATIO
BOARD_WIDTH = 650*RATIO
BOARD_HEIGHT = 250*RATIO
HORSE_DRAW_LEFT = 290*RATIO
HORSE_DRAW_WIDTH = 200*RATIO
HORSE_DRAW_HEIGHT = 140*RATIO

CORNERS_REF = {
    7: np.array([[0,0],
                 [MARKER_SIZE,0],
                 [MARKER_SIZE,MARKER_SIZE],
                 [0,MARKER_SIZE]]),
    23: np.array([[BOARD_WIDTH-MARKER_SIZE,0],
                  [BOARD_WIDTH,0],
                  [BOARD_WIDTH,MARKER_SIZE],
                  [BOARD_WIDTH-MARKER_SIZE,MARKER_SIZE]]),
    27: np.array([[BOARD_WIDTH-MARKER_SIZE,BOARD_HEIGHT+MARKER_SIZE],
                  [BOARD_WIDTH,BOARD_HEIGHT+MARKER_SIZE],
                  [BOARD_WIDTH,BOARD_HEIGHT+2*MARKER_SIZE],
                  [BOARD_WIDTH-MARKER_SIZE,BOARD_HEIGHT+2*MARKER_SIZE]]),
    42: np.array([[0,BOARD_HEIGHT+MARKER_SIZE],
                  [MARKER_SIZE,BOARD_HEIGHT+MARKER_SIZE],
                  [MARKER_SIZE,BOARD_HEIGHT+2*MARKER_SIZE],
                  [0,BOARD_HEIGHT+2*MARKER_SIZE]]),
}
BOARD_REF = np.array([[0,MARKER_SIZE],
                      [BOARD_WIDTH,MARKER_SIZE],
                      [BOARD_WIDTH,BOARD_HEIGHT+MARKER_SIZE],
                      [0,BOARD_HEIGHT+MARKER_SIZE]])
HORSE_DRAW_REF = np.array([[HORSE_DRAW_LEFT, 0],
                           [HORSE_DRAW_LEFT+HORSE_DRAW_WIDTH, 0],
                           [HORSE_DRAW_LEFT+HORSE_DRAW_WIDTH, HORSE_DRAW_HEIGHT],
                           [HORSE_DRAW_LEFT, HORSE_DRAW_HEIGHT]])
INFO_REF = np.array([BOARD_REF[0,0]+20,BOARD_REF[0,1]+20])

DEFAULT_PARAMS = {
    'bottom': {'x': 50, 'y': 70, 'theta': 0},
    'bottom_shoulder': {'theta': 0, 'l': 80}, # Main bone
    'shoulder_head': {'theta': -60, 'l': 50},
    'head_nose': {'theta': 60, 'l': 25},
    'bottom_wrt_shoulder': {'x': -80, 'y': 0, 'theta': 0},
    'bottom_wrt_shoulder_rear_foot': {'theta': 90, 'l': 60},
    'shoulder_front_foot': {'theta': 90, 'l': 60},
    'bottom_wrt_shoulder_tail': {'theta': 150, 'l': 40},
}

TEST_PARAMS = [
    DEFAULT_PARAMS,
    {
        'bottom': {'x': 50, 'y': 70, 'theta': 0},
        'bottom_shoulder': {'theta': 0, 'l': 80}, # Main bone
        'shoulder_head': {'theta': -30, 'l': 50},
        'head_nose': {'theta': 60, 'l': 25},
        'bottom_wrt_shoulder': {'x': -80, 'y': 0, 'theta': 0},
        'bottom_wrt_shoulder_rear_foot': {'theta': 130, 'l': 60},
        'shoulder_front_foot': {'theta': 50, 'l': 60},
        'bottom_wrt_shoulder_tail': {'theta': 210, 'l': 40},
    },
    DEFAULT_PARAMS,
    {
        'bottom': {'x': 50, 'y': 70, 'theta': 0},
        'bottom_shoulder': {'theta': 0, 'l': 80}, # Main bone
        'shoulder_head': {'theta': -30, 'l': 50},
        'head_nose': {'theta': 60, 'l': 25},
        'bottom_wrt_shoulder': {'x': -80, 'y': 0, 'theta': 0},
        'bottom_wrt_shoulder_rear_foot': {'theta': 50, 'l': 60},
        'shoulder_front_foot': {'theta': 130, 'l': 60},
        'bottom_wrt_shoulder_tail': {'theta': 120, 'l': 40},
    },
]


# def deg2rad(deg):
#     return deg/180*np.pi
#
# # Transformation between two coordinate system
# def t(x, y, theta):
#     return np.array([[np.cos(theta),-np.sin(theta),x],
#                      [np.sin(theta),np.cos(theta),y],
#                      [0,0,1]])
#
# # Transformation from a line/bone
# # theta wrt +x axis, l length of line
# def t_line(l, theta):
#     return t(l*np.cos(theta), l*np.sin(theta), theta)

def params2bones(params):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    T_bottom2w = t(params['bottom']['x'],params['bottom']['y'], deg2rad(params['bottom']['theta']))
    T_shoulder2bottom = t_line(params['bottom_shoulder']['l'], deg2rad(params['bottom_shoulder']['theta']))
    T_head2shoulder = t_line(params['shoulder_head']['l'], deg2rad(params['shoulder_head']['theta']))
    T_nose2head = t_line(params['head_nose']['l'], deg2rad(params['head_nose']['theta']))

    T_bottom_wrt_shoulder2shoulder = t(params['bottom_wrt_shoulder']['x'],params['bottom_wrt_shoulder']['y'], deg2rad(params['bottom_wrt_shoulder']['theta']))
    T_rear_foot2bottom_wrt_shoulder = t_line(params['bottom_wrt_shoulder_rear_foot']['l'], deg2rad(params['bottom_wrt_shoulder_rear_foot']['theta']))
    T_front_foot2shoulder = t_line(params['shoulder_front_foot']['l'], deg2rad(params['shoulder_front_foot']['theta']))
    T_tail2bottom_wrt_shoulder = t_line(params['bottom_wrt_shoulder_tail']['l'], deg2rad(params['bottom_wrt_shoulder_tail']['theta']))

    p_bottom_world = (T_bottom2w @ origin)[0:2].reshape(2)
    p_shoulder_world = (T_bottom2w @ T_shoulder2bottom @ origin)[0:2].reshape(2)
    p_head_world = (T_bottom2w @ T_shoulder2bottom @ T_head2shoulder @ origin)[0:2].reshape(2)
    p_nose_world = (T_bottom2w @ T_shoulder2bottom @ T_head2shoulder @ T_nose2head @ origin)[0:2].reshape(2)

    # p_bottom_wrt_shoulder_world should be the same as p_bottom_world
    p_rear_foot_world = (T_bottom2w @ T_shoulder2bottom @ T_bottom_wrt_shoulder2shoulder @ T_rear_foot2bottom_wrt_shoulder @ origin)[0:2].reshape(2)
    p_front_foot_world = (T_bottom2w @ T_shoulder2bottom @ T_front_foot2shoulder @ origin)[0:2].reshape(2)
    p_tail_world = (T_bottom2w @ T_shoulder2bottom @ T_bottom_wrt_shoulder2shoulder @ T_tail2bottom_wrt_shoulder @ origin)[0:2].reshape(2)

    bones = np.array([
        [p_bottom_world, p_shoulder_world], # bottom_shoulder
        [p_shoulder_world, p_head_world], # shoulder_head
        [p_head_world, p_nose_world], # head_nose
        [p_bottom_world, p_rear_foot_world], # bottom_wrt_shoulder_rear_foot
        [p_shoulder_world, p_front_foot_world], # bottom_front_foot
        [p_bottom_world, p_tail_world], # bottom_wrt_shoulder_tail
    ]).reshape((-1,4))

    bones = bones*RATIO

    return bones

def params2frames(params):
    return params2bones_with_params2bones(params, params2bones)
