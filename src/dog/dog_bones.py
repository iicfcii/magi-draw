import numpy as np
import copy

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
    'base': {'x': 50, 'y': 70, 'theta': 0},
    'base_neck': {'theta': 0, 'l': 80}, # Main bone
    'neck_head': {'theta': -60, 'l': 45},
    'head_nose': {'theta': 60, 'l': 25},
    'hip': {'x': -80, 'y': 10, 'theta': 0},
    'hip_rear_foot': {'theta': 90, 'l': 50},
    'shoulder': {'x': 0, 'y': 10, 'theta': 0},
    'shoulder_front_foot': {'theta': 90, 'l': 50},
    'bottom': {'x': -80, 'y': 0, 'theta': 0},
    'bottom_tail': {'theta': 150, 'l': 40},
    'breast': {'x': -10, 'y': 15, 'theta': 0},
    'breast_belly': {'theta': 180, 'l': 50},
}

WALK_RIGHT_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
WALK_RIGHT_PARAMS['hip_rear_foot']['theta'] = 60
WALK_RIGHT_PARAMS['shoulder_front_foot']['theta'] = 60
WALK_RIGHT_PARAMS['bottom_tail']['theta'] = 180
WALK_RIGHT_PARAMS['neck_head']['theta'] = -40
WALK_RIGHT_PARAMS['head_nose']['theta'] = 30

WALK_LEFT_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
WALK_LEFT_PARAMS['hip_rear_foot']['theta'] = 120
WALK_LEFT_PARAMS['shoulder_front_foot']['theta'] = 120
WALK_LEFT_PARAMS['bottom_tail']['theta'] = 180
WALK_LEFT_PARAMS['neck_head']['theta'] = -40
WALK_LEFT_PARAMS['head_nose']['theta'] = 30

WALK_FRONT_PARAMS = [
    DEFAULT_PARAMS,
    WALK_RIGHT_PARAMS,
    DEFAULT_PARAMS,
    WALK_LEFT_PARAMS,
]

WALK_BACK_PARAMS = [
    DEFAULT_PARAMS,
    WALK_LEFT_PARAMS,
    DEFAULT_PARAMS,
    WALK_RIGHT_PARAMS,
]

RUN_1_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
RUN_1_PARAMS['hip_rear_foot']['theta'] = 45
RUN_1_PARAMS['shoulder_front_foot']['theta'] = 120

RUN_2_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
RUN_2_PARAMS['hip_rear_foot']['theta'] = 120
RUN_2_PARAMS['shoulder_front_foot']['theta'] = 45

RUN_3_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
RUN_3_PARAMS['hip_rear_foot']['theta'] = 90
RUN_3_PARAMS['shoulder_front_foot']['theta'] = 90

RUN_1_BACK_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
RUN_1_BACK_PARAMS['hip_rear_foot']['theta'] = 60
RUN_1_BACK_PARAMS['shoulder_front_foot']['theta'] = 135

RUN_2_BACK_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
RUN_2_BACK_PARAMS['hip_rear_foot']['theta'] = 135
RUN_2_BACK_PARAMS['shoulder_front_foot']['theta'] = 60

RUN_3_BACK_PARAMS = copy.deepcopy(DEFAULT_PARAMS)
RUN_3_BACK_PARAMS['hip_rear_foot']['theta'] = 90
RUN_3_BACK_PARAMS['shoulder_front_foot']['theta'] = 90

RUN_FRONT_PARAMS = [
    RUN_1_PARAMS,
    RUN_2_PARAMS,
    RUN_3_PARAMS,
]

RUN_BACK_PARAMS = [
    RUN_1_BACK_PARAMS,
    RUN_2_BACK_PARAMS,
    RUN_3_BACK_PARAMS,
]


def params2bones(params):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    T_base2w = t(params['base']['x'],params['base']['y'], deg2rad(params['base']['theta']))
    T_neck2base = t_line(params['base_neck']['l'], deg2rad(params['base_neck']['theta']))
    T_head2neck = t_line(params['neck_head']['l'], deg2rad(params['neck_head']['theta']))
    T_nose2head = t_line(params['head_nose']['l'], deg2rad(params['head_nose']['theta']))

    T_hip2neck = t(params['hip']['x'],params['hip']['y'], deg2rad(params['hip']['theta']))
    T_rear_foot2hip = t_line(params['hip_rear_foot']['l'], deg2rad(params['hip_rear_foot']['theta']))
    T_shoulder2neck = t(params['shoulder']['x'],params['shoulder']['y'], deg2rad(params['shoulder']['theta']))
    T_front_foot2shoulder = t_line(params['shoulder_front_foot']['l'], deg2rad(params['shoulder_front_foot']['theta']))
    T_bottom2neck = t(params['bottom']['x'],params['bottom']['y'], deg2rad(params['bottom']['theta']))
    T_tail2bottom = t_line(params['bottom_tail']['l'], deg2rad(params['bottom_tail']['theta']))

    T_breast2neck = t(params['breast']['x'],params['breast']['y'], deg2rad(params['breast']['theta']))
    T_belly2breast = t_line(params['breast_belly']['l'], deg2rad(params['breast_belly']['theta']))

    p_base_world = (T_base2w @ origin)[0:2].reshape(2)
    p_neck_world = (T_base2w @ T_neck2base @ origin)[0:2].reshape(2)
    p_head_world = (T_base2w @ T_neck2base @ T_head2neck @ origin)[0:2].reshape(2)
    p_nose_world = (T_base2w @ T_neck2base @ T_head2neck @ T_nose2head @ origin)[0:2].reshape(2)

    # p_hip_world should be the same as p_base_world
    p_hip_world = (T_base2w @ T_neck2base @ T_hip2neck @ origin)[0:2].reshape(2)
    p_rear_foot_world = (T_base2w @ T_neck2base @ T_hip2neck @ T_rear_foot2hip @ origin)[0:2].reshape(2)
    p_shoulder_world = (T_base2w @ T_neck2base @ T_shoulder2neck @ origin)[0:2].reshape(2)
    p_front_foot_world = (T_base2w @ T_neck2base @ T_shoulder2neck @ T_front_foot2shoulder @ origin)[0:2].reshape(2)

    p_bottom_world = (T_base2w @ T_neck2base @ T_bottom2neck @ origin)[0:2].reshape(2)
    p_tail_world = (T_base2w @ T_neck2base @ T_bottom2neck @ T_tail2bottom @ origin)[0:2].reshape(2)

    p_breast_world = (T_base2w @ T_neck2base @ T_breast2neck @ origin)[0:2].reshape(2)
    p_belly_world = (T_base2w @ T_neck2base @ T_breast2neck @ T_belly2breast @ origin)[0:2].reshape(2)

    # Bones with larger index apear in the front
    bones = np.array([
        [p_base_world, p_neck_world], # base_neck
        [p_neck_world, p_head_world], # neck_head
        [p_head_world, p_nose_world], # head_nose
        [p_hip_world, p_rear_foot_world], # hip_rear_foot
        [p_shoulder_world, p_front_foot_world], # shoulder_front_foot
        [p_bottom_world, p_tail_world], # bottom_tail
        [p_belly_world, p_breast_world], # breast_belly
    ]).reshape((-1,4))

    bones = bones*RATIO

    return bones

def params2frames(params):
    return params2bones_with_params2bones(params, params2bones)
