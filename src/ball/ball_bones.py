import numpy as np
import copy

from animator.bone import *

PINK_COLOR = (131, 124, 237)
GREEN_COLOR = (88, 206, 169)

# Adjust ratio to use full resolution of the drawing
# Without ratio, dimensions matches the AI design file
RATIO = 2.0

MARKER_SIZE = 120*RATIO
MARKER_Y_SPACE = 260*RATIO
BOARD_WIDTH = 400*RATIO
BOARD_HEIGHT = 300*RATIO
BALL_DRAW_LEFT = 530*RATIO
BALL_DRAW_TOP = 200*RATIO
BALL_DRAW_SIZE = 100*RATIO
BEAM_WIDTH = 300*RATIO
BEAM_HEIGHT = 60*RATIO

CORNERS_REF = {
    24: np.array([[0,0],
                  [MARKER_SIZE,0],
                  [MARKER_SIZE,MARKER_SIZE],
                  [0,MARKER_SIZE]]),
    6: np.array([[BOARD_WIDTH+MARKER_SIZE,0],
                 [BOARD_WIDTH+2*MARKER_SIZE,0],
                 [BOARD_WIDTH+2*MARKER_SIZE,MARKER_SIZE],
                 [BOARD_WIDTH+MARKER_SIZE,MARKER_SIZE]]),
    4: np.array([[BOARD_WIDTH+MARKER_SIZE,MARKER_SIZE+MARKER_Y_SPACE],
                 [BOARD_WIDTH+2*MARKER_SIZE,MARKER_SIZE+MARKER_Y_SPACE],
                 [BOARD_WIDTH+2*MARKER_SIZE,2*MARKER_SIZE+MARKER_Y_SPACE],
                 [BOARD_WIDTH+MARKER_SIZE,2*MARKER_SIZE+MARKER_Y_SPACE]]),
    46: np.array([[0,MARKER_SIZE+MARKER_Y_SPACE],
                  [MARKER_SIZE,MARKER_SIZE+MARKER_Y_SPACE],
                  [MARKER_SIZE,2*MARKER_SIZE+MARKER_Y_SPACE],
                  [0,2*MARKER_SIZE+MARKER_Y_SPACE]]),
}
BOARD_REF = np.array([[MARKER_SIZE,0],
                      [MARKER_SIZE+BOARD_WIDTH,0],
                      [MARKER_SIZE+BOARD_WIDTH,BOARD_HEIGHT],
                      [MARKER_SIZE,BOARD_HEIGHT]])
BALL_DRAW_REF = np.array([[BALL_DRAW_LEFT, BALL_DRAW_TOP],
                          [BALL_DRAW_LEFT+BALL_DRAW_SIZE, BALL_DRAW_TOP],
                          [BALL_DRAW_LEFT+BALL_DRAW_SIZE, BALL_DRAW_TOP+BALL_DRAW_SIZE],
                          [BALL_DRAW_LEFT, BALL_DRAW_TOP+BALL_DRAW_SIZE]])
INFO_REF = np.array([BOARD_REF[0,0]+20,BOARD_REF[0,1]+20])

# PARAMS dimension matches AI design file
# Times ratio to match the size of board
DEFAULT_PARAMS = {
    'base': {'x': 50, 'y': 50, 'theta': 0},
    'base_top': {'theta': -90, 'l': 30},
}

CCW_PARAMS = [
]
CCW_NUM_STEP = 6
for i in range(CCW_NUM_STEP):
    ps = copy.deepcopy(DEFAULT_PARAMS)
    ps['base_top']['theta'] = 90-(360/CCW_NUM_STEP)*i
    CCW_PARAMS.append(ps)

CW_PARAMS = [
]
for i in range(CCW_NUM_STEP):
    ps = copy.deepcopy(DEFAULT_PARAMS)
    ps['base_top']['theta'] = 90+(360/CCW_NUM_STEP)*i
    CW_PARAMS.append(ps)


def params2bones(params):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    T_base2w = t(params['base']['x'],params['base']['y'], deg2rad(params['base']['theta']))
    T_top2base = t_line(params['base_top']['l'], deg2rad(params['base_top']['theta']))

    p_base_world = (T_base2w @ origin)[0:2].reshape(2)
    p_top_world = (T_base2w @ T_top2base @ origin)[0:2].reshape(2)

    # Bones with larger index apear in the front
    bones = np.array([
        [p_base_world, p_top_world],
    ]).reshape((-1,4))

    bones = bones*RATIO

    return bones

def params2frames(params):
    return params2bones_with_params2bones(params, params2bones)
