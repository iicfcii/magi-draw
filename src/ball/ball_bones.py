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

# # PARAMS dimension matches AI design file
# # Times ratio to match the size of board
# DEFAULT_PARAMS = {
#     'base': {'x': 60, 'y': 65, 'theta': 0},
# }
#
# def params2bones(params):
#     origin = [[0],[0],[1]] # Origin of each coordinate system
#
#     T_base2w = t(params['base']['x'],params['base']['y'], deg2rad(params['base']['theta']))
#     T_neck2base = t_line(params['base_neck']['l'], deg2rad(params['base_neck']['theta']))
#     T_head2neck = t_line(params['neck_head']['l'], deg2rad(params['neck_head']['theta']))
#     T_nose2head = t_line(params['head_nose']['l'], deg2rad(params['head_nose']['theta']))
#
#     T_hip2neck = t(params['hip']['x'],params['hip']['y'], deg2rad(params['hip']['theta']))
#     T_rear_foot2hip = t_line(params['hip_rear_foot']['l'], deg2rad(params['hip_rear_foot']['theta']))
#     T_shoulder2neck = t(params['shoulder']['x'],params['shoulder']['y'], deg2rad(params['shoulder']['theta']))
#     T_front_foot2shoulder = t_line(params['shoulder_front_foot']['l'], deg2rad(params['shoulder_front_foot']['theta']))
#     T_bottom2neck = t(params['bottom']['x'],params['bottom']['y'], deg2rad(params['bottom']['theta']))
#     T_tail2bottom = t_line(params['bottom_tail']['l'], deg2rad(params['bottom_tail']['theta']))
#
#     T_breast2neck = t(params['breast']['x'],params['breast']['y'], deg2rad(params['breast']['theta']))
#     T_belly2breast = t_line(params['breast_belly']['l'], deg2rad(params['breast_belly']['theta']))
#
#     p_base_world = (T_base2w @ origin)[0:2].reshape(2)
#     p_neck_world = (T_base2w @ T_neck2base @ origin)[0:2].reshape(2)
#     p_head_world = (T_base2w @ T_neck2base @ T_head2neck @ origin)[0:2].reshape(2)
#     p_nose_world = (T_base2w @ T_neck2base @ T_head2neck @ T_nose2head @ origin)[0:2].reshape(2)
#
#     # p_hip_world should be the same as p_base_world
#     p_hip_world = (T_base2w @ T_neck2base @ T_hip2neck @ origin)[0:2].reshape(2)
#     p_rear_foot_world = (T_base2w @ T_neck2base @ T_hip2neck @ T_rear_foot2hip @ origin)[0:2].reshape(2)
#     p_shoulder_world = (T_base2w @ T_neck2base @ T_shoulder2neck @ origin)[0:2].reshape(2)
#     p_front_foot_world = (T_base2w @ T_neck2base @ T_shoulder2neck @ T_front_foot2shoulder @ origin)[0:2].reshape(2)
#
#     p_bottom_world = (T_base2w @ T_neck2base @ T_bottom2neck @ origin)[0:2].reshape(2)
#     p_tail_world = (T_base2w @ T_neck2base @ T_bottom2neck @ T_tail2bottom @ origin)[0:2].reshape(2)
#
#     p_breast_world = (T_base2w @ T_neck2base @ T_breast2neck @ origin)[0:2].reshape(2)
#     p_belly_world = (T_base2w @ T_neck2base @ T_breast2neck @ T_belly2breast @ origin)[0:2].reshape(2)
#
#     # Bones with larger index apear in the front
#     bones = np.array([
#         [p_base_world, p_neck_world], # base_neck
#         [p_neck_world, p_head_world], # neck_head
#         [p_head_world, p_nose_world], # head_nose
#         [p_hip_world, p_rear_foot_world], # hip_rear_foot
#         [p_shoulder_world, p_front_foot_world], # shoulder_front_foot
#         [p_bottom_world, p_tail_world], # bottom_tail
#         [p_belly_world, p_breast_world], # breast_belly
#     ]).reshape((-1,4))
#
#     bones = bones*RATIO
#
#     return bones
#
# def params2frames(params):
#     return params2bones_with_params2bones(params, params2bones)
