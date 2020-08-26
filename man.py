import numpy as np
import cv2

# Man bones
def t_bottom2world(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_neck2bottom(theta, l):
    return np.array([[np.cos(theta),-np.sin(theta),l*np.cos(theta)],
                     [np.sin(theta),np.cos(theta),l*np.sin(theta)],
                     [0,0,1]])

def t_head2neck(theta, l):
    return np.array([[np.cos(theta),-np.sin(theta),l*np.cos(theta)],
                     [np.sin(theta),np.cos(theta),l*np.sin(theta)],
                     [0,0,1]])

def t_leftShoulder2neck(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_leftHand2leftShoulder(theta, l):
    return np.array([[np.cos(theta),-np.sin(theta),l*np.cos(theta)],
                     [np.sin(theta),np.cos(theta),l*np.sin(theta)],
                     [0,0,1]])

def t_rightShoulder2neck(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_rightHand2rightShoulder(theta, l):
    return np.array([[np.cos(theta),-np.sin(theta),l*np.cos(theta)],
                     [np.sin(theta),np.cos(theta),l*np.sin(theta)],
                     [0,0,1]])

def t_leftHip2world(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_leftHip2leftFoot(theta, l):
    return np.array([[np.cos(theta),-np.sin(theta),l*np.cos(theta)],
                     [np.sin(theta),np.cos(theta),l*np.sin(theta)],
                     [0,0,1]])

def t_rightHip2world(x, y, theta):
    return np.array([[np.cos(theta),-np.sin(theta),x],
                     [np.sin(theta),np.cos(theta),y],
                     [0,0,1]])

def t_rightHip2rightFoot(theta, l):
    return np.array([[np.cos(theta),-np.sin(theta),l*np.cos(theta)],
                     [np.sin(theta),np.cos(theta),l*np.sin(theta)],
                     [0,0,1]])

def deg2rad(deg):
    return deg/180*np.pi

def bones(bones_parameters):
    origin = [[0],[0],[1]] # Origin of each coordinate system

    t_b2w = t_bottom2world(bones_parameters['bottom']['x'],bones_parameters['bottom']['y'],deg2rad(bones_parameters['bottom']['theta']))
    t_n2b = t_neck2bottom(deg2rad(bones_parameters['bottom_neck']['theta']),bones_parameters['bottom_neck']['l'])
    t_h2n = t_head2neck(deg2rad(bones_parameters['neck_head']['theta']),bones_parameters['neck_head']['l'])
    p_bottom_world = (t_b2w @ origin)[0:2].reshape(2)
    p_neck_world = (t_b2w @ t_n2b @ origin)[0:2].reshape(2)
    p_head_world = (t_b2w @ t_n2b @ t_h2n @ origin)[0:2].reshape(2)

    t_ls2n = t_leftShoulder2neck(bones_parameters['leftShoulder']['x'],bones_parameters['leftShoulder']['y'],deg2rad(bones_parameters['leftShoulder']['theta']))
    t_lh2ls = t_leftHand2leftShoulder(deg2rad(bones_parameters['leftShoulder_leftHand']['theta']),bones_parameters['leftShoulder_leftHand']['l'])
    p_leftShoulder_world = (t_b2w @ t_n2b @ t_ls2n @ origin)[0:2].reshape(2)
    p_leftHand_world = (t_b2w @ t_n2b @ t_ls2n @ t_lh2ls @ origin)[0:2].reshape(2)

    t_rs2n = t_rightShoulder2neck(bones_parameters['rightShoulder']['x'],bones_parameters['rightShoulder']['y'],deg2rad(bones_parameters['rightShoulder']['theta']))
    t_rh2rs = t_rightHand2rightShoulder(deg2rad(bones_parameters['rightShoulder_rightHand']['theta']),bones_parameters['rightShoulder_rightHand']['l'])
    p_rightShoulder_world = (t_b2w @ t_n2b @ t_rs2n @ origin)[0:2].reshape(2)
    p_rightHand_world = (t_b2w @ t_n2b @ t_rs2n @ t_rh2rs @ origin)[0:2].reshape(2)

    t_lh2w = t_leftHip2world(bones_parameters['leftHip']['x'],bones_parameters['leftHip']['y'],deg2rad(bones_parameters['leftHip']['theta']))
    t_lh2lf = t_leftHip2leftFoot(deg2rad(bones_parameters['leftHip_leftFoot']['theta']),bones_parameters['leftHip_leftFoot']['l'])
    p_leftHip_world = (t_lh2w @ origin)[0:2].reshape(2)
    p_leftFoot_world = (t_lh2w @ t_lh2lf @ origin)[0:2].reshape(2)

    t_rh2w = t_rightHip2world(bones_parameters['rightHip']['x'],bones_parameters['rightHip']['y'],deg2rad(bones_parameters['rightHip']['theta']))
    t_rh2rf = t_rightHip2rightFoot(deg2rad(bones_parameters['rightHip_rightFoot']['theta']),bones_parameters['rightHip_rightFoot']['l'])
    p_rightHip_world = (t_rh2w @ origin)[0:2].reshape(2)
    p_rightFoot_world = (t_rh2w @ t_rh2rf @ origin)[0:2].reshape(2)

    bone_bottom2neck = np.array([p_bottom_world,p_neck_world])
    bone_neck2head = np.array([p_neck_world,p_head_world])
    bone_leftShoulder2leftHand = np.array([p_leftShoulder_world,p_leftHand_world])
    bone_rightShoulder2rightHand = np.array([p_rightShoulder_world,p_rightHand_world])
    bone_leftHip2leftFoot = np.array([p_leftHip_world,p_leftFoot_world])
    bone_rightHip2rightFoot = np.array([p_rightHip_world,p_rightFoot_world])

    bones = np.array([
        bone_bottom2neck,
        bone_neck2head,
        bone_leftShoulder2leftHand,
        bone_rightShoulder2rightHand,
        bone_leftHip2leftFoot,
        bone_rightHip2rightFoot
    ]).reshape((-1,4))

    return bones

img = cv2.imread('img/man.jpg', 1)

# Default bones parameters for each frame
# bones_default_parameters = (200,200,0,-90,100,0,40,165,120,0,140,80)
bones_default_parameters = {
    'bottom': {'x':200, 'y': 200, 'theta': -90},
    'bottom_neck': {'theta': 0, 'l': 100},
    'neck_head': {'theta': 0, 'l': 40},
    'leftShoulder': {'x':-20, 'y': -35, 'theta': -130},
    'leftShoulder_leftHand': {'theta': 0, 'l': 80},
    'rightShoulder': {'x':-20, 'y': 35, 'theta': 130},
    'rightShoulder_rightHand': {'theta': 0, 'l': 80},
    'leftHip': {'x':180, 'y': 200, 'theta': 95},
    'leftHip_leftFoot': {'theta': 0, 'l': 140},
    'rightHip': {'x':220, 'y': 200, 'theta': 85},
    'rightHip_rightFoot': {'theta': 0, 'l': 140},
}
bones_default = bones(bones_default_parameters)

bones_frames_parameters = [
    bones_default_parameters,
    {
        'bottom': {'x':200, 'y': 200, 'theta': -90},
        'bottom_neck': {'theta': 0, 'l': 100},
        'neck_head': {'theta': 0, 'l': 40},
        'leftShoulder': {'x':-20, 'y': -35, 'theta': -130},
        'leftShoulder_leftHand': {'theta': -40, 'l': 80},
        'rightShoulder': {'x':-20, 'y': 35, 'theta': 130},
        'rightShoulder_rightHand': {'theta': 40, 'l': 80},
        'leftHip': {'x':180, 'y': 200, 'theta': 90},
        'leftHip_leftFoot': {'theta': 0, 'l': 140},
        'rightHip': {'x':220, 'y': 200, 'theta': 90},
        'rightHip_rightFoot': {'theta': 0, 'l': 140},
    },
    {
        'bottom': {'x':200, 'y': 200, 'theta': -90},
        'bottom_neck': {'theta': 0, 'l': 100},
        'neck_head': {'theta': 0, 'l': 40},
        'leftShoulder': {'x':-20, 'y': -35, 'theta': -130},
        'leftShoulder_leftHand': {'theta': 60, 'l': 80},
        'rightShoulder': {'x':-20, 'y': 35, 'theta': 130},
        'rightShoulder_rightHand': {'theta': -60, 'l': 80},
        'leftHip': {'x':180, 'y': 200, 'theta': 90},
        'leftHip_leftFoot': {'theta': 20, 'l': 140},
        'rightHip': {'x':220, 'y': 200, 'theta': 90},
        'rightHip_rightFoot': {'theta': -20, 'l': 140},
    },
    bones_default_parameters,
]
bones_frames = []
for para in bones_frames_parameters:
    bones_frames.append(bones(para))

for frame in bones_frames:
    img_tmp = img.copy()
    for bone in frame:
        bone = bone.astype(np.int32)
        cv2.polylines(img_tmp, [bone.reshape((2,2)).astype(np.int32)], True, (255,0,0), 2)
        cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
        cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
    cv2.imshow('Man',img_tmp)
    cv2.waitKey(0)
cv2.destroyAllWindows()
