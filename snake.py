import numpy as np
import cv2

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

def bones(bones_parameters):
    x = bones_parameters[0]
    y = bones_parameters[1]
    theta = deg2rad(bones_parameters[2])
    theta1 = deg2rad(bones_parameters[3])
    l1 = bones_parameters[4]
    theta2 = deg2rad(bones_parameters[5])
    l2 = bones_parameters[6]
    theta3 = deg2rad(bones_parameters[7])
    l3 = bones_parameters[8]
    theta4 = deg2rad(bones_parameters[9])
    l4 = bones_parameters[10]

    origin = [[0],[0],[1]] # Origin of each coordinate system
    p_tail_world = (t_tail2world(x,y,theta) @ origin)[0:2].reshape(2)
    p_a_world = (t_tail2world(x,y,theta) @ t_a2tail(theta1,l1) @ origin)[0:2].reshape(2)
    p_b_world = (t_tail2world(x,y,theta) @ t_a2tail(theta1,l1) @ t_b2a(theta2,l2) @ origin)[0:2].reshape(2)
    p_c_world = (t_tail2world(x,y,theta) @ t_a2tail(theta1,l1) @ t_b2a(theta2,l2) @ t_c2b(theta3,l3) @ origin)[0:2].reshape(2)
    p_head_world = (t_tail2world(x,y,theta) @ t_a2tail(theta1,l1) @ t_b2a(theta2,l2) @ t_c2b(theta3,l3) @ t_head2c(theta4,l4) @ origin)[0:2].reshape(2)

    bone_tail2a = np.array([p_tail_world,p_a_world])
    bone_a2b = np.array([p_a_world,p_b_world])
    bone_b2c = np.array([p_b_world,p_c_world])
    bone_c2head = np.array([p_c_world,p_head_world])
    bones = np.array([bone_tail2a,bone_a2b,bone_b2c,bone_c2head]).reshape((-1,4))

    return bones

img = cv2.imread('img/snake.jpg', 1)

# Default bones parameters for each frame
# x y theta theta1 l1 theta2 l2 theta3 l3 theta3 l3 theta4 l4
bones_default_parameters = (60,100,0,0,80,0,100,0,100,0,80)
bones_default = bones(bones_default_parameters)

bones_frames_parameters = [
    (60,100,0,0,80,0,100,0,100,0,80),
    (60,100,0,15,80,-30,100,30,100,-30,80),
    (60,100,0,30,80,-60,100,60,100,-60,80),
    (60,100,0,15,80,-30,100,30,100,-30,80),
    (60,100,0,0,80,0,100,0,100,0,80),
    (60,100,0,-15,80,30,100,-30,100,30,80),
    (60,100,0,-30,80,60,100,-60,100,60,80),
    (60,100,0,-15,80,30,100,-30,100,30,80),
    (60,100,0,0,80,0,100,0,100,0,80),
    (60,100,0,0,80,0,100,30,100,-30,80),
    (60,100,0,0,80,0,100,0,100,0,80),
    (60,100,0,0,80,0,100,-30,100,30,80),
    (60,100,0,0,80,0,100,0,100,0,80),
]
bones_frames = []
for para in bones_frames_parameters:
    bones_frames.append(bones(para))

# img_bones = np.zeros(img.shape, np.uint8)
# img_bones[:,:] = (255,255,255)
# for bone in bones_default:
#     bone = bone.astype(np.int32)
#     cv2.polylines(img_bones, [bone.reshape((2,2)).astype(np.int32)], True, (255,0,0), 2)
#     cv2.circle(img_bones, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
#     cv2.circle(img_bones, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
# cv2.imwrite('img/snake_bones.jpg',img_bones)

# for frame in bones_frames:
#     img_tmp = img.copy()
#     for bone in frame:
#         bone = bone.astype(np.int32)
#         cv2.polylines(img_tmp, [bone.reshape((2,2)).astype(np.int32)], True, (255,0,0), 2)
#         cv2.circle(img_tmp, tuple(bone[0:2]), 5, (0,0,0), thickness=-1)
#         cv2.circle(img_tmp, tuple(bone[2:4]), 5, (0,0,0), thickness=-1)
#     cv2.imshow('Snake',img_tmp)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()
