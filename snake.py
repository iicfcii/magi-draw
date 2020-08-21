import numpy as np
import cv2

img_src = cv2.imread('img/snake.jpg', 1)

# Bone parameters for each frame
# xw yw thetaw theta1 a1 theta2 a2 theta3 a3
bones_parameters = [
    (260,450,0/180*np.pi,-60/180*np.pi,320,-100/180*np.pi,200,-10/180*np.pi,400),
    (260,450,0/180*np.pi,-70/180*np.pi,320,-90/180*np.pi,200,-10/180*np.pi,400),
    (260,450,0/180*np.pi,-80/180*np.pi,320,-80/180*np.pi,200,-10/180*np.pi,400),
    (260,450,0/180*np.pi,-90/180*np.pi,320,-70/180*np.pi,200,-10/180*np.pi,400),
    (260,450,0/180*np.pi,-80/180*np.pi,320,-80/180*np.pi,200,-10/180*np.pi,400),
    (260,450,0/180*np.pi,-70/180*np.pi,320,-90/180*np.pi,200,-10/180*np.pi,400),
    (260,450,0/180*np.pi,-60/180*np.pi,320,-100/180*np.pi,200,-10/180*np.pi,400),
]

# Transformation
# Coordinate system is assgined so that
# Positive x axis is the bone direction
def t0W(xw, yw, thetaw):
    return np.array([[np.cos(thetaw),-np.sin(thetaw),xw],
                     [np.sin(thetaw),np.cos(thetaw),yw],
                     [0,0,1]])

def t10(theta1,a1):
    return np.array([[np.cos(theta1),-np.sin(theta1),a1*np.cos(theta1)],
                     [np.sin(theta1),np.cos(theta1),a1*np.sin(theta1)],
                     [0,0,1]])

def t21(theta2,a2):
    return np.array([[np.cos(theta2),-np.sin(theta2),a2*np.cos(theta2)],
                     [np.sin(theta2),np.cos(theta2),a2*np.sin(theta2)],
                     [0,0,1]])

# Tail
def t_tail2base(theta3,a3):
    return np.array([[np.cos(theta3),-np.sin(theta3),a3*np.cos(theta3)],
                     [np.sin(theta3),np.cos(theta3),a3*np.sin(theta3)],
                     [0,0,1]])


bones_frames = []
for para in bones_parameters:
    # Bone position
    xw = para[0]
    yw = para[1]
    thetaw = para[2]
    theta1 = para[3]
    a1 = para[4]
    theta2 = para[5]
    a2 = para[6]
    theta3 = para[7]
    a3 = para[8]

    origin = [[0],[0],[1]] # Origin of each coordinate system
    p0W = (t0W(xw,yw,thetaw) @ origin)[0:2].reshape(2)
    p1W = (t0W(xw,yw,thetaw) @ t10(theta1, a1) @ origin)[0:2].reshape(2)
    p2W = (t0W(xw,yw,thetaw) @ t10(theta1, a1) @ t21(theta2, a2) @ origin)[0:2].reshape(2)
    p_tail_world = (t0W(xw,yw,thetaw) @ t_tail2base(theta3, a3) @ origin)[0:2].reshape(2)

    bone1 = np.array([p0W,p1W])
    bone2 = np.array([p1W,p2W])
    bone3 = np.array([p0W,p_tail_world])
    bones = np.array([bone1,bone2,bone3]).reshape((-1,4))

    bones_frames.append(bones)

# print(bones_frames)
