import numpy as np
import cv2

# Bone parameters for each frame
# xw yw thetaw theta1 a1 theta2 a2
bones_parameters = [
    (120,70,0/180*np.pi,10/180*np.pi,180,105/180*np.pi,320),
    (120,70,0/180*np.pi,30/180*np.pi,180,105/180*np.pi,320),
    (120,70,0/180*np.pi,30/180*np.pi,180,75/180*np.pi,320),
    (200,200,0/180*np.pi,30/180*np.pi,180,75/180*np.pi,320),
]

def t0W(xw, yw, thetaw):
    return np.array([[np.cos(thetaw),-np.sin(thetaw),0,xw],
                     [np.sin(thetaw),np.cos(thetaw),0,yw],
                     [0,0,1,0],
                     [0,0,0,1]])
# Transformationa1
def t10(theta1,a1):
    return np.array([[np.cos(theta1),-np.sin(theta1),0,a1*np.cos(theta1)],
                     [np.sin(theta1),np.cos(theta1),0,a1*np.sin(theta1)],
                     [0,0,1,0],
                     [0,0,0,1]])

def t21(theta2,a2):
    return np.array([[np.cos(theta2),-np.sin(theta2),0,a2*np.cos(theta2)],
                     [np.sin(theta2),np.cos(theta2),0,a2*np.sin(theta2)],
                     [0,0,1,0],
                     [0,0,0,1]])

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

    origin = [[0],[0],[0],[1]] # Origin of each coordinate system
    p0W = (t0W(xw,yw,thetaw) @ origin)[0:2].reshape(2)
    p1W = (t0W(xw,yw,thetaw) @ t10(theta1, a1) @ origin)[0:2].reshape(2)
    p2W = (t0W(xw,yw,thetaw) @ t10(theta1, a1) @ t21(theta2, a2) @ origin)[0:2].reshape(2)

    bone1 = np.array([p0W,p1W])
    bone2 = np.array([p1W,p2W])
    bones = np.array([bone1,bone2]).reshape((-1,2))

    bones_frames.append(bones)
