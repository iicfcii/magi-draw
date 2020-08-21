import numpy as np
import cv2
import triangulation
import snake

# Calculate transformation matrix from two line segments
# Line segment represents the positive x axis wrt world
# Return matrix that transforms coordinate wrt next to coordinate wrt current
def calcTransMat(current, next):
    theta_c = np.arctan2(current[3]-current[1],current[2]-current[0])
    theta_n = np.arctan2(next[3]-next[1],next[2]-next[0])
    theta_n2c = theta_n-theta_c

    x_n2c = (next[0]-current[0])*np.cos(theta_c)+(next[1]-current[1])*np.sin(theta_c)
    y_n2c = -(next[0]-current[0])*np.sin(theta_c)+(next[1]-current[1])*np.cos(theta_c)

    r_n2c = np.array([[np.cos(theta_n2c),-np.sin(theta_n2c)],
                      [np.sin(theta_n2c),np.cos(theta_n2c)]])
    p_n2c = np.array([[x_n2c],
                      [y_n2c]])

    # r_c2n = np.transpose(r_n2c)
    # p_c2n = -r_c2n @ p_n2c

    t_n2c = np.array([[r_n2c[0,0],r_n2c[0,1],p_n2c[0,0]],
                      [r_n2c[1,0],r_n2c[1,1],p_n2c[1,0]],
                      [0,0,1]])
    # t_c2n = np.array([[r_c2n[0,0],r_c2n[0,1],p_c2n[0,0]],
    #                   [r_c2n[1,0],r_c2n[1,1],p_c2n[1,0]],
    #                   [0,0,1]])

    return t_n2c

def calcTransMatBetweenFrame(x_c, x_n):
    # Length may be useful for scaling
    length = np.sqrt((x_c[3]-x_c[1])**2+(x_c[2]-x_c[0])**2)
    x_world = [0,0,length,0]

    t_c2w = calcTransMat(x_world,x_c)
    t_w2c = calcTransMat(x_c,x_world)
    t_n2c = calcTransMat(x_c,x_n)

    return t_c2w @ t_n2c @ t_w2c

# Minimum distance between a point and line
def calcPointLineDistance(p, x, y):
    xp = p-x
    xy = y-x

    cross = xp[0]*xy[1]-xp[1]*xy[0]

    d = np.abs(cross/np.sqrt(xy[0]**2+xy[1]**2))

    return d
# print(calcPointLineDistance(np.array([0,0]),np.array([1,0]),np.array([0,1])))
# print(calcPointLineDistance(np.array([0,0]),np.array([1,1]),np.array([0,1])))

def animate(bones_c, bones_n, triangles):
    num_bones = len(bones_c)

    # Calculate weight for each point wrt each bone in first frame
    weights_and_position = {}
    for triangle in triangles:
        for point in triangle:
            point_key = tuple(point)
            if point_key in weights_and_position:
                pass
            else:
                w = np.zeros(num_bones)
                for i, bone in enumerate(bones_c):
                    dist = calcPointLineDistance(point,bone[0:2],bone[2:4])
                    w[i] = 1/(dist**2) # w = 1/d^2

                w = w/np.sum(w) # Make sure sum of weights is 1
                weights_and_position[point_key] = {'weight': w}

    # Calculate transformation matrix to next frame for each bone
    transformations = []
    for i in range(num_bones):
        bone_c = bones_c[i]
        bone_n = bones_n[i]
        t = calcTransMatBetweenFrame(bone_c,bone_n) # Bone is always along positive x of its own coordinate system
        # print(bone_n)
        # print(t @ np.array([[bone_c[0]],[bone_c[1]],[1]]))
        # print(t @ np.array([[bone_c[2]],[bone_c[3]],[1]]))

        transformations.append(t)
    # print(transformations)
    transformations = np.concatenate(transformations, axis=0) # Shape 3n*3

    # Calculate next frame position for each point
    for point_key in weights_and_position.keys():
        point_c = np.array([[point_key[0]],[point_key[1]],[1]])
        point_n = transformations @ point_c
        point_n = np.transpose(point_n).reshape((2,3))[:,0:2]
        w = weights_and_position[point_key]['weight']
        position = w @ point_n
        # print(point_n)
        # print(w)
        # print(position)
        weights_and_position[point_key]['position'] = position

    triangles_n = []
    for triangle in triangles:
        triangle_n = np.zeros((3,2))
        for i, point in enumerate(triangle):
            point_key = tuple(point)
            triangle_n[i,:] = weights_and_position[point_key]['position']

        triangles_n.append(triangle_n.astype(np.float32))

    return triangles_n
