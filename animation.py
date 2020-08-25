import numpy as np
import cv2
import triangulation

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

def calcPointProjectionOutsideLine(p, x, y):
    xp = p-x
    xy = y-x

    dot = xp[0]*xy[0]+xp[1]*xy[1]
    l = np.sqrt(xy[0]**2+xy[1]**2)
    proj = dot/l

    if proj > l: return 1 # Outside y
    if proj < 0: return -1 # Outside x
    return 0 # On line
# print(calcPointProjectionOutsideLine(np.array([0,0]),np.array([1,0]),np.array([0,1])))
# print(calcPointProjectionOutsideLine(np.array([0,0]),np.array([1,0]),np.array([2,-1])))
# print(calcPointProjectionOutsideLine(np.array([0,0]),np.array([1,1]),np.array([0,1])))
# print(calcPointProjectionOutsideLine(np.array([0,0]),np.array([1,1]),np.array([1,2])))
# print(calcPointProjectionOutsideLine(np.array([0,0]),np.array([1,1]),np.array([2,1])))
# print(calcPointProjectionOutsideLine(np.array([0,0]),np.array([-2,1]),np.array([-1,1])))

def calcPointPointDistance(x, y):
    xy = y-x
    d = np.sqrt(xy[0]**2+xy[1]**2)
    return d

def calcWeight(p, bone):
    outside = calcPointProjectionOutsideLine(p, bone[0:2], bone[2:4])

    if outside == 1:
        d = calcPointPointDistance(p, bone[2:4])
    elif outside == -1:
        d = calcPointPointDistance(p, bone[0:2])
    else:
        d = calcPointLineDistance(p,bone[0:2], bone[2:4])
    weight = np.exp(-0.05*d) # w =e^(-Cd)

    return weight

# TODO: Points near bone should have weights closer to 1
def calcWeights(bones_default, triangles):
    weights = {}
    for triangle in triangles:
        for point in triangle:
            point_key = tuple(point)
            if point_key in weights:
                pass
            else:
                w = np.zeros(len(bones_default))
                for i, bone in enumerate(bones_default):
                    w[i] = calcWeight(point, bone)
                w = w/np.sum(w) # Make sure sum of weights is 1
                weights[point_key] = {'weight': w}

    return weights

def animate(bones_default, bones_n, triangles, weights):
    # Calculate transformation matrix to next frame for each bone
    transformations = []
    for i in range(len(bones_default)):
        bone_c = bones_default[i]
        bone_n = bones_n[i]
        t = calcTransMatBetweenFrame(bone_c,bone_n) # Bone is always along positive x of its own coordinate system
        # print(bone_n)
        # print(t @ np.array([[bone_c[0]],[bone_c[1]],[1]]))
        # print(t @ np.array([[bone_c[2]],[bone_c[3]],[1]]))

        transformations.append(t)

    transformations = np.concatenate(transformations, axis=0) # Shape 3n*3

    # Calculate next frame position for each point
    for point_key in weights.keys():
        point_c = np.array([[point_key[0]],[point_key[1]],[1]])
        point_n = transformations @ point_c
        # print(point_n)
        point_n = np.transpose(point_n).reshape((-1,3))[:,0:2]
        w = weights[point_key]['weight']
        position = w @ point_n
        # print(point_n)
        # print(w)
        # print(position)
        weights[point_key]['position'] = position

    triangles_n = []
    for triangle in triangles:
        triangle_n = np.zeros((3,2))
        for i, point in enumerate(triangle):
            point_key = tuple(point)
            triangle_n[i,:] = weights[point_key]['position']

        triangles_n.append(triangle_n.astype(np.float32))

    return triangles_n

def warp(img, triangles, triangles_next, bone):
    rect_n = cv2.boundingRect(np.concatenate(triangles_next,axis=0)) # x, y, w, h
    img_n = np.zeros((rect_n[3],rect_n[2],3), np.uint8)
    img_n[:,:] = (255,255,255)

    # Anchor position wrt to image
    # Anchor is the first point of first bone
    anchor = np.array([bone[0]-rect_n[0],bone[1]-rect_n[1]],dtype=np.int32)

    rect_offset = 5 # Offset bounding rectangle so won't pick black pixle
    for triangle_c, triangle_n in zip(triangles, triangles_next):
        # Get signle bounding rectangle for current and next
        # Take out of image into account
        (x,y,w,h) = cv2.boundingRect(np.concatenate((triangle_c,triangle_n),axis=0))
        x -= rect_offset
        y -= rect_offset
        w += rect_offset*2
        h += rect_offset*2
        x_valid = np.maximum(x,0)
        y_valid = np.maximum(y,0)
        w_valid = np.minimum(w,img.shape[1]-x)
        h_valid = np.minimum(h,img.shape[0]-y)

        # Warp from current to next
        # Calculate coordinate wrt to bounding rectangle
        triangle_c_offset = triangle_c-np.tile(np.array([x,y]),(3,1)).astype(np.float32)
        triangle_n_offset = triangle_n-np.tile(np.array([x,y]),(3,1)).astype(np.float32)

        img_triangle_c = np.zeros((h,w,3), np.uint8)
        # print(img_triangle_c[y_valid-y:y_valid-y+h_valid-(y_valid-y), x_valid-x:x_valid-x+w_valid-(x_valid-x)].shape)
        # print(img[y_valid-y+y:y_valid+h_valid-(y_valid-y), x_valid-x+x:x_valid+w_valid-(x_valid-x)].shape)
        img_triangle_c[y_valid-y:h_valid, x_valid-x:w_valid] = img[y_valid:h_valid+y, x_valid:w_valid+x]

        warp_mat  = cv2.getAffineTransform(triangle_c_offset, triangle_n_offset)
        img_triangle_n = cv2.warpAffine(img_triangle_c, warp_mat, (w, h))

        # Copy the pixle value from warpped image to the entire image
        mask_img_triangle_n = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask_img_triangle_n, triangle_n_offset.astype(np.int32), 255)
        indices = (mask_img_triangle_n > 0).nonzero()
        indices_offset = (indices[0]+y-rect_n[1], indices[1]+x-rect_n[0])
        img_n[indices_offset]=img_triangle_n[indices]

    return (img_n, anchor)
