import numpy as np
import cv2
import time

import animator.triangulation as triangulation

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
    length_c = np.sqrt((x_c[3]-x_c[1])**2+(x_c[2]-x_c[0])**2)
    length_n = np.sqrt((x_n[3]-x_n[1])**2+(x_n[2]-x_n[0])**2)

    x_world = [0,0,length_c,0]

    t_c2w = calcTransMat(x_world,x_c)
    t_w2c = calcTransMat(x_c,x_world)
    t_n2c = calcTransMat(x_c,x_n)

    # Allow scale along bone direction
    s_c2n = np.identity(3)
    s_c2n[0,0] = length_n/length_c

    return t_c2w @ t_n2c @ s_c2n @ t_w2c

def calcPointPointDistance(x, y):
    xy = y-x
    d = np.sqrt(xy[0]**2+xy[1]**2)
    return d

# Minimum distance between a point and line
def calcPointLineMinDistance(p, x, y):
    xp = p-x
    xy = y-x

    cross = xp[0]*xy[1]-xp[1]*xy[0]

    d = np.abs(cross/np.sqrt(xy[0]**2+xy[1]**2))

    return d
# print(calcPointLineMinDistance(np.array([0,0]),np.array([1,0]),np.array([0,1])))
# print(calcPointLineMinDistance(np.array([0,0]),np.array([1,1]),np.array([0,1])))

def calcPointLineDistance(p, x, y):
    xp = p-x
    xy = y-x

    dot = xp[0]*xy[0]+xp[1]*xy[1]
    l = np.sqrt(xy[0]**2+xy[1]**2)
    proj = dot/l

    if proj > l:
        # Outside y
        return calcPointPointDistance(p, y)
    if proj < 0:
        # Outside x
        return calcPointPointDistance(p, x)

    return calcPointLineMinDistance(p, x, y)
# print(calcPointLineDistance(np.array([0,0]),np.array([1,0]),np.array([0,1])))
# print(calcPointLineDistance(np.array([0,0]),np.array([1,1]),np.array([0,1])))
# print(calcPointLineDistance(np.array([0,0]),np.array([1,0]),np.array([2,0])))
# print(calcPointLineDistance(np.array([0,0]),np.array([-1,0]),np.array([-2,0])))
# print(calcPointLineDistance(np.array([0,0]),np.array([1,1]),np.array([2,1])))

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

def calcWeight(p, bone, triangles):
    d = calcPointLineDistance(p, bone[0:2], bone[2:4])

    # NOTE: 150 may be too big if image too small and vice versa
    # However, this can speed up the calculation
    if d > 150: return 0 # Too far away, nearly zero

    path = findPath(p, bone, triangles)
    if path is None: print('No path found!')

    d = 0
    for i in range(1,len(path)):
        d = d + calcPointPointDistance(path[i-1,:],path[i,:])
    d = d + calcPointLineDistance(path[-1,:], bone[0:2], bone[2:4])
    weight = np.exp(-0.03*d) # w =e^(-Cd)

    return weight

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
                    w[i] = calcWeight(point, bone, triangles)
                w = w/np.sum(w) # Make sure sum of weights is 1
                weights[point_key] = {'weight': w}

    return weights

# Return sorted triangles according to which bone each triangle belongs to
def sortTriangles(triangles, weights):
    relations = []

    for triangle in triangles:
        w_sum = None
        for point in triangle:
            point_key = tuple(point)
            w = weights[point_key]['weight']

            if w_sum is None:
                w_sum = w
            else:
                w_sum = w_sum + w
         # index of max w indicates which bone this triangle belongs to
        relations.append((triangle, np.argmax(w_sum)))

    relations.sort(key=lambda r:r[1])

    return [r[0] for r in relations], [r[1] for r in relations]

def findPath(start, bone, triangles):
    start_key = tuple(start)
    next = [start_key] # sort by f, lowest index is 0
    parent = {}
    g = {start_key: 0} # cost to current
    f = {start_key: 0+calcPointLineDistance(start,bone[0:2],bone[2:4])} # g + h

    while len(next) != 0:
        current_key = next.pop(0)
        current = np.asarray(current_key)
        if calcPointLineDistance(current,bone[0:2],bone[2:4]) < triangulation.MESH_DIST:
            path = [current_key]
            while current_key in parent:
                path.append(parent[current_key])
                current_key = parent[current_key]
            return np.array(path)


        # Get all neighbors
        match = triangulation.match_point2triangle(current, triangles)
        neighbor_keys = [] # list of point tuples
        for i in match:
            triangle = triangles[i]
            # cv2.polylines(img, [triangle.astype(np.int32)], True, (0,0,255))
            for point in triangle:
                if not (point == current).all():
                    neighbor_key = tuple(point)
                    if neighbor_key not in neighbor_keys:
                        neighbor_keys.append(neighbor_key)

        for neighbor_key in neighbor_keys:
            neighbor = np.asarray(neighbor_key)
            g_neighbor = g[current_key] + calcPointPointDistance(current, neighbor)

            if neighbor_key not in g or g_neighbor < g[neighbor_key]:
                parent[neighbor_key] = current_key
                g[neighbor_key] = g_neighbor
                f[neighbor_key] = g_neighbor + calcPointLineDistance(neighbor,bone[0:2],bone[2:4])

                if neighbor_key not in next:
                    # Make sure low f is in the front of the list
                    index = 0
                    for i in range(len(next)-1,-1,-1):
                        if f[neighbor_key] > f[next[i]]:
                            index = i+1
                            break
                    next.insert(index, neighbor_key)
    return None

# rect: (x,y,w,h)
def union_rects(rect1, rect2):
    x = np.maximum(rect1[0],rect2[0])
    y = np.maximum(rect1[1],rect2[1])
    w = np.minimum(rect1[0]+rect1[2],rect2[0]+rect2[2])-x
    h = np.minimum(rect1[1]+rect1[3],rect2[1]+rect2[3])-y

    if w > 0 and h > 0:
        return (x,y,w,h)
    else:
        # No union
        return None

def animate(bones_default, bones_n, triangles, weights):
    # Calculate transformation matrix to next frame for each bone
    transformations = []
    for i in range(len(bones_default)):
        bone_c = bones_default[i]
        bone_n = bones_n[i]
        t = calcTransMatBetweenFrame(bone_c,bone_n) # Bone is always along positive x of its own coordinate system
        # print(t)
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

def warp(img, triangles, triangles_next, bone, relations, hide=[]):
    rect_n = cv2.boundingRect(np.concatenate(triangles_next,axis=0)) # x, y, w, h
    img_n = np.zeros((rect_n[3],rect_n[2],3), np.uint8)
    mask_img_n = np.zeros((rect_n[3],rect_n[2],1), np.uint8)

    # Anchor position wrt to image
    # Anchor is the first point of first bone
    anchor = np.array([bone[0]-rect_n[0],bone[1]-rect_n[1]],dtype=np.int32)

    rect_offset = 5 # Offset bounding rectangle so won't pick black pixle
    for i, triangle_c, triangle_n in zip(range(len(triangles)), triangles, triangles_next):
        if relations[i] in hide: continue # Skip triangles that show be hidden based on their bones

        # Get signle bounding rectangle for current and next
        # Take out of image into account
        (x,y,w,h) = cv2.boundingRect(np.concatenate((triangle_c,triangle_n),axis=0))
        x -= rect_offset
        y -= rect_offset
        w += rect_offset*2
        h += rect_offset*2
        rect_valid= union_rects((x,y,w,h), (0,0,img.shape[1],img.shape[0]))

        # Warp from current to next
        # Calculate coordinate wrt to bounding rectangle
        triangle_c_offset = triangle_c-np.tile(np.array([x,y]),(3,1)).astype(np.float32)
        triangle_n_offset = triangle_n-np.tile(np.array([x,y]),(3,1)).astype(np.float32)

        img_triangle_c = np.zeros((h,w,3), np.uint8)
        if rect_valid is not None:
            x_valid, y_valid, w_valid, h_valid = rect_valid
            img_triangle_c[y_valid-y:y_valid-y+h_valid, x_valid-x:x_valid-x+w_valid] = img[y_valid:y_valid+h_valid, x_valid:x_valid+w_valid]

        warp_mat  = cv2.getAffineTransform(triangle_c_offset, triangle_n_offset)
        img_triangle_n = cv2.warpAffine(img_triangle_c, warp_mat, (w, h))

        # Copy the pixle value from warpped image to the entire image
        mask_img_triangle_n = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask_img_triangle_n, triangle_n_offset.astype(np.int32), 255)
        indices = (mask_img_triangle_n > 0).nonzero()
        indices_offset = (indices[0]+y-rect_n[1], indices[1]+x-rect_n[0])

        # # NOTE: if fillConvexPoly uses lineType=cv2.LINE_AA, indices offset may be out of boundary
        # # Following code delete those indices
        # invalid = np.logical_or.reduce(((indices_offset[1] > rect_n[2]-1),
        #                               (indices_offset[0] > rect_n[3]-1),
        #                               (indices_offset[1] < 0),
        #                               (indices_offset[0] < 0))).nonzero()
        # indices_valid = (np.delete(indices[0], invalid),np.delete(indices[1], invalid))
        # indices_offset_valid = (np.delete(indices_offset[0], invalid),np.delete(indices_offset[1], invalid))
        # img_n[indices_offset_valid]=img_triangle_n[indices_valid]
        # mask_img_n[indices_offset_valid] = 255

        img_n[indices_offset]=img_triangle_n[indices]
        mask_img_n[indices_offset] = 255

    # Erode so that no background pixles are shown
    # NOTE: mask is smaller than the black area of the image
    mask_img_n = cv2.erode(mask_img_n,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), borderType=cv2.BORDER_CONSTANT, borderValue=0)

    return (img_n, anchor, mask_img_n)

def merge_frames(front, back):
    img, anchor, mask = front
    img_alt, anchor_alt, mask_alt = back

    # Coordinates of four corners of two images based on anchor
    pts = np.array([
        [-anchor[0],-anchor[1]],
        [-anchor[0]+img.shape[1]-1,-anchor[1]+img.shape[0]-1],
        [-anchor_alt[0],-anchor_alt[1]],
        [-anchor_alt[0]+img_alt.shape[1]-1,-anchor_alt[1]+img_alt.shape[0]-1]
    ], dtype=np.int32)
    # Bounding rect based on anchor
    rect = cv2.boundingRect(pts)

    # Coordinates of two images in the new bounding rect
    anchor_new = (-rect[0],-rect[1])
    p_new = (pts[0,0]+anchor_new[0],pts[0,1]+anchor_new[1]) # top left corner
    p_alt_new = (pts[2,0]+anchor_new[0],pts[2,1]+anchor_new[1])
    w_new, h_new = rect[2:4]

    # Dialte the mask so that when masked, no black border will be shown
    mask_dilated = cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)), borderValue=0)

    # Apply back image first then front image with dilated mask
    img_new = np.zeros((h_new,w_new,3), np.uint8)
    img_new[p_alt_new[1]:p_alt_new[1]+img_alt.shape[0],p_alt_new[0]:p_alt_new[0]+img_alt.shape[1]] = img_alt
    img_new[p_new[1]:p_new[1]+img.shape[0],p_new[0]:p_new[0]+img.shape[1]][mask_dilated > 0] = img[mask_dilated > 0]

    # Similar process but with just mask
    mask_new = np.zeros((h_new,w_new), np.uint8)
    mask_new[p_alt_new[1]:p_alt_new[1]+mask_alt.shape[0],p_alt_new[0]:p_alt_new[0]+mask_alt.shape[1]] = mask_alt
    mask_new[p_new[1]:p_new[1]+mask.shape[0],p_new[0]:p_new[0]+mask.shape[1]][mask > 0] = mask[mask > 0]

    # cv2.circle(img_new, anchor_new, 5, (255,0,0), thickness=-1)
    # cv2.imshow('Img',img_new)
    # cv2.imshow('Mask',mask_new)
    # cv2.waitKey(0)

    return img_new, anchor_new, mask_new

def flip_frame(frame):
    img, anchor, mask = frame
    img_new = cv2.flip(img, 1)
    anchor_new = (img.shape[1]-anchor[0]-1,anchor[1])
    mask_new = cv2.flip(mask, 1)

    return (img_new, anchor_new, mask_new)
