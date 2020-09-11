import numpy as np
import cv2

MESH_DIST = 20

# Whether line (o1, p1) and (o2, p2) intersects
# End points touching means not intersect
# Parallel or collinear also means not intersect
# o1, p1, o2, p2 are tuples (x,y)
# https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
# https://stackoverflow.com/questions/7446126/opencv-2d-line-intersection-helper-function
def intersection(o1, p1, o2, p2):
    x = (o2[0]-o1[0], o2[1]-o1[1])
    d1 = (p1[0]-o1[0], p1[1]-o1[1])
    d2 = (p2[0]-o2[0], p2[1]-o2[1])

    cross = d1[0]*d2[1]-d1[1]*d2[0];
    if (abs(cross) < 1e-8): # eps
        # Parallel or collinear
        return False

    t = (x[0]*d2[1]-x[1]*d2[0])/cross
    u = (x[0]*d1[1]-x[1]*d1[0])/cross
    r = (o1[0]+d1[0]*t,o1[1]+d1[1]*t)

    if t<1 and t>0 and u<1 and u>0:
        return True
    else:
        # Not parallel and not intersect
        return False

# print(intersection((0,0),(1,1),(0,0),(-1,1)))
# print(intersection((0,0),(1,1),(0,1),(1,0)))
# print(intersection((0,0),(1,1),(2,2),(3,3)))
# print(intersection((0,0),(1,1),(-0.6,0.5),(-1,1)))
# print(intersection((0,0),(1,1),(-0.5,0.5),(-1,1)))

def intersection_contour(edge, contour):
    o1 = tuple(edge[0:2])
    p1 = tuple(edge[2:4])
    # Check if intersects contour
    intersects = False
    for i in range(len(contour)):
        o2 = tuple(contour[i,:])
        if i == len(contour)-1:
            p2 = tuple(contour[0,:])
        else:
            p2 = tuple(contour[i+1,:])

        if intersection(o1, p1, o2, p2):
            intersects = True
            break

    return intersects

# Triangles in shape (n, 3, 2)
# Edge in shape (1, 4)
# Match is (i, j)
def match_edge2triangle(edge, triangles):
    # Match points for each triangle
    match = np.logical_or((triangles == edge[0:2]).all(axis=2), (triangles == edge[2:4]).all(axis=2))
    # Count triangle with both points and get index
    match = (np.count_nonzero(match,axis=1) == 2).nonzero()[0]
    return match
# print(match_edge2triangle(np.array([0,0,1,1]),np.array([[[0,0],[1,1],[1,0]],[[0,0],[1,1],[0,1]],[[0,0],[2,2],[0,1]]])))

def match_point2triangle(point, triangles):
    match = (triangles == point).all(axis=2)
    match = (np.count_nonzero(match,axis=1) == 1).nonzero()[0]
    return match
# print(match_point2triangle(np.array([0,0]),np.array([[[0,0],[1,1],[1,0]],[[0,0],[1,1],[0,1]],[[4,7],[2,2],[0,1]]])))

def swap_diagonal(edge, match, triangles):
    triangle_a = triangles[match[0],:,:].copy()
    triangle_b = triangles[match[1],:,:].copy()

    # Swap first point of edge in first triangle
    first_point_index_a = (triangle_a == edge[0:2]).all(axis=1).nonzero()[0][0]
    other_point_index_a = np.logical_and((triangle_b != edge[0:2]).any(axis=1),(triangle_b != edge[2:4]).any(axis=1)).nonzero()[0][0]
    # other_point_index_a = np.logical_and((triangle_b != edge[0:2]),(triangle_b != edge[2:4])).any(axis=1).nonzero()[0][0]

    # Swap second point of edge in second triangle
    first_point_index_b = (triangle_b == edge[2:4]).all(axis=1).nonzero()[0][0]
    other_point_index_b = np.logical_and((triangle_a != edge[0:2]).any(axis=1),(triangle_a != edge[2:4]).any(axis=1)).nonzero()[0][0]
    # other_point_index_b = np.logical_and((triangle_a != edge[0:2]),(triangle_a != edge[2:4])).any(axis=1).nonzero()[0][0]

    edge_new = triangle_b[other_point_index_a,:]
    edge_new = np.concatenate((edge_new,triangle_a[other_point_index_b,:]))

    # Test convex quadrilateral
    intersects = intersection(tuple(edge[0:2]),tuple(edge[2:4]),tuple(edge_new[0:2]),tuple(edge_new[2:4]))
    if not intersects: return None # Not convex

    # Modify triangles
    triangles[match[0],:,:][first_point_index_a,:] = triangle_b[other_point_index_a,:]
    triangles[match[1],:,:][first_point_index_b,:] = triangle_a[other_point_index_b,:]

    return edge_new

def contour(img_gray):
    # Threshold
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 151, 10)
    # Offset contour and close holes
    img_close = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    img_dialate = cv2.dilate(img_close,cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))

    # cv2.imshow('bin',img_bin)
    # cv2.imshow('close',img_close)
    # cv2.imshow('dialate',img_dialate)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    contours, hierarchy = cv2.findContours(img_dialate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 1:
        # Only take the one with biggest area
        area_max = 0
        index_max = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > area_max:
                area_max = area
                index_max = i

        contours = [contours[index_max]]

    assert len(contours) == 1
    contour = cv2.approxPolyDP(contours[0],1,True).reshape((-1,2)).astype(np.int32)

    # Add points to controur
    points = []
    for i in range(len(contour)):
        p1 = contour[i,:]
        if i+1 >= len(contour):
            p2 = contour[0,:]
        else:
            p2 = contour[i+1,:]
        points.append(p1.reshape((1,2)))

        d = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
        if d > MESH_DIST:
            num = np.ceil(d/MESH_DIST)
            xs = np.linspace(p1[0],p2[0],num.astype(np.int32)).reshape((-1,1))
            ys = np.linspace(p1[1],p2[1],num.astype(np.int32)).reshape((-1,1))
            pts = np.concatenate((xs, ys), axis=1).astype(np.int32)
            points.append(pts)

        points.append(p2.reshape((1,2)))

    contour = np.concatenate(points)
    # print("Contour Points Number", len(contour))

    return contour

def keypoints_uniform(img_gray, contour):
    (x,y,w,h) = cv2.boundingRect(contour)
    xs = np.linspace(x,x+w,np.ceil(w/MESH_DIST).astype(np.int32))
    ys = np.linspace(y,y+h,np.ceil(h/MESH_DIST).astype(np.int32))
    xs_tmp = np.repeat(xs, ys.shape[0]).reshape(1,-1)
    ys_tmp = np.tile(ys, (1,xs.shape[0]))
    points = np.transpose(np.concatenate((xs_tmp, ys_tmp), axis=0)).astype(np.int32)

    # Remove points too close to contour
    keypoints = []
    for kp in points:
        tooClose = False
        for pt in contour:
            if (kp[0]-pt[0])**2+(kp[1]-pt[1])**2 < (MESH_DIST/2)**2:
                tooClose = True
                continue

        outside = cv2.pointPolygonTest(contour, tuple(kp), False) == -1

        if not tooClose and not outside:
            keypoints.append(kp)

    keypoints = np.asarray(keypoints, dtype=np.int32)

    return keypoints

def triangulate(contour, keypoints):
    # Form triangles
    if len(keypoints) != 0:
        points = np.concatenate([contour,keypoints],axis=0)
    else:
        points = contour
    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points.tolist())
    edges = subdiv.getEdgeList()
    triangles = subdiv.getTriangleList()
    triangles = triangles.reshape((-1,3,2))

    return (triangles, edges)

def constrain(contour, triangles, edges):
    # Find intersecting edges
    edges_intersects = []
    for edge in edges:
        # NOTE: Cant skip if both end points are not on contour
        # Extreme convexity is the exception

        match = match_edge2triangle(edge, triangles)
        if len(match) == 0:
            # Outside contour
            pass
        else:
            if len(match) != 1:
                # Inside contour
                if intersection_contour(edge, contour):
                    edges_intersects.append(edge)
            else:
                # On contour
                pass

    # Swap intersecting edges
    edges_new = []
    while len(edges_intersects) != 0:
        # Check edges process
        # img_tmp = img.copy()
        # for triangle in triangles:
        #     cv2.polylines(img_tmp, [triangle.astype(np.int32)], True, (0,0,255))
        # for edge in edges_intersects:
        #     cv2.polylines(img_tmp, [edge.reshape((2,2)).astype(np.int32)], True, (0,0,0))
        # for point in contour:
        #     cv2.circle(img_tmp, tuple(point.astype(np.int32)), 2, (255,0,0), thickness=-1)
        # cv2.imshow('Triangulation',img_tmp)
        # cv2.waitKey(0)

        edge = edges_intersects.pop(0)
        match = match_edge2triangle(edge, triangles)
        if len(match) != 2:
            continue
        edge_new = swap_diagonal(edge, match, triangles)

        if edge_new is not None:
            if intersection_contour(edge_new, contour):
                edges_intersects.append(edge_new)
            else:
                edges_new.append(edge_new)
        else:
            edges_intersects.append(edge)

    # If the newly formed quadrilateral is a triangle, remove both triangles
    # and add the bigger ones

    # TODO: Check delaunay triangulation criterion for new edges(omit edge on contour)
    # Refer to: A FAST ALGORITHM FOR GENERATING CONSTRAINED DELAUNAY TRIANGULATIONS

    # Remove triangles outside contour
    triangles_inside = []
    for triangle in triangles:
        # Make sure edge is inside contours
        triangle_rolled = np.roll(triangle,2)
        mid_points = (triangle+triangle_rolled)/2

        outside = False
        for pt in mid_points:
            dist = cv2.pointPolygonTest(contour,(pt[0],pt[1]),False)
            if dist == -1.0: outside = True

        if not outside: triangles_inside.append(triangle)

    return triangles_inside
