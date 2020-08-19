import numpy as np
import cv2

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
def match_triangle(edge, triangles):
    # Match points for each triangle
    match = np.logical_or(triangles == edge[0:2], triangles == edge[2:4]).all(axis=2)
    # Count triangle with both points and get index
    match = (np.count_nonzero(match,axis=1) == 2).nonzero()[0]
    return match

def swap_diagonal(edge, triangles):
    match = match_triangle(edge, triangles)

    if len(match) != 2: return # Should not happen?

    triangle_a = triangles[match[0],:,:].copy()
    triangle_b = triangles[match[1],:,:].copy()

    # Swap first point of edge in first triangle
    first_point_index = (triangle_a == edge[0:2]).all(axis=1).nonzero()[0][0]
    other_point_index = np.logical_and((triangle_b != edge[0:2]),(triangle_b != edge[2:4])).all(axis=1).nonzero()[0][0]
    triangles[match[0],:,:][first_point_index,:] = triangle_b[other_point_index,:] # Modify triangles
    edge_new = triangle_b[other_point_index,:]
    # Swap second point of edge in second triangle
    first_point_index = (triangle_b == edge[2:4]).all(axis=1).nonzero()[0][0]
    other_point_index = np.logical_and((triangle_a != edge[0:2]),(triangle_a != edge[2:4])).all(axis=1).nonzero()[0][0]
    triangles[match[1],:,:][first_point_index,:] = triangle_a[other_point_index,:] # Modify triangles
    # New edge
    edge_new = np.concatenate((edge_new,triangle_a[other_point_index,:]))
    # cv2.polylines(img_color,[edge_new.reshape((2,2)).astype(np.int32)],True,(0,0,0),2)
    # print(triangle_a,triangles[match[0],:,:])
    # print(triangle_b,triangles[match[1],:,:])

    return edge_new

def triangulate(img_color):
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


    # Find offset contour
    ret,img_bin = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)

    img_dilation = cv2.dilate(img_bin,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=1)
    contours, hierarchy = cv2.findContours(img_dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contour = contours[0]
    contour_simple = cv2.approxPolyDP(contour,3,True).reshape((-1,2)).astype(np.int32)
    print("Contour Points Number", len(contour_simple))
    for pt in contour_simple:
        cv2.circle(img_color,(pt[0],pt[1]),2,(255, 0, 0),cv2.FILLED)

    # Find feature points
    fast = cv2.FastFeatureDetector_create(threshold=10)
    keypoints = fast.detect(img_gray,None)

    # Filter feature points too close to the contour
    # and too close to each other
    keypoints_inside = []
    for kp in keypoints:
        dist = cv2.pointPolygonTest(contour_simple,kp.pt,True)
        if dist > 20:
            tooClose = False
            for pt in keypoints_inside:
                if (pt[0]-kp.pt[0])**2+(pt[1]-kp.pt[1])**2 < 20**2:
                    tooClose = True
                    break

            if not tooClose: #  and kp.response > 10
                keypoints_inside.append(kp.pt)

    keypoints_inside = np.array(keypoints_inside,dtype=np.int32)
    print("Key points inside number", len(keypoints_inside))

    # for kp in keypoints_inside:
    #     cv2.circle(img_color,(kp[0],kp[1]),2,(0, 255, 0),cv2.FILLED)

    # Form triangles
    points = np.append(contour_simple,keypoints_inside,axis=0)
    # points = np.array([[0,0],[100,100],[50,100],[150,100]])
    rect = cv2.boundingRect(points)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(points.tolist())
    edges = subdiv.getEdgeList()
    triangles = subdiv.getTriangleList()

    # Constrain triangles
    triangles = triangles.reshape((-1,3,2)) # Reshape to match points

    # Find intersecting edges
    edges_intersects = []
    for edge in edges:
        first_point_is_contour = len((contour_simple == edge[0:2]).all(axis=1).nonzero()[0]) == 1
        second_point_is_contour = len((contour_simple == edge[2:4]).all(axis=1).nonzero()[0]) == 1
        # print(first_point_is_contour, second_point_is_contour)

        # Skip if both end points are not on contour
        if not first_point_is_contour and not second_point_is_contour: continue

        match = match_triangle(edge, triangles)
        if len(match) == 0:
            # print('Outside contour')
            pass
        else:
            # cv2.polylines(img_color,[edge.reshape((2,2)).astype(np.int32)],True,(0,0,255),1)
            if len(match) != 1:
                # print('Inside contour',match[0][0],match[0][1])
                if intersection_contour(edge, contour_simple):
                    # cv2.polylines(img_color,[edge.reshape((2,2)).astype(np.int32)],True,(255,0,255),2)
                    edges_intersects.append(edge)
            else:
                # print('On contour',match[0][0])
                pass

    # Swap intersecting edges
    edges_new = []
    while len(edges_intersects) != 0:
        edge = edges_intersects.pop(0)
        edge_new = swap_diagonal(edge, triangles)

        if intersection_contour(edge_new, contour_simple):
            print('New edge still intersects')
            edges_intersects.append(edge_new)
        else:
            edges_new.append(edge_new)

    # TODO: Check delaunay triangulation criterion for new edges(omit edge on contour)
    # Refer to: A FAST ALGORITHM FOR GENERATING CONSTRAINED DELAUNAY TRIANGULATIONS

    # Remove triangles outside contour
    triangles_inside = []
    for triangle in triangles:
        # # Make sure edge is inside contours
        triangle_rolled = np.roll(triangle,2)
        mid_points = (triangle+triangle_rolled)/2
        # print(triangle,triangle_rolled, mid_points)

        outside = False
        for pt in mid_points:
            dist = cv2.pointPolygonTest(contour_simple,(pt[0],pt[1]),False)
            if dist == -1.0: outside = True

        if not outside: triangles_inside.append(triangle)

    # # Draw triangles and edges
    # for edge in edges_new:
    #     cv2.polylines(img_color,[edge.reshape((2,2)).astype(np.int32)],True,(255,0,255),2)
    # for triangle in triangles_inside:
    #     cv2.polylines(img_color, [triangle.astype(np.int32)], True, (0,0,255))
    # cv2.imshow('color',img_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return triangles_inside
