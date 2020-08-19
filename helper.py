import numpy as np

# Whether line (o1, p1) and (o2, p2) intersects
# End points touching means not intersect
# Parallel or collinear also means not intersect
# o1, p1, o2, p2 are tuples (x,y)
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
