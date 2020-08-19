import numpy as np
import cv2

image_name = 'cat.jpg'

img_color = cv2.imread(image_name, 1)
img_gray = cv2.imread(image_name, 0)


# Find offset contour
ret,img_bin = cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
cv2.imshow('bin',img_bin)
# cv2.waitKey(0)

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

        if not tooClose:
            keypoints_inside.append(kp.pt)


keypoints_inside = np.array(keypoints_inside,dtype=np.int32)
print("Key points inside number", len(keypoints_inside))

for kp in keypoints_inside:
    cv2.circle(img_color,(kp[0],kp[1]),2,(0, 255, 0),cv2.FILLED)


# Form triangles
points = np.append(contour_simple,keypoints_inside,axis=0)
# points = np.array([[0,0],[100,100],[50,100],[150,100]])
rect = cv2.boundingRect(points)
subdiv = cv2.Subdiv2D(rect)
subdiv.insert(points.tolist())
triangles = subdiv.getTriangleList().astype(np.int32)

triangles_inside = []
for triangle in triangles:
    # Make sure edge is inside contours
    triangle_rolled = np.roll(triangle,2)
    mid_points = (triangle+triangle_rolled)/2

    outside = False
    for pt in mid_points.reshape((3,2)):
        dist = cv2.pointPolygonTest(contour_simple,(pt[0],pt[1]),False)
        # if dist == -1.0: outside = True

    if not outside: triangles_inside.append(triangle)

for triangle in triangles_inside:
    cv2.polylines(img_color, [triangle.reshape((3,2))], True, (0,0,255))

triangles_inside = np.array(triangles_inside,dtype=np.int32)
print(triangles.shape,triangles_inside.shape)

cv2.imshow('color',img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
