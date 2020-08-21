import numpy as np
import cv2
import triangulation
import animation
import snake

# Prepare source image, triangulation, and default bones
img_c = snake.img_src.copy() # Source image
img_gray = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

img_contour = img_gray.copy()
contour = triangulation.preprocess(img_gray)
keypoints = triangulation.keypoints(img_gray, contour)
triangles = triangulation.triangulate(contour, keypoints)
bones_c = snake.bones_frames[0]

cv2.polylines(img_contour, [contour.astype(np.int32)], True, (0,0,255))
for triangle in triangles:
    cv2.polylines(snake.img_src, [triangle.astype(np.int32)], True, (0,0,255))
for bone in bones_c:
    cv2.polylines(snake.img_src, [bone.reshape((2,2)).astype(np.int32)], True, (255,0,0), 2)
cv2.imshow('contour',img_contour)
cv2.imshow('current',snake.img_src)
cv2.waitKey(0)

for i in range(len(snake.bones_frames)):
    bones_n = snake.bones_frames[i]
    triangles_next = animation.animate(bones_c,bones_n,triangles)

    rect_n = cv2.boundingRect(np.concatenate(triangles_next,axis=0)) # x, y, w, h
    img_n = np.zeros((rect_n[3],rect_n[2],3), np.uint8)
    img_n[:,:] = (255,255,255)

    rect_offset = 5 # Offset bounding rectangle so won't pick black pixle
    for triangle_c, triangle_n in zip(triangles, triangles_next):
        # Get signle bounding rectangle for current and next
        (x,y,w,h) = cv2.boundingRect(np.concatenate((triangle_c,triangle_n),axis=0))
        x -= rect_offset
        y -= rect_offset
        w += rect_offset*2
        h += rect_offset*2
        x_valid = np.maximum(x,0)
        y_valid = np.maximum(y,0)
        w_valid = np.minimum(w,img_c.shape[1]-x)
        h_valid = np.minimum(h,img_c.shape[0]-y)

        # Warp from current to next
        # Calculate coordinate wrt to bounding rectangle
        triangle_c_offset = triangle_c-np.tile(np.array([x,y]),(3,1)).astype(np.float32)
        triangle_n_offset = triangle_n-np.tile(np.array([x,y]),(3,1)).astype(np.float32)

        img_triangle_c = np.zeros((h,w,3), np.uint8)
        # print(img_triangle_c[y_valid-y:y_valid-y+h_valid-(y_valid-y), x_valid-x:x_valid-x+w_valid-(x_valid-x)].shape)
        # print(img_c[y_valid-y+y:y_valid+h_valid-(y_valid-y), x_valid-x+x:x_valid+w_valid-(x_valid-x)].shape)
        img_triangle_c[y_valid-y:h_valid, x_valid-x:w_valid] = img_c[y_valid:h_valid+y, x_valid:w_valid+x]

        warp_mat  = cv2.getAffineTransform(triangle_c_offset, triangle_n_offset)
        img_triangle_n = cv2.warpAffine(img_triangle_c, warp_mat, (w, h))

        # Copy the pixle value from warpped image to the entire image
        mask_img_triangle_n = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask_img_triangle_n, triangle_n_offset.astype(np.int32), 255)
        indices = (mask_img_triangle_n > 0).nonzero()
        indices_offset = (indices[0]+y-rect_n[1], indices[1]+x-rect_n[0])
        img_n[indices_offset]=img_triangle_n[indices]

    cv2.imshow('next',img_n)
    cv2.waitKey(0)

cv2.destroyAllWindows()
