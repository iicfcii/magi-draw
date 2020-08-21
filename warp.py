import numpy as np
import cv2
import triangulation
import animation
import snake

img_c = snake.img_src.copy()
triangles = triangulation.triangulate(img_c)
bones_c = snake.bones_frames[0]
bones_n = snake.bones_frames[1]
triangles_next = animation.animate(bones_c,bones_n,triangles)

# rect_n = cv2.boundingRect(np.concatenate(triangles_next,axis=0))
img_n = np.zeros(img_c.shape, np.uint8)
img_n[:,:] = (255,255,255)

# Draw both frames
for bone in bones_c:
    cv2.polylines(snake.img_src, [bone.reshape((-1,2)).astype(np.int32)], True, (255,0,0), 2)
for triangle in triangles:
    cv2.polylines(snake.img_src, [triangle.astype(np.int32)], True, (0,0,255))

for bone in bones_n:
    cv2.polylines(img_n, [bone.reshape((-1,2)).astype(np.int32)], True, (255,0,0), 2)
for triangle in triangles_next:
    cv2.polylines(img_n, [triangle.astype(np.int32)], True, (0,0,255))

cv2.imshow('current',snake.img_src)
cv2.imshow('next',img_n)
cv2.waitKey(0)

rect_offset = 2 # Offset bounding rectangle so won't pick black pixle
for triangle_c, triangle_n in zip(triangles, triangles_next):
    # Get signle bounding rectangle for current and next
    (x,y,w,h) = cv2.boundingRect(np.concatenate((triangle_c,triangle_n),axis=0))
    x -= rect_offset
    y -= rect_offset
    w += rect_offset*2
    h += rect_offset*2

    # Calculate coordinate wrt to bounding rectangle
    triangle_c_offset = triangle_c-np.tile(np.array([x,y]),(3,1)).astype(np.float32)
    triangle_n_offset = triangle_n-np.tile(np.array([x,y]),(3,1)).astype(np.float32)

    # Warp from current to next
    img_triangle_c = img_c[y:y+h, x:x+w]
    warp_mat  = cv2.getAffineTransform(triangle_c_offset, triangle_n_offset)
    img_triangle_n = cv2.warpAffine(img_triangle_c, warp_mat, (w, h))

    # Copy the pixle value from warpped image to the entire image
    mask_img_triangle_n = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask_img_triangle_n, triangle_n_offset.astype(np.int32), 255)
    indices = (mask_img_triangle_n > 0).nonzero()
    indices_offset = (indices[0]+y, indices[1]+x)
    img_n[indices_offset]=img_triangle_n[indices]

    # cv2.polylines(img_triangle_n, [triangle_n_offset.astype(np.int32)], True, (0,0,255))
    # cv2.imshow('warpped',img_triangle_n)
    # cv2.imshow('next',img_n)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

cv2.imshow('current',img_c)
cv2.imshow('next',img_n)
cv2.waitKey(0)
cv2.destroyAllWindows()
