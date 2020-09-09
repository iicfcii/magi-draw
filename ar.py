import cv2
import numpy as np
import animation

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARA = cv2.aruco.DetectorParameters_create()
PARA.adaptiveThreshWinSizeMin = 3
PARA.adaptiveThreshWinSizeMax = 53
PARA.adaptiveThreshWinSizeStep = 10

def findHomography(img, corners_ref):
    # Detect markers
    corners, ids, rejects = cv2.aruco.detectMarkers(img, DICT, parameters=PARA) # Default parameters
    if ids is None:
        # print('No ids')
        return None
    # img_tmp = img.copy()
    # img_tmp = cv2.aruco.drawDetectedMarkers(img_tmp, corners, ids)

    # Find Homography
    corners_dst = []
    corners_src = []
    for i, id in enumerate(ids):
        if id[0] in corners_ref:
            corners_src.append(corners_ref[id[0]])
            corners_dst.append(corners[i])

    if len(corners_src) == 0:
        # print('No correct ids')
        return None # Detected wrong markers

    corners_dst = np.concatenate(corners_dst).reshape(-1,2)
    corners_src = np.concatenate(corners_src).reshape(-1,2)

    M, mask = cv2.findHomography(corners_src,corners_dst)
    return M

def getDrawing(img, M, draw_ref):
    # Warp drawing
    size = (int(draw_ref[2,0]),int(draw_ref[2,1]))
    img_drawing = cv2.warpPerspective(img, M, size, flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)

    # cv2.imshow('Drawing', img_drawing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Crop drawing
    x = int(draw_ref[0,0])
    y = int(draw_ref[0,1])
    w = int(draw_ref[2,0]-draw_ref[0,0])
    h = int(draw_ref[2,1]-draw_ref[0,1])
    img_drawing = img_drawing[y:y+h,x:x+w]

    return img_drawing

def render(render, img, mask, position, M):
    # Construct scene and mask
    x_scene = 0
    y_scene = 0
    w_scene = int(position[0]+mask.shape[1])
    h_scene = int(position[1]+mask.shape[0])

    mask_scene = np.zeros((h_scene, w_scene), np.uint8)
    mask_scene[position[1]:h_scene,position[0]:w_scene] = mask
    img_scene = np.zeros((h_scene,w_scene,3),np.uint8)

    # Render scene before warp
    x_snake = position[0]
    y_snake = position[1]
    w_snake = img.shape[1]
    h_snake = img.shape[0]
    rect = animation.union_rects((x_snake,y_snake,w_snake,h_snake), (x_scene,y_scene,w_scene,h_scene))
    if rect is not None:
        x,y,w,h = rect
        img_scene[y:y+h,x:x+w] = img[y-y_snake:y-y_snake+h,x-x_snake:x-x_snake+w]

    # Warp and mask
    w_render = render.shape[1]
    h_render = render.shape[0]
    img_scene_warpped = cv2.warpPerspective(img_scene, M, (w_render,h_render), flags=cv2.INTER_LINEAR)
    mask_scene_warpped = cv2.warpPerspective(mask_scene, M, (w_render,h_render), flags=cv2.INTER_LINEAR)
    img_render = render.copy()
    img_render[mask_scene_warpped>0] = img_scene_warpped[mask_scene_warpped>0]

    return img_render
