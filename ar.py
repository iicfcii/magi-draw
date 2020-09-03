import cv2
import numpy as np
import animation

 # Ratio between design pixel size and desired size
 # Camera needs to be high res otherwise image is scaled during warpPerspective
 # Adjust ratio to get desired drawing size for animation
RATIO = 2.0
MARKER_SIZE = 144*RATIO
BOARD_SIZE = 500*RATIO
DRAW_WIDTH = 250*RATIO
DRAW_HEIGHT = 100*RATIO
CORNERS_REF = {
    7: np.array([[0,0],
                 [MARKER_SIZE,0],
                 [MARKER_SIZE,MARKER_SIZE],
                 [0,MARKER_SIZE]]),
    23: np.array([[BOARD_SIZE-MARKER_SIZE,0],
                  [BOARD_SIZE,0],
                  [BOARD_SIZE,MARKER_SIZE],
                  [BOARD_SIZE-MARKER_SIZE,MARKER_SIZE]]),
    27: np.array([[BOARD_SIZE-MARKER_SIZE,BOARD_SIZE-MARKER_SIZE],
                  [BOARD_SIZE,BOARD_SIZE-MARKER_SIZE],
                  [BOARD_SIZE,BOARD_SIZE],
                  [BOARD_SIZE-MARKER_SIZE,BOARD_SIZE]]),
    42: np.array([[0,BOARD_SIZE-MARKER_SIZE],
                  [MARKER_SIZE,BOARD_SIZE-MARKER_SIZE],
                  [MARKER_SIZE,BOARD_SIZE],
                  [0,BOARD_SIZE]]),
}
BOARD_REF = np.array([[0,0],
                      [BOARD_SIZE,0],
                      [BOARD_SIZE,BOARD_SIZE],
                      [0,BOARD_SIZE]])
DRAW_REF = np.array([[(BOARD_SIZE-DRAW_WIDTH)/2,(BOARD_SIZE-DRAW_HEIGHT)/2],
                     [(BOARD_SIZE+DRAW_WIDTH)/2,(BOARD_SIZE-DRAW_HEIGHT)/2],
                     [(BOARD_SIZE+DRAW_WIDTH)/2,(BOARD_SIZE+DRAW_HEIGHT)/2],
                     [(BOARD_SIZE-DRAW_WIDTH)/2,(BOARD_SIZE+DRAW_HEIGHT)/2]])

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
PARA = cv2.aruco.DetectorParameters_create()
PARA.adaptiveThreshWinSizeMin = 3
PARA.adaptiveThreshWinSizeMax = 53
PARA.adaptiveThreshWinSizeStep = 10

def findHomography(img):
    # Detect markers
    corners, ids, rejects = cv2.aruco.detectMarkers(img, DICT, parameters=PARA) # Default parameters
    if ids is None:
        # print('No ids')
        return None
    # img_tmp = img.copy()
    # img_tmp = cv2.aruco.drawDetectedMarkers(img_tmp, corners, ids)

    # Find Homography
    corners_dst = []
    corners_ref = []
    for i, id in enumerate(ids):
        if id[0] in CORNERS_REF:
            corners_ref.append(CORNERS_REF[id[0]])
            corners_dst.append(corners[i])

    if len(corners_ref) == 0:
        # print('No correct ids')
        return None # Detected wrong markers

    corners_dst = np.concatenate(corners_dst).reshape(-1,2)
    corners_ref = np.concatenate(corners_ref).reshape(-1,2)

    M, mask = cv2.findHomography(corners_ref,corners_dst)
    return M

def getDrawing(img, M):
    # Warp drawing
    size = (int(2*MARKER_SIZE+BOARD_SIZE),int(BOARD_SIZE))
    img_drawing = cv2.warpPerspective(img, M, size, flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)

    # cv2.imshow('Drawing', img_drawing)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Crop drawing
    x = int(DRAW_REF[0,0])
    y = int(DRAW_REF[0,1])
    w = int(DRAW_WIDTH)
    h = int(DRAW_HEIGHT)
    img_drawing = img_drawing[y:y+h,x:x+w]

    return img_drawing

def render(render, img, position, M):
    # Construct scene and mask
    x_scene = 0
    y_scene = 0
    w_scene = int(BOARD_SIZE)
    h_scene = int(BOARD_SIZE)
    mask_scene = np.zeros((h_scene, w_scene), np.uint8)
    mask_scene[:,:] = 255
    img_scene = np.zeros((h_scene,w_scene,3),np.uint8)
    img_scene[:,:] = (255,255,255)

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


class HomographyInterpolater:
    def __init__(self):
        self.MAT_PREV_COUNT = 2
        self.mat_prev_ptr = 0
        self.mats_prev = [(None, False)] * self.MAT_PREV_COUNT # (mat, interpolated)
        self.interpolation_count = 0

    def increment_ptr(self, ptr):
        ptr += 1
        if ptr >= self.MAT_PREV_COUNT:
            ptr = 0
        return ptr

    def decrement_ptr(self, ptr):
        ptr -= 1
        if ptr <  0:
            ptr = self.MAT_PREV_COUNT-1
        return ptr

    def estimate(self, mat):
        interpolated = False

        if mat is not None:
            self.interpolation_count = 0
        else:
            if self.interpolation_count < 1:
                mat_1 = self.mats_prev[self.mat_prev_ptr][0]
                mat_2 = self.mats_prev[self.decrement_ptr(self.mat_prev_ptr)][0]

                if mat_1 is not None and mat_2 is not None:
                    mat = (mat_1+mat_1-mat_2)*0.5+mat_1*0.5
                    interpolated = True
                    if self.mats_prev[self.mat_prev_ptr][1]: self.interpolation_count += 1

        self.mat_prev_ptr = self.increment_ptr(self.mat_prev_ptr)
        self.mats_prev[self.mat_prev_ptr] = (mat, interpolated)

        return (mat, interpolated)
