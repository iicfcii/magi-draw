import cv2
import numpy as np

 # Ratio between design pixel size and desired size
 # Camera needs to be high res otherwise image is scaled during warpPerspective
 # Adjust ratio to get desired drawing size for animation
RATIO = 2.0
MARKER_SIZE = 72*RATIO
BOARD_SIZE = 500*RATIO
DRAW_WIDTH = 250*RATIO
DRAW_HEIGHT = 100*RATIO
CORNERS_REF = {
    283: np.array([[0,0],
                   [MARKER_SIZE,0],
                   [MARKER_SIZE,MARKER_SIZE],
                   [0,MARKER_SIZE]]),
    430: np.array([[MARKER_SIZE+BOARD_SIZE,0],
                   [2*MARKER_SIZE+BOARD_SIZE,0],
                   [2*MARKER_SIZE+BOARD_SIZE,MARKER_SIZE],
                   [MARKER_SIZE+BOARD_SIZE,MARKER_SIZE]]),
    866: np.array([[0,BOARD_SIZE-MARKER_SIZE],
                   [MARKER_SIZE,BOARD_SIZE-MARKER_SIZE],
                   [MARKER_SIZE,BOARD_SIZE],
                   [0,BOARD_SIZE]]),
    935: np.array([[MARKER_SIZE+BOARD_SIZE,BOARD_SIZE-MARKER_SIZE],
                   [2*MARKER_SIZE+BOARD_SIZE,BOARD_SIZE-MARKER_SIZE],
                   [2*MARKER_SIZE+BOARD_SIZE,BOARD_SIZE],
                   [MARKER_SIZE+BOARD_SIZE,BOARD_SIZE]])
}
BOARD_REF = np.array([[MARKER_SIZE,0],
                      [MARKER_SIZE+BOARD_SIZE,0],
                      [MARKER_SIZE+BOARD_SIZE,BOARD_SIZE],
                      [MARKER_SIZE,BOARD_SIZE]])
DRAW_REF = np.array([[(BOARD_SIZE-DRAW_WIDTH)/2+MARKER_SIZE,(BOARD_SIZE-DRAW_HEIGHT)/2],
                      [(BOARD_SIZE+DRAW_WIDTH)/2+MARKER_SIZE,(BOARD_SIZE-DRAW_HEIGHT)/2],
                      [(BOARD_SIZE+DRAW_WIDTH)/2+MARKER_SIZE,(BOARD_SIZE+DRAW_HEIGHT)/2],
                      [(BOARD_SIZE-DRAW_WIDTH)/2+MARKER_SIZE,(BOARD_SIZE+DRAW_HEIGHT)/2]])

def findHomography(img):
    # Detect markers
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
    corners, ids, rejects = cv2.aruco.detectMarkers(img, dictionary) # Default parameters

    if ids is None: return None

    # img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
    # cv2.imshow('Detection', img_markers)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Find Homography
    corners = np.concatenate(corners).reshape(-1,2)
    corners_ref = []
    for id in ids:
        corners_ref.append(CORNERS_REF[id[0]])
    corners_ref = np.concatenate(corners_ref).reshape(-1,2)

    M, mask = cv2.findHomography(corners_ref,corners)

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
