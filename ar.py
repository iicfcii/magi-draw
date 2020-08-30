import cv2
import numpy as np

MARKER_SIZE = 72.
BOARD_SIZE = 500.
DRAW_WIDTH = 250.
DRAW_HEIGHT = 100
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

img = cv2.imread('img/snake_game_1.jpg')

# Detect markers
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
corners, ids, rejects = cv2.aruco.detectMarkers(img, dictionary) # Default parameters

img_markers = cv2.aruco.drawDetectedMarkers(img.copy(), corners, ids)
cv2.imshow('Detection', img_markers)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Find Homography
corners = np.concatenate(corners).reshape(-1,2)
corners_ref = []
for id in ids:
    corners_ref.append(CORNERS_REF[id[0]])
corners_ref = np.concatenate(corners_ref).reshape(-1,2)

M, mask = cv2.findHomography(corners_ref,corners)

# Draw board
points_ref = BOARD_REF.reshape(-1,1,2)
points_board = cv2.perspectiveTransform(points_ref,M)

img_board = cv2.polylines(img.copy(), [points_board.astype(np.int32)], True, (0,0,255),5)
cv2.imshow('Homography', img_board)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Warp drawing
img_scene_size = (int(2*MARKER_SIZE+BOARD_SIZE),int(BOARD_SIZE))
img_scene = cv2.warpPerspective(img.copy(), M, img_scene_size, flags=cv2.WARP_INVERSE_MAP+cv2.INTER_LINEAR)
cv2.imshow('Drawing', img_scene)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Crop drawing
x = int(DRAW_REF[0,0])
y = int(DRAW_REF[0,1])
w = int(DRAW_WIDTH)
h = int(DRAW_HEIGHT)
img_draw = img_scene[y:y+h,x:x+w]
img_draw = cv2.resize(img_draw, None, fx=2, fy=2)
cv2.imshow('Cropped', img_draw)
cv2.imwrite('img/snake_test.jpg', img_draw)
cv2.waitKey(0)
cv2.destroyAllWindows()
