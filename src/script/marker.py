import cv2
import numpy as np

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

img_marker = np.zeros((200, 200), dtype=np.uint8)
marker_count = 4
for i in range(marker_count):
    index = np.random.randint(50)
    img_marker = cv2.aruco.drawMarker(dictionary, index, 200, None, 1);
    file_name = 'img/marker_4x4_{}.jpg'.format(index)
    cv2.imshow(file_name, img_marker)
    cv2.imwrite(file_name, img_marker);
    cv2.waitKey(0)
    cv2.destroyAllWindows()
