import numpy as np
import cv2

img_m = cv2.imread('img/snake_icon_marker2.jpg',0)
img_s = cv2.imread('img/snake_scene_cropped.jpg',0)

orb = cv2.ORB_create()

kp_m, des_m = orb.detectAndCompute(img_m,None)
kp_s, des_s = orb.detectAndCompute(img_s,None)

img_kp_m = cv2.drawKeypoints(img_m, kp_m, None)
img_kp_s = cv2.drawKeypoints(img_s, kp_s, None)
cv2.imshow('Marker key points',img_kp_m)
cv2.imshow('Scene key points',img_kp_s)
cv2.waitKey(0)

# index_params= dict(algorithm=6, table_number=12, key_size=20, multi_probe_level=2)
# search_params = dict(checks=50)
# flann = cv2.FlannBasedMatcher(index_params, search_params)
# matches = flann.knnMatch(des_m,des_s,k=2)
# matches_good = []
# for match in matches:
#     if len(match) != 2: continue
#     if match[0].distance < 0.7*match[1].distance:
#         matches_good.append([match[0]])
# img_matches = cv2.drawMatchesKnn(img_m,kp_m,img_s,kp_s,matches_good,None)
# cv2.imshow('Matches',img_matches)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des_m,des_s)
matches = sorted(matches, key=lambda x:x.distance)
img_matches = cv2.drawMatches(img_m,kp_m,img_s,kp_s,matches[0:10],None)
cv2.imshow('Matches',img_matches)
cv2.waitKey(0)
cv2.destroyAllWindows()
