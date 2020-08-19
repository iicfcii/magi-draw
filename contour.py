import numpy as np
import cv2

img_color = cv2.imread('snake.jpg', 1)
img_gray = cv2.imread('snake.jpg', 0)
ret,img_bin = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)

img_dilation = cv2.dilate(img_bin,cv2.getStructuringElement(cv2.MORPH_RECT,(5,5)),iterations=2)

contours, hierarchy = cv2.findContours(img_dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
contour = contours[0]

print(contour.shape)
contour_simple = cv2.approxPolyDP(contour,2,True)
print(contour_simple.shape)
cv2.polylines(img_color, [contour_simple], True, (255,0,0), 2)

cv2.imshow('Snake Color',img_color)
cv2.imshow('Snake Gray',img_gray)
cv2.imshow('Snake Binary',img_bin)
cv2.imshow('Snake Dilation',img_dilation)
cv2.waitKey(0)
cv2.destroyAllWindows()
