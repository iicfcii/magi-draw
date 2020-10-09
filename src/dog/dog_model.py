import cv2

from dog.dog_bones import *

HEAD_OFFSET = 310
TAIL_OFFSET = 120
DOG_HEIGHT = 300
MAX_VX = 60

class DogModel:
    def __init__(self):
        self.x = BOARD_REF[3,0]+DEFAULT_PARAMS['base']['x']*RATIO
        self.y = BOARD_REF[3,1]-(DOG_DRAW_HEIGHT-DEFAULT_PARAMS['base']['y']*RATIO)
        self.vx = 0

        self.head_right = True

    def update(self):
        self.x = self.x + self.vx


        if self.head_right:
            if self.x+HEAD_OFFSET > BOARD_WIDTH:
                self.x = BOARD_WIDTH-HEAD_OFFSET
                self.vx = 0
            if self.x-TAIL_OFFSET < 0:
                self.x = TAIL_OFFSET
                self.vx = 0
        else:
            if self.x+TAIL_OFFSET > BOARD_WIDTH:
                self.x = BOARD_WIDTH-TAIL_OFFSET
                self.vx = 0
            if self.x-HEAD_OFFSET < 0:
                self.x = HEAD_OFFSET
                self.vx = 0

    def move(self, img, mat):
        center = self.find_goal(img, mat)
        if center is None:
            self.vx = 0
            return

        dist = 0
        if self.head_right:
            dist_head = center[0] - (self.x+HEAD_OFFSET)
            dist_tail = center[0] - (self.x-TAIL_OFFSET)

            if dist_head > 0: dist = dist_head
            if dist_tail < 0:  dist = dist_tail
        else:
            dist_head = center[0] - (self.x-HEAD_OFFSET)
            dist_tail = center[0] - (self.x+TAIL_OFFSET)

            if dist_head < 0: dist = dist_head
            if dist_tail > 0:  dist = dist_tail

        # print(center, dist)

        if dist != 0:
            if dist > 0:
                vx = MAX_VX
            else:
                vx = -MAX_VX
            self.vx = 0.5*vx+0.5*self.vx # Simple filter for sudden changes

            if dist < 0 and self.head_right: self.head_right = False
            if dist > 0 and not self.head_right: self.head_right = True
        else:
            self.vx = 0

        if np.absolute(self.vx) < 1: self.vx = 0
        if self.vx > MAX_VX: self.vx = MAX_VX
        if self.vx < -MAX_VX: self.vx = -MAX_VX

    def find_goal(self, img, mat):
        if mat is None: return None

        # Find black circle
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ret, img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 401, 60)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), borderValue=0, iterations=5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), borderValue=0, iterations=5)

        center = None
        contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        rects = []
        for contour in contours:
            rect = cv2.boundingRect(contour)
            # img = cv2.rectangle(img, rect, 127)
            aspect_ratio = rect[2]/rect[3]
            # area = cv2.contourArea(contour)
            if rect[2] > 20 and \
               rect[3] > 20 and \
               rect[2] < 100 and \
               rect[3] < 100 and \
               aspect_ratio < 2 and \
               aspect_ratio > 0.8:
                rects.append(rect)
            else:
                img = cv2.fillPoly(img, [contour], 0)

        rect = None
        if len(rects) > 1:
            dist = []
            for i in range(0,len(rects)-1):
                for j in range(i+1,len(rects)):
                    center_i = (rects[i][0]+rects[i][2]/2,rects[i][1]+rects[i][3]/2)
                    center_j = (rects[j][0]+rects[j][2]/2,rects[j][1]+rects[j][3]/2)

                    # Calculate actual distance based on rect size
                    size_avg = (rects[i][2]+rects[i][3]+rects[j][2]+rects[j][3])/4 # Average width or height

                    # difference between actual distance and distance bewteen rects
                    dist.append((np.absolute((center_i[0]-center_j[0])**2+(center_i[1]-center_j[1])**2-(size_avg*1.5)**2),i,j,size_avg))
            dist.sort(key=lambda d:d[0])
            if dist[0][0] < (dist[0][3]*1.5)**2:
                rect_a = np.array(rects[dist[0][1]])
                rect_b = np.array(rects[dist[0][2]])
                rect = (rect_a+rect_b)/2

        center_board = None
        if rect is not None:
            center = np.array([[[rect[0]+rect[2]/2,rect[1]+rect[3]/2]]], dtype=np.float) # Assume only one circle will be found
            center_board = cv2.perspectiveTransform(center, np.linalg.inv(mat)).reshape((2,1))

        # if rect is not None: img = cv2.circle(img,(int(center[0,0,0]),int(center[0,0,1])),3,255,-1)
        # cv2.imshow('img', img)
        # cv2.waitKey(20)

        return center_board
