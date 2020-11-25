import cv2
import numpy as np

from dog.dog_bones import *

HEAD_OFFSET = 310
TAIL_OFFSET = 120
DOG_HEIGHT = 300

class DogModel:
    def __init__(self):
        self.x = BOARD_REF[3,0]+DEFAULT_PARAMS['base']['x']*RATIO
        self.y = BOARD_REF[3,1]-(DOG_DRAW_HEIGHT-DEFAULT_PARAMS['base']['y']*RATIO)
        self.vx = 0
        self.walk_count = 0
        self.max_walk_count = None
        self.MAX_VX = 40
        self.WALK_VX = 8

        self.head_right = True
        self.look = None

        d = int(WAND_CIRCLE_DIAMETER)
        r = int(d/2)
        self.circle_template = np.zeros((d,d), dtype=np.uint8)
        self.circle_template = cv2.circle(self.circle_template,(r,r),r,255,-1)
        # cv2.imshow('template', self.circle_template)
        # cv2.waitKey()

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

    def set_head_right(self):
        if self.vx < 0 and self.head_right: self.head_right = False
        if self.vx > 0 and not self.head_right: self.head_right = True

    def move(self, img, mat):
        center = self.find_goal(img, mat)

        if center is None:
            self.look = None

            # Walk left/right or rest for a random period
            if self.max_walk_count is None:
                self.max_walk_count = np.floor(np.random.rand()*20+30)
                value = np.random.rand()
                if value < 1/3:
                    self.vx = 0
                elif value < 2/3:
                    self.vx = self.WALK_VX
                else:
                    self.vx = -self.WALK_VX
            else:
                if self.walk_count > self.max_walk_count:
                    self.vx = 0
                    self.walk_count = 0
                    self.max_walk_count = None
                else:
                    self.walk_count += 1
                    # print(self.walk_count, self.max_walk_count)
            self.set_head_right()
        else:
            self.walk_count = 0
            self.max_walk_count = None
            # Run toward target
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

            if dist != 0:
                if dist > 0:
                    vx = self.MAX_VX
                else:
                    vx = -self.MAX_VX
                self.vx = 0.5*vx+0.5*self.vx # Simple filter for sudden changes
                self.set_head_right()
            else:
                self.vx = 0
                # print(center)
                if center[1] < 350:
                    self.look = 'up'
                elif center[1] > 850:
                    self.look = 'down'
                else:
                    self.look = 'happy'

            if np.absolute(self.vx) < 1: self.vx = 0

    def find_goal(self, img, mat):
        if mat is None: return None

        dot_size = 40*RATIO
        dot_size_lb = dot_size*0.8
        dot_size_ub = dot_size*1.2
        padding = 200 # So can look for wand around board

        # Find black circle
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mat_translated = np.array([[1,0,-padding],[0,1,-padding],[0,0,1]])
        mat_padded =  mat.copy() @ mat_translated
        size = (int(BOARD_WIDTH+padding*2),int(BOARD_HEIGHT+MARKER_SIZE*2+padding*2))
        img = cv2.warpPerspective(img, mat_padded, size, flags=cv2.WARP_INVERSE_MAP)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 401, 60)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), borderValue=0, iterations=5)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)), borderValue=0, iterations=5)

        contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        similarity = []
        for contour in contours:
            x,y,w,h = cv2.boundingRect(contour)
            if w > dot_size_lb and h > dot_size_lb and \
               w < dot_size_ub and h < dot_size_ub:
                r = int(WAND_CIRCLE_DIAMETER/2)
                x_center = int(x+w/2)
                y_center = int(y+h/2)

                if x_center-r < 0 or y_center-r < 0 or \
                   x_center+r > img.shape[1]-1 or y_center+r > img.shape[0]-1:
                   continue # Size not correct

                img_contour = img[y_center-r:y_center+r,x_center-r:x_center+r]
                img_contour = cv2.bitwise_xor(img_contour, self.circle_template)
                similarity.append(((x_center,y_center),np.sum(img_contour)/255))
                # cv2.imshow('img contour', img_contour)
                # cv2.waitKey()

        similarity.sort(key=lambda r:r[1])

        if len(similarity) >= 2:
            p1 = similarity[0][0]
            p2 = similarity[1][0]


            center = ((p1[0]+p2[0])/2-padding, (p1[1]+p2[1])/2-padding)
        else:
            center = None

        # if center is not None: img = cv2.circle(img,(int(center[0]+padding),int(center[1]+padding)),5,255,-1)
        # img = cv2.resize(img, None, fx=0.5,fy=0.5)
        # cv2.imshow('img', img)
        # cv2.waitKey(10)

        return center
