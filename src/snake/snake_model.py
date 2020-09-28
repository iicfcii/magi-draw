from snake.snake_bones import *

class SnakeModel:
    def __init__(self):
        # Make sure snake is at bottom
        self.RECT = (GAME_X,GAME_Y,GAME_WIDTH,GAME_HEIGHT)
        self.X_DEFAULT = self.RECT[0]+self.RECT[2]/2
        self.Y_DEFAULT = self.RECT[1]+self.RECT[3]
        self.SPEED = 45
        self.ACC = self.SPEED/len(TURN_LEFT_PARAMS) # Sync with number of animation frames

        self.x = self.X_DEFAULT
        self.y = self.Y_DEFAULT
        self.v = 0.0 # Horizontal speed
        self.eat_counter = 3 # Number of frames of eat animation

    # Physics
    def update(self):
        self.x = self.x + self.v # Velocity
        self.v = self.v - np.sign(self.v)*self.ACC # Acceleration
        if np.abs(self.v) < 1: self.v = 0

        self.eat_counter = self.eat_counter + 1
        if self.eat_counter > 3: self.eat_counter = 3

        # Simple constrain
        if self.x > self.RECT[0]+self.RECT[2]:
            self.x = self.RECT[0]+self.RECT[2]
        if self.x < self.RECT[0]:
            self.x = self.RECT[0]

    def move(self, key):
        if self.v == 0:
            if key == 65: # a
                self.v = -self.SPEED
            if key == 68: # d
                self.v = self.SPEED

    def constrain(self, frame):
        img, anchor, mask = frame
        # Two rectangles
        x,y,w,h = self.RECT
        x_snake = self.x-anchor[0]
        y_snake = self.y-anchor[1]
        w_snake = img.shape[1]
        h_snake = img.shape[0]

        rect_u = animation.union_rects((x,y,w,h), (x_snake,y_snake,w_snake,h_snake))
        if rect_u is not None:
            x_u,y_u,w_u,h_u = rect_u
            dw = w_snake-w_u
            dh = h_snake-h_u
            if dw > 0:
                if x_u == x:
                    self.x += dw
                else:
                    self.x -= dw
            if dh > 0:
                if y_u == y:
                    self.y += dh
                else:
                    self.y -= dh
        else:
            self.x = self.X_DEFAULT
            self.y = self.Y_DEFAULT
