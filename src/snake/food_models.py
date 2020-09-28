from snake.snake_bones import *

class FoodModels:
    def __init__(self):
        self.RECT = (GAME_X,GAME_Y,GAME_WIDTH,GAME_HEIGHT)
        self.X_RANGE = tuple(np.arange(GAME_X,GAME_X+GAME_WIDTH+1,GAME_STEP))

        self.models = []
        self.models.append(FoodModel(self.random_x(), self.RECT))

        self.eat_counter = 0
        self.frame_counter = 0

    def random_x(self):
        return self.X_RANGE[np.random.randint(0, len(self.X_RANGE))]

    def update(self, snake_model):
        self.frame_counter += 1

        # Remove bottom ones and eaten ones
        for i in range(len(self.models)-1,-1,-1):
            model = self.models[i]
            model.update()

            if model.v == 0:
                self.models.pop(i)
            else:
                if model.x == snake_model.x and model.y > snake_model.y-GAME_SNAKE_LENGTH and model.y < snake_model.y-GAME_SNAKE_BODY_LENGTH:
                    self.models.pop(i)
                    self.eat_counter += 1
                    snake_model.eat_counter = -1 # Start eat animation

        # Generate at even speed
        if self.frame_counter % 20 == 0:
            self.models.append(FoodModel(self.random_x(), self.RECT))

class FoodModel:
    def __init__(self, x, rect):
        self.RECT = rect

        self.x = x
        self.y = self.RECT[1]
        self.v = 20 # Vertical speed

    # Physics
    def update(self):
        self.y = self.y + self.v # Velocity
        if self.y > self.RECT[1]+self.RECT[3]:
            self.v = 0
