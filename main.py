from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import numpy as np

import ar
import triangulation
import animation
import snake

GAME_VIEW_WIDTH = 1280
GAME_VIEW_HEIGHT = 720

class App:
    def __init__(self):
        self.window = Tk()
        self.window.title('Orimagi Draw Demo')


        self.key_manager = KeyManager()
        self.window.bind("<Key>", self.key_manager.set)

        self.vid = VideoCapture(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT)

        self.home_view = HomeView(self.window)
        self.snake_view = SnakeView(self.window, self.vid, self.key_manager)
        self.menu_view = MenuView(self.window, self.show_view, self.key_manager, self.vid)
        self.show_view('home')

        self.window.mainloop()

    def show_view(self, view):
        if view == 'snake':
            self.home_view.frame.pack_forget()
            self.snake_view.frame.pack()

        if view == 'home':
            self.home_view.frame.pack()
            self.snake_view.frame.pack_forget()

class MenuView:
    def __init__(self, window, show_view, key_manager, vid):
        self.key_manager = key_manager
        self.vid = vid

        self.frame = Frame(window)
        self.frame.pack(fill=X, side=BOTTOM)

        self.home_button = Button(self.frame, text="Home", font=('Arial', '12'), command=self.show_home)
        self.home_button.pack(padx=5, pady=5, side=LEFT)
        self.snake_button = Button(self.frame, text="Snake", font=('Arial', '12'), command=self.show_snake)
        self.snake_button.pack(padx=5, pady=5, side=LEFT)


        self.vid_id = IntVar(self.frame)
        if len(self.vid.available) > 0:
            self.vid_id.set(self.vid.available[0])
            self.vid.start(self.vid.available[0])
        else:
            self.vid_id.set(-1)
        self.options = OptionMenu(self.frame, self.vid_id, *self.vid.available, command=self.on_vid_change)
        self.options.pack(side=RIGHT)

        self.label = Label(self.frame, text="Set video device", font=('Arial', '12'))
        self.label.pack(side=RIGHT)

        self.show_view = show_view

    def on_vid_change(self, value):
        self.vid.start(value)

    def show_snake(self):
        self.key_manager.set(None)
        self.show_view('snake')

    def show_home(self):
        self.key_manager.set(None)
        self.show_view('home')

class HomeView:
    def __init__(self, window):
        self.frame = Frame(window)
        self.frame.pack()

        self.canvas = Canvas(self.frame, width=GAME_VIEW_WIDTH, height=GAME_VIEW_HEIGHT)
        self.canvas.pack(side=BOTTOM)

        self.canvas.create_text((GAME_VIEW_WIDTH/2,GAME_VIEW_HEIGHT/2),
                                text='Welcome to MagiDraw!',
                                justify=CENTER,
                                font= ('Arial', '32'))

class SnakeView:
    def __init__(self, window, vid, key_manager):
        self.vid = vid

        self.key_manager = key_manager

        self.frame = Frame(window)
        self.frame.pack()

        self.canvas = Canvas(self.frame, width=GAME_VIEW_WIDTH, height=GAME_VIEW_HEIGHT)
        self.canvas.pack(side=TOP)

        self.game = snake.SnakeGame()

        self.update()

    def update(self):
        if self.frame.winfo_ismapped():
            img = self.game.update(self.vid.get_frame(), self.key_manager.get())
            self.key_manager.set(None)

            # Draw on canvas
            if img is not None:
                self.img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.img, anchor=NW)
        else:
            self.game.reset()

        self.frame.after(10, self.update)

class KeyManager:
    def __init__(self):
        self.keycode = None

    def set(self, event):
        if event is None:
            self.keycode = None
        else:
            self.keycode = event.keycode

    def get(self):
        return self.keycode

class VideoCapture:
    def __init__(self, width, height):
        # self.vid = cv2.VideoCapture('img/snake_game_video_4.MOV')

        self.available = []
        for i in range(5):
            vid = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if vid.isOpened():
                self.available.append(i)
                vid.release()

        self.vid = None

    def start(self, id):
        if self.vid is not None and self.vid.isOpened():
             self.vid.release()

        self.vid = cv2.VideoCapture(id, cv2.CAP_DSHOW)
        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, GAME_VIEW_WIDTH)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, GAME_VIEW_HEIGHT)

    def get_fake_frame(self):
        img = cv2.imread('img/snake_game_2.jpg')
        img = cv2.resize(img, (int(GAME_VIEW_WIDTH), int(GAME_VIEW_HEIGHT)))

        return img

    def get_frame(self):
        if self.vid is not None and self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                if GAME_VIEW_WIDTH != self.vid.get(cv2.CAP_PROP_FRAME_WIDTH) or GAME_VIEW_HEIGHT != self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT):
                    frame = cv2.resize(frame, (int(GAME_VIEW_WIDTH), int(GAME_VIEW_HEIGHT)))
                return frame

        return None


if __name__ == '__main__':
    App()
