from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk
import webbrowser
import os
import sys

from snake.snake_game import SnakeGame
from dog.dog_game import DogGame
from game_view import GameView

GAME_VIEW_WIDTH = 1280
GAME_VIEW_HEIGHT = 720

class App:
    def __init__(self):
        self.window = Tk()
        self.window.title('MagiDraw Demo')


        self.key_manager = KeyManager()
        self.window.bind("<Key>", self.key_manager.set)

        self.vid = VideoCapture(GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT)

        self.home_view = HomeView(self.window)
        self.snake_view = GameView(self.window, self.vid, self.key_manager, SnakeGame, (GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT))
        self.dog_view = GameView(self.window, self.vid, self.key_manager, DogGame, (GAME_VIEW_WIDTH, GAME_VIEW_HEIGHT))
        self.menu_view = MenuView(self.window, self.show_view, self.key_manager, self.vid)
        self.show_view('home')

        self.window.mainloop()

    def show_view(self, view):
        if view == 'snake':
            self.dog_view.frame.pack_forget()
            self.home_view.frame.pack_forget()
            self.snake_view.frame.pack()

        if view == 'dog':
            self.snake_view.frame.pack_forget()
            self.home_view.frame.pack_forget()
            self.dog_view.frame.pack()

        if view == 'home':
            self.home_view.frame.pack()
            self.snake_view.frame.pack_forget()
            self.dog_view.frame.pack_forget()

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
        self.dog_button = Button(self.frame, text="Dog", font=('Arial', '12'), command=self.show_dog)
        self.dog_button.pack(padx=5, pady=5, side=LEFT)

        if len(self.vid.available) > 0:
            self.vid_id = IntVar(self.frame)
            self.vid_id.set(self.vid.available[0])
            self.vid.start(self.vid.available[0])
            self.options = OptionMenu(self.frame, self.vid_id, *self.vid.available, command=self.on_vid_change)
            self.options.pack(side=RIGHT)
            msg = "Set video device"
        else:
            msg = "No video device"

        self.label = Label(self.frame, text=msg, font=('Arial', '12'))
        self.label.pack(side=RIGHT)

        self.show_view = show_view

    def on_vid_change(self, value):
        self.vid.start(value)

    def show_snake(self):
        self.key_manager.set(None)
        self.show_view('snake')

    def show_dog(self):
        self.key_manager.set(None)
        self.show_view('dog')

    def show_home(self):
        self.key_manager.set(None)
        self.show_view('home')

class HomeView:
    def __init__(self, window):
        self.frame = Frame(window)
        self.frame.pack()

        w = 600
        h = 100
        self.canvas = Canvas(self.frame, width=w, height=h)
        self.canvas.pack()
        self.canvas.create_text((w/2,h/2),
                                text='Welcome to MagiDraw!',
                                justify=CENTER,
                                font= ('Arial', '32'))

        self.board_frame = Frame(self.frame)
        self.board_frame.pack(pady=(0,20),side=BOTTOM)

        self.label = Label(self.board_frame, text='Open Draw Board', font=('Arial', '10'))
        self.label.pack(padx=5, pady=5, side=LEFT)

        self.snake_button = Button(self.board_frame, text="Snake", font=('Arial', '10'),command=self.open_snake)
        self.snake_button.pack(padx=5, pady=5, side=LEFT)

        self.dog_button = Button(self.board_frame, text="Dog", font=('Arial', '10'),command=self.open_dog)
        self.dog_button.pack(padx=5, pady=5, side=LEFT)


    def open_snake(self):
        # For debugging, make sure script is ran from magi-draw
        # Build should be fine
        path = resource_path(os.path.join('.','img','snake.pdf'))
        webbrowser.open(path)

    def open_dog(self):
        path = resource_path(os.path.join('.','img','dog.pdf'))
        webbrowser.open(path)

class KeyManager:
    def __init__(self):
        self.keycode = None

    def set(self, event):
        # print(event.keysym)
        if event is None:
            self.keycode = None
        else:
            self.keycode = (event.keycode, event.keysym)

    def get(self):
        return self.keycode

class VideoCapture:
    def __init__(self, width, height):
        self.available = []
        for i in range(5):
            vid = cv2.VideoCapture(i)
            if vid.isOpened():
                self.available.append(i)
                vid.release()

        self.vid = None

    def start(self, id):
        if self.vid is not None and self.vid.isOpened():
             self.vid.release()

        self.vid = cv2.VideoCapture(id)
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

def resource_path(relative_path):
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

if __name__ == '__main__':
    App()
