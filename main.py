import tkinter
import cv2
import PIL.Image, PIL.ImageTk

import numpy as np
import ar
import triangulation
import animation
import snake

class App:
    def __init__(self):
        self.window = tkinter.Tk()
        self.window.title('Draw and Play')
        self.window.bind("<Key>", self.key)

        self.width = 1280
        self.height = 720

        self.vid = VideoCapture(self.width, self.height)

        self.canvas = tkinter.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.pack()

        self.game = snake.SnakeGame()
        self.key = None

        self.update()
        self.window.mainloop()

    def key(self, event):
        self.key = event.keycode

    def update(self):
        # print(self.key)
        frame = self.game.update(self.vid.get_frame(), self.key)
        self.key = None

        # Draw on canvas
        if frame is not None:
            self.img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.img, anchor=tkinter.NW)

        self.window.after(15, self.update)

class VideoCapture:
    def __init__(self, width, height):
        self.vid = self.vid = cv2.VideoCapture(0)
        if not self.vid.isOpened():
            print('Video source not correct')

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if self.width != width or self.height != height:
            print('Video dimension not correct')

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                return frame

        return None

    def release(self):
        if self.vid.isOpened():
            self.vid.release()

App()
