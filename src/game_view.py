from tkinter import *
import cv2
import PIL.Image, PIL.ImageTk

from dog.dog_game import DogGame

class GameView:
    def __init__(self, window, vid, key_manager, game, size):
        self.vid = vid

        self.key_manager = key_manager

        self.frame = Frame(window)
        self.frame.pack()

        self.canvas = Canvas(self.frame, width=size[0], height=size[1])
        self.canvas.pack(side=TOP)

        self.game = game()

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
