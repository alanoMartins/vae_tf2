import tkinter as tk
import tensorflow as tf
from PIL import ImageTk, Image


def start_gui(model):

    def show_values():
        example = tf.constant([[w1.get() / 1000, w2.get() / 1000]])
        predictions = model.decoder(example)
        array_image = predictions.numpy() * 255
        ai = array_image.reshape((28,28))
        image1 = Image.fromarray(ai).resize(size=(100,100))
        test = ImageTk.PhotoImage(image1)

        label1 = tk.Label(image=test)
        label1.image = test

        # Position image
        label1.place(x= 75, y = 125)


    master = tk.Tk()
    w1 = tk.Scale(master, from_=0, to=1000, orient=tk.HORIZONTAL)
    w1.pack()
    w2 = tk.Scale(master, from_=0, to=1000, orient=tk.HORIZONTAL)
    w2.pack()
    tk.Button(master, text='Show', command=show_values).pack()

    master.geometry("250x250")
    tk.mainloop()