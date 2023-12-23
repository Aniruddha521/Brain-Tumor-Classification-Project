import numpy as np
import tensorflow as tf
import tkinter as tk
import math
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Add, ReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import MaxPooling2D
from tkinter import font
from tkinter import filedialog
from tkinter import ttk
from PIL import Image, ImageTk


class inception_module():
    def __init__(self, filters, **kwargs):
        super(inception_module, self).__init__(**kwargs)
        self.filters = filters
        # First middle layer
        self.conv_1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')
        ###########################################################################
        # Second middle layer
        self.conv_2 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')
        self.conv_3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')
        ############################################################################
        # Third middle layer
        self.conv_4 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')
        self.conv_5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')
        ##############################################################################
        # Fourth middle layer
        self.maxpool_build = MaxPooling2D((3, 3), strides=(1, 1), padding='same')
        self.conv_6 = Conv2D(filters[5], (1, 1), padding='same', activation='relu')
    def get_config(self):
        config = super(inception_module, self).get_config()
        config.update({'filters': self.filters})
        return config
    def call(self,x):
        conv1 = self.conv_1(x)
        conv2 = self.conv_2(x)
        conv3 = self.conv_3(conv2)
        conv4 = self.conv_4(x)
        conv5 = self.conv_5(conv4)
        maxpool = self.maxpool_build(x)
        conv6 = self.conv_6(maxpool)

        inception_block = concatenate([conv1, conv3, conv5, conv6], axis=-1)
        return inception_block

class residual_block(tf.keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(residual_block, self).__init__(**kwargs)
        self.filters = filters
        element_sum = filters[0] + filters[2] + filters[4] + filters[5]
        self.conv_shortcut = Conv2D(element_sum, (1, 1), padding='same')
        self.norm = tf.keras.layers.BatchNormalization()
        self.mid_conv_1 = Conv2D(element_sum, (3, 3), padding='same', activation='relu')
        self.mid_conv_2 = Conv2D(element_sum, (5, 5), padding='same', activation='relu')
        self.inception_1 = inception_module(filters=filters)
        self.inception_2 = inception_module(filters=filters)
        self.inception_3 = inception_module(filters=filters)
    def get_config(self):
        config = super(residual_block, self).get_config()
        config.update({'filters': self.filters})
        return config
    @tf.function(experimental_relax_shapes=True)
    def call(self, x):
        shortcut = self.conv_shortcut(x)

        # Layer 1
        incep_1 = self.inception_1.call(x)
        inception_1_norm = self.norm(incep_1)
        x = ReLU()(inception_1_norm)

        # Layer 2
        mid_layer_1 = self.mid_conv_1(x)
        mid_layer_2 = self.mid_conv_2(mid_layer_1)
        mid_norm = self.norm(mid_layer_2)
        x = ReLU()(mid_norm)

        # Layer 3
        incep_3 = self.inception_3.call(x)
        inception_3_norm = self.norm(incep_3)
        x = ReLU()(inception_3_norm)

        # Output
        x = Add()([x, shortcut])
        x = ReLU()(x)
        return x
    
def preprocess_image(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)
    
    return image, label

def browse_file():
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg"),("Image files", "*.png"),("Image files", "*.jepg")])
        if file_path:
            load_and_display_image(file_path)
            image_path.set(file_path)
            main.set(True)

def reset_image():
    global accuracy
    canvas.delete("reset_image")
    entry_var.set("")
    if main.get():
        if image_path.get() != "":
            for i in range(math.ceil(accuracy)+1,0,-1):
                progress_var.set(i)
                percentage_label.config(text=f"{i}%")
                root.update_idletasks()
                root.after(10)
            percentage_label.config(text=f"{0}%")
            root.update_idletasks()
        else:
             shaker(10)
        image_path.set("")
    else:
         shaker(10)

def detect():
    global prediction, accuracy
    if main.get():
        try:
            img = load_img(image_path.get(), color_mode='grayscale', target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            img_preprocessed, _ = preprocess_image(img_array, label=None)
            prediction = model.predict(img_preprocessed)
            classes = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']
            accuracy = prediction[0][np.argmax(prediction)]*100
            entry_var.set(classes[np.argmax(prediction)])
            for i in range(math.ceil(accuracy)+1):
                progress_var.set(i)
                percentage_label.config(text=f"{i}%")
                root.update_idletasks()
                root.after(20)
            percentage_label.config(text=f"{accuracy}%")
            root.update_idletasks()
        except:
             shaker(10)
    else:
        shaker(10)

def load_and_display_image(file_path):
        canvas.delete("invalid_warning")
        pil_image = Image.open(file_path)
        resized_image = pil_image.resize((680, 460))
        tk_image = ImageTk.PhotoImage(resized_image)
        canvas.image = tk_image
        canvas.create_image(350, 80, anchor=tk.NW, image=tk_image, tags="reset_image")

def close_window():
    root.destroy()

def shaker(n): 
    main.set(False)
    num = -10
    for j in range(n):
        next_num = -1*num
        canvas.delete("imagecontainer")
        canvas.create_rectangle(320-num,50,1060-num,570, fill="green", tags="imagecontainer")
        if j == 0 :
            num = 2*next_num
        else:
             num = next_num
        canvas.update_idletasks()
        canvas.after(100)
    canvas.delete("imagecontainer")
    canvas.create_rectangle(320,50,1060,570, fill="green", tags="imagecontainer")
    canvas.create_text(650,300,text="Please select an Image", fill='red', font=font.Font(family="Helvetica", size=18, weight="bold"), tags="invalid_warning")

# path of the model
model_path = "/home/aniruddha/Brain Tumor Classification Project/model.h5"

# loading model with custom object residual block
custom_objects = {'residual_block': residual_block}
with tf.keras.utils.custom_object_scope(custom_objects):
    model = tf.keras.models.load_model(model_path)
# Root and Canvas
root = tk.Tk()
canvas = tk.Canvas(root,width=1080, height=720 ,bg="lightblue")
root.title("Tumor Detector")
root.geometry("1080x720")
root.minsize(1080,720)
# All Variables
progress_var = tk.DoubleVar()
entry_var = tk.StringVar()
image_path = tk.StringVar()
main = tk.BooleanVar()
# Designing in the whole canvas
custom_font = font.Font(family="Helvetica", size=18, weight="bold")
entry = tk.Entry(root, width=25, textvariable=entry_var)
progress_bar = ttk.Progressbar(root, variable=progress_var, length=400, mode="determinate")
button1 = tk.Button(root, text="Select Images", background="yellow", fg="black", padx= 30, pady=10, font=custom_font, command=browse_file)
button2 = tk.Button(root, text="Detect", background="yellow", fg="black", padx= 80, pady=10, font=custom_font, command=detect)
button3 = tk.Button(root, text="Reset", background="yellow", fg="black", padx= 80, pady=10, font=custom_font, command= reset_image)
button4 = tk.Button(root, text="Exit", background="yellow", fg="black", padx= 90, pady=10, font=custom_font, command=close_window)
percentage_label = tk.Label(root, text="0%", font=("Helvetica", 14))
label = tk.Label(root,text="  Brain Tumor Detector and Classifier  ", background="#FF9999", fg="black",font=font.Font(family="Helvetica", size=20, weight="bold"))
canvas.create_rectangle(20,50,300,650, fill="green")
canvas.create_rectangle(320,50,1060,570, fill="green", tags="imagecontainer")
canvas.create_rectangle(320,580,570,680, fill="green")
canvas.create_rectangle(580,580,1060,680, fill="green")
button_window1 = canvas.create_window(40, 100, anchor=tk.NW, window=button1)
button_window2 = canvas.create_window(40, 250, anchor=tk.NW, window=button2)
button_window3 = canvas.create_window(40, 400, anchor=tk.NW, window=button3)
button_window4 = canvas.create_window(40, 550, anchor=tk.NW, window=button4)
progress_window = canvas.create_window(600, 620, anchor=tk.NW, window=progress_bar)
entry_window1 = canvas.create_window(340,620, anchor=tk.NW, window=entry)
percentage_label_window = canvas.create_window(1010,625, anchor=tk.NW, window=percentage_label)
label = canvas.create_window(420,10, anchor=tk.NW, window=label)
canvas.create_text(160,70,text="<<<-----------Menu----------->>>",fill='White', font=font.Font(family="Helvetica", size=16, weight="bold"))
canvas.create_text(430,600,text="Detection Result",fill='yellow', font=font.Font(family="Helvetica", size=18, weight="bold"))
canvas.create_text(830,600,text="Accuracy ",fill='yellow', font=font.Font(family="Helvetica", size=18, weight="bold"))
# Design packed
canvas.pack()
root.mainloop()