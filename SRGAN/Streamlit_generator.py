from keras.models import load_model
import cv2
import imageio    
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Generator():
    def __init__(self, img_object, height, width):
        self.img_object = img_object
        self.height = height
        self.width = width
        

    def rescale_img(self):
        """
        Rescale input images before prediction (generation) in order to meet the expected input shape and
            value range of the generator model.

        ==Input==
          img_path: filepath to the .jpg input image in relation to this notebook.

        ==Output==
          img_lr: rescaled LR image.
          img_sr: generated super resolution image.
          img_hr: original HR image.

        """
        # Load pre-trained model
        a = load_model('saved_model/generator.h5', compile=False)
        
        # Assign img variable differently depending on if it's a user-uploaded image or not
        try:
            img = imageio.imread(self.img_object, pilmode='RGB').astype(float)
        except:
            img = np.array(Image.open(self.img_object)).astype(float)
        
        img_res = (256, 256)        
        h, w = img_res
        low_h, low_w = int(h / 4), int(w / 4)

        #img_hr = img
        img_res = (178, 218)
        img_hr = cv2.resize(img, img_res)
        img_lr = cv2.resize(img, (low_w, low_h))


        #img_lr = np.array(img_lr) / 127.5 - 1.
        img_lr = np.array(img_lr)/255

        #img_hr = np.array(img_hr) / 127.5 - 1.
        img_hr = np.array(img_hr)/255

        # Expand dimensions to match generator's expected input dimensions, and predict
        img_lr2 = np.expand_dims(img_lr, axis=0)
        img_sr = a.predict(img_lr2)[0]
        
        # Resize img_lr and img_sr to match the original HR aspect ratio.
        #img_res = (self.width, self.height)
        img_lr = cv2.resize(img_lr, img_res)
        img_sr = cv2.resize(img_sr, img_res)
        
        return img_lr, img_sr, img_hr