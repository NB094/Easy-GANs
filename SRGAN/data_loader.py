#import scipy
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import imageio
import cv2

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        
        path = glob('./datasets/%s/*' % (self.dataset_name))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_hr = []
        imgs_lr = []
        for img_path in batch_images:
            #print(f'image path: {img_path}')   # Uncomment this line to display the filename of the image being processed
            
            img = self.imread(img_path)

            h, w = self.img_res
            low_w, low_h = int(w / 4), int(h / 4)

            img_hr = cv2.resize(img, self.img_res)
            img_lr = cv2.resize(img, (low_w, low_h))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_hr = np.fliplr(img_hr)
                img_lr = np.fliplr(img_lr)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)

        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def imread(self, path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)
