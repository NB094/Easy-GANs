from keras.preprocessing.image import img_to_array
import tensorflow as tf
import numpy as np

class Generator():
    def __init__(self, img_object, subject):
        self.img_object = img_object        
        self.subject = subject

    def load_image(self, filename, size=(256,256)):
        # convert to numpy array
        pixels = img_to_array(filename)

        # scale from [0,255] to [-1,1]
        pixels = (pixels - 127.5) / 127.5
        # reshape to 1 sample
        pixels = np.expand_dims(pixels, 0)
        return pixels



    def generate_image(self, model):
        src_image = self.load_image(self.img_object)

        # generate image from source
        gen_image = model.predict(src_image)

        gen_image = tf.reshape(gen_image, [256, 256, 3])
        # scale from [-1,1] to [0,1]
        gen_image = (gen_image + 1) / 2.0

        return np.asarray(gen_image)


