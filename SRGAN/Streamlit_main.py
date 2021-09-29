import streamlit as st
import numpy as np
from Streamlit_generator import Generator
import urllib.request
from PIL import Image

# Page intro
st.title('Enhance Low Resolution Images with Machine Learning!')

st.text('')
st.text('Want to see a computer add more detail to a pixelated image?')
st.text('')
st.text('Choose an example image below or upload your own, and view a live demo!')
st.text('')

expander = st.sidebar.expander("Frequently-Asked Questions", expanded=True)
expander.write("**What type of machine learning is being used?** \n\n \
The model's architecture is based on a Super Resolution Generative Adversarial Network, or SRGAN. \n\n &nbsp \n\n \
**How do GANs work?** \n\n \
There are two main components to GAN models: a *discriminator* and a *generator*. \n\n \
The purpose of the discriminator is to classify images presented to it as real or fake. \
The purpose of the generator is to create plausible images to fool the discriminator. \n\n \
After many cycles of training, the skill of the generator improves enough to produce some impressive results!     \n\n &nbsp \n\n \
**What is the difference between a GAN and an SRGAN?** \n\n \
The basic idea behind SRGANs is the same. The primary difference is way the model improves after each cycle, which is based on \
a *loss* calculation. For SRGANs, this calculation is designed to maximize the perceptual quality of \
upscaled images. \n\n &nbsp \n\n \
**Where can I read more about SRGANs?** \n\n \
For more information on SRGANs, check out [this paper.](https://arxiv.org/abs/1609.04802)     ")

st.text('')
st.text('')


##### CODE FOR EXAMPLE IMAGES #####

# Format widgets and photos side by side
left_column, right_column = st.columns([2,1])
left_column2, middle_column2, right_column2 = st.columns(3)


# Selection Box for Example Images
image_selection = left_column.selectbox(label = 'Select example image', options = ['Celebrity A', 'Celebrity B', 'Celebrity C', 'Celebrity D', 'Celebrity E'])


# Load all required files from URLs
urllib.request.urlretrieve('https://raw.githubusercontent.com/NB094/Easy-GANs/main/SRGAN/datasets/img_align_celeba/130428.jpg', '130428.jpg')
urllib.request.urlretrieve('https://raw.githubusercontent.com/NB094/Easy-GANs/main/SRGAN/datasets/img_align_celeba/130082.jpg', '130082.jpg')
urllib.request.urlretrieve('https://raw.githubusercontent.com/NB094/Easy-GANs/main/SRGAN/datasets/img_align_celeba/130526.jpg', '130526.jpg')
urllib.request.urlretrieve('https://raw.githubusercontent.com/NB094/Easy-GANs/main/SRGAN/datasets/img_align_celeba/130779.jpg', '130779.jpg')
urllib.request.urlretrieve('https://raw.githubusercontent.com/NB094/Easy-GANs/main/SRGAN/datasets/img_align_celeba/130869.jpg', '130869.jpg')


# Insert original HR image
image_dict = {'Celebrity A': '130428.jpg', \
              'Celebrity B': '130082.jpg', \
              'Celebrity C': '130526.jpg', \
              'Celebrity D': '130779.jpg', \
              'Celebrity E': '130869.jpg'}

img = Image.open(image_dict[image_selection])
height = img.height
width = img.width

left_column2.image(image = image_dict[image_selection], caption = 'Original Image')


# Generate LR and SR images
gen = Generator(image_dict[image_selection], height, width)
img_lr, img_sr, _ = gen.rescale_img()


# Insert downscaled LR image
middle_column2.image(image = img_lr, caption = 'Downscaled Image')


# Define function for inserting predicted SR image
def create_SR():    
    return right_column2.image(image = img_sr, caption = 'Enhanced Image', clamp=True)


# Button for Example Images
right_column.text('')
right_column.text('')
if right_column.button(label = 'Click to Enhance Image!', key=1):
    create_SR()







##### CODE FOR USER-UPLOADED IMAGES #####

st.text('')
st.text('')
st.text('')


# Create additional columns for formatting layout
left_column3, right_column3 = st.columns([2,1])
left_column4, middle_column4, right_column4 = st.columns(3)


# Upload own image
user_img = left_column3.file_uploader('Upload your own face...', type=['jpg', 'jpeg'], help='For the best results, try to crop your image to look similar to the example images above.')




if user_img is not None:
    #left_column4.image(image = user_img, caption = 'Original Image')

    img2 = Image.open(user_img)
    height2 = img2.height
    width2 = img2.width
    
    gen2 = Generator(user_img, height2, width2)
    img_lr2, img_sr2, img_hr2 = gen2.rescale_img()

    left_column4.image(image = img_hr2, caption = 'Original Image')
    middle_column4.image(image = img_lr2, caption = 'Downscaled Image')


# Define function for inserting predicted SR image
def create_SR2():    
    return right_column4.image(image = img_sr2, caption = 'Enhanced Image', clamp=True)


# Button for own images
right_column3.text('')
right_column3.text('')
if right_column3.button(label = 'Click to Enhance Image!', key=2):
    create_SR2()