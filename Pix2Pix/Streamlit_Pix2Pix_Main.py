import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from Streamlit_Pix2Pix_Generator import Generator
import numpy as np
import matplotlib.pyplot as plt 

# Page intro
st.title('Pix2Pix')


st.text('')
st.text('Sketch out an object, and see it brought to life with the power of machine learning!')
st.text('')
st.text('')


##### CODE FOR Pix2Pix #####


# Set default subject and colors
subject = 'Human'
stroke_color = '#FFFFFF'
background_color='#005000'

# Define page layout
left_column, right_column = st.columns([2,1])

# Create selection box and logic for various sketch subjects.
subject_selection = left_column.selectbox(label = 'Select what you wish to draw...', options = ['Human', 'Cat', 'Shoe', 'Handbag'], index = 0)
if subject_selection == 'Human':
    subject = 'Human'
    stroke_color = '#FFFFFF'
    background_color='#000000'
if subject_selection == 'Cat':
    subject = 'Cat'
    stroke_color = '#000000'
    background_color='#FFFFFF'
if subject_selection == 'Shoe':
    subject = 'Shoe'
    stroke_color = '#000000'
    background_color='#FFFFFF'
if subject_selection == 'Handbag':
    subject = 'Handbag'
    stroke_color = '#000000'
    background_color='#FFFFFF'


bg_image = st.sidebar.file_uploader("Upload an image to trace:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "polygon", "transform")
)
right_column.text('')
right_column.text('')
#realtime_update = st.sidebar.checkbox("Update in realtime", True)


# Create the drawing canvas
if 1 + 1 == 3:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Fill colors have full transparency
        stroke_width=1,
        stroke_color=stroke_color,
        background_color=background_color,
        background_image=Image.open(bg_image) if bg_image else None,
        #update_streamlit=realtime_update,
        height=256,
        width=256,
        drawing_mode=drawing_mode,
        key="canvas")




# right_column.button(label='Generate Real Image!')


##### SKETCH PROCESSING #####

# Store canvas sketch data into a variable
drawn_image = canvas_result.image_data

# Convert sketch data into parseable numpy array
drawn_image = np.array(Image.fromarray((drawn_image * 255).astype(np.uint8)).resize((256, 256)).convert('RGB'))
drawn_image = (drawn_image * 255).astype(np.uint8)

# Convert array into image and save
# downloaded_image = Image.fromarray(drawn_image)
# downloaded_image.save('Sam.png')

# drawn_image = Image.open('Sam.png')
# drawn_image = np.asarray(drawn_image)

# Pass numpy array into generator, and predict
gen = Generator(drawn_image, subject)
gen_image = gen.generate_image()

# Display prediction
st.image(gen_image)