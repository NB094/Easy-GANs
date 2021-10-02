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



subject = 'Human'
stroke_color = '#FFFFFF'
background_color='#005000'


left_column, right_column = st.columns([2,1])

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


bg_image = st.sidebar.file_uploader("Upload trace:", type=["png", "jpg"])
drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("freedraw", "line", "rect", "circle", "polygon", "transform")
)
right_column.text('')
right_column.text('')
#realtime_update = st.sidebar.checkbox("Update in realtime", True)


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
    key="canvas",
)

# right_column.button(label='Generate Real Image!')

# Do something interesting with the image data and paths


# if canvas_result.image_data is not None:




drawn_image = st.image(canvas_result.image_data, output_format = 'JPEG')

drawn_image = canvas_result.image_data
drawn_image = Image.fromarray(drawn_image, 'RGB')
drawn_image = np.asarray(drawn_image)
# drawn_image = np.reshape(drawn_image, (1, 256, 256, 3))

st.text(f'drawn_image array shape: {drawn_image.shape}')
st.text(f'drawn_image array: {drawn_image}')


gen = Generator(drawn_image, subject)
gen_image = gen.generate_image()
st.image(gen_image)


# st.text(f'gen_image array: {gen_image}')

#st.pyplot(gen_image)

# plt.imshow(gen_image)
# plt.axis('off')
# plt.show()



# right_column.button(label='Generate', on_click=main_Pix2Pix())

# if canvas_result.image_data is not None:
#     drawn_image = st.image(canvas_result.image_data)
#     gen = Generator(drawn_image, subject)
#     gen_image = gen.generate_image()
#     st.image(gen_image)

# if canvas_result.json_data is not None:
#     objects = pd.json_normalize(canvas_result.json_data["objects"])
#     for col in objects.select_dtypes(include=['object']).columns:
#         objects[col] = objects[col].astype("str")
#     st.dataframe(objects)



# Integrate model code!