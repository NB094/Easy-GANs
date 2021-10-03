from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from Streamlit_Pix2Pix_Generator import Generator
import numpy as np
import urllib.request
from keras.preprocessing.image import load_img
from keras.models import load_model

# Page intro
st.title('Pix2Pix – See Your Sketches Brought to Life!')

st.text('')
st.markdown('Sketch out an object using the canvas below, and let your computer do the rest of the heavy lifting.')
st.text('')
st.text('')


# Links and FAQ section
st.sidebar.markdown("### [SRGANs Web Page](dummy)")
st.sidebar.markdown("### [Number Generator Web Page](dummy)")
st.sidebar.text('')

expander = st.sidebar.expander("Pix2Pix Frequently-Asked Questions", expanded=True)
expander.write("**What type of machine learning is being used?** \n\n \
The model's architecture is based on solving image-to-image translation with a Conditional Generative Adversarial Network, or cGAN. \n\n &nbsp \n\n \
**How do GANs work?** \n\n \
There are two main components to GAN models: a *discriminator* and a *generator*. \n\n \
The purpose of the discriminator is to classify images presented to it as real or fake. \
The purpose of the generator is to create plausible images to fool the discriminator. \n\n \
After many cycles of training, the skill of the generator improves enough to produce some impressive results!     \n\n &nbsp \n\n \
**What is the difference between a GAN and a cGAN?** \n\n \
The basic idea behind cGANs is the same. The primary difference is way the model improves after each cycle, which is based on \
a *loss* calculation. For cGANs, this calculation optimizes the structure or joint configuration of the output. \n\n &nbsp \n\n \
**What are the possible applications of cGANs?** \n\n \
cGANs have been used in self-driving cars, creating maps from satellite images, colorizing black and white photos, and much more. \n\n &nbsp \n\n \
**Where can I read more about cGANs?** \n\n \
For more information on cGANs, check out [this paper.](https://arxiv.org/abs/1611.07004)   \n\n &nbsp \n\n \
**Who developed this web page?** \n\n \
This web page and the underlying models were developed by Niklas Bergen with the help of some additional resources. \
Check out the [GitHub repo](https://github.com/NB094/Easy-GANs) for more information.")




##### CODE FOR Pix2Pix #####



# Define page layout
left_column, right_column = st.columns([2,1])


# Create selection box and logic for various sketch subjects.
subject_selection = left_column.selectbox(label = 'Select what you wish to draw...', options = ['Human', 'Shoe', 'Handbag'], index = 0)
if subject_selection == 'Human':
    stroke_color = '#F44F36'
    background_color='#000000'
else:
    stroke_color = '#F44F36'
    background_color='#FFFFFF'


# Initialize a random number in the session state. Used to randomize examples shown.
if 'random_num' not in st.session_state:
        st.session_state.random_num = 1


# Change the random example number whenever the radio buttons are changed.
def random_num():
    st.session_state.random_num = np.random.randint(1,5+1)
    return


# Retrieve a randomly-selected example image
urllib.request.urlretrieve(f'https://github.com/NB094/Easy-GANs/raw/main/Pix2Pix/example_images_streamlit/example_{str.lower(subject_selection)}{st.session_state.random_num}.jpg?raw=true', \
                            'example_img.jpg')


# Create more options menus
canvas_mode = st.radio(label = 'Select canvas mode...', options = ('Draw on a blank canvas', 'View an example sketch', 'Try tracing an example sketch'), \
                       index = 1, help='Example sketches are chosen randomly out of 5 options.', on_change=random_num)
drawing_mode = right_column.selectbox(label = "Drawing tool:", options = ("freedraw", "line", "rect", "circle", "polygon", "transform"), index = 0)


# Create the drawing canvas                            
if canvas_mode == 'View an example sketch':    
    st.image('example_img.jpg')
else:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.0)",  # Fill colors from shape objects have full transparency
        stroke_width=1,
        stroke_color=stroke_color,
        background_color=background_color,
        background_image=Image.open('example_img.jpg') if canvas_mode == 'Try tracing an example sketch' else None,
        height=256,
        width=256,
        drawing_mode=drawing_mode,
        key="canvas")





##### SKETCH PROCESSING #####

if canvas_mode == 'View an example sketch':
    drawn_image = load_img('example_img.jpg')

else:    
    # Store canvas sketch data into a variable
    drawn_image = canvas_result.image_data
    
    # Insert try/except loop to prevent website from temporarily throwing error when unchecking the box.
    # try:

    # Convert sketch data into parseable numpy array
    drawn_image = np.array(Image.fromarray((drawn_image * 255).astype(np.uint8)).resize((256, 256)).convert('RGB'))
    drawn_image = (drawn_image * 255).astype(np.uint8)
    
    # If needed, convert black background to white before passing image to generator.
    if subject_selection != 'Human':
        drawn_image[drawn_image == 0] = 255
    
    # except:
        # pass


# Load and cache model files due to large file sizes
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def cache_all_models():
    st.text('Missed cache')
    urllib.request.urlretrieve(f'https://github.com/NB094/Easy-GANs/blob/main/Pix2Pix/saved_model/humans_fully_trained.h5?raw=true', 'humans_fully_trained.h5')
    urllib.request.urlretrieve(f'https://github.com/NB094/Easy-GANs/blob/main/Pix2Pix/saved_model/shoes_fully_trained.h5?raw=true', 'shoes_fully_trained.h5')
    urllib.request.urlretrieve(f'https://github.com/NB094/Easy-GANs/blob/main/Pix2Pix/saved_model/handbags_fully_trained.h5?raw=true', 'handbags_fully_trained.h5')

    humans_model = load_model('humans_fully_trained.h5', compile=False)
    shoes_model = load_model('shoes_fully_trained.h5', compile=False)
    handbags_model = load_model('handbags_fully_trained.h5', compile=False)
    return humans_model, shoes_model, handbags_model

humans_model, shoes_model, handbags_model = cache_all_models()

if subject_selection=='Human':
    model = humans_model
elif subject_selection=='Shoe':
    model = shoes_model
elif subject_selection=='Handbag':
    model = handbags_model    



# Insert try/except loop to prevent website from temporarily throwing error when unchecking the box.
# try:

# Pass numpy array into generator, and predict
gen = Generator(drawn_image, subject_selection)
gen_image = gen.generate_image(model)

# Display prediction
st.image(gen_image)

# except:
#     pass