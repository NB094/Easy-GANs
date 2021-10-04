import streamlit as st
import urllib.request
from keras.models import load_model
from numpy import asarray
from numpy.random import randn
from matplotlib import pyplot

# Page intro
st.title('NumGen – Generate Completely Unique Handwritten Numbers!')

st.text('')
st.markdown('This tool is simpler than the other two; just click the button to have the model generate randomized "handwritten" numbers.')
st.text('')
st.text('')


# Links and FAQ section
st.sidebar.markdown("### [SRGANs Web Page](https://share.streamlit.io/nb094/easy-gans/main/SRGAN/Streamlit_SRGAN_Main.py)")
st.sidebar.markdown("### [Pix2Pix Web Page](https://share.streamlit.io/nb094/easy-gans/main/Pix2Pix/Streamlit_Pix2Pix_Main.py)")
st.sidebar.text('')

expander = st.sidebar.expander("NumGen Frequently-Asked Questions", expanded=True)
expander.write("**What type of machine learning is being used?** \n\n \
The model's architecture is based on a regular Generative Adversarial Network, or GAN. \n\n &nbsp \n\n \
**How do GANs work?** \n\n \
There are two main components to GAN models: a *discriminator* and a *generator*. \n\n \
The purpose of the discriminator is to classify images presented to it as real or fake. \
The purpose of the generator is to create plausible images to fool the discriminator. \n\n \
After many cycles of training, the skill of the generator improves enough to produce some impressive results!     \n\n &nbsp \n\n \
**What are the possible applications of GANs?** \n\n \
GANs similar to this model could be used for [CAPTCHAs](https://en.wikipedia.org/wiki/CAPTCHAhttps://en.wikipedia.org/wiki/CAPTCHA). \
Conversely, images of numbers and letters could be fed into a model and classified for \
[OCR](https://en.wikipedia.org/wiki/Optical_character_recognition) purposes.  \n\n &nbsp \n\n \
**Where can I learn more about cGANs?** \n\n \
For a quick overview of GANs, check out [this video.](https://www.youtube.com/watch?v=-Upj_VhjTBs)   \n\n &nbsp \n\n \
**Who developed this web page?** \n\n \
This web page and the underlying models were developed by Niklas Bergen with the help of some additional resources. \
Check out the [GitHub repo](https://github.com/NB094/Easy-GANs) for more information.")




##### CODE FOR NumGen #####



st.sidebar.text('')
st.sidebar.text('')


# This function will be called upon button press.
def generate_number():
    
    # Load pre-trained model
    urllib.request.urlretrieve('https://github.com/NB094/Easy-GANs/blob/main/NumGen/saved_model/generator_model_100.h5?raw=true', 'generatorX.h5')
    model = load_model('generatorX.h5', compile=False)

    # Create random latent space array.
    vector = asarray([[randn() for _ in range(100)]])

    X = model.predict(vector)
    X = X[0, :, :, 0]

    return X
    

clicked = st.button(label = 'Generate Number!')


# If the button was clicked, generate a number and plot it.
if clicked:
    X = generate_number()

    st.sidebar.text('')
    st.sidebar.text('')

    fig = pyplot.figure()
    pyplot.imshow(X, cmap='gray_r')
    pyplot.axis('off')
    pyplot.show()
    fig.savefig('test.png')
    
    st.image('test.png')