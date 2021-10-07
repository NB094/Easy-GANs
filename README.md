# Easy GANs
### A Showcase of Three Different Types of GANs.
‎

## Table of Contents
1. [Project Overview](#project-overview)
2. [How do GANs work?](#how-do-gans-work)
3. [The Premise of Easy GANs](#the-premise-of-easy-gans)
    1. [Project Setup Instructions](#project-setup-instructions)
    2. [NumGen](#numgen)
    3. [Pix2Pix](#pix2pix)
    4. [SRGAN](#srgan)
4. [Further Development](#further-development)
5. [References](#references)


## Project Overview
Easy GANs focuses on neural network architectures commonly referred to as Generative Adversarial Networks (GANs). With the advent of neural networks, the potential of machine learning has greatly increased. The architecture of a neural network plays a central part in its functionality, and many variants have been created for all kinds of use-cases. GANs are typically used to create generators that perform some kind of image processing on a given input. Easy GANs is a showcase of three different types of GANs:
* **NumGen (regular GAN):** a traditional GAN architecture to produce handwritten numbers.
* **Pix2Pix (Conditional GAN):** a modified GAN architecture to predict photos from sketches.
* **SRGAN (Super Resolution GAN):** a modified GAN architecture to enhance image quality.

<sub>[back to top](#easy-gans)</sub>
‎
## How do GANs work?
There are two main components to GANs: a *discriminator* and a *generator*. The purpose of the discriminator is to classify images presented to it as real or fake. The purpose of the generator is to create plausible images to fool the discriminator. After each round between the discriminator and generator, the weights of both components are updated based on their loss calculations. In other words, after many rounds (or matches) between the discriminator or generator, the skill of both components greatly improves. From there, we can save the generator model and use it for predictions outside of the dataset.

![image](https://user-images.githubusercontent.com/84533378/136279528-7763c4f3-252e-414b-9760-a34b1b722ab9.png)

<sup>[Source](https://www.researchgate.net/figure/The-architecture-of-vanilla-GANs_fig1_340458845)</sup>


For more information on GANs, you can check out [this video](https://www.youtube.com/watch?v=-Upj_VhjTBs) for a high-level overview and [this video](https://www.youtube.com/watch?v=9JpdAg6uMXs) / [this paper](https://arxiv.org/abs/1701.00160) for a more in-depth look into their architecture.

<sub>[back to top](#easy-gans)</sub>
‎
## The Premise of Easy GANs
The objective of Easy GANs is to seamlessly demonstrate the usefulness and functionalities of various GANs. Additionally, it has been designed to accomodate users unfamiliar with GANs; there is a significant amount of comments, docstrings, and instructions throughout the code so that everything is always well-defined. I have [deployed all of the models onto Streamlit](https://share.streamlit.io/nb094/easy-gans/main/Pix2Pix/Streamlit_Pix2Pix_Main.py), an online platform for showing Python apps, so users can jump right into the action and test things out. 

No matter if you decide to run this code locally or try out the Streamlit web pages, Easy GANs aims to make the experience as straightforward as possible.

‎
### Project Setup Instructions
To run one of these projects locally, please create a new virtual environment using the respective `requirements.txt` file provided in the parent folders. I recommend [Anaconda](https://www.anaconda.com/products/individual) for environment management. To read the `.ipynb` notebook files, open them in either Jupyter Lab or Jupyter Notebook. Each notebook contains detailed instructions on how to run the code, where to get the datasets, and more. Training was conducted locally on a Nvidia RTX 2080 Super GPU, and estimated training times are provided below. CPU training is not recommended for the Pix2Pix or SRGAN models.

To check out the projects online, click these links to access their Streamlit pages:

Numgen: [https://share.streamlit.io/nb094/easy-gans/main/NumGen/Streamlit_NumGen_Main.py](https://share.streamlit.io/nb094/easy-gans/main/NumGen/Streamlit_NumGen_Main.py)

Pix2Pix: [https://share.streamlit.io/nb094/easy-gans/main/Pix2Pix/Streamlit_Pix2Pix_Main.py](https://share.streamlit.io/nb094/easy-gans/main/Pix2Pix/Streamlit_Pix2Pix_Main.py)

SRGAN: [https://share.streamlit.io/nb094/easy-gans/main/SRGAN/Streamlit_SRGAN_Main.py](https://share.streamlit.io/nb094/easy-gans/main/SRGAN/Streamlit_SRGAN_Main.py)

<sub>[back to top](#easy-gans)</sub>
‎
### NumGen
Numgen, or Number generator, is a basic GAN that produces handwritten digits. It is trained by using the Modified National Institute of Standards and Technology (MNIST) Handwritten Digits dataset. With a trained model, the code is set up to generate a grid of handwritten digits, or single handwritten digits. Handwritten number generation is useful in systems such as [CAPTCHAs](https://en.wikipedia.org/wiki/CAPTCHAhttps://en.wikipedia.org/wiki/CAPTCHA) to verify if a user is human or not, in order to prevent spam.

Training time: 1 hour

![image](https://user-images.githubusercontent.com/84533378/136296369-edae4ea4-c173-48de-ba12-37171b6b57e3.png)

<sub>[back to top](#easy-gans)</sub>
‎
### Pix2Pix
Pix2Pix is an image-to-image translation tool which trains a model to predict a photo when provided a sketch as an input. In reality, the term "Pix2Pix" encompasses many different kinds of image-to-image translation applications (as described in [this paper](https://arxiv.org/abs/1611.07004)). Easy GANs integrates the Edges2Photo application, which was trained on datasets of human faces, handbags, and shoes.

Edges2Photo has a wide variety of use-cases such as drafting building architectures, visualizing suspects of a crime, and creating new clothing ideas. Pix2Pix functions off of a Condition GAN (or cGAN) architecture, which applies loss based on an image's structural similarity to the expected result.

Training time: 5 hours

![image](https://user-images.githubusercontent.com/84533378/136296539-dbc07aef-7f86-4f99-98d7-8fe9004a0ee5.png)

<sub>[back to top](#easy-gans)</sub>
‎
### SRGAN
SRGAN is an image enhancement tool for adding more detail to low-resolution images. Like the previous GANs, there are a lot of interesting use-cases for this application: enhancing surveillance camera footage, restoring old images and videos to current standards, or getting that extra bit of detail from medical imagery – just to name a few. The loss calculation for SRGANs is designed to optimize for the perceptual quality of upscaled images (more info in [this paper](https://arxiv.org/abs/1611.07004)).

This SRGAN notebook takes a high-resolution image as input, and returns grid of images (Low Res - Super Res - High Res). Only the low-resolution image is used for predicting the super-resolution image. Practically, the real use of this model would be to pass a low-resolution image as input and directly receive a super-resolution output; but as a proof-of-concept, this feature has not yet been implemented into the notebook.

Depending on the level of quality you are looking to achieve, training a SRGAN can be quite an intensive process. To limit the scope of the project, this model was only trained on a dataset of human faces (celebrities). Therefore, I've set up the notebook to support checkpoints so you can resume training at another time if it was suspended.

Training time: 15 hours

![image](https://user-images.githubusercontent.com/84533378/136296767-36f153db-055a-41d4-99c7-0458101e211d.png)

<sub>[back to top](#easy-gans)</sub>
‎
## Further Development
In the future, I hope to implement an [Enhanced SRGAN (ESRGAN)](https://arxiv.org/abs/1809.00219v1) architecture to improve the quality of the output images, and to eliminate the artifacts that sometimes appear in the SRGAN results – a known problem with the architecture.

Lastly, it would be interesting to train Pix2Pix on some more datasets such as animals.

Thank you for checking out this repo, and please let me know if you have any questions or concerns!

<sub>[back to top](#easy-gans)</sub>
‎
## References
[https://machinelearningmastery.com/](https://machinelearningmastery.com/)

[https://www.tensorflow.org/tutorials/generative/pix2pix](https://www.tensorflow.org/tutorials/generative/pix2pix)

[https://github.com/eriklindernoren/Keras-GAN](https://github.com/eriklindernoren/Keras-GAN)
