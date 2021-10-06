# Easy GANs

## Table of Contents
1. [Project Overview](#Project Overview)
2. [How do GANs work?](#How do GANs work?)
3. [The Premise of Easy GANs](#The Premise of Easy GANs)
4. [NumGen](#NumGen)

## Project Overview
With the advent of neural networks, the potential of machine learning has greatly increased. The architecture of a neural network plays a central part in its functionality, and many variants have been created for all kinds of use-cases. This project, Easy GANs, focuses on architectures commonly referred to as Generative Adversarial Networks (GANs). GANs are generally used to create generators that perform some kind of image processing on a given input. Easy GANs is a showcase of three different types of GANs:
* **NumGen (regular GAN):** a traditional GAN architecture to produce handwritten numbers. A good starting place for those new to GANs.
* **Pix2Pix (Conditional GAN):** a modified GAN architecture to predict photos from sketches.
* **SRGAN (Super Resolution GAN):** a modified GAN architecture to enhance image quality.

## How do GANs work?
There are two main components to GANs: a *discriminator* and a *generator*. The purpose of the discriminator is to classify images presented to it as real or fake. The purpose of the generator is to create plausible images to fool the discriminator. After each round between the discriminator and generator, the weights of both components are updated based on their loss calculations. In other words, after many rounds (or matches) between the discriminator or generator, the skill of both components greatly improves. From there, we can save the generator model and use it for predictions outside of the dataset.

![image](https://user-images.githubusercontent.com/84533378/136279528-7763c4f3-252e-414b-9760-a34b1b722ab9.png)

[Source](https://www.researchgate.net/figure/The-architecture-of-vanilla-GANs_fig1_340458845)


For more information on GANs, you can check out [this video](https://www.youtube.com/watch?v=-Upj_VhjTBs) for a high-level overview and [this video](https://www.youtube.com/watch?v=9JpdAg6uMXs) for a more in-depth look into their architecture.

## The Premise of Easy GANs
I wanted to make Easy GANs to seamlessly demonstrate the usefulness and functionalities of GANs. Additionally, it has been designed for users unfamiliar with GANs as a way for them better understand how they work. I have [deployed all of the models onto Streamlit](https://share.streamlit.io/nb094/easy-gans/main/Pix2Pix/Streamlit_Pix2Pix_Main.py), an online platform for showing Python apps, so users can jump right into the action and test things out. Easy GANs was also developed with clarity in mind; I've added a significant amount of comments, docstrings, and instructions throughout the code so that everything is always well-defined.

No matter if you decide to run this code locally or try out the Streamlit web pages, Easy GANs aims to make the experience as straightforward as possible.

### NumGen
Numgen
### Pix2Pix
Pix2Pix
### SRGAN
SRGAN
