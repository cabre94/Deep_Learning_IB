#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 21-11-2020
File: adversarial_example.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:

# Aca hay citados algunos papers
https://jvmohr.github.io/post/adversarial-machine-learning/

# Esto es una libreria para hacer ataques??
https://foolbox.readthedocs.io/en/latest/modules/attacks.html#foolbox.attacks.L2AdditiveGaussianNoiseAttack

https://www.kaggle.com/josephvm/generating-adversarial-examples-with-foolbox


# Aca hay ejemplos de como defenderse de adversarials examples
https://medium.com/analytics-vidhya/implementing-adversarial-attacks-and-defenses-in-keras-tensorflow-2-0-cab6120c5715

Este esta en chino
https://www.kaggle.com/kenichinakatani/create-adversarial-examples

https://foolbox.readthedocs.io/en/latest/modules/models.html#foolbox.models.TensorFlowModel

Aca estan los distintos preprocesados de las redes
https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L21
y aca un ejemplo de una persona que hizo la inversion
https://stackoverflow.com/questions/55987302/reversing-the-image-preprocessing-of-vgg-in-keras-to-return-original-image



# Adversarial examples con audio
https://nicholas.carlini.com/code/audio_adversarial_examples



-----------------------------------
Ataque adversario con tensorflow
https://www.tensorflow.org/tutorials/generative/adversarial_fgsm
https://www.tensorflow.org/tutorials/generative/deepdream
"""

# TODO chequear si es necesario pasar a array la imagen

import os
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.applications import inception_v3, inception_resnet_v2


model = inception_v3.InceptionV3()

model.trainable = False  # XXX No se si hace falta esto


def preProcess(img):

    img = image.img_to_array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float64)
    img = inception_v3.preprocess_input(img)
    img = img[None, ...]

    return img

def invert_preProcess(x):
    x += 1.0
    x *= 127.5
    return tf.cast(x, tf.uint8)

def get_labels_predictions(img):
    predict = model.predict(img)
    for val in inception_v3.decode_predictions(predict)[0]:
        print("{} ---> {}".format(val[1], val[2]))

def get_imagenet_label(img):
    predict = model.predict(img)
    return inception_v3.decode_predictions(predict, top=1)[0][0]


imagen = image.load_img("Cerveza.jpg", target_size=(299, 299))

imagen = preProcess(imagen)

get_labels_predictions(imagen)


plt.figure()
plt.imshow(invert_preProcess(imagen)[0]) # To change [-1, 1] to [0,1]
_, image_class, class_confidence = get_imagenet_label(imagen)
plt.title('{} : {:.2f}% Confidence'.format(image_class, class_confidence*100))
plt.show()


loss_object = tf.keras.losses.CategoricalCrossentropy()

def create_adversarial_pattern(input_image, input_label):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        prediction = model(input_image)
        loss = loss_object(input_label, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_image)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad

# Get the input label of the image.
image_probs = model.predict(imagen)

labrador_retriever_index = 441
label = tf.one_hot(labrador_retriever_index, image_probs.shape[-1])
label = tf.reshape(label, (1, image_probs.shape[-1]))

perturbations = create_adversarial_pattern(imagen, label)
plt.imshow(invert_preProcess(perturbations)[0]) # To change [-1, 1] to [0,1]
plt.show()


def display_images(img, description):
    _, label, confidence = get_imagenet_label(img)
    plt.figure()
    plt.imshow(invert_preProcess(img)[0])
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description,
                                                   label, confidence*100))
    plt.show()

epsilons = [0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]
descriptions = [('Epsilon = {:0.3f}'.format(eps) if eps else 'Input')
                for eps in epsilons]

for i, eps in enumerate(epsilons):
    adv_x = imagen + eps*perturbations
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    display_images(adv_x, descriptions[i])
# plt.show()