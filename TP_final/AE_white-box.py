#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 18-11-2020
File: AE_white.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:

https://arxiv.org/abs/1602.02697

https://bair.berkeley.edu/blog/2017/12/30/yolo-attack/

https://arxiv.org/abs/1712.09665

https://openai.com/blog/adversarial-example-research/

Valores de las clases de ImageNEt
https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
"""

import os
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19, inception_resnet_v2
from tensorflow.keras.applications import inception_v3


# Cargo una red, creo que voy a arrancar con la VGG16

# model = keras.applications.VGG16()
# model = inception_resnet_v2.InceptionResNetV2()
model = inception_v3.InceptionV3()

imagen = image.load_img("Cerveza.jpg", target_size=(299,299))

arr_img = image.img_to_array(imagen)   # imagen como array


# arr_img = np.expand_dims(arr_img, axis=0)
arr_img = arr_img[None,...]

# Preprocesamos la imagen
arr_img = inception_v3.preprocess_input(arr_img)

tensor_img = tf.convert_to_tensor(arr_img, dtype=tf.float32)

def invert_preProcess(x):
    x += 1.0
    x *= 127.5
    return tf.cast(x, tf.uint8)




# Hacemos una prediccion
y = model.predict(arr_img)

# Con esto sacamos los 5 (pero se pueden pedir mas), valores mas probables
inception_v3.decode_predictions(y)
# decode_predictions(y)

"""Aca empieza lo importante"""
# Tomamos la entrada y la salida del modelo
Input = model.layers[0].input
Output = model.layers[-1].output

target = 7  # Creo que es una referencia a un cock en V3

# De la salida, nos interesa la componente correspondiente a nuestra imagen
# loss = Output[0, target]
# target_output = Output[0, target]
target_output = Output[:, target]   #XXX chequear si cambia en algo esto

# Creamos el gradiente pero no entre el error y los parametros, si no sobre la variable de entrada
# gradient = K.gradients(loss, Input)[0]

kk = tf.image.convert_image_dtype(arr_img, dtype=tf.float32)

# with tf.GradientTape() as tape:
#     tape.watch(Input)

# # gradient = tape.gradient(loss, Input)
# # gradient = tape.gradient(loss, arr_img)
# # gradient = tape.gradient(loss, img)
# gradient = tape.gradient(loss, kk)

#########################################################
feature_extractor = keras.Model(inputs=Input, outputs=target_output)

def compute_loss(img):
    score = feature_extractor(img)
    return tf.reduce_mean(score)


@tf.function
def gradient_ascent_step(adversarial, perturbation, epsilon):
    
    with tf.GradientTape() as tape:
        tape.watch(adversarial)
        loss = compute_loss(adversarial)
    
    # Calculo el gradiente
    grad = tape.gradient(loss, adversarial)

    # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

    # Limitamos cuanto puede variar la imagen
    # grad = tf.clip_by_value(grad, clip_value_min=-1*epsilon, clip_value_max=epsilon)

    adversarial += grad
    perturbation += grad

    return loss, adversarial, perturbation

def initialize_img_wZeros(img):
    zeros = tf.zeros_like(img, dtype=tf.float64)
    return zeros

def generate_adversarial_example(adversarial):

    perturbation = initialize_img_wZeros(adversarial)

    epsilon = 1
    # epsilon = 0.01

    loss = compute_loss(adversarial).numpy()

    steps = 0

    while loss < 0.95:

        # import ipdb; ipdb.set_trace(context=15)  # XXX BREAKPOINT

        loss, adversarial, perturbation = gradient_ascent_step(adversarial, perturbation, epsilon)

        loss = loss.numpy()
        steps += 1

        # print("Step {} - loss: {:.3f}".format(steps, loss))
        print("Step {} - loss: {}".format(steps, loss))
    
    return adversarial, perturbation


adver, pert = generate_adversarial_example(tensor_img)
