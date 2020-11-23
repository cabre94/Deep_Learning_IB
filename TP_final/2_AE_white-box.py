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

# Mirar esto
https://www.tensorflow.org/api_docs/python/tf/GradientTape

# Aca esta indicado que es cada label de Imagenet
https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a
"""

import os
import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19
from tensorflow.keras.applications import inception_v3, inception_resnet_v2


# Cargo una red, creo que voy a arrancar con la VGG16

# model = keras.applications.VGG16()
# model = inception_resnet_v2.InceptionResNetV2()
model = inception_v3.InceptionV3()

imagen = image.load_img("Cerveza.jpg", target_size=(299, 299))

arr_img = image.img_to_array(imagen)  # imagen como array

# arr_img = np.expand_dims(arr_img, axis=0)
arr_img = arr_img[None, ...]

# Preprocesamos la imagen
arr_img = inception_v3.preprocess_input(arr_img)

tensor_img = tf.convert_to_tensor(arr_img, dtype=tf.float64)


def invert_preProcess(x):
    x += 1.0
    x *= 127.5
    return tf.cast(x, tf.uint8)


# Hacemos una prediccion
y = model.predict(arr_img)

# Con esto sacamos los 5 (pero se pueden pedir mas), valores mas probables
print(inception_v3.decode_predictions(y))
# decode_predictions(y)

"""Aca empieza lo importante"""
# Tomamos la entrada y la salida del modelo
Input = model.layers[0].input
Output = model.layers[-1].output

target = 7  # Creo que es una referencia a un cock en V3

# De la salida, nos interesa la componente correspondiente a nuestra imagen
# target_output = Output[0, target]
target_output = Output[:, target]  # XXX chequear si cambia en algo esto

# Creamos un modelo auxiliar para tomar solo la salida que nos importa
feature_extractor = keras.Model(inputs=Input, outputs=target_output)


def compute_loss(img):
    score = feature_extractor(img)
    return tf.reduce_mean(score)
    # return score


@tf.function
def gradient_ascent_step(adversarial, perturbation, epsilon, min_pert, max_pert):

    with tf.GradientTape() as tape:
        tape.watch(adversarial)
        loss = compute_loss(adversarial)

    # Calculo el gradiente
    grad = tape.gradient(loss, adversarial)

    # Limitamos cuanto puede variar la imagen
    # grad = tf.clip_by_value(grad, clip_value_min=-1*epsilon, clip_value_max=epsilon)

    adversarial += grad
    perturbation += grad

    # Limitamos la perturbacion en la imagen
    adversarial = tf.clip_by_value(adversarial, min_pert, max_pert)
    perturbation = tf.clip_by_value(perturbation, -epsilon, epsilon)

    return loss, adversarial, perturbation


def initialize_tensor_wZeros(img):
    zeros = tf.zeros_like(img, dtype=tf.float64)
    return zeros


def generate_adversarial_example(adversarial, epsilon=1):

    perturbation = initialize_tensor_wZeros(adversarial)

    perturbation = inception_v3.preprocess_input(perturbation) #XXX Ojo si cambio la red

    max_pert = adversarial + epsilon
    min_pert = adversarial - epsilon

    loss = compute_loss(adversarial).numpy()

    steps = 0

    while loss < 0.95:

        loss, adversarial, perturbation = gradient_ascent_step(
            adversarial, perturbation, epsilon, min_pert, max_pert 
        )

        loss = loss.numpy()
        steps += 1

        # print("Step {} - loss: {:.3f}".format(steps, loss))
        print("Step {} - loss: {}".format(steps, loss))

    return adversarial, perturbation


adver, pert = generate_adversarial_example(tensor_img)

inverted_img = invert_preProcess(tensor_img)
inverted_adver = invert_preProcess(adver)
inverted_pert = invert_preProcess(pert)

fig, axs = plt.subplots(1, 3, figsize=(15, 3), sharey=True)
axs[0].imshow(inverted_img[0])
axs[1].imshow(inverted_pert[0])
axs[2].imshow(inverted_adver[0])
fig.suptitle("Ataque adversario")
plt.savefig("Final.pdf", format='pdf')
plt.show()

prediccion = model.predict(adver)
for val in inception_v3.decode_predictions(prediccion)[0]:
    print(val)
print("-----------------------")
prediccion = model.predict(tensor_img)
for val in inception_v3.decode_predictions(prediccion)[0]:
    print(val)

# Probamos que predicen otras redes con esta imagen modificada
# Inception Resner v2
model2 = inception_resnet_v2.InceptionResNetV2()

prediccion = model2.predict(adver)
for val in inception_resnet_v2.decode_predictions(prediccion)[0]:
    print(val)

# VGG19
model3 = vgg19.VGG19()

ej = tf.image.resize(inverted_adver, [224, 224])
ej = vgg19.preprocess_input(ej)
prediccion = model3.predict(ej)
for val in vgg19.decode_predictions(prediccion)[0]:
    print(val)