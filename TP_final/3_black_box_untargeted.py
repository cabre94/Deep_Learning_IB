#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 24-11-2020
File: 3_black_box.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
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

SAVE_FOLDER = "Resultados"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

PATH_FOLDER = os.path.join(SAVE_FOLDER, "3_untargeted")
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

model = inception_v3.InceptionV3()

model.trainable = False  # XXX No se si hace falta esto

# Funcion para preprocesar la imagen para la red Inception V3
def preProcess(img):

    img = image.img_to_array(img)
    img = tf.convert_to_tensor(img, dtype=tf.float64)
    img = inception_v3.preprocess_input(img)
    img = img[None, ...]

    return img


# Funcion para invertir el preprocesado para la red Inception V3
# Esto puede variar segun la red. Solo sirve si el preprocesado
# de la funcion preprocess_input pasa como argumento 'tf'.
# https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py#L21
def invert_preProcess(x):
    x += 1.0
    x *= 127.5
    return tf.cast(x, tf.uint8)


# Funcion para ver las 5 clases mas probables que predice la red
def print_predictions(img):
    predict = model.predict(img)
    for val in inception_v3.decode_predictions(predict)[0]:
        print("{:<20s}: {:.3f}".format(val[1], val[2]))


# Funcion que retorna la clase mas probable
def get_imagenet_label(img):
    predict = model.predict(img)
    return inception_v3.decode_predictions(predict, top=1)[0][0]


# Abrimos la imagen
imagen = image.load_img("Perro.jpg", target_size=(299, 299))
# La preprocesamos
imagen = preProcess(imagen)
# Printeamos las 5 probabilidades mas grandes
print_predictions(imagen)

# ----------------------------------
# ----------------------------------

golden_index = 207  # label en Imagenet correspondiente a un golden


# Funcion para inicializar una imagenes random
def initialize_randomTensor2(img, stddev):
    random_img = tf.random.normal(img.shape, stddev=stddev, dtype=tf.float64)
    return random_img


# Funcion para inicializar una imagenes con ceros
def initialize_tensor_wZeros(img):
    zeros = tf.zeros_like(img, dtype=tf.float64)
    return zeros


# Funcion para inicializar una imagenes random
def initialize_randomTensor(img, threshold):
    random_img = tf.random.uniform(img.shape, -1, 1)
    aux = random_img.numpy()
    aux[aux > threshold] = 1
    aux[aux < -threshold] = -1
    aux[np.bitwise_and(aux < threshold, aux > -threshold)] = 0
    random_img = tf.convert_to_tensor(aux, dtype=tf.float64)
    return random_img


# Funcion que modifica de manera aleatoria la imagen y acepta el cambio en caso de
# disminuir el score de la clase apuntada
def step(adversarial, min_pert, max_pert, lr):

    prob = model.predict(adversarial)[0, golden_index]

    random_pert = initialize_randomTensor(adversarial, 0.5)

    new_adv = adversarial + random_pert * lr
    new_adv = tf.clip_by_value(new_adv, min_pert, max_pert)
    new_adv = tf.clip_by_value(new_adv, -1, 1)

    new_prob = model.predict(new_adv)[0, golden_index]

    if new_prob < prob:
        return new_prob, new_adv
    else:
        new_adv = adversarial - random_pert * lr
        new_adv = tf.clip_by_value(new_adv, min_pert, max_pert)
        new_adv = tf.clip_by_value(new_adv, -1, 1)

        new_prob = model.predict(new_adv)[0, golden_index]

        if new_prob > prob:
            # En este caso no nos sirve y no hacemos nada. Retornamos la imagen original
            # con la esperanza que la siguiente inicializacion aleatoria de mejor.
            return prob, adversarial

        return new_prob, new_adv


def generate_adversarial_example(adversarial, epsilon=1, lr=1e-2):

    # Determinamos los limites de la imagen perturbada
    max_pert = adversarial + epsilon
    min_pert = adversarial - epsilon

    # Tomamos la probabilidad inicial para la clase correspondiente
    # prob = compute_loss(adversarial).numpy()
    prob = model.predict(adversarial)[0, golden_index]
    steps = 0

    # Iteramos hasta obtener una probabilidad de 0.01
    while prob > 0.01:

        prob, adversarial = step(adversarial, min_pert, max_pert, lr)

        steps += 1
        print("Step {} - prob: {}".format(steps, prob))

    return adversarial


adver = generate_adversarial_example(imagen, epsilon=0.1, lr=1e-2)
pert = imagen - adver

image.save_img(os.path.join(PATH_FOLDER, "adver_e=1.jpg"), adver[0])
image.save_img(os.path.join(PATH_FOLDER, "pert_e=1.jpg"), pert[0])

plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(pert)[0])
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.1.jpg"), format="jpg")
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.1.pdf"), format="pdf")
plt.show()

_, clase, probabilidad = get_imagenet_label(adver)
plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(adver)[0])
plt.title(
    "$\epsilon$=0.1\n{}: {:.2f}% de Probabilidad".format(clase, probabilidad * 100),
    fontsize=20,
)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.1.jpg"), format="jpg")
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.1.pdf"), format="pdf")
plt.show()
