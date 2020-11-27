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

PATH_FOLDER = os.path.join(SAVE_FOLDER, "2")
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
imagen = preProcess(imagen)  # La preprocesamos
print_predictions(imagen)  # Printeamos las 5 probabilidades mas grandes

# Graficamos la imagen original con su prediccion
plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(imagen)[0])
_, clase, prob = get_imagenet_label(imagen)
plt.title("{} : {:.2f}%".format(clase, prob * 100))
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, "Golden_retriver.pdf"), format="pdf", fontsize=20)
plt.show()


# Tomamos la entrada y la salida del modelo
Input = model.layers[0].input
Output = model.layers[-1].output

pizza_index = 963  # Creo que es una referencia a una pizza

# De la salida, nos interesa la componente correspondiente a nuestra imagen
target_output = Output[:, pizza_index]

# Creamos un modelo auxiliar para tomar solo la salida que nos importa
feature_extractor = keras.Model(inputs=Input, outputs=target_output)


def compute_loss(img):
    score = feature_extractor(img)
    return tf.reduce_mean(score)


# Funcion que nos devuelve la funcion de costo CCE que vamos a buscar maximizar
# respecto a la clase correspondiente al indice idx
def compute_loss2(img, idx):
    loss_CCE = keras.losses.CategoricalCrossentropy()

    image_probs = model(img)  # Calculamos las probabilidades de c/ clase

    label = tf.one_hot(idx, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    # Calculamos el costo respecto a la clase
    loss = loss_CCE(label, image_probs)

    return loss


# Funcion que nos retorna el signo del gradiente que maximiza el error
# def gradient_step(adversarial, perturbation, epsilon, min_pert, max_pert):
def gradient_step(adversarial, min_pert, max_pert, lr):

    with tf.GradientTape() as tape:
        tape.watch(adversarial)
        loss = compute_loss(adversarial)

    # Calculamos el gradiente
    grad = tape.gradient(loss, adversarial) * lr

    # Sumamos el gradiente al adversarial example y a la perturbacion
    adversarial += grad

    # Limitamos la perturbacion en la imagen
    adversarial = tf.clip_by_value(adversarial, min_pert, max_pert)
    adversarial = tf.clip_by_value(adversarial, -1, 1)

    # Por ultimo, tomamos la probabilidad/confianza de la prediccion
    # de la red para la clase que le estamos indicando
    prob = loss.numpy()

    # return prob, adversarial, perturbation
    return prob, adversarial


# Funcion para inicializar una imagenes con ceros
def initialize_tensor_wZeros(img):
    zeros = tf.zeros_like(img, dtype=tf.float64)
    return zeros


def generate_adversarial_example(adversarial, epsilon=1, lr=10):

    # Determinamos los limites de la imagen perturbada
    max_pert = adversarial + epsilon
    min_pert = adversarial - epsilon

    # Tomamos la probabilidad inicial para la clase correspondiente
    prob = compute_loss(adversarial).numpy()
    steps = 0

    # Iteramos hasta obtener una probabilidad de 0.95
    while prob < 0.95:

        prob, adversarial = gradient_step(adversarial, min_pert, max_pert, lr)

        steps += 1
        print("Step {} - prob: {}".format(steps, prob))

    return adversarial


# ---------------
#   epsilon = 1
# ---------------
adver = generate_adversarial_example(imagen, epsilon=1, lr=10)
pert = imagen - adver

image.save_img(os.path.join(PATH_FOLDER, "adver_e=1.jpg"), adver[0])
image.save_img(os.path.join(PATH_FOLDER, "pert_e=1.jpg"), pert[0])

plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(pert)[0])
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=1.pdf"), format="pdf")
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=1.jpg"), format="jpg")
plt.show()

_, clase, probabilidad = get_imagenet_label(adver)
plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(adver)[0])
plt.title(
    "$\epsilon$=1\n{}: {:.2f}% de Probabilidad".format(clase, probabilidad * 100),
    fontsize=20,
)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=1.pdf"), format="pdf")
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=1.jpg"), format="jpg")
plt.show()

# ---------------
#   epsilon = 0.1
# ---------------
adver = generate_adversarial_example(imagen, epsilon=0.1, lr=10)
pert = imagen - adver

image.save_img(os.path.join(PATH_FOLDER, "adver_e=0.1.jpg"), adver[0])
image.save_img(os.path.join(PATH_FOLDER, "pert_e=0.1.jpg"), pert[0])

plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(pert)[0])
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.1.pdf"), format="pdf")
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.1.jpg"), format="jpg")
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
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.1.pdf"), format="pdf")
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.1.jpg"), format="jpg")
plt.show()

# ---------------
#   epsilon = 0.01
# ---------------
adver = generate_adversarial_example(imagen, epsilon=0.01, lr=5)
pert = imagen - adver

image.save_img(os.path.join(PATH_FOLDER, "adver_e=0.01.jpg"), adver[0])
image.save_img(os.path.join(PATH_FOLDER, "pert_e=0.01.jpg"), pert[0])

plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(pert)[0])
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.01.pdf"), format="pdf")
plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.01.jpg"), format="jpg")
plt.show()

_, clase, probabilidad = get_imagenet_label(adver)
plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(adver)[0])
plt.title(
    "$\epsilon$=0.01\n{}: {:.2f}% de Probabilidad".format(clase, probabilidad * 100),
    fontsize=20,
)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.01.pdf"), format="pdf")
plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.01.jpg"), format="jpg")
plt.show()

# # ---------------
# #   epsilon = 0.001
# # ---------------
# adver = generate_adversarial_example(imagen, epsilon=0.005, lr=1)
# pert = imagen - adver

# image.save_img(os.path.join(PATH_FOLDER, 'adver_e=0.005.jpg'),adver[0])
# image.save_img(os.path.join(PATH_FOLDER, 'pert_e=0.005.jpg'),pert[0])

# plt.figure(figsize=(5, 5))
# plt.imshow(invert_preProcess(pert)[0])
# plt.axis("off")
# plt.tight_layout()
# plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.005.pdf"), format="pdf")
# plt.savefig(os.path.join(PATH_FOLDER, "Perturbacion_e=0.005.jpg"), format="jpg")
# plt.show()

# _, clase, probabilidad = get_imagenet_label(adver)
# plt.figure(figsize=(5, 5))
# plt.imshow(invert_preProcess(adver)[0])
# plt.title(
#     "$\epsilon$=0.005\n{}: {:.2f}% de Probabilidad".format(clase, probabilidad * 100),
#     fontsize=20,
# )
# plt.axis("off")
# plt.tight_layout()
# plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.005.pdf"), format="pdf")
# plt.savefig(os.path.join(PATH_FOLDER, "Adv_example_e=0.005.jpg"), format="jpg")
# plt.show()

