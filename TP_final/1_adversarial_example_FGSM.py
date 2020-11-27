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

FGSM_FOLDER = os.path.join(SAVE_FOLDER, "1-FGSM")
if not os.path.exists(FGSM_FOLDER):
    os.makedirs(FGSM_FOLDER)

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

# Graficamos la imagen original con su prediccion
plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(imagen)[0])
_, clase, prob = get_imagenet_label(imagen)
plt.title("{} : {:.2f}%".format(clase, prob * 100))
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_FOLDER, "Golden_retriver.jpg"), format="jpg", fontsize=22)
plt.show()

# Funcion que nos devuelve la funcion de costo CCE que vamos a buscar maximizar
# respecto a la clase correspondiente al indice idx
def compute_loss(img, idx):
    loss_CCE = keras.losses.CategoricalCrossentropy()

    image_probs = model(img)  # Calculamos las probabilidades de c/ clase

    label = tf.one_hot(idx, image_probs.shape[-1])
    label = tf.reshape(label, (1, image_probs.shape[-1]))

    # Calculamos el costo respecto a la clase
    loss = loss_CCE(label, image_probs)

    return loss


# Funcion que nos retorna el signo del gradiente que maximiza el error
def grad_FGSM(input_image, idx):

    with tf.GradientTape() as tape:
        tape.watch(input_image)
        loss = compute_loss(input_image, idx)

    # Calculamos el gradiente
    gradient = tape.gradient(loss, input_image)
    # Tomamos solo el signo del gradiente
    signed_grad = tf.sign(gradient)
    return signed_grad


# Tomamos el label correcto de la imagen
golden_retriever_index = 207

# Obtenemos la perturbacion para reducir la confianza de la prediccion
# de la red para la clase que tomamos
perturbations = grad_FGSM(imagen, golden_retriever_index)

# Lo graficamos
plt.figure(figsize=(5, 5))
plt.imshow(invert_preProcess(perturbations)[0])  # To change [-1, 1] to [0,1]
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FGSM_FOLDER, "Perturbacion.jpg"), format="jpg")
plt.show()

# Funcion para graficar las imagenes perturbadas, con su nueva prediccion y probabilidad
def display_images(img, epsilon):
    # Obtenemos la clase predicha y su probabilidad
    _, clase, probabilidad = get_imagenet_label(img)

    plt.figure(figsize=(5, 5))
    plt.imshow(invert_preProcess(img)[0])
    plt.title(
        "$\epsilon$={} \n {}: {:.2f}% de Probabilidad".format(
            epsilon, clase, probabilidad * 100
        ),
        fontsize=22,
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(
        os.path.join(FGSM_FOLDER, "epsilon={}.jpg".format(epsilon)), format="jpg"
    )
    plt.show()


epsilons = [0, 0.01, 0.1, 0.15, 0.2, 0.25, 0.3]

for i, eps in enumerate(epsilons):
    # Agregamos la perturbacion a la imagen original
    adv_x = imagen + eps * perturbations
    # Limitamos los valores al rango adecuado [-1,1]
    adv_x = tf.clip_by_value(adv_x, -1, 1)
    # Graficamos
    display_images(adv_x, epsilons[i])

# Veamos que pasa si usamos estas imagenes modificadas en otras redes
# a ver si tambien logramos engañarlas
adver = imagen + 0.1 * perturbations
adver = tf.clip_by_value(adver, -1, 1)

# InceptionResnetV2
model2 = inception_resnet_v2.InceptionResNetV2()
y = model2.predict(adver)
inception_resnet_v2.decode_predictions(y)

# VGG19
model = vgg19.VGG19()
y = model.predict(
    tf.image.resize(vgg19.preprocess_input(invert_preProcess(adver)), [224, 224])
)
vgg19.decode_predictions(y)  # 'vizsla', 0.1885828

# VGG16
model = vgg16.VGG16()
y = model.predict(
    tf.image.resize(vgg16.preprocess_input(invert_preProcess(adver)), [224, 224])
)
vgg16.decode_predictions(y)  # Chesapeake_Bay_retriever', 0.14962395

