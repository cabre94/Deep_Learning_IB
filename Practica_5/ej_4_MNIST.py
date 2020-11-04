#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 01-11-2020
File: ej_4.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description: https://keras.io/examples/vision/visualizing_what_convnets_learn/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras

# Defino constantes
epochs = 200
learning_rate = 1.0

# Dimensiones de la imagen de entrada (segun VGG16)
img_width = 28
img_height = 28

# Funcion loss que vamos a querer maximizar. En este caso es la media de la
# activacion para un filtro de la layer que tenemo. No se bien porque pero
# no consideramos los bordes
def compute_loss(input_image, filter_index,feature_extractor):
    activation = feature_extractor(input_image)
    #filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    filter_activation = activation[:, :, :, filter_index]
    return tf.reduce_mean(filter_activation)

# Funcion que realiza un paso del gradient ascent
@tf.function
def gradient_ascent_step(img, filter_index, lr,feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index,feature_extractor)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += lr * grads
    return loss, img

# Funcion para desnormalizar
def deprocess_image(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15
    # Center crop
    #img = img[25:-25, 25:-25, :]#
    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img

# funcion para inicializar una imagen en gris
def initialize_image():
    img = tf.random.uniform((1, img_width, img_height, 1))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25

# 
def visualize_filter(filter_index,feature_extractor):
    img = initialize_image()
    for iteration in range(epochs):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate,feature_extractor)
    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img

mnist_model_path = os.path.join(os.getcwd(), "Datos", "Conv_MNIST", "trained_model.h5")
model = keras.models.load_model(mnist_model_path)
#model = keras.applications.VGG16(False)
#model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)

# Carpeta donde guardamos las imagenes
img_folder = os.path.join('Figuras', '4_MNIST')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Constantes para que los graficos queden lindos
margin = 1
n = 4   # Numero de filas
m = 6   # Numero de columnas
cropped_width = 28
cropped_height = 28
width = n * cropped_width + (n - 1) * margin
height = m * cropped_height + (m - 1) * margin


for layer in model.layers:
    if 'conv' in layer.name: # Solo buscamos las convolucionales
        layer_name = layer.name
        layer = model.get_layer(name=layer_name)

        feature_extractor = keras.Model(inputs=model.inputs, outputs=layer.output)

        random_idx = np.random.choice(range(layer.filters),n*m,replace=False)

        all_imgs = []
        for filter_index in random_idx:
            print("Layer {} - Filtro {}".format(layer_name,filter_index))
            loss, img = visualize_filter(filter_index, feature_extractor)
            all_imgs.append(img)

        stitched_filters = np.zeros((width, height, 1))

        for i in range(n):
            for j in range(m):
                img = all_imgs[i * m + j]
                stitched_filters[
                    (cropped_width + margin) * i : (cropped_width + margin) * i + cropped_width,
                    (cropped_height + margin) * j : (cropped_height + margin) * j
                    + cropped_height,
                    :,
                ] = img

        save_path = os.path.join(img_folder, "{}.pdf".format(layer_name))
        keras.preprocessing.image.save_img(save_path, stitched_filters)

feature_extractor_mnist = keras.Model(inputs=model.inputs, outputs=model.outputs)

fig = plt.figure(figsize=(19,3))
for i in range(10):
    loss, img = visualize_filter(i, feature_extractor_mnist)
    ax = plt.subplot(1, 10, i+1)
    plt.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title(str(i), fontsize=14, wrap=True)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.suptitle('Clases interpretadas por la red neuronal para MNIST.', fontsize=16)
save_path = os.path.join(img_folder, "Entrada.pdf")
plt.savefig(save_path, format='pdf')
plt.show()
