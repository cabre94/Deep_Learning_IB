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

from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import vgg16, vgg19, inception_resnet_v2


# Cargo una red, creo que voy a arrancar con la VGG16

# model = keras.applications.VGG16()
model = inception_resnet_v2.InceptionResNetV2()

# imagen = image.load_img("Cerveza.jpg")
img = image.load_img("Cerveza.jpg", target_size=(299,299))
# imagen_3 = image.load_img("Cerveza.jpg", target_size=(1024,1024))

arr_img = image.img_to_array(img)
# imagen_2_arr = keras.preprocessing.image.img_to_array(imagen_2)
# imagen_3_arr = keras.preprocessing.image.img_to_array(imagen_3)

# imagen_arr = imagen_arr.reshape( (-1,imagen_arr.shape[0], imagen_arr.shape[1], imagen_arr.shape[2]) )
# imagen_2_arr = imagen_2_arr.reshape( (-1,imagen_2_arr.shape[0], imagen_2_arr.shape[1], imagen_2_arr.shape[2]) )
# imagen_3_arr = imagen_3_arr.reshape( (-1,imagen_3_arr.shape[0], imagen_3_arr.shape[1], imagen_3_arr.shape[2]) )

arr_img = np.expand_dims(arr_img, axis=0)

# Preprocesamos la imagen
arr_img = inception_resnet_v2.preprocess_input(arr_img)


# Hacemos una prediccion
y = model.predict(arr_img)

# Con esto sacamos los 5 (pero se pueden pedir mas), valores mas probables
inception_resnet_v2.decode_predictions(y)
# decode_predictions(y)

"""Aca empieza lo importante"""
# Tomamos la entrada y la salida del modelo
Input = model.layers[0].input
Output = model.layers[-1].output

