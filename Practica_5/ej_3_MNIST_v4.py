#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 31-10-2020
File: ej_3_MNIST.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Script propio para pasar argumentos por linea de comandos
from utils import lr, rf, epochs, batch_size, description

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

# Importo los datos
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Los junto porque creo que no estan bien distribuidos
x_train, y_train = np.vstack((x_train, x_test)), np.hstack((y_train, y_test))
# Separo los datos de test
x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                    y_train,
                                                    test_size=15000,
                                                    stratify=y_train)
# Ahora separo entre training y validacion
x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=15000,
                                                  stratify=y_train)

# Normalizacion
x_train = x_train.reshape((-1, 28, 28, 1)) / 255
x_test = x_test.reshape((-1, 28, 28, 1)) / 255
x_val = x_val.reshape((-1, 28, 28, 1)) / 255

# Paso los labels a one-hot representation
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
y_val = keras.utils.to_categorical(y_val, 10)

# path del modelo entrenado para fashion-mnist
trained_model = os.path.join(os.getcwd(), "Fashion-MNIST", "trained_model.h5")
# Cargo el modelo
fashion_model = keras.models.load_model(trained_model)
# No dejo que se entrene
fashion_model.trainable = False

mnist_model = keras.models.Sequential(name='Mnist')

for layer in fashion_model.layers[:-6]:
    mnist_model.add(layer)

# Pruebo a ver que pasa si solo cambio la ultima capa densa (la de clasificacion)
mnist_model.add(layers.Dropout(0.2,name='n1'))
mnist_model.add(layers.BatchNormalization(name='n2'))
mnist_model.add(layers.Dense(64, 'linear', kernel_regularizer=l2(rf), name='n3'))
mnist_model.add(layers.Dropout(0.2,name='n4'))
mnist_model.add(layers.BatchNormalization(name='n5'))
mnist_model.add(layers.Dense(10, 'linear', kernel_regularizer=l2(rf), name='n6'))

mnist_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
              loss=losses.CategoricalCrossentropy(from_logits=True),
              metrics=[metrics.CategoricalAccuracy(name='acc')])

# Callbacks
earlystop = keras.callbacks.EarlyStopping(patience=10)
lrr = keras.callbacks.ReduceLROnPlateau('val_acc',0.5,5,1,min_lr=1e-5)
#callbacks = [earlystop, lrr]
callbacks = [lrr]

hist = mnist_model.fit(x_train,
                 y_train,
                 epochs=epochs,
                 validation_data=(x_val, y_val),
                 batch_size=batch_size,
                 callbacks = callbacks,
                 verbose=1)

# Calculo la loss y Accuracy para los datos de test
test_loss, test_acc = mnist_model.evaluate(x_test, y_test)
hist.history['test_loss'] = test_loss
hist.history['test_acc'] = test_acc

# Guardo los datos
data_folder = os.path.join('Datos', 'MNIST')
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
mnist_model.save(os.path.join(data_folder, '{}.h5'.format(description)))
np.save(os.path.join(data_folder, '{}.npy'.format(description)), hist.history)

# Guardo las imagenes
img_folder = os.path.join('Figuras', 'MNIST')
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.plot(hist.history['loss'], label="Loss Training")
plt.plot(hist.history['val_loss'], label="Loss Validation")
plt.title("Acc Test: {:.3f}".format(test_acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Loss_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()

plt.plot(hist.history['acc'], label="Acc. Training")
plt.plot(hist.history['val_acc'], label="Acc. Validation")
plt.title("Acc Test: {:.3f}".format(test_acc))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig(os.path.join(img_folder, 'Acc_{}.png'.format(description)),
            format="png",
            bbox_inches="tight")
plt.close()
