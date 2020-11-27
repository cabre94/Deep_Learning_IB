#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 26-11-2020
File: 5_defense.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
import random
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten

SAVE_FOLDER = "Resultados"
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

PATH_FOLDER = os.path.join(SAVE_FOLDER, "5_Defense")
if not os.path.exists(PATH_FOLDER):
    os.makedirs(PATH_FOLDER)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols, channels = 28, 28, 1
num_classes = 10

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape((-1, img_rows, img_cols, channels))
x_test = x_test.reshape((-1, img_rows, img_cols, channels))

y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

model = keras.models.Sequential(name="MNIST_Conv")
model.add(
    Conv2D(
        32,
        3,
        3,
        padding="same",
        activation="relu",
        input_shape=(img_rows, img_cols, channels),
    )
)
model.add(Conv2D(64, 3, 3, padding="same", activation="relu"))
model.add(Conv2D(64, 3, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(32))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation="softmax"))

model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_test, y_test))

print(
    "Precision final para los datos de test:",
    model.evaluate(x=x_test, y=y_test, verbose=0)[1],
)


def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad


image = x_train[0]
image_label = y_train[0]

perturbations = adversarial_pattern(
    image.reshape((1, img_rows, img_cols, channels)), image_label
).numpy()

adversarial = image + perturbations * 0.1


Correcto = model.predict(image.reshape((1, img_rows, img_cols, channels))).argmax()
Prediccion = model.predict(adversarial).argmax()

plt.imshow(adversarial.reshape((img_rows, img_cols)))
plt.title(
    "Predict: {}    Correcto: {}".format(Prediccion, Correcto), fontsize=20,
)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Ejemplo.jpg"), format="jpg")
plt.show()


def generate_adversarials_train(batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            N = random.randint(0, x_train.shape[0] - 1)
            # N = random.randint(0, 100)

            label = y_train[N]
            image = x_train[N]

            perturbations = adversarial_pattern(
                image.reshape((1, img_rows, img_cols, channels)), label
            ).numpy()

            epsilon = 0.1
            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_train[N])

        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)

        yield x, y


def generate_adversarials_test(batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            N = random.randint(0, x_test.shape[0] - 1)

            label = y_test[N]
            image = x_test[N]

            perturbations = adversarial_pattern(
                image.reshape((1, img_rows, img_cols, channels)), label
            ).numpy()

            epsilon = 0.1
            adversarial = image + perturbations * epsilon

            x.append(adversarial)
            y.append(y_test[N])

        x = np.asarray(x).reshape((batch_size, img_rows, img_cols, channels))
        y = np.asarray(y)

        yield x, y


x_adversarial_test, y_adversarial_test = next(generate_adversarials_test(10000))
print(
    "Precision para imagenes adversarias de test:",
    model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1],
)

# Nos quedamos con un pequeño conjunto que vamos a usar para graficar
aux = x_adversarial_test[:12]
aux_label = y_adversarial_test[:12]

Prediccion = model.predict(aux)

fig, axs = plt.subplots(
    nrows=3, ncols=4, figsize=(12, 8), subplot_kw={"xticks": [], "yticks": []}
)

for ax, predict, label, imagen in zip(axs.flat, Prediccion, aux_label, aux):
    predict = predict.argmax()
    label = label.argmax()
    ax.imshow(imagen.reshape((img_rows, img_cols)))
    ax.set_title("Predict: {}  Correcto: {}".format(predict, label))

plt.savefig(os.path.join(PATH_FOLDER, "12_Test.jpg"), format="jpg")
plt.tight_layout()
plt.show()

# Nos generamos un conjunto de training de imagenes adversarias
x_adversarial_train, y_adversarial_train = next(generate_adversarials_train(20000))

hist = model.fit(
    x_adversarial_train,
    y_adversarial_train,
    batch_size=32,
    epochs=20,
    validation_data=(x_test, y_test),
)

print(
    "Precision para imagenes adversarias de test:",
    model.evaluate(x=x_adversarial_test, y=y_adversarial_test, verbose=0)[1],
)
print(
    "Precision para imagenes originales de test:",
    model.evaluate(x=x_test, y=y_test, verbose=0)[1],
)

# Veamos si cambio la performance del conjunto de test
plt.plot(hist.history["accuracy"], label="Adv. Training")
plt.plot(hist.history["val_accuracy"], label="Acc. Validation")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Accuracy", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FOLDER, "Acc.jpg"), format="jpg", bbox_inches="tight")
plt.show()

# Volvemos a predecir las imagenes adversarias que no habiamos podido
# clasificar correctamente
Prediccion = model.predict(aux)

fig, axs = plt.subplots(
    nrows=3, ncols=4, figsize=(12, 8), subplot_kw={"xticks": [], "yticks": []}
)

for ax, predict, label, imagen in zip(axs.flat, Prediccion, aux_label, aux):
    predict = predict.argmax()
    label = label.argmax()
    ax.imshow(imagen.reshape((img_rows, img_cols)))
    ax.set_title("Predict: {}  Correcto: {}".format(predict, label))

plt.savefig(os.path.join(PATH_FOLDER, "12_Test_AfterTraining.jpg"), format="jpg")
plt.tight_layout()
plt.show()
