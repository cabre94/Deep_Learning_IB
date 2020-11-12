#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 10-11-2020
File: Cabrera.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers

import seaborn as snn

snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

# -------------------------------------
# 1 - Arbo los datos
# -------------------------------------
data = pd.read_csv("airline-passengers.csv", header=0)

data = data["Passengers"].values
data = data.reshape((data.shape[0], 1))

# -------------------------------------
# 2 - Normalizacion
# -------------------------------------
normalization = MinMaxScaler(feature_range=(0, 1))
data_normalized = normalization.fit_transform(data)

# -------------------------------------
# 3 - Ruido Gaussiano
# -------------------------------------

# Voy a agregarle el ruido antes de formateear los datos, porque para mi tiene mas
# sentido que un valor x[i] tenga el mismo ruido en cualquier set
noise = np.random.normal(loc=0, scale=0.02, size=data_normalized.shape)
data_noise = data_normalized + noise

# -------------------------------------
# 2 - Formateo de los datos
# -------------------------------------
# Funcion para formatear los datos segun el enunciado
def formatData(dataset, l=1):
    X, Y = [], []
    for i in range(len(dataset) - l):
        a = dataset[i : (i + l)]  ###i=0, 0,1,2,3
        X.append(np.copy(a))
        Y.append(np.copy(dataset[i + l]))
    return np.array(X), np.array(Y)


l = 1
X, Y = formatData(data_noise, l)

# -------------------------------------
# 4 - Spliteo de los datos
# -------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.3)

# -------------------------------------
# 5 - Reshape de los datos para LSTM
# -------------------------------------

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)

# -------------------------------------
# 6 - Modelo con LSTM
# -------------------------------------


def LSTM_model():
    model = keras.models.Sequential(name="LSTM")
    model.add(layers.Input(shape=X_train.shape[1:]))
    model.add(layers.LSTM(4, return_sequences=False, name="LSTM"))
    model.add(layers.Dense(1, name="Dense"))
    model.summary()
    return model


model = LSTM_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4), loss=losses.MeanSquaredError(),
)

# -------------------------------------
# 7 - Entrenamiento de la red
# -------------------------------------

hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=100,
    batch_size=1,
    verbose=2,
)

# Calculo la loss y Accuracy para los datos de test
test_loss = model.evaluate(X_test, Y_test)
hist.history["test_loss"] = test_loss

# Guardo las imagenes
img_folder = "Figuras"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.figure()
plt.plot(hist.history["loss"], label="Loss Training")
plt.plot(hist.history["val_loss"], label="Loss Test")
# plt.title("Loss Test: {:.3f}".format(test_loss))
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, "Loss.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# -------------------------------------
# 8 - Calculamos las predicciones
# -------------------------------------

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invertimos la normalizacion
train_predict = normalization.inverse_transform(train_predict)
test_predict = normalization.inverse_transform(test_predict)
Y_train = normalization.inverse_transform(Y_train)
Y_test = normalization.inverse_transform(Y_test)

# Calculo el error cuadratico medio
train_mse = mean_squared_error(Y_train, train_predict)
test_mse = mean_squared_error(Y_test, test_predict)
print("MSE final de train: {:.2f}".format(train_mse))
print("MSE final de test:  {:.2f}".format(test_mse))

# Ploteo los resultados
month = np.arange(len(data))

plt.figure()
plt.plot(normalization.inverse_transform(data_noise), label="Datos+Ruido")
plt.plot(np.arange(len(train_predict)) + l, train_predict, label="Training")
plt.plot(
    np.arange(len(test_predict)) + len(train_predict) + l, test_predict, label="Test"
)
plt.xlabel("Meses", fontsize=15)
plt.ylabel("N° de pasajeros", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(img_folder, "Predict.pdf"), format="pdf", bbox_inches="tight")
plt.close()

# -------------------------------------
# 9 - Analizamos el efecto de l
# -------------------------------------

l_log = []
mse_test = []
mse_train = []

for l in range(1, 37):

    X, Y = formatData(data_noise, l)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, shuffle=False, test_size=0.3
    )

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)

    model = LSTM_model()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4), loss=losses.MeanSquaredError(),
    )

    hist = model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=100,
        batch_size=1,
        verbose=2,
    )

    # Grafico
    plt.figure()
    plt.plot(hist.history["loss"], label="Loss Training")
    plt.plot(hist.history["val_loss"], label="Loss Test")
    # plt.title("Loss Test: {:.3f}".format(test_loss))
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(img_folder, "Loss_l={}.pdf".format(l)),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invertimos la normalizacion
    train_predict = normalization.inverse_transform(train_predict)
    test_predict = normalization.inverse_transform(test_predict)
    Y_train = normalization.inverse_transform(Y_train)
    Y_test = normalization.inverse_transform(Y_test)

    # Calculo el error cuadratico medio
    train_mse = mean_squared_error(Y_train, train_predict)
    test_mse = mean_squared_error(Y_test, test_predict)
    print("l={} MSE train: {:.2f}".format(l, train_mse))
    print("l={} MSE test:  {:.2f}".format(l, test_mse))

    # Ploteo los resultados
    month = np.arange(len(data_noise))

    plt.figure()
    plt.plot(normalization.inverse_transform(data_noise), label="Datos+Ruido")
    plt.plot(np.arange(len(train_predict)) + l, train_predict, label="Training")
    plt.plot(
        np.arange(len(test_predict)) + len(train_predict) + l,
        test_predict,
        label="Test",
    )
    plt.xlabel("Meses", fontsize=15)
    plt.ylabel("N° de pasajeros", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(img_folder, "Predict_l={}.pdf".format(l)),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    l_log.append(l)
    mse_test.append(test_mse)
    mse_train.append(train_mse)

l_log = np.array(l_log)
mse_test = np.array(mse_test)
mse_train = np.array(mse_train)


# Grafico
plt.figure()
plt.plot(l_log, mse_train, label="Training")
plt.xlabel("l", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "MSE_LSTM_Train.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()

plt.figure()
plt.plot(l_log, mse_test, label="Test")
plt.xlabel("l", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "MSE_LSTM_Test.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()

# -------------------------------------
# 10 - Comparar con una red de Densas
# -------------------------------------


def Dense_model():
    model = keras.models.Sequential(name="Dense")
    model.add(layers.Input(shape=X_train.shape[1:]))
    model.add(layers.Dense(4, name="Dense_hidden"))
    model.add(layers.Dense(1, name="Dense"))
    model.summary()
    return model


l = 1
X, Y = formatData(data_noise, l)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.3)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1])

model = Dense_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4), loss=losses.MeanSquaredError(),
)

hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=100,
    batch_size=1,
    verbose=2,
)

# Calculo la loss y Accuracy para los datos de test
test_loss = model.evaluate(X_test, Y_test)
hist.history["test_loss"] = test_loss

# Guardo las imagenes
img_folder = "Figuras"
if not os.path.exists(img_folder):
    os.makedirs(img_folder)

# Grafico
plt.figure()
plt.plot(hist.history["loss"], label="Loss Training")
plt.plot(hist.history["val_loss"], label="Loss Test")
plt.xlabel("Epocas", fontsize=15)
plt.ylabel("Loss", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "Dense_Loss.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invertimos la normalizacion
train_predict = normalization.inverse_transform(train_predict)
test_predict = normalization.inverse_transform(test_predict)
Y_train = normalization.inverse_transform(Y_train)
Y_test = normalization.inverse_transform(Y_test)

# Calculo el error cuadratico medio
train_mse = mean_squared_error(Y_train, train_predict)
test_mse = mean_squared_error(Y_test, test_predict)
print("MSE final de train: {:.2f}".format(train_mse))
print("MSE final de test:  {:.2f}".format(test_mse))

# Ploteo los resultados
month = np.arange(len(data))

plt.figure()
plt.plot(normalization.inverse_transform(data_noise), label="Datos+Ruido")
plt.plot(np.arange(len(train_predict)) + l, train_predict, label="Training")
plt.plot(
    np.arange(len(test_predict)) + len(train_predict) + l, test_predict, label="Test"
)
plt.xlabel("Meses", fontsize=15)
plt.ylabel("N° de pasajeros", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "Dense_Predict.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()


dense_l_log = []
dense_mse_test = []
dense_mse_train = []

for l in range(1, 37):

    X, Y = formatData(data_noise, l)

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, shuffle=False, test_size=0.3
    )

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], X_train.shape[1])

    model = Dense_model()

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4), loss=losses.MeanSquaredError(),
    )

    hist = model.fit(
        X_train,
        Y_train,
        validation_data=(X_test, Y_test),
        epochs=100,
        batch_size=1,
        verbose=2,
    )

    # Grafico
    plt.figure()
    plt.plot(hist.history["loss"], label="Loss Training")
    plt.plot(hist.history["val_loss"], label="Loss Test")
    # plt.title("Loss Test: {:.3f}".format(test_loss))
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Loss", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(img_folder, "Dense_Loss_l={}.pdf".format(l)),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Invertimos la normalizacion
    train_predict = normalization.inverse_transform(train_predict)
    test_predict = normalization.inverse_transform(test_predict)
    Y_train = normalization.inverse_transform(Y_train)
    Y_test = normalization.inverse_transform(Y_test)

    # Calculo el error cuadratico medio
    train_mse = mean_squared_error(Y_train, train_predict)
    test_mse = mean_squared_error(Y_test, test_predict)
    print("l={} MSE train: {:.2f}".format(l, train_mse))
    print("l={} MSE test:  {:.2f}".format(l, test_mse))

    # Ploteo los resultados
    month = np.arange(len(data_noise))

    plt.figure()
    plt.plot(normalization.inverse_transform(data_noise), label="Datos+Ruido")
    plt.plot(np.arange(len(train_predict)) + l, train_predict, label="Training")
    plt.plot(
        np.arange(len(test_predict)) + len(train_predict) + l,
        test_predict,
        label="Test",
    )
    plt.xlabel("Meses", fontsize=15)
    plt.ylabel("N° de pasajeros", fontsize=15)
    plt.legend(loc="best")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        os.path.join(img_folder, "Dense_Predict_l={}.pdf".format(l)),
        format="pdf",
        bbox_inches="tight",
    )
    plt.close()

    dense_l_log.append(l)
    dense_mse_test.append(test_mse)
    dense_mse_train.append(train_mse)

dense_l_log = np.array(dense_l_log)
dense_mse_test = np.array(dense_mse_test)
dense_mse_train = np.array(dense_mse_train)

# Grafico
plt.figure()
plt.plot(dense_l_log, dense_mse_train, label="Training")
plt.xlabel("l", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "MSE_Dense_Train.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()

plt.figure()
plt.plot(dense_l_log, dense_mse_test, label="Test")
plt.xlabel("l", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "MSE_Dense_Test.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()

# Comparacion

plt.figure()
plt.plot(dense_l_log, dense_mse_test, label="Dense - Test")
plt.plot(l_log, mse_test, label="LSTM - Test")
plt.plot(dense_l_log, dense_mse_train, label="Dense - Train")
plt.plot(l_log, mse_train, label="LSTM - Train")
plt.xlabel("l", fontsize=15)
plt.ylabel("MSE", fontsize=15)
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(img_folder, "Comparacion.pdf"), format="pdf", bbox_inches="tight"
)
plt.close()
