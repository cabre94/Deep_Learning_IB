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
from tensorflow.keras.regularizers import l2

import seaborn as snn

snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

# Abro el csv
data = pd.read_csv("airline-passengers.csv", header=0)

data = data["Passengers"].values
data = data.reshape((data.shape[0], 1))

# Normalizacion
normalization = MinMaxScaler(feature_range=(0, 1))
data_normalized = normalization.fit_transform(data)

# Voy a agregarle el ruido antes de formateear los datos, porque para mi tiene mas
# sentido que un valor x[i] tenga el mismo ruido en cualquier set
noise = np.random.normal(loc=0, scale=0.02, size=data_normalized.shape)
data_noise = data_normalized + noise

# Funcion para formatear los datos segun el enunciado
def formatData(dataset, l=1):
    X, Y = [], []
    for i in range(len(dataset) - l):
        a = dataset[i : (i + l)]  ###i=0, 0,1,2,3
        X.append(np.copy(a))
        Y.append(np.copy(dataset[i + l]))
    return np.array(X), np.array(Y)


l = 36
X, Y = formatData(data_noise, l)

# Spliteo los datos
# train_size = int(len(X) * 0.7)

# X_train, X_test = X[:train_size, :], X[train_size:, :]
# Y_train, Y_test = Y[:train_size, :], Y[train_size:, :]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.3)


# -------------------------------------
# 5
# -------------------------------------

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)

# X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1])
# X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1])


def LSTM_model():
    model = keras.models.Sequential(name="LSTM")
    model.add(layers.Input(shape=X_train.shape[1:]))
    model.add(layers.LSTM(4, return_sequences=False, name="LSTM"))
    model.add(layers.Dense(1, name="Dense"))
    model.summary()
    return model


model = LSTM_model()

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3), loss=losses.MeanSquaredError(),
)

hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=50,
    batch_size=1,
    verbose=2,
)

# Calculo la loss y Accuracy para los datos de test
test_loss = model.evaluate(X_test, Y_test)
hist.history["test_loss"] = test_loss
# hist.history['test_acc'] = test_acc

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
plt.tight_layout()
plt.savefig(os.path.join(img_folder, "Loss.pdf"), format="pdf", bbox_inches="tight")


train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = normalization.inverse_transform(train_predict)
test_predict = normalization.inverse_transform(test_predict)
Y_train = normalization.inverse_transform(Y_train)
Y_test = normalization.inverse_transform(Y_test)


train_mse = mean_squared_error(Y_train, train_predict)
test_mse = mean_squared_error(Y_test, test_predict)

print("MSE final de train: {:.2f}".format(train_mse))
print("MSE final de test:  {:.2f}".format(test_mse))


month = np.arange(len(data))

plt.figure()
# plt.plot(normalization.inverse_transform(passengers), label='Datos')
plt.plot(data, label='Datos+Ruidos')

# plt.plot(month[l:len(train_predict)+l], train_predict)
plt.plot(np.arange(len(train_predict)) +l, train_predict, label='Training')

plt.plot(np.arange(len(test_predict))+len(train_predict)+l, test_predict, label='Test')
# plt.plot(month[-len(test_predict)-l:-l], test_predict, label='1')
# plt.plot(len(train_predict)+np.arange(0,len(test_predict)), test_predict, label='2')


plt.grid(True)
plt.legend(loc='best')
plt.show()