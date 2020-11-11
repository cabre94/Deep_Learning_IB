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

# Abro el csv
data = pd.read_csv("airline-passengers.csv", header=0)

month = data["Month"].values
passengers = data["Passengers"].values
passengers = passengers.reshape((passengers.shape[0],1))

# -------------------------------------
# 2
# -------------------------------------

# Normalizacion
normalization = MinMaxScaler(feature_range=(0, 1))
passengers = normalization.fit_transform(passengers)

# Normalizo
# minimo = np.min(passengers)
# passengers -= minimo

# maximo = np.max(passengers)
# passengers = passengers / maximo

# plt.plot(passengers)
# plt.show()


# converting an array of values into a dataset matrix
def formatData(data, l=1):
    X, Y = [], []
    for i in range(len(data) - l - 1):
        a = data[i : (i + l)]  ###i=0, 0,1,2,3
        X.append(np.copy(a))
        Y.append(np.copy(data[i + l]))
    return np.array(X), np.array(Y)


l = 24

X, Y = formatData(passengers, l)

# -------------------------------------
# 3
# -------------------------------------

noise = np.random.normal(loc=0, scale=0.02, size=X.shape)

X += noise

# -------------------------------------
# 4 - Spliteo los datos
# -------------------------------------

# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=False, test_size=0.5)

umbral = int(len(X) * 0.7)

X_train, X_test = X[:umbral, :], X[umbral:, :]
Y_train, Y_test = Y[:umbral, :], Y[umbral:, :]

# -------------------------------------
# 5
# -------------------------------------

# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_train.shape[1], 1)

X_train = X_train.reshape(X_train.shape[0],1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0],1, X_test.shape[1])

# -------------------------------------
# 6
# -------------------------------------

lr = 1e-3
batch_size = 1
epochs = 100

model = keras.models.Sequential(name="LSTM")
model.add(layers.Input(shape=X_train.shape[1:]))

# model.add(layers.LSTM(100, return_sequences=True, name="LSTM_1"))
# model.add(layers.LSTM(100, return_sequences=True, name="LSTM_2"))
# model.add(layers.LSTM(100, return_sequences=True, name="LSTM_3"))
# model.add(layers.LSTM(100, return_sequences=False, name="LSTM_4"))
model.add(layers.LSTM(4, return_sequences=False, name="LSTM_4"))

model.add(layers.Dense(1, name="Dense"))

model.summary()

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    loss=losses.MeanSquaredError(),
    # loss=losses.CategoricalCrossentropy(from_logits=True),
    # metrics=[metrics.CategoricalAccuracy(name="acc")],
)

# -------------------------------------
# 7
# -------------------------------------

hist = model.fit(
    X_train,
    Y_train,
    validation_data=(X_test, Y_test),
    epochs=epochs,
    batch_size=batch_size,
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
# plt.close()


# -------------------------------------
# 8
# -------------------------------------

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = normalization.inverse_transform(train_predict)
test_predict = normalization.inverse_transform(test_predict)
Y_train = normalization.inverse_transform(Y_train)
Y_test = normalization.inverse_transform(Y_test)

passengers = normalization.inverse_transform(passengers)

# train_predict = train_predict * maximo + minimo
# test_predict = test_predict * maximo + minimo

# Y_train = Y_train * maximo + minimo
# Y_test = Y_test * maximo + minimo

train_mse = mean_squared_error(Y_train, train_predict)
test_mse = mean_squared_error(Y_test, test_predict)

print("MSE final de train: {:.2f}".format(train_mse))
print("MSE final de test:  {:.2f}".format(test_mse))

# -------------------------------------
# Ploteo el resultado
# -------------------------------------

# trainPredictPlot = np.empty_like(passengers)
# trainPredictPlot[:] = np.nan
# trainPredictPlot[l:len(train_predict)+l] = train_predict
# # shift test predictions for plotting
# testPredictPlot = np.empty_like(data)
# testPredictPlot[:] = np.nan
# testPredictPlot[len(train_predict)+(l*2)+1:len(data)-1] = test_predict

# # passengers = passengers * maximo + minimo

# plt.plot(passengers)
# # plt.plot(train_predict, label="Train")
# plt.plot(testPredictPlot, label="Test")
# plt.show()

month = np.arange(len(passengers))


plt.figure()
# plt.plot(passengers, label='Datos')
plt.plot(normalization.inverse_transform(X[:,0]), label='Datos+Ruidos')

plt.plot(month[:len(train_predict)], train_predict)

plt.plot(month[-len(test_predict)-l:-l], test_predict, label='1')
plt.plot(len(train_predict)+np.arange(0,len(test_predict)), test_predict, label='2')

plt.legend(loc='best')
plt.show()

# -------------------------------------
# 9
# -------------------------------------
