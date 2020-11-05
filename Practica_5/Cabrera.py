#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 22-10-2020
File: Cabrera.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""

import os
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses, metrics, optimizers
from tensorflow.keras.regularizers import l2

import seaborn as snn
snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

#--------------------------------------
#           Ejercicio 3
#--------------------------------------


def ej3TraininigFashionMNIST():
    lr = 1e-3
    rf = 1e-4
    epochs = 100
    batch_size = 256

    # Importo los datos
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # Los junto porque creo que no estan bien distribuidos
    x_train, y_train = np.vstack((x_train, x_test)), np.hstack(
        (y_train, y_test))
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=10000,
                                                        stratify=y_train)
    # Ahora separo entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=10000,
                                                      stratify=y_train)

    # Normalizacion
    x_train = x_train.reshape((-1, 28, 28, 1)) / 255
    x_test = x_test.reshape((-1, 28, 28, 1)) / 255
    x_val = x_val.reshape((-1, 28, 28, 1)) / 255

    # Paso los labels a one-hot representation
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_val = keras.utils.to_categorical(y_val, 10)

    # Arquitectura de la red con capas densas
    model = keras.models.Sequential(name='Fashion-MNIST_Conv')
    model.add(layers.Input(shape=x_train.shape[1:]))

    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, 3, activation='relu', padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(96, 3, activation='relu', padding='same'))
    model.add(layers.MaxPool2D())

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256, 'linear', kernel_regularizer=l2(rf)))
    model.add(layers.Dropout(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(10, 'linear', kernel_regularizer=l2(rf)))

    model.summary()

    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=[metrics.CategoricalAccuracy(name='acc')])

    # Callbacks
    lrr = keras.callbacks.ReduceLROnPlateau('val_acc', 0.5, 8, 1, min_lr=1e-5)
    callbacks = [lrr]

    hist = model.fit(x_train,
                     y_train,
                     epochs=epochs,
                     validation_data=(x_val, y_val),
                     batch_size=batch_size,
                     callbacks=callbacks,
                     verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_acc = model.evaluate(x_test, y_test)
    hist.history['test_loss'] = test_loss
    hist.history['test_acc'] = test_acc

    # Guardo los datos
    data_folder = 'Fashion-MNIST'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    model.save(os.path.join(data_folder, 'trained_model.h5'))
    np.save(os.path.join(data_folder, 'Fashion-MNIST.npy'))

    # Guardo las imagenes
    img_folder = 'Fashion-MNIST'
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
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    plt.plot(hist.history['acc'], label="Acc. Training")
    plt.plot(hist.history['val_acc'], label="Acc. Validation")
    plt.title("Acc Test: {:.3f}".format(test_acc))
    plt.xlabel("Epocas", fontsize=15)
    plt.ylabel("Accuracy", fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()


def ej3TransferToMNIST():
    lr = 1e-5
    rf = 1e-4
    epochs = 100
    batch_size = 25
    description = "lr={}_rf={}_bs={}".format(lr, rf, batch_size)

    # Importo los datos
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Los junto porque creo que no estan bien distribuidos
    x_train, y_train = np.vstack((x_train, x_test)), np.hstack(
        (y_train, y_test))
    # Separo los datos de test
    x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                        y_train,
                                                        test_size=34500,
                                                        stratify=y_train)
    # Ahora separo entre training y validacion
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=34500,
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
    trained_model = os.path.join(os.getcwd(), "Fashion-MNIST",
                                 "trained_model.h5")
    # Cargo el modelo
    fashion_model = keras.models.load_model(trained_model)
    # No dejo que se entrene
    fashion_model.trainable = False

    mnist_model = keras.models.Sequential(name='Mnist')

    # Sacamos las capas densas de clasificacion
    for layer in fashion_model.layers[:-6]:
        mnist_model.add(layer)

    # Pruebo a ver que pasa si solo cambio la ultima capa densa (la de clasificacion)
    mnist_model.add(layers.Dropout(0.2, name='n4'))
    mnist_model.add(layers.BatchNormalization(name='n5'))
    mnist_model.add(
        layers.Dense(10, 'linear', kernel_regularizer=l2(rf), name='n6'))

    mnist_model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                        loss=losses.CategoricalCrossentropy(from_logits=True),
                        metrics=[metrics.CategoricalAccuracy(name='acc')])

    # Callbacks
    earlystop = keras.callbacks.EarlyStopping(patience=10)
    lrr = keras.callbacks.ReduceLROnPlateau('val_acc', 0.5, 5, 1, min_lr=1e-5)
    #callbacks = [earlystop, lrr]
    callbacks = [lrr]

    hist = mnist_model.fit(x_train,
                           y_train,
                           epochs=epochs,
                           validation_data=(x_val, y_val),
                           batch_size=batch_size,
                           callbacks=callbacks,
                           verbose=2)

    # Calculo la loss y Accuracy para los datos de test
    test_loss, test_acc = mnist_model.evaluate(x_test, y_test)
    hist.history['test_loss'] = test_loss
    hist.history['test_acc'] = test_acc

    # Guardo los datos
    data_folder = os.path.join('Datos', 'MNIST')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    mnist_model.save(os.path.join(data_folder, '{}.h5'.format(description)))
    np.save(os.path.join(data_folder, '{}.npy'.format(description)),
            hist.history)

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
    plt.show()

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
    plt.show()


#--------------------------------------
#   Func. auxiliares para el ej 4
#--------------------------------------

# Basado en el ejemplo de
# https://keras.io/examples/vision/visualizing_what_convnets_learn/


# Funcion loss que vamos a querer maximizar. En este caso es la media de la
# activacion para un filtro de la layer que tenemo. No se bien porque pero
# no consideramos los bordes
def compute_loss(input_image, filter_index, feature_extractor):
    activation = feature_extractor(input_image)
    # No consideramos los bordes
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)


# Funcion que realiza un paso del gradient ascent
@tf.function
def gradient_ascent_step(img, filter_index, lr, feature_extractor):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = compute_loss(img, filter_index, feature_extractor)
    # Calcula el gradiente
    grads = tape.gradient(loss, img)
    # Regularizacion
    grads = tf.math.l2_normalize(grads)
    img += lr * grads
    return loss, img


# Funcion para desnormalizar
def deprocess_image(img):
    # Ajusta la imagen centrandola en cero y ajustandole la std a 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15
    # Sacamos los bordes
    img = img[25:-25, 25:-25, :]
    # Clip a [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)
    # Pasamos a RGB
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


# funcion para inicializar una imagen en gris
def initialize_image(img_width, img_height):
    img = tf.random.uniform((1, img_width, img_height, 3))
    # El input de la red ResNet50V2 esta entre [-1, +1].
    # Escalamos a [-0.125, +0.125]
    return (img - 0.5) * 0.25


# Para un dado filtro de una dada capa, calcula la imagen que maximiza
# su activacion. Tambien retorna la loss correspondiente
def visualize_filter(filter_index, feature_extractor, epochs, learning_rate,
                     img_width, img_height):
    img = initialize_image(img_width, img_height)
    for iteration in range(epochs):
        loss, img = gradient_ascent_step(img, filter_index, learning_rate,
                                         feature_extractor)
    # Decode the resulting input image
    img = deprocess_image(img[0].numpy())
    return loss, img


#--------------------------------------
#           Ejercicio 4
#--------------------------------------


def ej4(model='VGG16'):
    # Defino constantes
    epochs = 200  # Yo lo calcule con mas, pero tardaba un siglo
    learning_rate = 1

    # Dimensiones de la imagen de entrada
    img_width = 224
    img_height = 224

    # Cargamos el modelo eentrenado de Keras. Por defecto carga los pesos
    # entrenamos para ImageNet. Ademas le decimos que no nos cargue las
    # capas densas de clasificacion ya que no nos interesan
    if model == 'VGG16':
        model = keras.applications.VGG16(False)
        img_folder = os.path.join('Figuras', 'VGG16')
    elif model == 'ResNet':
        model = keras.applications.ResNet50V2(False)
        img_folder = os.path.join('Figuras', 'ResNet50v2')

    # Carpeta donde guardamos las imagenes
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Constantes para que los graficos queden lindos
    margin = 1
    n = 4  # Numero de filas
    m = 6  # Numero de columnas
    cropped_width = img_width - 25 * 2
    cropped_height = img_height - 25 * 2
    width = n * cropped_width + (n - 1) * margin
    height = m * cropped_height + (m - 1) * margin

    for layer in model.layers:
        if '_conv' in layer.name:  # Solo buscamos las convolucionales
            layer_name = layer.name
            layer = model.get_layer(name=layer_name)

            feature_extractor = keras.Model(inputs=model.inputs,
                                            outputs=layer.output)

            # Me gusta agarrar filtros aleatorios y no solo los primeros
            random_idx = np.random.choice(range(layer.filters),
                                          n * m,
                                          replace=False)

            all_imgs = []
            for filter_index in random_idx:
                print("Layer {} - Filtro {}".format(layer_name, filter_index))
                loss, img = visualize_filter(filter_index, feature_extractor,
                                             epochs, learning_rate, img_width,
                                             img_height)
                all_imgs.append(img)

            stitched_filters = np.zeros((width, height, 3))

            for i in range(n):
                for j in range(m):
                    img = all_imgs[i * m + j]
                    stitched_filters[(cropped_width + margin) *
                                     i:(cropped_width + margin) * i +
                                     cropped_width, (cropped_height + margin) *
                                     j:(cropped_height + margin) * j +
                                     cropped_height, :, ] = img

            save_path = os.path.join(img_folder, "{}.pdf".format(layer_name))
            keras.preprocessing.image.save_img(save_path, stitched_filters)


#--------------------------------------
# Funcion para graficar
#--------------------------------------


def graficosFashionMNIST():
    path_data = os.path.join(os.getcwd(), "Fashion-MNIST",
                                 "trained_model.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_FashionMNIST')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:50], label="Loss Training")
    plt.plot(data['val_loss'][:50], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.plot(data['acc'][:50], label="Loss Training")
    plt.plot(data['val_acc'][:50], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

def graficosTransferToMNIST():
    path_data = os.path.join(os.getcwd(), "MNIST",
                                 "transferedModel.npy")

    data = np.load(path_data, allow_pickle=True).item()

    # Grafico y guardo figuras
    img_folder = os.path.join('Informe', 'Figuras', '3_MNIST')
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Grafico
    plt.plot(data['loss'][:50], label="Loss Training")
    plt.plot(data['val_loss'][:50], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas",fontsize=15)
    plt.ylabel("Loss",fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Loss.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

    # Grafico
    plt.plot(data['acc'][:50], label="Loss Training")
    plt.plot(data['val_acc'][:50], label="Loss Validation")
    # plt.title("Acc Test: {:.3f}".format(test_Acc))
    plt.xlabel("Epocas",fontsize=15)
    plt.ylabel("Accuracy",fontsize=15)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(img_folder, 'Acc.pdf'),
                format="pdf",
                bbox_inches="tight")
    plt.show()

if __name__ == "__main__":

    # ej3TraininigFashionMNIST()
    # ej3TransferToMNIST()

    graficosFashionMNIST()

    graficosTransferToMNIST()

    # ej4(model='VGG16')
    # ej4(model='ResNet')
