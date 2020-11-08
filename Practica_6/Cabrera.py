#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
date: 05-11-2020
File: Cabrera.py
Author : Facundo Martin Cabrera
Email: cabre94@hotmail.com facundo.cabrera@ib.edu.ar
GitHub: https://github.com/cabre94
GitLab: https://gitlab.com/cabre94
Description:
"""
# from sklearn.metrics import classification_report, accuracy_score
# from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import metrics

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import tree, ensemble
import graphviz

import seaborn as snn

snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

# seed = np.random.randint(1e3, size=1)[0]
seed = 972
np.random.seed(seed)

print("Semilla: {}\n".format(seed))

FONTSIZE = 15
SAVE_PATH = "Figuras"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)


# Funcion para crea la nueva variable 'High'
def newVariable(row):
    if row["Sales"] >= 8:
        return "Yes"
    else:
        return "No"


def item_B(train, test, plot=False):
    # Eliminamos las variables continuas
    x_train, y_train = train.drop(["High", "Sales"], axis=1), train["High"]
    x_test, y_test = test.drop(["High", "Sales"], axis=1), test["High"]

    treeClassifier = tree.DecisionTreeClassifier()
    treeClassifier = treeClassifier.fit(x_train, y_train)

    print("\nItem B - Resultados para Tree Classifier")
    print("Score Train: {:.2f}".format(treeClassifier.score(x_train, y_train)))
    print("Score Test: {:.2f}".format(treeClassifier.score(x_test, y_test)))

    dot_data = tree.export_graphviz(
        treeClassifier,
        out_file=None,
        max_depth=3,
        filled=True,
        rounded=True,
        label="root",
        leaves_parallel=False,
        rotate=False,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(SAVE_PATH, "B_Arbol_croped"))

    return treeClassifier


def item_C(train, test, plot=False):
    # Eliminamos la variable discreta del dataset
    x_train, y_train = train.drop(["High", "Sales"], axis=1), train["Sales"]
    x_test, y_test = test.drop(["High", "Sales"], axis=1), test["Sales"]

    treeRegressor = tree.DecisionTreeRegressor()
    treeRegressor = treeRegressor.fit(x_train, y_train)

    print("\nItem C - Resultados para Tree Regressor")
    print("Score Train: {:.2f}".format(treeRegressor.score(x_train, y_train)))
    print("Score Test: {:.2f}".format(treeRegressor.score(x_test, y_test)))

    dot_data = tree.export_graphviz(
        treeRegressor,
        max_depth=3,
        out_file=None,
        filled=True,
        rounded=True,
        label="root",
        leaves_parallel=False,
        rotate=False,
        special_characters=True,
    )

    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(SAVE_PATH, "C_Arbol_croped"))

    predict_train = treeRegressor.predict(x_train)
    predict_test = treeRegressor.predict(x_test)

    if plot:
        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_train, predict_train, "or", label="Predic Train", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "C_Train.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_test, predict_test, "ob", label="Predic Test", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "C_Test.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

    return treeRegressor


def item_D(train, test, tClasifier, tRegressor):

    x_train, y_train = train.drop(["High", "Sales"], axis=1), train["High"]
    x_test, y_test = test.drop(["High", "Sales"], axis=1), test["High"]

    acc_train = metrics.accuracy_score(y_train, tClasifier.predict(x_train))
    acc_test = metrics.accuracy_score(y_test, tClasifier.predict(x_test))

    print("\nItem D - Error en Tree Classifier")
    print("Error de training: {:.2f}".format(1 - acc_train))
    print("Error de test: {:.2f}".format(1 - acc_test))

    x_train, y_train = train.drop(["High", "Sales"], axis=1), train["Sales"]
    x_test, y_test = test.drop(["High", "Sales"], axis=1), test["Sales"]

    mse_train = metrics.mean_squared_error(y_train, tRegressor.predict(x_train))
    mse_test = metrics.mean_squared_error(y_test, tRegressor.predict(x_test))

    print("\nItem D - Error en Tree Regressor")
    print("MSE de training: {:.2f}".format(mse_train))
    print("MSE de test: {:.2f}".format(mse_test))


def printFeatureImportances(x_train, feature_importances):
    zipped = zip(feature_importances, x_train.keys())
    sort = sorted(zipped, reverse=True)
    tuples = zip(*sort)

    importance, features = [list(tuple) for tuple in tuples]

    print("Atributo\tImportancia")
    for i in range(len(features)):
        print("{}\t{:.3f}".format(features[i], importance[i]))


def item_E(x_train, y_train, x_test, y_test, plot=False):

    treeRegressor = tree.DecisionTreeRegressor()

    parameters = {"max_depth": np.arange(1, 20, 1), "ccp_alpha": np.linspace(0, 1, 201)}

    gsCV = GridSearchCV(treeRegressor, parameters, return_train_score=True)
    gsCV.fit(x_train, y_train)

    print("\nItem E")
    print("Mejores parámetros obtenidos del GridSearchCV:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Resultados con modelo optimizado con GridSearchCV")
    print("Score Train: {:.2f}".format(final_model.score(x_train, y_train)))
    print("Score Test: {:.2f}".format(final_model.score(x_test, y_test)))

    predict_train = final_model.predict(x_train)
    predict_test = final_model.predict(x_test)

    mse_train = metrics.mean_squared_error(y_train, predict_train)
    mse_test = metrics.mean_squared_error(y_test, predict_test)

    print("\nItem E - Error con modelo optimizado con GridSearchCV")
    print("MSE de training: {:.2f}".format(mse_train))
    print("MSE de test: {:.2f}".format(mse_test))

    if plot:
        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_train, predict_train, "or", label="Predic Train", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "E_Train.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_test, predict_test, "ob", label="Predic Test", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "E_Test.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

    return final_model


def item_F(x_train, y_train, x_test, y_test, plot=True):

    treeRegressor = tree.DecisionTreeRegressor()
    ensembleBagging = ensemble.BaggingRegressor(treeRegressor)

    parameters = {
        "n_estimators": np.arange(30, 80, 10),
        "max_samples": np.random.uniform(0.2, 0.99, 50),
        "bootstrap": ["True"],
    }

    gsCV = GridSearchCV(ensembleBagging, parameters, verbose=1, return_train_score=True)
    gsCV.fit(x_train, y_train)

    print("\nItem F")
    print("Mejores parámetros obtenidos del Bagging:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Resultados con modelo optimizado con Bagging")
    print("Score Train: {:.2f}".format(final_model.score(x_train, y_train)))
    print("Score Test: {:.2f}".format(final_model.score(x_test, y_test)))

    predict_train = final_model.predict(x_train)
    predict_test = final_model.predict(x_test)

    mse_train = metrics.mean_squared_error(y_train, predict_train)
    mse_test = metrics.mean_squared_error(y_test, predict_test)

    print("\nItem F - Error con modelo optimizado con Bagging")
    print("MSE de training: {:.2f}".format(mse_train))
    print("MSE de test: {:.2f}".format(mse_test))

    if plot:
        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_train, predict_train, "or", label="Predic Train", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "F_Train.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_test, predict_test, "ob", label="Predic Test", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "F_Test.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

    feature_importance = np.zeros(10)

    for estimator in final_model.estimators_:
        feature_importance += estimator.feature_importances_
    
    feature_importance /= len(final_model.estimators_)

    printFeatureImportances(x_train, feature_importance)

    return final_model


def item_G_Mejora(x_train, y_train, x_test, y_test, plot=False):

    randomForest = ensemble.RandomForestRegressor()

    parameters = {
        "max_depth": np.arange(5, 15, 1),
        "ccp_alpha": np.linspace(0, 0.5, 21),
        "n_estimators": [70],
        "max_samples": np.random.uniform(0.5, 0.99, 10),
    }

    gsCV = GridSearchCV(randomForest, parameters, verbose=1, return_train_score=True)
    gsCV.fit(x_train, y_train)

    print("\nItem G")
    print("Mejores parámetros obtenidos del Random Forest:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Resultados con modelo optimizado con Random Forest")
    print("Score Train: {:.2f}".format(final_model.score(x_train, y_train)))
    print("Score Test: {:.2f}".format(final_model.score(x_test, y_test)))

    predict_train = final_model.predict(x_train)
    predict_test = final_model.predict(x_test)

    mse_train = metrics.mean_squared_error(y_train, predict_train)
    mse_test = metrics.mean_squared_error(y_test, predict_test)

    print("\nItem G - Error con modelo optimizado con Random Forest")
    print("MSE de training: {:.2f}".format(mse_train))
    print("MSE de test: {:.2f}".format(mse_test))

    if plot:
        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_train, predict_train, "or", label="Predic Train", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "G_Train.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_test, predict_test, "ob", label="Predic Test", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "G_Test.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

    printFeatureImportances(x_train, final_model.feature_importances_)

    return final_model


def item_G_Barrido(x_train, y_train, x_test, y_test, plot=False):
    mse_train = np.array([])
    mse_test = np.array([])

    for i in range(x_train.shape[1]):

        randomForest = ensemble.RandomForestRegressor(max_features=i + 1)
        randomForest = randomForest.fit(x_train, y_train)

        predict_train = randomForest.predict(x_train)
        predict_test = randomForest.predict(x_test)

        mse_train_i = metrics.mean_squared_error(y_train, predict_train)
        mse_test_i = metrics.mean_squared_error(y_test, predict_test)

        mse_train = np.append(mse_train, mse_train_i)
        mse_test = np.append(mse_test, mse_test_i)

    if plot:
        plt.plot(
            np.arange(1, 11, 1), mse_train, drawstyle="steps-post", label="Training"
        )
        plt.plot(np.arange(1, 11, 1), mse_test, drawstyle="steps-post", label="Test")
        plt.xlabel("Max features", fontsize=FONTSIZE)
        plt.ylabel("MSE", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "G_Barrido_MaxFeatures.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.show()

    mse_train = np.array([])
    mse_test = np.array([])

    for i in range(30):

        randomForest = ensemble.RandomForestRegressor(max_depth=i + 1)
        randomForest = randomForest.fit(x_train, y_train)

        predict_train = randomForest.predict(x_train)
        predict_test = randomForest.predict(x_test)

        mse_train_i = metrics.mean_squared_error(y_train, predict_train)
        mse_test_i = metrics.mean_squared_error(y_test, predict_test)

        mse_train = np.append(mse_train, mse_train_i)
        mse_test = np.append(mse_test, mse_test_i)

    if plot:
        plt.plot(
            np.arange(1, 31, 1), mse_train, drawstyle="steps-post", label="Training"
        )
        plt.plot(np.arange(1, 31, 1), mse_test, drawstyle="steps-post", label="Test")
        plt.xlabel("Max Depth", fontsize=FONTSIZE)
        plt.ylabel("MSE", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "G_Barrido_MaxDepth.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        plt.show()


def item_H(x_train, y_train, x_test, y_test, final_model, plot=True):

    adaBoost = ensemble.AdaBoostRegressor(final_model)

    parameters = {
        "learning_rate": np.linspace(1e-5, 5, 21),
    }

    gsCV = GridSearchCV(adaBoost, parameters, verbose=1, return_train_score=True)
    gsCV.fit(x_train, y_train)

    print("\nItem H")
    print("Mejores parámetros obtenidos del AdaBoost:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Resultados con modelo optimizado con AdaBoost")
    print("Score Train: {:.2f}".format(final_model.score(x_train, y_train)))
    print("Score Test: {:.2f}".format(final_model.score(x_test, y_test)))

    predict_train = final_model.predict(x_train)
    predict_test = final_model.predict(x_test)

    mse_train = metrics.mean_squared_error(y_train, predict_train)
    mse_test = metrics.mean_squared_error(y_test, predict_test)

    print("\nItem H - Error con modelo optimizado con AdaBoost")
    print("MSE de training: {:.2f}".format(mse_train))
    print("MSE de test: {:.2f}".format(mse_test))

    if plot:
        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_train, predict_train, "or", label="Predic Train", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "H_Train.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

        plt.plot([0, max(y_test)], [0, max(y_test)], "--k", label="Target")
        plt.plot(y_test, predict_test, "ob", label="Predic Test", alpha=0.6)
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE - 2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(SAVE_PATH, "H_Test.pdf"), format="pdf", bbox_inches="tight"
        )
        plt.show()

    printFeatureImportances(x_train, final_model.feature_importances_)


if __name__ == "__main__":

    # Abro los datos
    data = pd.read_csv("Carseats.csv", header=0)
    # Genero la nueva variable 'High'
    data["High"] = data.apply(lambda row: newVariable(row), axis=1)

    # Reemplazo valores con formato str a int
    data.replace(("Yes", "No"), (1, 0), inplace=True)
    data.replace(("Good", "Medium", "Bad"), (2, 1, 0), inplace=True)

    # item a: Spliteo los datos
    train, test = train_test_split(data, test_size=0.3, stratify=data["High"])

    tClasifier = item_B(train, test, plot=True)

    tRegressor = item_C(train, test, plot=True)

    item_D(train, test, tClasifier, tRegressor)

    # De ahora en mas solo hacemos regresiones, asi que ya saco la variable
    # 'High' para siempre
    x_train, y_train = train.drop(["High", "Sales"], axis=1), train["Sales"]
    x_test, y_test = test.drop(["High", "Sales"], axis=1), test["Sales"]

    item_E(x_train, y_train, x_test, y_test, plot=True)

    item_F(x_train, y_train, x_test, y_test, plot=True)

    randomForest = item_G_Mejora(x_train, y_train, x_test, y_test, plot=True)

    item_G_Barrido(x_train, y_train, x_test, y_test, plot=True)

    item_H(x_train, y_train, x_test, y_test, randomForest, plot=True)
