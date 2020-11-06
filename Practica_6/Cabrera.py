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

seed = np.random.randint(1e3, size=1)[0]
np.random.seed(seed)

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
        max_depth=4,
        filled=True,
        rounded=True,
        label="root",
        leaves_parallel=False,
        rotate=False,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render(os.path.join(SAVE_PATH, "B_Arbol_croped"))

    # print('Resultados sobre los datos de training:')
    # y_true, y_pred = y_train, treeClassifier.predict(x_train)
    # print(metrics.classification_report(y_true, y_pred))

    # print('Resultados sobre los datos de test:')
    # y_true, y_pred = y_test, treeClassifier.predict(x_test)
    # print(metrics.classification_report(y_true, y_pred))

    # y_scores = treeClassifier.predict_proba(x_train)
    # fpr, tpr, thresholds = metrics.roc_curve(y_train, y_scores[:, 1])
    # print('AUC training ', metrics.roc_auc_score(y_train, y_scores[:, 1]))
    # y_scores_test = treeClassifier.predict_proba(x_test)
    # fpr_test, tpr_test, thresholds = metrics.roc_curve(y_test, y_scores_test[:, 1])
    # print('AUC test', metrics.roc_auc_score(y_test, y_scores_test[:, 1]))

    # fig = plt.figure(figsize=(8,6))
    # plt.plot(fpr, tpr,lw=2, label='Training')
    # plt.plot(fpr_test, tpr_test, label='Test')
    # plt.title('ROC curve', fontsize=16)
    # plt.xlabel('False positive ratio', fontsize=14)
    # plt.ylabel('True positive ratio')
    # plt.legend(fontsize=14)
    # #plt.savefig('ROC_b_1.pdf', format='pdf')
    # plt.show()

    # fig = metrics.plot_roc_curve(treeClassifier, x_train, y_train, label='Training')
    # metrics.plot_roc_curve(treeClassifier, x_test, y_test, ax=fig.ax_, label='Test')
    # plt.show()

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
        max_depth=4,
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
        plt.plot(y_test, predict_test, "ob", label="Predicciones")
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, "C.pdf"), format="pdf", bbox_inches="tight")
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

    print("\nItem D - Error en Tree Classifier")
    print("MSE de training: {:.2f}".format(mse_train))
    print("MSE de test: {:.2f}".format(mse_test))


def item_E(x_train, y_train, x_test, y_test, plot=False):

    treeRegressor = tree.DecisionTreeRegressor()

    parameters = {"max_depth": np.arange(1, 20, 1), "ccp_alpha": np.linspace(0, 1, 101)}

    gsCV = GridSearchCV(treeRegressor, parameters, return_train_score=True)
    gsCV.fit(x_train, y_train)

    print("\nItem E")
    print("Mejores parámetros obtenido del GridSearchCV:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Error con modelo optimizado con GridSearchCV")
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
        plt.plot(y_test, predict_test, "ob", label="Predicciones")
        plt.xlabel("Sales reales", fontsize=FONTSIZE)
        plt.ylabel("Sales predichos", fontsize=FONTSIZE)
        plt.legend(loc="best", fontsize=FONTSIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_PATH, "E.pdf"), format="pdf", bbox_inches="tight")
        plt.show()
    
    return final_model


def item_F(x_train, y_train, x_test, y_test):

    treeRegressor = tree.DecisionTreeRegressor()
    ensembleBagging = ensemble.BaggingRegressor(treeRegressor)

    parameters = {
        "n_estimators": np.arange(10, 100, 5),
        "max_samples": np.random.uniform(0, 1, 100),
        "bootstrap": ["True"]
        # 'ccp_alpha': np.linspace(0, 2, 100)
    }

    gsCV = GridSearchCV(ensembleBagging, parameters, verbose=1, return_train_score=True)

    gsCV.fit(x_train, y_train)

    print("Mejores parámetros:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Resultados para Bagging")
    print(final_model.score(x_test, y_test))

    pesos = np.zeros(10)

    for trees in final_model.estimators_:
        pesos += trees.feature_importances_

    pesos /= len(final_model.estimators_)
    print(pesos)


def item_G(x_train, y_train, x_test, y_test):
    scores_train = np.array([])
    scores_test = np.array([])

    pesos = np.zeros(10)

    for i in range(x_train.shape[1]):

        randomForest = ensemble.RandomForestRegressor(max_features=i + 1)

        randomForest = randomForest.fit(x_train, y_train)

        scores_train = np.append(scores_train, randomForest.score(x_train, y_train))
        scores_test = np.append(scores_test, randomForest.score(x_test, y_test))

        pesos += randomForest.feature_importances_

    plt.figure()
    plt.plot(
        np.arange(1, 11, 1), scores_train, drawstyle="steps-post", label="Training"
    )
    plt.plot(np.arange(1, 11, 1), scores_test, drawstyle="steps-post", label="Test")
    plt.show()

    pesos /= x_train.shape[1]
    print("Los pesos son")
    print(pesos)

    print("Ahora variando el max deep")

    scores_train = np.array([])
    scores_test = np.array([])

    pesos = np.zeros(10)

    for i in range(30):

        randomForest = ensemble.RandomForestRegressor(max_depth=i + 1)

        randomForest = randomForest.fit(x_train, y_train)

        scores_train = np.append(scores_train, randomForest.score(x_train, y_train))
        scores_test = np.append(scores_test, randomForest.score(x_test, y_test))

        pesos += randomForest.feature_importances_

    pesos /= 30
    print("Los pesos son")
    print(pesos)

    plt.figure()
    plt.plot(
        np.arange(1, 31, 1), scores_train, drawstyle="steps-post", label="Training"
    )
    plt.plot(np.arange(1, 31, 1), scores_test, drawstyle="steps-post", label="Test")
    plt.show()


def item_H(x_train, y_train, x_test, y_test):
    scores_train = np.array([])
    scores_test = np.array([])

    pesos = np.zeros(10)

    for i in range(x_train.shape[1]):

        treeRegressor = tree.DecisionTreeRegressor(max_features=i + 1)
        adaBoost = ensemble.AdaBoostRegressor(treeRegressor)

        adaBoost = adaBoost.fit(x_train, y_train)

        scores_train = np.append(scores_train, adaBoost.score(x_train, y_train))
        scores_test = np.append(scores_test, adaBoost.score(x_test, y_test))

        pesos += adaBoost.feature_importances_

    plt.figure()
    plt.plot(
        np.arange(1, 11, 1), scores_train, drawstyle="steps-post", label="Training"
    )
    plt.plot(np.arange(1, 11, 1), scores_test, drawstyle="steps-post", label="Test")
    plt.show()

    pesos /= x_train.shape[1]
    print("Los pesos son")
    print(pesos)

    print("Ahora variando el max deep")

    scores_train = np.array([])
    scores_test = np.array([])

    pesos = np.zeros(10)

    for i in range(30):

        treeRegressor = tree.DecisionTreeRegressor(max_depth=i + 1)
        adaBoost = ensemble.AdaBoostRegressor(treeRegressor)

        adaBoost = adaBoost.fit(x_train, y_train)

        scores_train = np.append(scores_train, adaBoost.score(x_train, y_train))
        scores_test = np.append(scores_test, adaBoost.score(x_test, y_test))

        pesos += adaBoost.feature_importances_

    pesos /= 30
    print("Los pesos son")
    print(pesos)

    plt.figure()
    plt.plot(
        np.arange(1, 31, 1), scores_train, drawstyle="steps-post", label="Training"
    )
    plt.plot(np.arange(1, 31, 1), scores_test, drawstyle="steps-post", label="Test")
    plt.show()


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

