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

# random.seed(10)


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
    # x_val, y_val = val.drop(['High', 'Sales'], axis=1), val['High']

    # Creo el arbol
    treeClassifier = tree.DecisionTreeClassifier()
    treeClassifier = treeClassifier.fit(x_train, y_train)

    print("Resultados para Tree Classifier")
    print(treeClassifier.score(x_train, y_train))
    print(treeClassifier.score(x_test, y_test))
    # print(treeClassifier.score(x_val,y_val))

    dot_data = tree.export_graphviz(
        treeClassifier,
        out_file=None,
        filled=True,
        rounded=True,
        label="root",
        leaves_parallel=False,
        rotate=False,
        special_characters=True,
    )
    graph = graphviz.Source(dot_data)
    graph.render("b")







# ------------------------------
print("Inciso C")
# ------------------------------

def item_C(x_train, y_train):
    
    # x_val, y_val = val.drop(['High', 'Sales'], axis=1), val['High']

    # Creo el arbol
    treeRegressor = tree.DecisionTreeRegressor()
    treeRegressor = treeRegressor.fit(x_train, y_train)

    print("Resultados para Tree Regressor")
    print(treeRegressor.score(x_train, y_train))
    print(treeRegressor.score(x_test, y_test))
    # print(treeClassifier.score(x_val,y_val))

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
    graph.render("c")

    predict_train = treeRegressor.predict(x_train)
    predict_test = treeRegressor.predict(x_test)

    mse_train = np.linalg.norm(y_train - predict_train) / len(y_train)
    mse_tests = np.linalg.norm(y_test - predict_test) / len(y_test)

    # print(mse_tests)



def item_D(x_train, y_train):
    pass
    # print("Precision de Clasificador")
    # print("Train:", treeClassifier.score(x_train,y_train))
    # print("Test:",treeClassifier.score(x_test,y_test))

    # print("Precision de Clasificador")
    # print("Train:", treeRegressor.score(x_train,y_train))
    # print("Test:", treeRegressor.score(x_test,y_test))

    # predict_train = treeRegressor.predict(x_train)
    # predict_test = treeRegressor.predict(x_test)


def item_E(x_train, y_train):
    # clf = tree.DecisionTreeClassifier()
    # path = clf.cost_complexity_pruning_path(x_train, y_train)
    # ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # fig, ax = plt.subplots()
    # ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    # ax.set_xlabel("effective alpha")
    # ax.set_ylabel("total impurity of leaves")
    # ax.set_title("Total Impurity vs effective alpha for training set")
    # # plt.show()


    treeRegressor = tree.DecisionTreeRegressor()

    parameters = {
        "max_depth": np.arange(1, 20, 1),
        "ccp_alpha": np.linspace(0, 2, 100)
    }

    gsCV = GridSearchCV(treeRegressor,
                        parameters,
                        verbose=1,
                        return_train_score=True)

    gsCV.fit(x_train, y_train)

    print("Mejores parámetros:")
    print(gsCV.best_params_)
    final_model = gsCV.best_estimator_

    print("Resultados para Prunnig")
    print(final_model.score(x_test, y_test))


def item_F(x_train, y_train):

    treeRegressor = tree.DecisionTreeRegressor()
    ensembleBagging = ensemble.BaggingRegressor(treeRegressor)

    parameters = {
        "n_estimators": np.arange(10, 100, 5),
        "max_samples": np.random.uniform(0, 1, 100),
        "bootstrap": ["True"]
        # 'ccp_alpha': np.linspace(0, 2, 100)
    }

    gsCV = GridSearchCV(ensembleBagging,
                        parameters,
                        verbose=1,
                        return_train_score=True)

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


def item_G(x_train, y_train):
    scores_train = np.array([])
    scores_test = np.array([])

    pesos = np.zeros(10)

    for i in range(x_train.shape[1]):

        randomForest = ensemble.RandomForestRegressor(max_features=i + 1)

        randomForest = randomForest.fit(x_train, y_train)

        scores_train = np.append(scores_train,
                                randomForest.score(x_train, y_train))
        scores_test = np.append(scores_test, randomForest.score(x_test, y_test))

        pesos += randomForest.feature_importances_

    plt.figure()
    plt.plot(np.arange(1, 11, 1),
            scores_train,
            drawstyle='steps-post',
            label='Training')
    plt.plot(np.arange(1, 11, 1),
            scores_test,
            drawstyle='steps-post',
            label='Test')
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

        scores_train = np.append(scores_train,
                                randomForest.score(x_train, y_train))
        scores_test = np.append(scores_test, randomForest.score(x_test, y_test))

        pesos += randomForest.feature_importances_

    pesos /= 30
    print("Los pesos son")
    print(pesos)

    plt.figure()
    plt.plot(np.arange(1, 31, 1),
            scores_train,
            drawstyle='steps-post',
            label='Training')
    plt.plot(np.arange(1, 31, 1),
            scores_test,
            drawstyle='steps-post',
            label='Test')
    plt.show()


def item_H(x_train, y_train):
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
    plt.plot(np.arange(1, 11, 1),
            scores_train,
            drawstyle='steps-post',
            label='Training')
    plt.plot(np.arange(1, 11, 1),
            scores_test,
            drawstyle='steps-post',
            label='Test')
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
    plt.plot(np.arange(1, 31, 1),
            scores_train,
            drawstyle='steps-post',
            label='Training')
    plt.plot(np.arange(1, 31, 1),
            scores_test,
            drawstyle='steps-post',
            label='Test')
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
    # train, val = train_test_split(train, test_size=0.2, stratify=train['High'])

    item_B(train, test, plot=True)

    # De ahora en mas solo hacemos regresiones, asi que ya saco la variable
    # 'High' para siempre
    x_train, y_train = train.drop(["High", "Sales"], axis=1), train["Sales"]
    x_test, y_test = test.drop(["High", "Sales"], axis=1), test["Sales"]