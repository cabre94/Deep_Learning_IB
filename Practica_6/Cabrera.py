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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn import tree
from sklearn import ensemble
import graphviz

# import random

seed = np.random.randint(1e3, size=1)[0]
np.random.seed(seed)
# random.seed(10)

import seaborn as snn

snn.set(font_scale=1)
snn.set_style("darkgrid", {"axes.facecolor": ".9"})

# Abro los datos
data = pd.read_csv("Carseats.csv", header=0)


# Creo la nueva variable 'High'
def newVariable(row):
    if row["Sales"] >= 8:
        # return 1
        return "Yes"
    else:
        # return 0
        return "No"


# data['High'] = newVariable(data['Sales'])
data["High"] = data.apply(lambda row: newVariable(row), axis=1)
data.replace(("Yes", "No"), (1, 0), inplace=True)
data.replace(("Good", "Medium", "Bad"), (2, 1, 0), inplace=True)

# Spliteo los datos
train, test = train_test_split(data,
                               test_size=0.3,
                               random_state=seed,
                               stratify=data["High"])
# train, val = train_test_split(train, test_size=0.2, stratify=train['High'])

# ------------------------------
# b
# ------------------------------
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
# c
# ------------------------------
x_train, y_train = train.drop(["High", "Sales"], axis=1), train["Sales"]
x_test, y_test = test.drop(["High", "Sales"], axis=1), test["Sales"]
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

# ------------------------------
# d
# ------------------------------

# print("Precision de Clasificador")
# print("Train:", treeClassifier.score(x_train,y_train))
# print("Test:",treeClassifier.score(x_test,y_test))

# print("Precision de Clasificador")
# print("Train:", treeRegressor.score(x_train,y_train))
# print("Test:", treeRegressor.score(x_test,y_test))

# predict_train = treeRegressor.predict(x_train)
# predict_test = treeRegressor.predict(x_test)

# ------------------------------
# e
# ------------------------------

# clf = tree.DecisionTreeClassifier()
# path = clf.cost_complexity_pruning_path(x_train, y_train)
# ccp_alphas, impurities = path.ccp_alphas, path.impurities

# fig, ax = plt.subplots()
# ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
# ax.set_xlabel("effective alpha")
# ax.set_ylabel("total impurity of leaves")
# ax.set_title("Total Impurity vs effective alpha for training set")
# # plt.show()

from sklearn.model_selection import GridSearchCV

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

# ------------------------------
# f
# ------------------------------

from sklearn.model_selection import GridSearchCV

treeRegressor = tree.DecisionTreeRegressor()
ensembleBagging = ensemble.BaggingRegressor(treeRegressor)

parameters = {
    "n_estimators": np.arange(10, 100, 5),
    "max_samples": np.arange(1, 20, 1),
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

# ------------------------------
# g
# ------------------------------

# ------------------------------
# h
# ------------------------------
