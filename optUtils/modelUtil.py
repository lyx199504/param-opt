#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 15:39
# @Author : LYX-夜光

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from optUtils.pytorchModel import DLClassifier, DLRegressor, AutoEncoder, VariationalAutoEncoder

model_dict = {
    'knn_clf': KNeighborsClassifier,
    'knn_reg': KNeighborsRegressor,
    'svm_clf': SVC,
    'svm_reg': SVR,
    'lr_clf': LogisticRegression,
    'lr_reg': LinearRegression,
    'dt_clf': DecisionTreeClassifier,
    'dt_reg': DecisionTreeRegressor,
    'rf_clf': RandomForestClassifier,
    'rf_reg': RandomForestRegressor,
    'voting': VotingClassifier,
    'dl_clf': DLClassifier,
    'dl_reg': DLRegressor,
    'ae': AutoEncoder,
    'vae': VariationalAutoEncoder,
}

# 选择模型
def model_selection(model_name, **params):
    model = model_dict[model_name](**params)
    return model

