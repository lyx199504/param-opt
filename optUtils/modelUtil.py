#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 15:39
# @Author : LYX-夜光

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from optUtils.pytorchModel import DLClassifier, DLRegressor, AE

model_dict = {
    'knn': KNeighborsClassifier,
    'svc': SVC,
    'lr': LogisticRegression,
    'dt': DecisionTreeClassifier,
    'rf_clf': RandomForestClassifier,
    'voting': VotingClassifier,
    'dl_clf': DLClassifier,
    'dl_reg': DLRegressor,
    'ae': AE,
}

# 选择模型
def model_selection(model_name, **params):
    model = model_dict[model_name](**params)
    return model

