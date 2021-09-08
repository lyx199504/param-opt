#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 15:39
# @Author : LYX-夜光

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from example_dl_model import RNNClassifier
from optUtils.pytorchModel import PytorchClassifier


# 选择模型
def model_selection(model_name, **params):
    if model_name == 'svc':
        return SVC(**params)
    if model_name == 'lr':
        return LogisticRegression(**params)
    if model_name == 'rf_clf':
        return RandomForestClassifier(**params)
    if model_name == 'voting':
        return VotingClassifier(**params)
    if model_name == 'base_dl':
        return PytorchClassifier(**params)
    if model_name == 'rnn':
        return RNNClassifier(**params)
    return None

