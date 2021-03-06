#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/30 23:03
# @Author : LYX-夜光
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from example_dl_model import RNNClassifier
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_samples
from optUtils.metricsUtil import f1_micro_score, f1_macro_score, f1_weighted_score
from optUtils.modelUtil import model_registration
from optUtils.trainUtil import cv_train

if __name__ == "__main__":
    """
    本文件是使用模型进行N折交叉验证的demo
    """
    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']
    # 使用鸢尾花数据集
    X, y = load_iris()['data'], load_iris()['target']
    # 数据按折数分层排列
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)

    # 注册自己构造的模型
    model_registration(
        rnn_clf=RNNClassifier,
    )

    # 交叉验证
    model_name_list = ['svm_clf', 'rnn_clf']
    metrics_list = [accuracy_score, f1_micro_score, f1_macro_score, f1_weighted_score]
    for model_name in model_name_list:
        cv_train(X, y, model_name, model_param={'random_state': seed}, metrics_list=metrics_list)
