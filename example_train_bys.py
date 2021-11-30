#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/5 15:23
# @Author : LYX-夜光

from sklearn.datasets import load_iris

from example_dl_model import RNNClassifier
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_split
from optUtils.logUtil import get_best_param
from optUtils.modelUtil import model_selection, model_registration
from optUtils.trainUtil import bayes_search_train

if __name__ == "__main__":
    """
    本文件是使用贝叶斯搜索寻找模型参数的demo
    """
    n_splits, seed = yaml_config['bys_param']['fold'], yaml_config['cus_param']['seed']
    # 使用鸢尾花数据集
    X, y = load_iris()['data'], load_iris()['target']
    # 数据按折数分层排列
    X, y = stratified_shuffle_split(X, y, n_splits=n_splits, random_state=seed)

    # 注册自己构造的模型
    model_registration(
        rnn_clf=RNNClassifier,
    )

    # 训练配置文件[param.yaml]中定义的model
    for model_name, model_param in yaml_config['model']:
        bayes_search_train(X, y, model_name, model_param)

    # 训练配置文件[param.yaml]中定义的multi-model
    for multi_model_name, multi_model_param in yaml_config['multi-model']:
        # 融合svm_clf和dl_clf模型
        estimators = []
        for model_name in ['svm_clf', 'dl_clf']:
            # 获取分数最高的基模型参数
            params = get_best_param(model_name)
            param = params['best_param_']
            model = model_selection(model_name, **param)
            if model_name == "svm_clf":
                model.set_params(probability=True)
            print("分类器[%s]的参数：%s" % (model_name, param))
            estimators.append((model_name, model))

        multi_model = model_selection(multi_model_name, **{'estimators': estimators})
        # 由于这里使用的voting没必要搜索参数，故将迭代次数修改为1次
        yaml_config['bys_param']['n_iter'] = 1
        bayes_search_train(X, y, multi_model_name, multi_model_param, model=multi_model)
