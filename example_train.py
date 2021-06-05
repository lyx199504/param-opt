#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/5 15:23
# @Author : LYX-夜光

from sklearn.datasets import load_iris

from utils import yaml_config
from utils.dataUtil import stratified_shuffle_split
from utils.logUtil import get_param_from_log
from utils.modelUtil import model_selection
from utils.trainUtil import bayes_search_train

if __name__ == "__main__":
    # 用鸢尾花数据集测试
    X, y = load_iris()['data'], load_iris()['target']
    n_splits, seed = yaml_config['bys_param']['fold'], yaml_config['cus_param']['seed']
    X, y = stratified_shuffle_split(X, y, n_splits=n_splits, random_state=seed)

    # # 单个模型
    # for model_name, model_param in yaml_config['model']:
    #     bayes_search_train(X, y, model_name, model_param)

    # 融合多个模型
    for multi_model_name, multi_model_param in yaml_config['multi-model']:
        model_path_list = [
            # 'lr-1622908750',
            'svc-1622908751',
            # 'rf_clf-1622908755',
            'base_dl-1622908969',
            # 'rnn-1622910839'
        ]

        estimators = []
        for model_path in model_path_list:
            model_name = model_path.split('-')[0]
            param = get_param_from_log(model_name, model_path)
            if not param:
                print("没有找到对应的分类器，程序退出...")
                exit()
            model = model_selection(model_name, **param)
            if model_name == "svc":
                model.set_params(probability=True)
            print("分类器[%s]的参数：%s" % (model_name, param))
            estimators.append((model_name, model))

        multi_model = model_selection(multi_model_name, **{'estimators': estimators})
        yaml_config['bys_param']['n_iter'] = 1
        bayes_search_train(X, y, multi_model_name, multi_model_param, model=multi_model)
