#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/30 11:46
# @Author : LYX-夜光
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score

from example_dl_model import RNNClassifier
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_split
from optUtils.pytorchModel import DLRegressor

if __name__ == "__main__":
    """
    本文件是使用模型进行常规N折交叉验证的demo
    """
    # 使用鸢尾花数据集
    X, y = load_iris()['data'], load_iris()['target']
    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']
    # 数据按折数分层排列
    X, y = stratified_shuffle_split(X, y, n_splits=fold, random_state=seed)
    train_point = int(len(X) / fold)

    # 分类器训练演示
    model = RNNClassifier(learning_rate=0.011, epochs=100, batch_size=150, random_state=seed)
    model.model_name += '_common'  # 修改模型名称
    model.param_search = False  # 常规训练时将搜索参数模式关闭
    # model.save_model = True  # 常规训练时，可开启保存模型功能
    model.metrics_list = [f1_score]  # 添加多个评价指标
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])

    # 回归器训练演示
    model = DLRegressor(learning_rate=0.01, epochs=100, batch_size=150, random_state=seed)
    model.model_name += '_common'
    model.param_search = False
    y = y/2  # 修改标签
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])
