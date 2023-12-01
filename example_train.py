#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/9/30 11:46
# @Author : LYX-夜光
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score

from example_dl_model import FusionClassifier
from optUtils import yaml_config
from optUtils.dataUtil import stratified_shuffle_samples
from optUtils.metricsUtil import f1_micro_score, f1_macro_score
from optUtils.pytorchModel import DeepLearningClassifier, DeepLearningRegressor, SupervisedAutoEncoder
from optUtils.trainUtil import ml_train

if __name__ == "__main__":
    """
    本文件是使用模型进行常规训练的demo
    """
    fold, seed = yaml_config['cv_param']['fold'], yaml_config['cus_param']['seed']
    # 使用鸢尾花数据集
    X, y = load_iris()['data'], load_iris()['target']
    # 数据按折数分层排列
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)
    train_point = int(len(X) / fold)

    # 机器学习训练演示
    model_name_list = ['lr_clf', 'svm_clf', 'rf_clf']
    metrics_list = [f1_micro_score, f1_macro_score]  # 添加多个评价指标
    for model_name in model_name_list:
        model_param = {'random_state': seed}
        # 机器学习常规训练
        ml_train(X[train_point:], y[train_point:], X[:train_point], y[:train_point], model_name, model_param, metrics_list)

    # 深度学习分类器训练演示
    model = DeepLearningClassifier(learning_rate=0.01, epochs=200, batch_size=150, random_state=seed)
    model.param_search = False  # 常规训练时将搜索参数模式关闭
    # model.device = 'cuda'  # 使用GPU训练，默认使用CPU训练
    # model.save_model = True  # 常规训练时，可开启保存模型功能
    model.only_save_last_epoch = True  # 常规训练时，可开启仅保存最后一个epoch的功能
    model.shuffle_every_epoch = True  # 将每一个epoch的数据进行打乱
    # model.watch_epoch = True  # 观察每个epoch中的训练
    model.metrics_list = [f1_micro_score, f1_macro_score]  # 添加多个评价指标
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])

    # 深度学习回归器训练演示
    model = DeepLearningRegressor(learning_rate=0.01, epochs=100, batch_size=150, random_state=seed)
    model.param_search = False
    model.only_save_last_epoch = True
    y_ = y/2  # 修改标签
    model.fit(X[train_point:], y_[train_point:], X[:train_point], y_[:train_point])

    # 监督自编码器训练演示
    model = SupervisedAutoEncoder(learning_rate=0.03, epochs=100, batch_size=150, random_state=seed)
    model.param_search = False
    model.only_save_last_epoch = True
    y[y == 2] = 1  # 把第2类转为第1类，变成二分类
    model.metrics = roc_auc_score  # 主评价指标
    model.metrics_list = [f1_score, accuracy_score, precision_score, recall_score]  # 添加多个评价指标
    model.fit(X[train_point:], y[train_point:], X[:train_point], y[:train_point])

    # 使用手写数字数据集
    X, y = load_digits()['data'], load_digits()['target']
    # 数据按折数分层排列
    X = [X, X.reshape(-1, 8, 8)]  # 融合一维数据和二维数据
    X, y = stratified_shuffle_samples(X, y, n_splits=fold, random_state=seed)
    train_point = int(len(X) / fold) if type(X) != list else int(len(X[0]) / fold)

    # 融合分类器训练演示
    model = FusionClassifier(learning_rate=0.005, epochs=100, batch_size=200, random_state=seed)
    model.param_search = False
    model.only_save_last_epoch = True
    model.metrics_list = [f1_micro_score, f1_macro_score]  # 添加多个评价指标
    model.fit([x[train_point:] for x in X], y[train_point:], [x[:train_point] for x in X], y[:train_point])
