#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/3 22:01
# @Author : LYX-夜光
import time

import numpy as np

import joblib
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from skopt import BayesSearchCV

from optUtils import make_dir, yaml_config
from optUtils.logUtil import logging_config
from optUtils.modelUtil import model_selection

# 交叉验证
def cv_train(X, y, model_name, model_param=None, model=None, X_test=None, y_test=None):
    """
    :param X: 训练集的特征
    :param y: 训练集的标签
    :param model_name: 模型名称
    :param model_param: 模型参数
    :param model: 机器学习或深度学习模型，可缺省，默认根据模型名称获取模型
    :param X_test: 测试集的特征，可缺省
    :param y_test: 测试集的标签，可缺省
    :return: model，模型文件
    """

    model_dir, log_dir = yaml_config['dir']['model_dir'], yaml_config['dir']['log_dir']
    cus_param, cv_param = yaml_config['cus_param'], yaml_config['cv_param']

    if model is None:
        model = model_selection(model_name)
    if model_param is None:
        model_param = {}
    else:
        model.set_params(**model_param)

    def cv_score(model, X, y, scoring):
        if scoring == 'roc_auc':
            return roc_auc_score(y, model.predict_proba(X)[:, 1])
        if scoring == 'roc_auc_ovr':
            return roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
        return model.score(X, y)

    def get_score(train_index, val_index):
        start_time = time.time()
        model.fit(X[train_index], y[train_index])
        score = cv_score(model, X[val_index], y[val_index], scoring=cv_param['scoring'])
        run_time = int(time.time() - start_time)
        print("score: %.6f - time: %ds" % (score, run_time))
        return score

    print("参数设置：%s" % model_param)
    parallel = Parallel(n_jobs=cv_param['workers'], verbose=4)
    k_fold = KFold(n_splits=cv_param['fold'])
    score_list = parallel(
        delayed(get_score)(train, val) for train, val in k_fold.split(X, y))
    best_score = np.mean(score_list)

    model.fit(X, y)

    make_dir(model_dir)
    model_path = model_dir + '/%s-%s.model' % (model_name, int(time.time()))
    if 'device' in model.get_params():
        model.cpu()
        model.device = 'cpu'
    joblib.dump(model, model_path)

    # 配置日志文件
    make_dir(log_dir)
    logger = logging_config(model_name, log_dir + '/%s.log' % model_name)
    log_message = {
        "cus_param": cus_param,
        "cv_param": cv_param,
        "best_param_": model_param,
        "best_score_": best_score,
        "train_score": cv_score(model, X, y, scoring=cv_param['scoring']),
        "model_path": model_path,
    }
    if X_test and y_test:
        log_message.update({"test_score": model.score(X_test, y_test)})
    logger.info(log_message)

    return model


# 贝叶斯搜索
def bayes_search_train(X, y, model_name, model_param, model=None, X_test=None, y_test=None):
    """
    :param X: 训练集的特征
    :param y: 训练集的标签
    :param model_name: 模型名称
    :param model_param: 模型参数
    :param model: 机器学习或深度学习模型，可缺省，默认根据模型名称获取模型
    :param X_test: 测试集的特征，可缺省
    :param y_test: 测试集的标签，可缺省
    :return: 无，输出模型文件和结果日志
    """

    model_dir, log_dir = yaml_config['dir']['model_dir'], yaml_config['dir']['log_dir']
    cus_param, bys_param = yaml_config['cus_param'], yaml_config['bys_param']

    if not model:
        model = model_selection(model_name)

    # 将训练集分为cv折，进行cv次训练得到交叉验证分数均值，最后再训练整个训练集
    bys = BayesSearchCV(
        model,
        model_param,
        n_iter=bys_param['n_iter'],
        cv=bys_param['fold'],
        verbose=4,
        n_jobs=bys_param['workers'],
        random_state=cus_param['seed'],
    )

    bys.fit(X, y)

    make_dir(model_dir)
    model_path = model_dir + '/%s-%s.model' % (model_name, int(time.time()))
    if 'device' in bys.best_estimator_.get_params():
        bys.best_estimator_.cpu()
        bys.best_estimator_.device = 'cpu'
    model = bys.best_estimator_
    joblib.dump(model, model_path)

    # 配置日志文件
    make_dir(log_dir)
    logger = logging_config(model_name, log_dir + '/%s.log' % model_name)
    log_message = {
        "cus_param": cus_param,
        "bys_param": bys_param,
        "best_param_": dict(bys.best_params_),
        "best_score_": bys.best_score_,
        "train_score": bys.score(X, y),
        "model_path": model_path,
    }

    # 如果有测试集，则计算测试集分数
    if X_test and y_test:
        log_message.update({"test_score": bys.score(X_test, y_test)})
    logger.info(log_message)

    return model
