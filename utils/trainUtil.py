#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/3 22:01
# @Author : LYX-夜光
import time

import joblib
from skopt import BayesSearchCV

from utils import make_dir, yaml_config
from utils.logUtil import logging_config
from utils.modelUtil import model_selection


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
    if X_test and y_test:
        log_message.update({"test_score": bys.score(X_test, y_test)})
    logger.info(log_message)

    return model
