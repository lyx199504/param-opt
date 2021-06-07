#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/3 22:08
# @Author : LYX-夜光
import time

import numpy as np

import paddle
from paddle import nn
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from utils import set_seed, yaml_config, make_dir
from utils.logUtil import logging_config

# paddle随机种子
def paddle_set_seed(seed):
    if seed:
        set_seed(seed)
        paddle.seed(seed)


class PaddleClassifier(nn.Layer, BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0):
        super().__init__()
        self.model_name = "base_dl"
        self._estimator_type = "classifier"
        self.param_search = True
        self.logger = None

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    # numpy => tensor
    def to_tensor(self, data):
        dataType = 'float32' if data.dtype == 'float' else 'int64'
        return paddle.to_tensor(data).astype(dataType)

    # 组网
    def create_model(self, in_features, out_features):
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    # 前向推理
    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        return y

    # 训练
    def fit(self, X, y, X_val=None, y_val=None):
        # 设置随机种子
        paddle_set_seed(self.random_state)
        # 构建模型
        self.create_model(X.shape[1], len(set(y)))
        # 定义优化器，损失函数
        self.optimizer = paddle.optimizer.Adam(parameters=self.parameters(), learning_rate=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        # 初始化训练集
        X, y = self.to_tensor(X), self.to_tensor(y)
        train_index = list(range(0, X.shape[0], self.batch_size))
        # 若不进行超参数搜索，则初始化验证集
        val_index = []
        if not self.param_search and X_val is not None and y_val is not None:
            X_val, y_val = self.to_tensor(X_val), self.to_tensor(y_val)
            val_index = list(range(0, X_val.shape[0], self.batch_size))

        for epoch in range(self.epochs):
            start_time = time.time()

            train_loss, train_score = self.fit_step(X, y, train_index, train=True)
            massage = "epoch: %d/%d - train_loss: %.6f - train_score: %.6f" % (
                epoch + 1, self.epochs, train_loss, train_score)

            # 有输入验证集，则计算val_loss和val_score
            val_score = 0
            if val_index:
                val_loss, val_score = self.fit_step(X_val, y_val, val_index, train=False)
                massage += " - val_loss: %.6f - val_score: %.6f" % (val_loss, val_score)

            run_time = time.time() - start_time
            print(massage + " - time: %ds" % int(run_time))

            # 不进行超参数搜索，则存储每个epoch的模型和日志
            if not self.param_search:
                # 存储模型
                model_dir = yaml_config['dir']['model_dir']
                make_dir(model_dir)
                model_path = model_dir + '/%s-%03d-%s.model' % (self.model_name, epoch + 1, int(time.time()))
                paddle.save(self.state_dict(), model_path)
                # 存储日志
                log_dir = yaml_config['dir']['log_dir']
                make_dir(log_dir)
                if self.logger is None:
                    self.logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
                self.logger.info({
                    "epoch": epoch + 1,
                    "best_param_": self.get_params(),
                    "best_score_": val_score,
                    "train_score": train_score,
                    "model_path": model_path,
                })

    # 拟合步骤
    def fit_step(self, X, y, indexList, train):
        if train:
            self.train()  # 训练模式
        else:
            self.eval()  # 求值模式

        total_loss, y_prob = 0, []
        for i in indexList:
            X_batch = X[i:i + self.batch_size]
            y_batch = y[i:i + self.batch_size]
            y_prob_batch = self.forward(X_batch)

            loss = self.loss_fn(y_prob_batch, y_batch)
            total_loss += float(loss)

            y_prob.append(y_prob_batch.cpu().detach().numpy())

            if train:
                loss.backward()  # 梯度计算
                self.optimizer.step()  # 优化更新权值
                self.optimizer.clear_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

        mean_loss = total_loss / len(indexList)
        score = self.score(X, y.numpy(), np.vstack(y_prob))

        return mean_loss, score

    # 预测分类概率
    def predict_proba(self, X):
        self.eval()  # 求值模式
        self.to(self.device)
        X = self.to_tensor(X).to(self.device)
        batch_size = 20000
        y_prob = []
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_prob_batch = self.forward(X_batch).cpu().detach().numpy()
            y_prob.append(y_prob_batch)
        y_prob = np.vstack(y_prob)
        return y_prob

    # 预测分类标签
    def predict(self, X, y_prob=None):
        if y_prob is None:
            y_prob = self.predict_proba(X)
        return y_prob.argmax(1)

    # 评价指标，精确度：accuracy
    def score(self, X, y, y_prob=None):
        y_pred = self.predict(X, y_prob)
        return accuracy_score(y_pred, y)
