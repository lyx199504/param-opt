#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/4 19:08
# @Author : LYX-夜光
import time

import joblib
import numpy as np
import torch
from torch import nn

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error

from utils import set_seed, make_dir, yaml_config
from utils.logUtil import logging_config

# pytorch随机种子
def pytorch_set_seed(seed):
    if seed:
        set_seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed_all(seed)  # 并行gpu
        # torch.backends.cudnn.enabled = False
        # torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致
        # torch.backends.cudnn.benchmark = True  # 训练集变化不大时使训练加速


class PytorchClassifier(nn.Module, BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__()
        self.model_name = "base_dl"
        self._estimator_type = "classifier"
        self.param_search = True

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    # numpy => tensor
    def to_tensor(self, data):
        dataType = torch.float32 if data.dtype == 'float' else torch.int64
        return torch.tensor(data, dtype=dataType)

    # 组网
    def create_model(self, in_features, out_features):
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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
        pytorch_set_seed(self.random_state)
        # 构建模型
        self.create_model(X.shape[1], len(set(y)))
        self.to(self.device)
        # 定义优化器，损失函数
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
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
                epoch+1, self.epochs, train_loss, train_score)

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
                model_path = model_dir + '/%s-%03d-%s.model' % (self.model_name, epoch+1, int(time.time()))
                device = self.device
                self.device = 'cpu'
                self.to(self.device)
                joblib.dump(self, model_path)
                self.device = device
                self.to(self.device)
                # 存储日志
                log_dir = yaml_config['dir']['log_dir']
                make_dir(log_dir)
                logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
                logger.info({
                    "epoch": epoch+1,
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
            X_batch = X[i:i+self.batch_size].to(self.device)
            y_batch = y[i:i+self.batch_size].to(self.device)
            y_prob_batch = self.forward(X_batch)

            loss = self.loss_fn(y_prob_batch, y_batch)
            total_loss += loss.item()

            y_prob.append(y_prob_batch.cpu().detach().numpy())

            if train:
                loss.backward()  # 梯度计算
                self.optimizer.step()  # 优化更新权值
                self.optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

        mean_loss = total_loss/len(indexList)
        score = self.score(X, y.cpu().detach().numpy(), np.vstack(y_prob))

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


class PytorchRegressor(nn.Module, BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__()
        self.model_name = "base_dl"
        self._estimator_type = "regressor"
        self.param_search = True

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

    # numpy => tensor
    def to_tensor(self, data):
        dataType = torch.float32 if data.dtype == 'float' else torch.int64
        return torch.tensor(data, dtype=dataType)

    # 组网
    def create_model(self, in_features, out_features):
        if not self.param_search:
            print("设定超参数：", self.get_params())
        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

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
        pytorch_set_seed(self.random_state)
        # 构建模型
        self.create_model(X.shape[1], y.shape[1])
        self.to(self.device)
        # 定义优化器，损失函数
        self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
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

            train_loss = self.fit_step(X, y, train_index, train=True)
            massage = "epoch: %d/%d - train_loss: %.6f" % (
                epoch+1, self.epochs, train_loss)

            # 有输入验证集，则计算val_loss和val_score
            val_loss = 0
            if val_index:
                val_loss = self.fit_step(X_val, y_val, val_index, train=False)
                massage += " - val_loss: %.6f" % val_loss

            run_time = time.time() - start_time
            print(massage + " - time: %ds" % int(run_time))

            # 不进行超参数搜索，则存储每个epoch的模型和日志
            if not self.param_search:
                # 存储模型
                model_dir = yaml_config['dir']['model_dir']
                make_dir(model_dir)
                model_path = model_dir + '/%s-%03d-%s.model' % (self.model_name, epoch+1, int(time.time()))
                device = self.device
                self.device = 'cpu'
                self.to(self.device)
                joblib.dump(self, model_path)
                self.device = device
                self.to(self.device)
                # 存储日志
                log_dir = yaml_config['dir']['log_dir']
                make_dir(log_dir)
                logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
                logger.info({
                    "epoch": epoch+1,
                    "best_param_": self.get_params(),
                    "best_score_": val_loss,
                    "train_score": train_loss,
                    "model_path": model_path,
                })

    # 拟合步骤
    def fit_step(self, X, y, indexList, train):
        if train:
            self.train()  # 训练模式
        else:
            self.eval()  # 求值模式

        total_loss = 0
        for i in indexList:
            X_batch = X[i:i+self.batch_size].to(self.device)
            y_batch = y[i:i+self.batch_size].to(self.device)
            y_prob_batch = self.forward(X_batch)

            loss = self.loss_fn(y_prob_batch, y_batch)
            total_loss += loss.item()

            if train:
                loss.backward()  # 梯度计算
                self.optimizer.step()  # 优化更新权值
                self.optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

        mean_loss = total_loss/len(indexList)

        return mean_loss

    # 预测标签
    def predict(self, X):
        self.eval()  # 求值模式
        self.to(self.device)
        X = self.to_tensor(X).to(self.device)
        batch_size = 20000
        y_pred = []
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            y_pred_batch = self.forward(X_batch).cpu().detach().numpy()
            y_pred.append(y_pred_batch)
        y_pred = np.vstack(y_pred)
        return y_pred

    # 评价指标，损失
    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred)
