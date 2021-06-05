#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/4 19:08
# @Author : LYX-夜光
import time

import numpy as np
import torch
from torch import nn

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from utils import set_seed

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

        self._estimator_type = "classifier"

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
    def fit(self, X, y):
        # 设置随机种子
        pytorch_set_seed(self.random_state)
        # 构建模型
        self.create_model(X.shape[1], len(set(y)))
        # 定义优化器，损失函数
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        # 训练
        self.train()  # 训练模式
        self.to(self.device)
        X, y = self.to_tensor(X), self.to_tensor(y)
        indexList = list(range(0, X.shape[0], self.batch_size))
        for epoch in range(self.epochs):
            startTime = time.time()
            total_loss = 0
            for i in indexList:
                X_batch, y_batch = X[i:i+self.batch_size], y[i:i+self.batch_size]
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_pred_batch = self.forward(X_batch)

                loss = loss_fn(y_pred_batch, y_batch)
                total_loss += loss.item()

                loss.backward()  # 梯度计算
                optimizer.step()  # 优化更新权值
                optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

            runTime = time.time() - startTime
            print("epoch: %d/%d - loss: %.6f - time: %ds" % (epoch+1, self.epochs, total_loss/len(indexList), int(runTime)))

    # 预测分类标签
    def predict(self, X):
        y_prob = self.predict_proba(X)
        return y_prob.argmax(1)

    # 预测分类概率
    def predict_proba(self, X):
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

    # 评价指标，精确度：accuracy
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)
