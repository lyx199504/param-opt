#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/3 22:08
# @Author : LYX-夜光
import paddle
from paddle import nn
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from utils import set_seed
from utils.dataUtil import stratified_shuffle_split


def paddle_set_seed(seed):
    if seed:
        set_seed(seed)
        paddle.seed(seed)


class PaddleClassifier(paddle.nn.Layer, BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0):
        super().__init__()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state

    # numpy => tensor
    def to_tensor(self, data):
        dataType = 'float32' if data.dtype == 'float' else 'int64'
        return paddle.to_tensor(data).astype(dataType)

    def create_model(self, X, y):
        self.fc1 = nn.Linear(in_features=X.shape[1], out_features=len(set(y)))
        self.fc2 = nn.Linear(in_features=len(set(y)), out_features=len(set(y)))
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, X):
        X = self.to_tensor(X)
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        return y

    def fit(self, X, y):
        paddle_set_seed(self.random_state)
        # 构建模型
        self.create_model(X, y)
        # 定义优化器，损失函数
        optimizer = paddle.optimizer.Adam(parameters=self.parameters(), learning_rate=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        # 训练
        self.train()  # 训练开启
        y = self.to_tensor(y)
        for epoch in range(self.epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch, y_batch = X[i:i+self.batch_size], y[i:i+self.batch_size]
                y_pred_batch = self.forward(X_batch)
                loss = loss_fn(y_pred_batch, y_batch)

                loss.backward()  # 梯度计算
                optimizer.step()  # 优化更新权值
                optimizer.clear_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

    def predict(self, X):
        y_prob = self.predict_proba(X)
        return y_prob.argmax(1)

    def predict_proba(self, X):
        return self.forward(X).cpu().detach().numpy()

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y_pred, y)


if __name__ == "__main__":
    # 用鸢尾花数据集测试
    from sklearn.datasets import load_iris
    from utils.trainUtil import bayes_search_train
    X, y = load_iris()['data'], load_iris()['target']
    from utils import config
    n_splits = config['bys_param']['fold']
    X, y = stratified_shuffle_split(X, y, n_splits=n_splits, random_state=config['cus_param']['seed'])

    model = PaddleClassifier()
    from utils import config
    for model_name, model_param in config['model']:
        if model_name == 'base_dl':
            bayes_search_train(X, y, model_name, model_param, model=model)

    # model_name = "dl_test_1622803291"
    # para = paddle.load('./model/%s.model' % model_name)
    # model = PaddleClassifier(**{'batch_size': 200, 'epochs': 98, 'learning_rate': 0.06179521860151487, 'random_state': 100})
    # model.create_model(X, y)
    # model.set_state_dict(para)
    # print(model.score(X, y))
