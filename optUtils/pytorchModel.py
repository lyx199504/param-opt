#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/4 19:08
# @Author : LYX-夜光
import time

import joblib
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error

from optUtils import set_seed, make_dir, yaml_config
from optUtils.logUtil import logging_config

# pytorch随机种子
def pytorch_set_seed(seed):
    if seed:
        set_seed(seed)
        torch.manual_seed(seed)  # cpu
        torch.cuda.manual_seed(seed)  # gpu
        torch.cuda.manual_seed_all(seed)  # 并行gpu
        torch.backends.cudnn.deterministic = True  # cpu/gpu结果一致

# pytorch深度学习模型
class PytorchModel(nn.Module, BaseEstimator):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__()
        self.model_name = "base_dl"
        self.param_search = True  # 默认开启搜索参数功能
        self.save_model = False  # 常规训练中，默认关闭保存模型功能

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.device = device

        # 优化器、评价指标
        self.optim = torch.optim.Adam
        self.metrics = None
        self.metrics_list = []  # 多个评价指标

    # numpy => tensor
    def to_tensor(self, data):
        if type(data).__name__ == 'Tensor':
            return data
        dataType = torch.float32 if 'float' in str(data.dtype) else torch.int64
        return torch.tensor(data, dtype=dataType)

    # 训练
    def fit(self, X, y, X_val=None, y_val=None):
        # 设置随机种子
        pytorch_set_seed(self.random_state)
        # 构建模型
        self.create_model()
        # 初始化优化器
        self.optimizer = self.optim(params=self.parameters(), lr=self.learning_rate)
        # 初始化训练集
        X, y = self.to_tensor(X), self.to_tensor(y)
        # 若不进行超参数搜索，则初始化验证集
        if not self.param_search and X_val is not None and y_val is not None:
            X_val, y_val = self.to_tensor(X_val), self.to_tensor(y_val)

        # 训练每个epoch
        for epoch in range(self.epochs):
            start_time = time.time()

            self.to(self.device)
            train_loss, train_score, train_score_list = self.fit_epoch(X, y, train=True)
            train_score_dict = {self.metrics.__name__: train_score}
            for i, metrics in enumerate(self.metrics_list):
                train_score_dict.update({metrics.__name__: train_score_list[i]})
            massage = "epoch: %d/%d - train_loss: %.6f - train_score: %.6f" % (
                epoch + 1, self.epochs, train_loss, train_score)

            # 有输入验证集，则计算val_loss和val_score等
            val_score, val_score_dict = 0, {}
            if not self.param_search and X_val is not None and y_val is not None:
                val_loss, val_score, val_score_list = self.fit_epoch(X_val, y_val, train=False)
                val_score_dict = {self.metrics.__name__: val_score}
                for i, metrics in enumerate(self.metrics_list):
                    val_score_dict.update({metrics.__name__: val_score_list[i]})
                massage += " - val_loss: %.6f - val_score: %.6f" % (val_loss, val_score)

            run_time = time.time() - start_time
            print(massage + " - time: %ds" % int(run_time))

            # 不进行超参数搜索，则存储每个epoch的模型和日志
            if not self.param_search:
                # 存储模型
                model_path = None
                if self.save_model:
                    model_dir = yaml_config['dir']['model_dir']
                    make_dir(model_dir)
                    model_path = model_dir + '/%s-%03d-%s.model' % (self.model_name, epoch + 1, int(time.time()))
                    self.to('cpu')
                    joblib.dump(self, model_path)
                # 存储日志
                log_dir = yaml_config['dir']['log_dir']
                make_dir(log_dir)
                logger = logging_config(self.model_name, log_dir + '/%s.log' % self.model_name)
                logger.info({
                    "epoch": epoch + 1,
                    "best_param_": self.get_params(),
                    "best_score_": val_score,
                    "train_score": train_score,
                    "train_score_list": train_score_dict,
                    "val_score_list": val_score_dict,
                    "model_path": model_path,
                })

    # 每轮拟合
    def fit_epoch(self, X, y, train):
        mean_loss, y_hat = self.fit_step(X, y, train)

        y_numpy = y.cpu().detach().numpy()
        y_hat_numpy = np.hstack(y_hat) if len(y_hat[0].shape) == 1 else np.vstack(y_hat)
        score = self.score(X, y_numpy, y_hat_numpy)
        score_list = self.score_list(X, y_numpy, y_hat_numpy)

        return mean_loss, score, score_list

    # 拟合步骤
    def fit_step(self, X, y=None, train=True):
        self.train() if train else self.eval()

        total_loss, y_hat = 0, []
        indexList = range(0, X.shape[0], self.batch_size)
        for i in indexList:
            X_batch = X[i:i + self.batch_size].to(self.device)
            y_batch = X_batch if y is None else y[i:i + self.batch_size].to(self.device)
            output = self.forward(X_batch)

            loss = self.loss_fn(output, y_batch)
            total_loss += loss.item()

            y_hat_batch = output[0] if type(output) == tuple else output
            y_hat.append(y_hat_batch.cpu().detach().numpy())

            if train:
                loss.backward()  # 梯度计算
                self.optimizer.step()  # 优化更新权值
                self.optimizer.zero_grad()  # 求解梯度前需要清空之前的梯度结果（因为model会累加梯度）

        mean_loss = total_loss / len(indexList)

        return mean_loss, y_hat

    # 预测结果
    def predict_output(self, X, batch_size):
        self.eval()  # 求值模式
        self.to(self.device)
        X = self.to_tensor(X).to(self.device)
        y_hat = []
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i + batch_size]
            output = self.forward(X_batch)
            y_hat_batch = output[0] if type(output) == tuple else output
            y_hat_batch = y_hat_batch.cpu().detach().numpy()
            y_hat.append(y_hat_batch)
        y_hat = np.hstack(y_hat) if len(y_hat[0].shape) == 1 else np.vstack(y_hat)
        return y_hat

# 深度学习分类器
class DLClassifier(PytorchModel):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "dl_clf"
        self._estimator_type = "classifier"
        self.label_num = 2  # 默认二分类

        self.metrics = accuracy_score

    # 组网
    def create_model(self):
        self.fc1 = nn.Linear(in_features=4, out_features=3)
        self.fc2 = nn.Linear(in_features=3, out_features=self.label_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    # 损失函数
    def loss_fn(self, output, y_true):
        y_hat = output[0] if type(output) == tuple else output
        return F.cross_entropy(y_hat, y_true)

    # 前向推理
    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.softmax(y)
        return y

    def fit(self, X, y, X_val=None, y_val=None):
        self.label_num = len(set(y))
        super().fit(X, y, X_val, y_val)

    # 预测分类概率
    def predict_proba(self, X, batch_size=10000):
        y_prob = super().predict_output(X, batch_size)
        return y_prob

    # 预测分类标签
    def predict(self, X, y_prob=None, batch_size=10000):
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        return y_prob.argmax(axis=1)

    # 评价指标
    def score(self, X, y, y_prob=None):
        if y_prob is None:
            y_prob = self.predict_proba(X)
        y_pred = self.predict(X, y_prob)
        if self.label_num == 2 and 'auc' in self.metrics.__name__:
            return self.metrics(y, y_prob[:, 1]) if len(y_prob.shape) > 1 else self.metrics(y, y_prob)
        return self.metrics(y, y_pred)

    # 评价指标列表
    def score_list(self, X, y, y_prob=None):
        score_list = []
        if y_prob is None:
            y_prob = self.predict_proba(X)
        y_pred = self.predict(X, y_prob)
        for metrics in self.metrics_list:
            if self.label_num == 2 and 'auc' in metrics.__name__:
                score = metrics(y, y_prob[:, 1]) if len(y_prob.shape) > 1 else metrics(y, y_prob)
            else:
                score = metrics(y, y_pred)
            score_list.append(score)
        return score_list

# 深度学习回归器
class DLRegressor(PytorchModel):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "dl_reg"
        self._estimator_type = "regressor"

        self.metrics = mean_squared_error

    # 组网
    def create_model(self):
        self.fc1 = nn.Linear(in_features=4, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # 损失函数
    def loss_fn(self, output, y_true):
        y_hat = output[0] if type(output) == tuple else output
        return F.mse_loss(y_hat, y_true)

    # 前向推理
    def forward(self, X):
        y = self.fc1(X)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.squeeze(-1)
        return y

    # 预测标签
    def predict(self, X, batch_size=10000):
        y_pred = super().predict_output(X, batch_size)
        return y_pred

    # 评价指标
    def score(self, X, y, y_pred=None):
        if y_pred is None:
            y_pred = self.predict(X)
        return self.metrics(y, y_pred)

    # 评价指标列表
    def score_list(self, X, y, y_pred=None):
        score_list = []
        if y_pred is None:
            y_pred = self.predict(X)
        for metrics in self.metrics_list:
            score = metrics(y, y_pred)
            score_list.append(score)
        return score_list

# 自编码器
class AutoEncoder(DLClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "ae"
        self.alpha = 0  # 异常阈值

    # 组网
    def create_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=1),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=4)
        )

    # 损失函数
    def loss_fn(self, output, y_true):
        y_hat = output[0] if type(output) == tuple else output
        return F.mse_loss(y_hat, y_true)

    # 前向推理
    def forward(self, X):
        Z = self.encoder(X)
        X_hat = self.decoder(Z)
        return X_hat

    # 每轮拟合
    def fit_epoch(self, X, y, train):
        y_numpy = y.cpu().detach().numpy()
        if train:
            self.fit_step(X[y_numpy == 1], train=True)  # 只训练正常数据

        mean_loss, X_hat = self.fit_step(X, train=False)  # 不进行训练
        X_numpy, X_hat_numpy = X.cpu().detach().numpy(), np.vstack(X_hat)
        if train:
            self.alpha = self.get_alpha(X_numpy[y_numpy == 1], X_hat_numpy[y_numpy == 1])

        y_prob = self.get_proba_score(X_numpy, X_hat_numpy)  # y_pred取1的概率

        score = self.score(X, y_numpy, y_prob)
        score_list = self.score_list(X, y_numpy, y_prob)

        return mean_loss, score, score_list

    # 预测得分
    def get_proba_score(self, X, X_hat):
        # # 二范数，同np.sqrt(np.sum((X - X_hat) ** 2, axis=1))
        # errors = np.linalg.norm(X - X_hat, axis=1, ord=2)
        errors = np.sum(np.abs(X - X_hat), axis=1)
        scores = 1 / (errors + 1)  # 根据误差计算得分
        return scores

    # 阈值
    def get_alpha(self, X, X_hat):
        return min(self.get_proba_score(X, X_hat))

    # 预测概率
    def predict_proba(self, X, batch_size=10000):
        X_hat = super().predict_proba(X, batch_size)
        if type(X).__name__ == 'Tensor':
            X = X.cpu().detach().numpy()
        y_prob = self.get_proba_score(X, X_hat)
        return y_prob

    # 预测标签
    def predict(self, X, y_prob=None, batch_size=10000):
        if y_prob is None:
            y_prob = self.predict_proba(X, batch_size)
        y_pred = np.array([0 if score < self.alpha else 1 for score in y_prob])
        return y_pred

# 变分自编码
class VariationalAutoEncoder(AutoEncoder):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "vae"

    # 组网
    def create_model(self):
        self.encoder = nn.Sequential(
            nn.Linear(in_features=4, out_features=2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=1, out_features=4),
            nn.Sigmoid(),
        )

    # 损失函数
    def loss_fn(self, output, y_true):
        y_hat, mu, log_sigma = output
        BCE = F.binary_cross_entropy(y_hat, y_true, reduction='sum')
        D_KL = 0.5 * torch.sum(torch.exp(log_sigma) + torch.pow(mu, 2) - 1. - log_sigma)
        loss = BCE + D_KL
        return loss

    # 前向推理
    def forward(self, X):
        H = self.encoder(X)
        mu, log_sigma = H.chunk(2, dim=-1)
        Z = self.reparameterize(mu, log_sigma)
        X_hat = self.decoder(Z)
        return X_hat, mu, log_sigma

    # 重构Z层：均值+随机采样*标准差
    def reparameterize(self, mu, log_sigma):
        std = torch.exp(log_sigma * 0.5)
        esp = torch.randn(std.size())
        z = mu + esp * std
        return z
