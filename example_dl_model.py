#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/3/31 18:43
# @Author : LYX-夜光
from torch import nn

from optUtils.pytorchModel import DLClassifier

"""
本文件是模型构建的demo
"""

class RNNClassifier(DLClassifier):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=50, random_state=0, device='cpu'):
        super().__init__(learning_rate, epochs, batch_size, random_state, device)
        self.model_name = "rnn_clf"

    def create_model(self):
        hidden_size = 8
        self.rnn = nn.RNN(input_size=1, hidden_size=hidden_size, dropout=0.2)
        self.fc = nn.Linear(in_features=hidden_size, out_features=self.label_num)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, X):
        X = X.T.unsqueeze(-1)  # [time_step, batch_size, vector_size]
        # X => rnn
        _, hn = self.rnn(X)  # hn: [1, batch_size, hidden_size]
        y = hn.squeeze(0)  # [batch_size, hidden_size]
        # y = torch.cat(list(hn), dim=1)  # [batch_size, hidden_size]
        # rnn => fc
        y = self.relu(self.fc(y))
        y = self.softmax(y)
        # 返回分类结果
        return y
