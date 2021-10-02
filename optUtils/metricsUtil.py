#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/10/2 12:04
# @Author : LYX-夜光
from sklearn.metrics import f1_score

# f1_micro_score评价指标
def f1_micro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='micro')

# f1_macro_score评价指标
def f1_macro_score(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro')