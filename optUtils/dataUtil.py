#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/4 18:09
# @Author : LYX-夜光

import numpy as np

from optUtils import set_seed


# 数据分层随机排列
def stratified_shuffle_split(X, y, n_splits, random_state=0):
    if random_state:
        set_seed(random_state)
    # 每个标签下标列表，每个标签的样本个数
    indexList_label, len_label = {}, {}
    # 按标签取出样本，打乱样本下标
    for label in set(y):
        indexList_label[label] = np.where(y == label)[0]
        np.random.shuffle(indexList_label[label])
        len_label[label] = int(np.ceil(len(indexList_label[label]) / n_splits))
    # 每个标签的样本平均分成n_splits份，设标签个数为label_num，则共有label_num*n_splits份样本数据
    # 再以标签间隔的形式拼成新数据，即先拼接每个标签的第一份样本数据、再拼接每个标签的第二份样本数据、...
    # 其中每个标签的第i份样本数据拼接后再打乱
    indexList = np.array([], dtype=int)
    for i in range(n_splits):
        indexList_i = np.array([], dtype=int)
        for label in len_label:
            indexList_label_i = indexList_label[label][len_label[label] * i: len_label[label] * (i + 1)]
            indexList_i = np.append(indexList_i, indexList_label_i)
        np.random.shuffle(indexList_i)
        indexList = np.append(indexList, indexList_i)
    return X[indexList], y[indexList]