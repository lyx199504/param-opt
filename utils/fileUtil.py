#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 14:21
# @Author : LYX-夜光
import logging
import re


# 日志配置
def logging_config(logName, fileName):
    logger = logging.getLogger(logName)
    logger.setLevel('DEBUG')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)

    # 输出到控制台的handler
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    logger.addHandler(chlr)

    # 输出到文件的handler
    fhlr = logging.FileHandler(fileName, encoding='utf-8')
    fhlr.setFormatter(formatter)
    fhlr.setLevel('INFO')
    logger.addHandler(fhlr)

    return logger

# 读取日志列表
def read_log(logFile):
    with open(logFile, 'r') as log:
        logList = log.readlines()
    return list(map(lambda x: eval(re.findall(r"(?<=INFO:).*$", x)[0]), logList))

