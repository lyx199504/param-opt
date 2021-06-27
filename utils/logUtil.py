#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2021/6/2 14:21
# @Author : LYX-夜光
import logging
import re

from utils import yaml_config

# 日志配置
def logging_config(logName, fileName):
    logger = logging.getLogger(logName)

    if not logger.handlers:
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

# 按照模型关键字获取对应超参数
def get_param_from_log(model_name, model_key):
    paramList = read_log(yaml_config['dir']['log_dir'] + "/" + model_name + ".log")
    paramList = list(filter(lambda x: model_key in x['model_path'], paramList))
    return paramList[0] if paramList else None