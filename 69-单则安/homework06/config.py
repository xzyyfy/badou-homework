# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "train_data.csv",
    "valid_data_path": "test_data.csv",
    "max_length": 20,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 10,
    "batch_size": 64,
    "optimizer": "adam",
    "learning_rate": 2e-5,
    "pretrain_model_path":"../../bert-base-chinese",
    "seed": 987
}