# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.encoder = BertModel.from_pretrained(config["pretrain_model_path"])
        hidden_size = self.encoder.config.hidden_size
        self.classify = nn.Linear(hidden_size, 2)
        self.loss = nn.functional.cross_entropy  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        x = self.encoder(x)
        x = x[-1]
        predict = self.classify(x)   #input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict.squeeze(), target.squeeze())
        else:
            return predict
        
#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    Config["model_type"] = "bert"
    model = BertModel.from_pretrained(Config["pretrain_model_path"])
    x = torch.LongTensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    sequence_output, pooler_output = model(x)
    print(x[2], type(x[2]), len(x[2]))