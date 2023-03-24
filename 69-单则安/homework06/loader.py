# -*- coding: utf-8 -*-

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config

"""
数据加载
"""
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.load()

    def load(self):
        self.data = []
        datas = pd.read_csv(self.path)
        labels = datas['label']
        texts = datas['review']
        for i in range(len(datas)):
            label = labels[i]
            text = texts[i]
            input_id = self.tokenizer.encode(text, max_length=self.config["max_length"], pad_to_max_length=True)
            input_id = torch.LongTensor(input_id)
            label = torch.LongTensor([int(label)])
            self.data.append([input_id, label])
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl
