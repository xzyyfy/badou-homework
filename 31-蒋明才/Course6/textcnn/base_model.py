"""
 #
 # @Author: jmc
 # @Date: 2023/3/22 23:07
 # @Version: v1.0
 # @Description: 模型文件
"""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, input_dim, out_dim, filter_num, filter_size, vocab_size):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        self.filter_size = filter_size
        self.conv = nn.ModuleList([
            nn.Conv2d(1, filter_num, kernel_size=(size, input_dim)) for size in filter_size
        ])
        self.classifier = nn.Linear(filter_num * len(filter_size), out_dim)

    def forward(self, batch_sample):
        batch_sample = self.embedding(batch_sample)
        seq_len = batch_sample.size(1)
        batch_sample = batch_sample.unsqueeze(dim=1)
        pool_output = []
        for idx, conv in enumerate(self.conv):
            height = seq_len - self.filter_size[idx] + 1
            hidden = torch.relu(conv.forward(batch_sample))
            max_hidden = torch.max_pool2d(hidden, kernel_size=(height, 1))
            pool_output.append(max_hidden)
        pool_cat = torch.cat(pool_output, dim=1)
        pool_cat = torch.squeeze(pool_cat)
        out = self.classifier.forward(pool_cat)
        return out
