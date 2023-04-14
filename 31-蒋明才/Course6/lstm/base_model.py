"""
 #
 # @Author: jmc
 # @Date: 2023/3/22 23:07
 # @Version: v1.0
 # @Description: 模型文件
"""
import torch
import torch.nn as nn
import math


class BiLSTMAttention(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 vocab_size,
                 num_layers=1,
                 batch_first=True,
                 bidirectional=True,
                 dropout=0.1,
                 pool="sum"):
        super(BiLSTMAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim, padding_idx=0)
        self.pool = pool.lower()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=batch_first, bidirectional=bidirectional)

        if bidirectional:
            bi_flag = 2
        else:
            bi_flag = 1

        self.w_q = nn.Linear(hidden_dim * bi_flag, hidden_dim * bi_flag, bias=False)
        self.w_k = nn.Linear(hidden_dim * bi_flag, hidden_dim * bi_flag, bias=False)
        self.w_v = nn.Linear(hidden_dim * bi_flag, hidden_dim * bi_flag, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear(hidden_dim * bi_flag, output_dim)

    # 计算注意力得分
    def attention(self, q, k, v):
        d_k = q.size(-1)
        score = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(d_k)
        score = torch.softmax(score, dim=1)
        out = torch.matmul(score, v)
        if self.pool == "sum":
            out = out.sum(dim=1)
        elif self.pool == "avg" or self.pool == "mean":
            out = torch.mean(out, dim=1)
        elif self.pool == "max":
            out = torch.max(out, dim=1)[0]
        return out

    def forward(self, batch_sample):
        batch_sample = self.embedding(batch_sample)
        out, _ = self.lstm.forward(batch_sample)
        q = self.w_q.forward(out)
        k = self.w_k.forward(out)
        v = self.w_v.forward(out)
        out = self.attention(q, k, v)
        out = self.dropout.forward(out)
        out = self.fc.forward(out)
        return out
