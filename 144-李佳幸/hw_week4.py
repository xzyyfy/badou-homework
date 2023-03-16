import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
实现基于词表，对于输入文本输出所有可能的分词方式。尝试实现基于lstm的分词。
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_lstm_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab)+1, input_dim)
        self.lstm_layer = nn.LSTM(input_dim,hidden_size,batch_first=True,num_layers=num_lstm_layers)
        self.classify = nn.Linear(hidden_size,2)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, y=None):
        x = self.embedding(x)
        x, _ = self.lstm_layer(x)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss_func(y_pred.view(-1,2),y.view(-1))
        else:
            return y_pred

class Dataset:
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        self.load()
    
    def load(self):
        self.data = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_seqence(line, self.vocab)
                label = seqence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence,label])
                if len(self.data)>10000:
                    break

    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length-len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length-len(label))
        return sequence, label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]

def sentence_to_seqence(sentence, vocab):
    seqence = [vocab.get(char, vocab['unk']) for char in sentence]
    return seqence

def seqence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0]*len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer-1] = 1
    return label

def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index+1
    vocab['unk'] = len(vocab)+1
    return vocab

def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length)
    data_loader = DataLoader(dataset, shuffle=True,batch_size=batch_size)
    return data_loader

def main():
    epoch_num = 10
    batch_size = 20
    char_dim = 50
    hidden_size = 100
    num_lstm_layers = 3
    max_length = 20
    learning_rate = 1e-3
    vocab_path = "chars.txt"
    corpus_path = "corpus.txt"
    vocab = build_vocab(vocab_path)
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)
    model = TorchModel(char_dim, hidden_size, num_lstm_layers, vocab)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()
            loss = model(x,y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    torch.save(model.state_dict(), "my_model.pth")
    return

def predict(model_path, vocab_path, input_strings):
    char_dim = 50
    hidden_size = 100
    num_lstm_layers = 3
    vocab = build_vocab(vocab_path)
    model = TorchModel(char_dim, hidden_size, num_lstm_layers, vocab)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for input_string in input_strings:
        x = sentence_to_seqence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim = -1)
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end=" ")
                else:
                    print(input_string[index], end="")
            print()

if __name__ == "__main__":
    # main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    predict("my_model.pth", "chars.txt", input_strings)
