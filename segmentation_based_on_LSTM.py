#coding:utf8

import torch
import torch.nn as nn
import jieba
import numpy as np
import random
import json
from torch.utils.data import DataLoader

"""
基于pytorch的网络编写一个分词模型
我们使用jieba分词的结果作为训练数据
看看是否可以得到一个效果接近的神经网络模型
"""

class TorchModel(nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim) #shape=(vocab_size, dim)
        self.rnn_layer = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            num_layers=num_rnn_layers,
                            )
        self.classify = nn.Linear(hidden_size, 2)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, _ = self.rnn_layer(x)  #output shape:(batch_size, sen_len, hidden_size)
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, 2)
        if y is not None:
            # view(-1,2): (batch_size, sen_len, 2) ->  (batch_size * sen_len, 2)
            return self.loss_func(y_pred.view(-1, 2), y.view(-1))
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
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                self.data.append([sequence, label])
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(self.data) > 10000:
                    break

    #将文本截断或补齐到固定长度
    def padding(self, sequence, label):
        sequence = sequence[:self.max_length]
        sequence += [0] * (self.max_length - len(sequence))
        label = label[:self.max_length]
        label += [-100] * (self.max_length - len(label))
        return sequence, label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

#文本转化为数字序列，为embedding做准备
def sentence_to_sequence(sentence, vocab):
    sequence = [vocab.get(char, vocab['unk']) for char in sentence]
    return sequence

#基于结巴生成分级结果的标注
def sequence_to_label(sentence):
    words = jieba.lcut(sentence)
    label = [0] * len(sentence)
    pointer = 0
    for word in words:
        pointer += len(word)
        label[pointer - 1] = 1
    return label

#加载字表
def build_vocab(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line.strip()
            vocab[char] = index + 1   #每个字对应一个序号
    vocab['unk'] = len(vocab) + 1
    return vocab

#建立数据集
def build_dataset(corpus_path, vocab, max_length, batch_size):
    dataset = Dataset(corpus_path, vocab, max_length) #diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #torch
    return data_loader


def main():
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    char_dim = 50         #每个字的维度
    hidden_size = 100     #隐含层维度
    num_rnn_layers = 3    #rnn层数
    max_length = 20       #样本最大长度
    learning_rate = 1e-3  #学习率
    vocab_path = "chars.txt"  #字表文件路径
    corpus_path = "../corpus.txt"  #语料文件路径
    vocab = build_vocab(vocab_path)       #建立字表
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  #建立数据集
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)     #建立优化器
    #训练开始
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, y in data_loader:
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
    #保存模型
    torch.save(model.state_dict(), "model.pth")
    return

#最终预测
def predict(model_path, vocab_path, input_strings):
    #配置保持和训练时一致
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    vocab = build_vocab(vocab_path)       #建立字表
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab)   #建立模型
    model.load_state_dict(torch.load(model_path))   #加载训练好的模型权重
    model.eval()
    for input_string in input_strings:
        #逐条预测
        x = sentence_to_sequence(input_string, vocab)
        with torch.no_grad():
            result = model.forward(torch.LongTensor([x]))[0]
            result = torch.argmax(result, dim=-1)  #预测出的01序列
            #在预测为1的地方切分，将切分后文本打印出来
            for index, p in enumerate(result):
                if p == 1:
                    print(input_string[index], end="/")
                else:
                    print(input_string[index], end="")
            print()

#加载词典
def load_word_dict(path):
    max_word_length = 0
    word_dict = {}  #用set也是可以的。用list会很慢
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            word_dict[word] = 0
            max_word_length = max(max_word_length, len(word))
    return word_dict, max_word_length

#先确定最大词长度
#从长向短查找是否有匹配的词
#找到后移动窗口
def cut_method1(string, word_dict, max_len):
    words = []
    while string != '':
        lens = min(max_len, len(string))
        word = string[:lens]
        while word not in word_dict:
            if len(word) == 1:
                break
            word = word[:len(word) - 1]
        words.append(word)
        string = string[len(word):]
    str1 = ""
    for x in range(len(words)):
        if x < len(words) - 1:
            str1 += words[x] + "/"
        else:
            str1 += words[x]
    return str1

#加载词前缀词典
#用0和1来区分是前缀还是真词
#需要注意有的词的前缀也是真词，在记录时不要互相覆盖
def load_prefix_word_dict(path):
    prefix_dict = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            word = line.split()[0]
            for i in range(1, len(word)):
                if word[:i] not in prefix_dict: #不能用前缀覆盖词
                    prefix_dict[word[:i]] = 0  #前缀
            prefix_dict[word] = 1  #词
    return prefix_dict

#输入字符串和字典，返回词的列表
def cut_method2(string, prefix_dict):
    if string == "":
        return []
    words = []  # 准备用于放入切好的词
    start_index, end_index = 0, 1  #记录窗口的起始位置
    window = string[start_index:end_index] #从第一个字开始
    find_word = window  # 将第一个字先当做默认词
    while start_index < len(string):
        #窗口没有在词典里出现
        if window not in prefix_dict or end_index > len(string):
            words.append(find_word)  #记录找到的词
            start_index += len(find_word)  #更新起点的位置
            end_index = start_index + 1
            window = string[start_index:end_index]  #从新的位置开始一个字一个字向后找
            find_word = window
        #窗口是一个词
        elif prefix_dict[window] == 1:
            find_word = window  #查找到了一个词，还要在看有没有比他更长的词
            end_index += 1
            window = string[start_index:end_index]
        #窗口是一个前缀
        elif prefix_dict[window] == 0:
            end_index += 1
            window = string[start_index:end_index]
    #最后找到的window如果不在词典里，把单独的字加入切词结果
    if prefix_dict.get(window) != 1:
        words += list(window)
    else:
        words.append(window)
    str1 = ""
    for x in range(len(words)):
        if x < len(words) - 1:
            str1 += words[x] + "/"
        else:
            str1 += words[x]
    return str1


if __name__ == "__main__":
    #main()
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬",
                     "两大主力多头金鹏期货和成都倍特期货略微增几百手",
                     "金鹏期货北京海鹰路营业部总经理陈旭指出",
                     "截止北京时间昨日下午16点46分报241.5日元/公斤",
                     "该国天然橡胶库存较10月31日时的库存量下滑3.4%"]
    word_dict, max_len = load_word_dict("dict.txt")
    prefix_dict = load_prefix_word_dict("dict.txt")
    print("第一种实现方式")
    for x in input_strings:
        print(cut_method1(x, word_dict, max_len))
    print("第二种实现方式")
    for x in input_strings:
        print(cut_method2(x, prefix_dict))
    print("LSTM实现方式")
    predict("model.pth", "chars.txt", input_strings)

