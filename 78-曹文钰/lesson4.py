import torch
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np

def timer(func):
    def func_wrapper(self,*args, **kwargs):
        from time import time
        time_start = time()
        result = func(self,*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result
    return func_wrapper

class TorchModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_size, num_rnn_layers, vocab, model_path="model.pth"):
        super(TorchModel, self).__init__()
        self.embedding = torch.nn.Embedding(len(vocab) + 1, input_dim) #shape=(vocab_size, dim)
        self.lstm_layer = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True
        )
        self.classify = torch.nn.Linear(hidden_size, 2)
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.model_path = model_path
        self.cuda()

    def forward(self, x, y=None):
        x = self.embedding(x)  #input shape: (batch_size, sen_len), output shape:(batch_size, sen_len, input_dim)
        x, _ = self.lstm_layer(x)  #output shape:(batch_size, sen_len, hidden_size)
        y_pred = self.classify(x)   #output shape:(batch_size, sen_len, 2)
        return y_pred

    @timer
    def do_training(self, epoch_num=10, learning_rate=1e-3):
        self.train()
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  # 建立优化器
        # 训练开始
        for epoch in range(epoch_num):
            watch_loss = []
            for x_batch, y in data_loader:
                optim.zero_grad()  # 梯度归零
                y_pre = self.forward(x_batch)
                # view(-1,2): (batch_size, sen_len, 2) ->  (batch_size * sen_len, 2)
                loss = self.loss_func(y_pre.view(-1, 2), y.view(-1)) # 计算loss
                loss.backward()  # 计算梯度
                optim.step()  # 更新权重
                watch_loss.append(loss.item())
            print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        # 保存模型
        torch.save(model.state_dict(), self.model_path)
        return

    # 最终预测
    def predict(self, model_path, vocab, input_strings):
        self.eval()
        self.load_state_dict(torch.load(model_path))  # 加载训练好的模型权重
        for input_string in input_strings:
            # 逐条预测
            x = sentence_to_sequence(input_string, vocab)
            x = torch.LongTensor([x]).cuda()
            with torch.no_grad():
                result = model.forward(x)[0]
                result = torch.argmax(result, dim=-1)  # 预测出的01序列
                # 在预测为1的地方切分，将切分后文本打印出来
                for index, p in enumerate(result):
                    if p == 1:
                        print(input_string[index], end=" ")
                    else:
                        print(input_string[index], end="")
                print()

class DataSet():
    def __init__(self, corpus_path, vocab, max_length):
        self.vocab = vocab
        self.corpus_path = corpus_path
        self.max_length = max_length
        data,target = self.load()
        self.data, self.target = torch.stack(data).cuda(), torch.stack(target).cuda()

    def load(self):
        data,target = [],[]
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                sequence = sentence_to_sequence(line, self.vocab)
                label = sequence_to_label(line)
                sequence, label = self.padding(sequence, label)
                sequence = torch.LongTensor(sequence)
                label = torch.LongTensor(label)
                data.append(sequence)
                target.append(label)
                #使用部分数据做展示，使用全部数据训练时间会相应变长
                if len(data) > 10000:
                    break
        return data,target

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
        return self.data[item], self.target[item]

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
    dataset = DataSet(corpus_path, vocab, max_length) #diy __len__ __getitem__
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size) #torch
    return data_loader


if __name__ == "__main__":
    vocab_path = "chars.txt"  # 字表文件路径
    corpus_path = "../corpus.txt"  # 语料文件路径
    vocab = build_vocab(vocab_path)  # 建立字表
    
    batch_size = 20  # 每次训练样本个数
    max_length = 20  # 样本最大长度
    data_loader = build_dataset(corpus_path, vocab, max_length, batch_size)  # 建立数据集
    
    char_dim = 50  # 每个字的维度
    hidden_size = 100  # 隐含层维度
    num_rnn_layers = 3  # rnn层数
    model_path = "model.pth"
    model = TorchModel(char_dim, hidden_size, num_rnn_layers, vocab, model_path)  # 建立模型

    epoch_num = 10  # 训练轮数
    learning_rate = 1e-3  # 学习率
    model.do_training(epoch_num, learning_rate)
    
    
    input_strings = ["同时国内有望出台新汽车刺激方案",
                     "沪胶后市有望延续强势",
                     "经过两个交易日的强势调整后",
                     "昨日上海天然橡胶期货价格再度大幅上扬"]
    model.predict(model_path, vocab, input_strings)