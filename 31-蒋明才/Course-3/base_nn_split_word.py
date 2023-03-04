"""
 #
 # @Author: jmc
 # @Date: 2023/3/3 20:48
 # @Version: v1.0
 # @Description: 基于NN的分词
"""
import jieba
import torch
import torch.nn as nn
from base_math_split_word import load_corpus
from torch.utils.data import dataset, dataloader
from loguru import logger
from datetime import datetime
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score


# 加载字典
def load_vocabs(vocab_path):
    vocabs = []
    with open(vocab_path, encoding="utf-8") as file:
        for line in file.readlines():
            vocabs.append(line.replace("\n", ""))
    return vocabs


# 利用jieba分词构建数据集
def build_dataset(corpus: list):
    ds = []
    for ele in corpus:
        words = jieba.lcut(ele)
        chars, label = [], []
        for word in words:
            for idx, char in enumerate(word):
                chars.append(char)
                if idx == len(word) - 1:
                    label.append(1)
                else:
                    label.append(0)
        ds.append([chars, label])
    return ds


# 自定义数据集
class CustomDataset(dataset.Dataset):
    def __init__(self, ds: list):
        super(CustomDataset, self).__init__()
        self.ds = ds

    def __getitem__(self, item):
        x, y = self.ds[item]
        return x, y

    def __len__(self):
        return len(self.ds)


# 自定义dataloader
class CustomDataloader:
    def __init__(self, ds: list, vocab: list, batch_size=32, shuffle=True):
        self.ds = CustomDataset(ds)
        self.vocab = vocab
        self.batch_size = batch_size
        self.shuffle = shuffle

    def collate_fn(self, batch):
        x, y = [], []
        batch_len = 0
        for ele in batch:
            batch_len = max(batch_len, len(ele[0]))
            x.append(ele[0])
            y.append(ele[1])
        for idx, ele in enumerate(x):
            ele_id = []
            for char in ele:
                try:
                    ele_id.append(self.vocab.index(char))
                except:
                    ele_id.append(1)  # UKN的下标
            if len(ele_id) < batch_len:  # 1个Batch PAD到一个相同的长度
                diff = batch_len - len(ele_id)
                ele_id += [0] * diff  # PAD的下标
                y[idx] = y[idx] + [-1 for _ in range(diff)]
            x[idx] = ele_id
        return torch.LongTensor(x), torch.LongTensor(y)

    def gen_loader(self):
        return dataloader.DataLoader(
            dataset=self.ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )


# 构建模型
class SplitWordModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True):
        super(SplitWordModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=batch_first, bidirectional=bidirectional)
        self.classification = nn.Linear(2*hidden_dim, 2)

    def forward(self, batch):
        embed = self.embedding.forward(batch)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm.forward(embed)  # (batch,bi*hidden_dim)
        out = self.classification.forward(lstm_out)  # (batch, seq_len, 2)
        return out


# 训练
def model_train():
    corpus = load_corpus("./datasets/corpus.txt")
    vocab = load_vocabs("./datasets/chars.txt")
    ds = build_dataset(corpus)
    loader = CustomDataloader(ds, vocab, batch_size=64).gen_loader()
    lr = 2e-4
    epoch = 5
    save_step = 32
    save_path = "./checkpoint/split_words.pt"
    spm = SplitWordModel(4622, 256, 128)
    loss_func = nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(params=spm.parameters(), lr=lr)
    writer = SummaryWriter(logdir="./log/" + datetime.now().strftime("%Y-%m-%d"))
    count = 0
    best_f1 = 0
    pred, tgt = [], []
    for epo in range(epoch):
        for idx, (x, y) in enumerate(loader):
            out = spm.forward(x)
            out = out.reshape((-1, 2))
            y = y.reshape((1, -1)).squeeze()
            loss = loss_func.forward(out, y)
            pred += torch.argmax(torch.softmax(out, dim=1), dim=1).tolist()
            tgt += y.tolist()
            optimizer.zero_grad()
            loss.backward()
            if idx % save_step == 0:
                count += 1
                # 去除PAD位置
                new_pred, new_tgt = [], []
                for i, ele in enumerate(tgt):
                    if ele != -1:
                        new_pred.append(pred[i])
                        new_tgt.append(ele)
                score = f1_score(new_tgt, new_pred)
                logger.info(f"epoch:{epo+1}/{epoch}, idx:{idx+1}/{len(loader)}, train.f1:{score}, train.loss:{loss.item()}")
                if score > best_f1:
                    best_f1 = score
                    torch.save(spm.state_dict(), save_path)
                    logger.info("save model")
                pred, tgt = [], []
                writer.add_scalar("train/loss", loss.item(), count)
                writer.add_scalar("train/f1", score, count)
            optimizer.step()


# 模型测试
def model_test(text: str):
    spm = SplitWordModel(4622, 256, 128)
    spm.load_state_dict(torch.load("./checkpoint/split_words.pt"))
    vocab = load_vocabs("./datasets/chars.txt")
    text_id = []
    for char in text:
        try:
            text_id.append(vocab.index(char))
        except:
            text_id.append(1)
    text_id = torch.LongTensor(text_id)
    out = spm.forward(text_id)
    out = torch.argmax(torch.softmax(out, dim=1), dim=1)
    words = []
    word = ""
    for idx, label in enumerate(out):
        word += text[idx]
        if label == 1:
            words.append(word)
            word = ""
    if len(word) != 0:
        words.append(word)
    return words


# 语料库数据测试
def corpus_test():
    corpus = load_corpus("./datasets/corpus.txt")
    for text in corpus:
        words = model_test(text)
        print(f"原文：{text} \n 基于NN分词结果：{words}")


if __name__ == '__main__':
    # model_train()
    # model_test("昨日上海天然橡胶期货价格再度大幅上扬")
    corpus_test()