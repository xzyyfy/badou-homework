import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import autograd

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


class NGramDataset(Dataset):
    def __init__(self, corpus, vocab, n=3):
        super(NGramDataset, self).__init__()
        self.corpus = corpus
        self.vocab = vocab
        self.n = n

        self.sos = "[sos]"  # start of sentence，句子开始的标识符
        self.eos = "[eos]"  # end of sentence，句子结束的标识符
        self.unk = "[unk]"

        self.data = None
        self.label = None
        self.build_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.label[idx]
        return torch.LongTensor(data), torch.LongTensor([label])

    def build_data(self):
        self.data = []
        self.label = []
        for line in self.corpus:
            line = [self.sos] + list(line) + [self.eos]
            line = [word2id(word, self.vocab) for word in line]
            for i, word in enumerate(line):
                for windows in range(1, self.n + 1):
                    if i + windows + 1 > len(line):
                        continue
                    self.data.append(add_pad(line[i: i + windows]))
                    self.label.append(line[i + windows])


def word2id(word, vocab):
    return vocab.get(word, vocab['[unk]'])


def add_pad(iterat: list, pad=0, n=3):
    if len(iterat) >= n:
        return iterat[:n]
    else:
        for i in range(n - len(iterat)):
            iterat.insert(0, pad)
    return iterat


def clean(corpus):
    # 把中文标点符号清洗掉
    clean_dir = ['，', '。', '？', '“', "”", "：", "【", "】", "！", " ", "（", "）"]
    out_corpus = []
    for sent in corpus:
        # 1,先清除左右的空格并转换为字符列表
        sent = list(sent.strip())
        for word in sent:
            if word in clean_dir:
                sent.remove(word)
        out_corpus.append(''.join(sent))
    return out_corpus


def build_vocab(corpus, start="[sos]", end="[eos]", unknow="[unk]"):
    words = {}
    for line in corpus:
        for word in list(line):
            words[word] = words.get(word, 0) + 1
    words = sorted(words.items(), key=lambda x: x[1], reverse=True)
    vocab = {unknow: 0, start: 1, end: 2}
    for i, (word, freq) in enumerate(words):
        vocab[word] = i + 3
    return vocab


class NGramModel(nn.Module):
    def __init__(self, vocab, embedding_size=50, n=3):
        super(NGramModel, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed = nn.Embedding(self.vocab_size, embedding_size)
        self.n = n
        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(n * embedding_size, 128)
        self.fc2 = nn.Linear(128, self.vocab_size)

    def forward(self, inputs):
        # inputs: [b, 3]
        x = self.embed(inputs)  # [b, 3, 50]
        x = x.view(-1, self.n * self.embedding_size)  # [b, 3 * 50]
        x = F.relu(self.fc1(x))  # [b, 128]
        x = self.fc2(x)  # [b, vocab_size]
        return F.log_softmax(x)

    def predict(self, sentence: str):
        word_list = clean(sentence)
        # word_list = list[sentence]
        word_list = ["[sos]"] + word_list + ["[eos]"]
        word_list = [word2id(word, self.vocab) for word in word_list]
        sentence_prob = 0
        for index, word in enumerate(word_list):
            ngram = add_pad(word_list[max(0, index - self.n + 1):index + 1])
            prob = self.forward(torch.LongTensor(ngram)).view(-1).cpu().detach().numpy()
            # print(ngram, prob)
            sentence_prob += prob[word2id(word, self.vocab)]
        # return 2 ** (sentence_prob * (-1 / len(word_list)))
        return sentence_prob


def train(pretrained=None):
    with open("财经.txt", 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    corpus = clean(corpus)
    vocab = build_vocab(corpus)
    dataset = NGramDataset(corpus=corpus,
                           vocab=vocab,
                           n=3)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=512,
                            shuffle=True)
    model = NGramModel(vocab,
                       embedding_size=50,
                       n=3)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    opt = torch.optim.Adam(lr=0.001,
                           params=model.parameters())
    # 负对数似然损失loss
    loss_func = nn.NLLLoss()

    model.cuda()
    loss_func.cuda()
    step = 0
    for epoch in range(10):
        # 训练
        model.train()
        for data, target in dataloader:
            data = data.cuda()
            target = target.cuda()

            opt.zero_grad()
            pred = model(data)
            loss = loss_func(pred, autograd.Variable(target.view(-1)))

            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0:
                print("epoch: {}, step: {}, loss: {}".format(epoch, step, loss.cpu().detach().numpy()))

        # 测试
        model.eval()
        acc_list = []
        for data, target in dataloader:
            data = data.cuda()

            pred = model(data).cpu()
            pred = torch.argmax(pred, dim=-1)
            acc = torch.eq(pred, target.view(-1)).detach().numpy()
            acc_list.append(np.sum(acc) / len(acc))

        print("epoch: {}, acc: {}".format(epoch, np.mean(acc_list)))
    # 保存模型
    torch.save(model.state_dict(), "n_gram_model.pth")


if __name__ == '__main__':
    train(pretrained="./n_gram_model.pth")
