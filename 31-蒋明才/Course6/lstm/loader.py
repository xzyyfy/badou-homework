"""
 #
 # @Author: jmc
 # @Date: 2023/3/22 23:07
 # @Version: v1.0
 # @Description: 处理数据集的文件
"""
import json
from tqdm import tqdm
import torch
from torch.utils.data import dataset, dataloader
import jieba
jieba.initialize()


def load_label_vocab() -> list:
    with open("../dataset/tag_vocab.json", encoding="utf-8") as file:
        label_dict = json.loads(file.read())
        # print(len(label_dict["vocab"]))
    return label_dict["label"], label_dict["vocab"]


def load_dataset(file_path):
    labels, vocabs = load_label_vocab()
    ds = []
    with open(file_path, encoding="utf-8") as file:
        for line in tqdm(file.readlines(), desc=f"加载数据,路径为:{file_path}"):
            line = line.replace("\n", "")
            line = json.loads(line)
            tag, content = line["tag"], line["content"].lower()
            content = content[:380] + content[-120:]  # 长度 <= 500
            tag_idx = labels.index(tag)
            ds.append([content, tag_idx])
    # print(ds)
    return ds, vocabs


class CustomDataset(dataset.Dataset):
    def __init__(self, ds: list):
        super(CustomDataset, self).__init__()
        # ds [[sentence1, label], [sentence2, label], ......]
        self.ds = ds

    def __getitem__(self, item):
        x, y = self.ds[item]
        return x, y

    def __len__(self):
        return len(self.ds)


class CustomDataloader:
    def __init__(self, ds, vocabs, device="cpu", batch_size=32, shuffle=True, seq_max_len=512):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.vocabs = vocabs
        self.device = device
        self.ds = CustomDataset(ds)
        self.seq_max_len = seq_max_len

    def collate_fn(self, batch):
        x, y = [], []
        batch_max_len = 0
        for ele1, ele2 in batch:
            ele1 = jieba.lcut(ele1)
            x.append(ele1)
            y.append(ele2)
            batch_max_len = max(batch_max_len, len(ele1))
        # word -> id
        ids = []
        for ele in x:
            tmp = []
            for word in ele:
                try:
                    tmp.append(self.vocabs.index(word))
                except:
                    tmp.append(1)  # UKN
            if len(ele) != batch_max_len:
                tmp += [0] * (batch_max_len - len(ele))  # PAD
            ids.append(tmp)
        ids = torch.LongTensor(ids).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        return ids, y

    def gen_dataloader(self):
        return dataloader.DataLoader(
            dataset=self.ds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn
        )


if __name__ == '__main__':
    load_dataset("../dataset/train_tag_news.json")