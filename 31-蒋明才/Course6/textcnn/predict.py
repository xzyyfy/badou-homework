"""
 #
 # @Author: jmc
 # @Date: 2023/3/22 23:08
 # @Version: v1.0
 # @Description: 模型预测的文件
"""
import json
import torch
import jieba
from base_model import TextCNN
from tqdm import tqdm
import time
jieba.initialize()
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_label_vocab() -> list:
    with open("../dataset/tag_vocab.json", encoding="utf-8") as file:
        label_dict = json.loads(file.read())
        # print(len(label_dict["vocab"]))
    return label_dict["label"], label_dict["vocab"]


def preprocess(vocabs, text):
    text = text.lower()
    text = text[:380] + text[-120:]
    ids = []
    for word in jieba.lcut(text):
        try:
            ids.append(vocabs.index(word))
        except:
            ids.append(1)  # UKN
    return torch.LongTensor([ids]).to(device)


def predict(model, labels, vocabs, text):
    model.eval()
    ids = preprocess(vocabs, text)
    with torch.no_grad():
        out = model.forward(ids)
    pred_idx = torch.argmax(out, dim=0).item()
    return labels[pred_idx]


def main():
    labels, vocabs = load_label_vocab()
    model = TextCNN(256, 18, 3, [2, 3, 4], 64692).to(device)
    model.load_state_dict(torch.load("./checkpoint/textcnn.pt"))
    with open("../dataset/tag_news.json", encoding="utf-8") as file:
        start = time.time()
        for line in tqdm(file.readlines()[:100], desc=f"预测数据"):
            line = line.replace("\n", "")
            line = json.loads(line)
            tag, content = line["tag"], line["content"].lower()
            pred = predict(model, labels, vocabs, content)
            print(f"\n实际标签：{tag}, 预测标签：{pred}")
        print(f"100条数据预测时间：{time.time() - start}秒")


if __name__ == '__main__':
    main()
