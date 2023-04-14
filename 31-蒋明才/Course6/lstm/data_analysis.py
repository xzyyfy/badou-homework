"""
 #
 # @Author: jmc
 # @Date: 2023/3/25 20:25
 # @Version: v1.0
 # @Description: 数据分析
"""
import json
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import jieba
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
jieba.initialize()


def data_analysis():
    label = dict()
    contents = []
    with open("../dataset/train_tag_news.json", encoding="utf-8") as file:
        for line in tqdm(file.readlines()):
            line = line.replace("\n", "")
            line = json.loads(line)
            label[line["tag"]] = label.get(line["tag"], 0) + 1
            contents.append(line['content'].lower())
    print(f"总类别数：{len(label)}")
    df = pd.DataFrame()
    df["X"] = list(label.keys())
    df["Y"] = list(label.values())
    plt.figure(figsize=(15, 10))
    sns.barplot(df, x="X", y="Y", hue="X")  # 18类，每类数据比较均衡
    plt.tight_layout()
    plt.show()
    # 形成词典
    words = []
    for cnt in contents:
        words += jieba.lcut(cnt)
    words = list(set(words))
    words = ["PAD", "UKN"] + words
    # 保存tag和词典
    label_vocab = {"label": list(label.keys()), "vocab": words}
    with open("../dataset/tag_vocab.json", "w", encoding="utf-8") as file:
        json.dump(label_vocab, file, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    data_analysis()
