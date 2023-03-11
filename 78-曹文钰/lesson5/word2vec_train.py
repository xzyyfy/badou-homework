import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from collections import defaultdict


from db import Mysql

#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#dim指定词向量的维度，如100
def train_word2vec_model(corpus, dim):
    model = Word2Vec(corpus, vector_size=dim, sg=1)
    model.save("model.w2v")
    return model

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def build_corpus(asr=True):
    corpus = []
    news = Mysql().select()
    for i in news:
        content = jieba.lcut(i["title"] if i["title"] else "")
        if asr:
            content += jieba.lcut(i["asr"] if i["asr"] else "")
        corpus.append(content)
    print("获取新闻数量：", len(news))
    return corpus

def main():
    corpus = build_corpus()
    model = train_word2vec_model(corpus, 100)
    return


def test():
    model = load_word2vec_model("model.w2v")

    print(model.wv.most_similar(positive=["贫困"], negative=["小康"])) #类比

    while True:  #找相似
        string = input("input:")
        try:
            print(model.wv.most_similar(string))
        except KeyError:
            print("输入词不存在")

if __name__ == "__main__":
    main()  #训练
    #test()

