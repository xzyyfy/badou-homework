
import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from collections import defaultdict

'''
词向量模型的简单实现
作用：就是把一个文本里面的词语，反复映射，经过学习后，得到没歌词对应的向量。
'''

#训练模型
#corpus: [["cat", "say", "meow"], ["dog", "say", "woof"]]
#dim指定词向量的维度，如100
def train_word2vec_model(corpus, dim):                   #corpus 代表穿入的文本文件
    model = Word2Vec(corpus, vector_size=dim, sg=1)      #vector_size 代表每个词语要映射成几维的向量  sg=1 代表使用窗口映射
    model.save("model.w2v-5")                            #Word2Vec()就是一个专门用来把文本映射为向量的类
    return model

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def main():
    sentences = []
    with open("corpus-5.txt", encoding="utf8") as f:
        for line in f:
            sentences.append(jieba.lcut(line))      #使用jieba分词，对文本里面的每一行进行分词,然后组成一个很长的句子
    model = train_word2vec_model(sentences, 20)    #把句子中的每个词都映射成一个100维的向量，得到 len(sentences)*100的训练以后的矩阵
    return

if __name__ == "__main__":
    main()  #训练

    model = load_word2vec_model("model.w2v-5")  #加载训练好的词向量参数
    #试验：找出 与'男人、母亲' 很相似 且 与’女人‘很不相似的词
    print(model.wv.most_similar(positive=["男人", "母亲"], negative=["女人"]))   #model.wv.most_similar()就是一个找相似词的方法

    while True:  #找相似
        string = input("input:")
        try:
            print(model.wv.most_similar(string))
        except KeyError:
            print("输入词不存在")