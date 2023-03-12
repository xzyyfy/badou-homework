#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
import gensim
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

print(gensim.__version__)
#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

def distance( p1, p2):
    #计算两点间距
    tmp = 0
    for i in range(len(p1)):
        tmp += pow(p1[i] - p2[i], 2)
    return pow(tmp, 0.5)

def inner_distance(center, vectors):
    sum_distance = 0.0
    point_counts = len(vectors)
    for i in range(point_counts):
        sum_distance += distance(vectors[i], center)
    return sum_distance / point_counts

def main():

    sentence_vector_dict = defaultdict(list)
    label_center_dict = defaultdict(list)
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    index= 0
    for i in sentences:
        print(vectors)
        sentence_vector_dict[i] = vectors[index]
        index += 1
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    print("len(sentences), ", len(sentences))
    print("len(vectors), ", len(vectors))
    sentence_label_dict = defaultdict(list)

    label_array = []
    distance_array = []
    for sentence, label, centers in zip(sentences, kmeans.labels_, kmeans.cluster_centers_):  #取出句子和标签
        print("label, ", label)
        print("centers, ", centers)
        label_center_dict[label] = centers
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    for label, sentences in sentence_label_dict.items():
        # 转换成向量数组
        sentence_array = []
        for sen in range(len(sentences)):
            sentence_array.append(sentence_vector_dict[sentences[sen]])
        distance = inner_distance(label_center_dict[label], sentence_array)

        label_array.append(label)
        distance_array.append(distance)

        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")

    # 进行排序
    sorted_distance_array = np.argsort(distance_array)
    print("sorted_distance_array:", sorted_distance_array)

    #输出排序后的数据
    print("==============词向量KMeans分类之后的结果如下==============")
    classes = len(distance_array)
    for i in range(classes):
        print("排序后的第{}大的分类，为原始的第{}类，类标签为：{}，类内距离为：{}.".format(i + 1,
            sorted_distance_array[i], label_array[sorted_distance_array[i]], distance_array[sorted_distance_array[i]]))
        print("============================================")

if __name__ == "__main__":
    main()

