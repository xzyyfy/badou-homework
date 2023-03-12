#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

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

def cluster_distances(centers, sentences_dict, model):
    # 计算每个聚类中，各个句子到中心点的平均欧氏距离
    distances = []
    for label, sentences in sentences_dict.items():
        index = int(label)
        distance = 0
        sentence_vectors = sentences_to_vectors(sentences, model)
        for sentence_vector in sentence_vectors:
            distance += np.linalg.norm(centers[index]-sentence_vector)
        distances.append(distance/len(sentences))
    return distances

def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    # 得到最后分类的聚类中心
    final_centers = []
    for i, center in enumerate(kmeans.cluster_centers_):
        final_centers.append(center)
    # print(0 in kmeans.labels_)   # 判断0是否在标签当中 TRUE

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    distances = cluster_distances(final_centers, sentence_label_dict, model)  #计算聚类距离
    for index, distance in enumerate(distances):
        print("第 %d 类的平均中心距离为：" %(index), distance)
    
if __name__ == "__main__":
    main()

