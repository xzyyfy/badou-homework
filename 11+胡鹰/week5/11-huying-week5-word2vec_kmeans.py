#!/usr/bin/env python3  
#coding: utf-8

#*****************************************************
#准备好语料，然后用word2Vec进行词向量训练，得到训练好的词向量参数
#然后，用训练好的词向量参数，把需要聚类的目标语料映射为词向量；
#再用KMeans类对映射后的目标语料词向量进行聚类     2023.03.09
#*****************************************************
import math
import random
import sys
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import kmeans

#输入模型文件路径
#加载训练好的模型  这里使用的是前面用另外一份语料事先就训练好的词向量参数
def load_word2vec_model(path):
    model = Word2Vec.load(path)      #直接加载该训练好的词向量参数，就可以得到一个word2Vec模型
    return model

#加载目标语料，并把他用jieba分词先分好，然后放到一个集合中
def load_sentence(path):
    sentences = set()                                      #初始化一个存放每句话的集合
    with open(path, encoding="utf8") as f:
        for line in f:                                     #按行读取语料中的每一行
            sentence = line.strip()                        #去掉每句话两端的空格
            sentences.add(" ".join(jieba.cut(sentence)))   #每个句子里面的词用' '隔开
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []                                                #用列表来存放每句话得到词向量
    for sentence in sentences:
        words = sentence.split()                                #sentence是分好词的，空格分开(split()函数就是用空格来分开字符串的)
        vector = np.zeros(model.vector_size)                    #初始化的每个词的词向量都是0，维度就是模型中的向量维度
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)          #部分词在训练中未出现，用全0向量代替
        vectors.append(vector / len(words))                    #以该句子中的每个词的词向量的平均值作为整个句子的句向量
    return np.array(vectors)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#定义一个聚类用的类
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):                          #初始化
        self.ndarray = ndarray                                         #传入的句向量矩阵
        self.cluster_num = cluster_num                                 #聚类的类别数量
        self.points = self.__pick_start_point(ndarray, cluster_num)    #获得随机的cluster_num个中心点

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])                                          #先把每个类都给出列表来，一边后续存放该类下的点向量
        #*************************************************************
        #聚类的核心代码
        #*************************************************************
        for item in self.ndarray:
            distance_min = sys.maxsize                                 #初始化最小距离
            index = -1

            # 把每一个点都与K个中心点计算距离，找到距离最小的那个中心点，然后把该点归到该中心点的这一个类上来
            #++++++++++++++++++++++++++++++++++++++++++++++++++++
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
            #++++++++++++++++++++++++++++++++++++++++++++++++++++

        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())           #每一类计算自己的中心点
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            #return result, self.points, sum                            #达到稳定，则返回各个类，中心点，总距离
            return result
        self.points = np.array(new_center)                             #否则，更新中心点
        return self.cluster()

    # ****************************************************************
    #计算每一组的总距离和，用来验证是否达到稳定。因为均稳定的话，那么总距离之和不变
    # ****************************************************************
    def __sumdis(self,result):
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    # ****************************************************************
    # 计算该类的中心点，用平均值作为中心点
    # ****************************************************************
    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)                            #axis  轴

    # ****************************************************************
    # 计算两个向量之间的距离
    # ****************************************************************
    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)                              #pow（x,y）就是计算x的y次方
        return pow(tmp, 0.5)                                          #平方和再开平方根，得到两个点之间的距离

    #****************************************************************
    #获取开始的中心点
    #****************************************************************
    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")                            #raise关键字：用来手动抛出带有定制消息的异常  这里的异常可以是不同的类型
                                                                     #比如Exception类型，TypeError类型等等
                                                                     #raise关键字抛出异常后，会显示出我们设定的异常信息，程序会终止。
                                                                     #但如果raise后面的行行中有failly的话，它会执行failly代码块
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)   #随机取下标
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def main():
    model = load_word2vec_model("model.w2v-5")                 #加载词向量模型
    sentences = load_sentence("titles-5.txt")                  #加载目标语料
    vectors = sentences_to_vectors(sentences, model)           #将目标语料向量化，得到的是每句话的句向量

    #*********************************************************
    #KMeans 算法的核心步骤
    #*********************************************************
    #n_clusters = int(math.sqrt(len(sentences)))                #用句子数量的平方根作为需要聚成几个类,即类别数量K
    n_clusters = 10
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)                                #实例化一个KMeans类(指定聚类数量)的
    kmeans.fit(vectors)                                        #用KMeans的fit()方法进行聚类计算
    #==========================================================
    #用KMeansClusterer类来计算，得到的只是聚类以后的向量！不能带有标签！！
    #kmeans = KMeansClusterer(vectors, n_clusters)
    #result = kmeans.cluster()
    #*********************************************************

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):     #取出句子和标签写入字典    kmeans.labels_ 是聚类以后的标签
        sentence_label_dict[label].append(sentence)            #同标签的放到一起
    for label, sentences in sentence_label_dict.items():       #sentence_label_dict.items()就是每一个子项
        print("===========第 %s 类=============" % label)       #打印看看标签名字是什么
        for i in range(min(3, len(sentences))):               #随便打印几个，太多了看不过来  不足10个的全部打印，多余10个的就打前10个
            print(sentences[i].replace(" ", ""))               #把空格去掉后组成一个字符串打印
        print("---------")

    #***********************************************************
    #没有标签，输出的只是聚类以后的向量！
    #for i in range(len(result)):
    #    for j in range(min(10, len(result[i]))):
    #        #print(result[i][j].replace(" ", ""))
    #        print(result[i][j])
    #    print("---第 %s 类------" % i)
    #***********************************************************

if __name__ == "__main__":
    main()

