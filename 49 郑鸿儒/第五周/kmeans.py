#!/usr/bin/env python3
# coding: utf-8
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    return sentences


# 文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences, model)

    n_clusters = int(math.sqrt(len(sentences)))
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)

    distance_dict = {}
    for label, sentence in sentence_label_dict.items():
        cur_total_num = len(sentence)
        cur_center = kmeans.cluster_centers_[label]
        cur_distance = 0
        for i in range(cur_total_num):
            tmp = 0
            word_set = set()
            word_set.add(sentence[i])
            word_verctor = sentences_to_vectors(word_set, model)[0]
            for j in range(len(word_verctor)):
                tmp += pow(word_verctor[j] - cur_center[j], 2)
            cur_distance += pow(tmp, 0.5)
        distance_dict[label] = cur_distance/cur_total_num
    # print('distance_dict', distance_dict)
    for label, sentences in sentence_label_dict.items():
        print("cluster %s (类内平均距离 %f):" % (label, distance_dict[label]))
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
