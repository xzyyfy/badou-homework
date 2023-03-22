import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

'''
在基于词向量的kmeans聚类中，增加类内距离的计算
'''

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
        vectors.append(vector/len(words))
    return np.array(vectors)

def main():
    model = load_word2vec_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors = sentences_to_vectors(sentences,model)

    n_clusters = int(math.sqrt(len(sentences)))
    print("聚类数量：",n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentence_label_dict = defaultdict(list)
    vector_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):
        sentence_label_dict[label].append(sentence)
    for vector, label in zip (vectors, kmeans.labels_):
        vector_label_dict[label].append(vector)
    cluster_distance = defaultdict(list)
    for label, distances in vector_label_dict.items():
        center_distence = 0
        for distance in distances:
            center_distence += math.dist(distance, kmeans.cluster_centers_[label])
        cluster_distance[label]=center_distence/len(distances)
    distances = list(cluster_distance.values())
    distances.sort()
    new_dict = defaultdict(list)
    for label, sentence in sentence_label_dict.items():
        if cluster_distance[label] in distances[:10]:
            new_dict[label] = sentence
    for label, sentences in new_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10,len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("------------------")

if __name__ == "__main__":
    main()
    
