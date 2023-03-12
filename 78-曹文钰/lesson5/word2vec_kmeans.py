import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

from db import Mysql
from word2vec_train import load_word2vec_model, build_corpus

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
#将文本向量化
def news_to_vectors(news, model):
    vectors = []
    for new in news:
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in new:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(new))
    return np.stack(vectors)


def main():
    model = load_word2vec_model("model.w2v") #加载词向量模型
    news = build_corpus(asr=True)   
    titles = build_corpus(asr=False)
    vectors = news_to_vectors(news, model)   #将所有标题向量化

    kmeans = KMeans(20)  #定义一个kmeans计算类
    kmeans.fit(vectors)  #进行聚类计算

    #{label:[(new_index, sim), (new_index, sim)], label:[(new_index, sim), (new_index, sim)]}
    new_label_dict = defaultdict(list)
    for index, label in enumerate(kmeans.labels_):
        # print(np.asarray(vectors[index]),np.asarray(kmeans.cluster_centers_[label]))
        # print(np.asarray(vectors[index]).shape,np.asarray(kmeans.cluster_centers_[label].shape))
        sim = np.dot(np.asarray(vectors[index]),np.asarray(kmeans.cluster_centers_[label]).T)
        new_label_dict[label].append((index, sim))

    for label, ls in new_label_dict.items():
        print("cluster :", label)
        ls = sorted(ls, key=lambda x: x[1], reverse=True)
        i = 1
        for (new_index, sim) in ls[:10]:
            row_elem = [str(i), str(sim), "".join(titles[new_index]), "".join(news[new_index])[len("".join(titles[new_index])):len("".join(titles[new_index]))+15]+"......"]
            print(", ".join(row_elem))
            i += 1
        print("---------")

if __name__ == "__main__":
    main()

