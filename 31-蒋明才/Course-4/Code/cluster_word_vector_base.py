"""
 #
 # @Author: jmc
 # @Date: 2023/3/11 18:19
 # @Version: v1.0
 # @Description: 基于词向量的聚类
"""
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import jieba
jieba.initialize()
plt.rcParams["font.sans-serif"]=["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"]=False  # 解决图像中的“-”负号的乱码问题


# 加载word2vec模型
def load_model(model_path):
    model = Word2Vec.load(model_path)
    return model


# 获取句向量
def get_word_vec(model, sentence):
    words = jieba.lcut(sentence)
    vecs = np.array([.0] * 256)
    count = 0
    for word in words:
        if word != " ":
            try:
                count += 1
                vecs += model.wv[word]
            except:
                continue
    return vecs


# 绘制聚类的散点图
def cluster_graph(cluster_data):
    plt.figure(figsize=(10, 10))
    plt.title("聚类结果可视化")
    for idx, cluster in enumerate(cluster_data):
        x, y = [], []
        for data in cluster:
            x.append(data[0])
            y.append(data[1])
        plt.scatter(x, y, label="cluster_" + str(idx))
    plt.legend(loc="best")
    plt.show()


# 计算类内距离，抛弃距离质心较大的点
def cal_centroid_dis(cluster_data, cluster_centroid, cal_mode="eud"):
    if cal_mode not in ["eud", "cos"]:
        raise ValueError("cal_mode的取值只能为eud、cos")
    dis = []
    centroid = np.array(cluster_centroid)
    for ele in cluster_data:
        vec = np.array(ele)
        if cal_mode == "eud":  # 计算欧氏距离
            dis.append(np.linalg.norm(vec - centroid, ord=2))
        elif cal_mode == "cos":  # 计算余弦相似度
            num = float(np.dot(vec, centroid))  # 向量点乘
            denom = np.linalg.norm(vec) * np.linalg.norm(centroid)  # 求模长的乘积
            dis.append(0.5 + 0.5 * (num / denom) if denom != 0 else 0)
    # 抛弃大于90%分位数的值
    sort_dis = sorted(dis, reverse=False)
    val = np.percentile(np.array(sort_dis), q=90)
    save_data = []
    for idx, cur in enumerate(dis):
        if cur <= val:
            save_data.append(cluster_data[idx])
    return save_data


# kmeans聚类及可视化
def cluster_kmeans(model_path, ds_path, cluster_nums, is_drop=True, title="j聚类可视化-不抛弃类内距离大的元素"):
    vec_model = load_model(model_path)
    with open(ds_path, encoding="utf-8") as file:
        lines = file.readlines()
    ds_vecs = []
    for line in tqdm(lines, desc="计算数据集的向量空间"):
        line = line.replace("\n", "")
        ds_vecs.append(list(get_word_vec(vec_model, line)))
    ds_vecs = np.array(ds_vecs)
    clusters = KMeans(n_clusters=cluster_nums, random_state=0).fit(ds_vecs)
    # attr: cluster_centers_ labels_  ...
    centroid = clusters.cluster_centers_  # 质心
    labels = clusters.labels_  # 聚类的类别号
    cluster_data = [[] for _ in range(cluster_nums)]
    for idx, label in enumerate(labels):
        cluster_data[label].append(list(ds_vecs[idx, :]))
    new_labels = []
    for idx in range(cluster_nums):
        if is_drop:
            cluster_data[idx] = cal_centroid_dis(cluster_data[idx], centroid[idx, :], cal_mode="cos")
        new_labels += [idx] * len(cluster_data[idx])
        cluster_data[idx] = np.array(cluster_data[idx])
    # 拼接成数组模式
    cluster_data = np.vstack(cluster_data)

    cluster_data = cluster_data.transpose((1, 0))
    pca = PCA(n_components=2, random_state=0).fit(cluster_data)
    cluster_data = pca.components_.transpose((1, 0))

    pca_data = [[] for _ in range(cluster_nums)]
    for idx, label in enumerate(new_labels):
        pca_data[label].append(list(cluster_data[idx, :]))
    cluster_graph(pca_data)


if __name__ == '__main__':
    # get_word_vec(load_model("../Checkpoint/sg_softmax.model"), "新增资金入场 沪胶强势创年内新高")
    cluster_kmeans("../Checkpoint/sg_softmax.model", "../Datasets/titles.txt", 6, is_drop=False, title="聚类可视化-不抛弃类内距离大的元素")
