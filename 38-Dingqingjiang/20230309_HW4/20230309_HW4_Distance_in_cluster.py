import numpy as np
import random
import sys
'''
Kmeans算法实现
class KMeansClusterer中增加self.__inner_distance(self, result, cenpoints)方法，计算每个类内的平均距离
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num): # 初始化，属性ndarray,cluster_num
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num) # 初始化起始点列表，给每个类一个起始点

    def cluster(self): # 主要的聚类方法
        result = [] # 建一个结果列表
        for i in range(self.cluster_num): # 遍历每一个类
            result.append([]) # 在结果中为每个类建立一个空表
        for item in self.ndarray: # 遍历每一个输入点
            distance_min = sys.maxsize # 系统最大值
            index = -1
            for i in range(len(self.points)):
                distance = self.__distance(item, self.points[i])
                if distance < distance_min:
                    distance_min = distance
                    index = i
            result[index] = result[index] + [item.tolist()]
        new_center = []
        for item in result:
            new_center.append(self.__center(item).tolist())
        # 中心点未改变，说明达到稳态，结束递归
        if (self.points == new_center).all():
            sum = self.__sumdis(result)
            # 计算类内平均距离
            inner_distance = self.__inner_distance(result, self.points)
            return result, self.points, sum, inner_distance
        self.points = np.array(new_center)
        return self.cluster()

    def __sumdis(self,result):
        #计算总距离和
        sum=0
        for i in range(len(self.points)):
            for j in range(len(result[i])):
                sum+=self.__distance(result[i][j],self.points[i])
        return sum

    def __center(self, list):
        # 计算每一列的平均值
        return np.array(list).mean(axis=0)

    def __distance(self, p1, p2):
        #计算两点间距
        tmp = 0
        for i in range(len(p1)):
            tmp += pow(p1[i] - p2[i], 2)
        return pow(tmp, 0.5)

    def __inner_distance(self, result, cenpoints):
        #计算平均类内距离，点到类中心的距离
        dis = []
        for cluster_num in range(len(result)):
            tmp = 0
            for respoint in result[cluster_num]:
                tmp += self.__distance(respoint,cenpoints[cluster_num])
                dis.append([tmp])
        return dis

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]: # 设置簇数，在0-输入点总数之间
            raise Exception("簇数设置有误") # 否则报错
        # 随机取cluster_num个点的下标，即分成cluster_num类，每个类挑一个起始点
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        # np.arange()用于生成等差数组。三个参数分别是，起点、终点和步长
        # .tolist()将numpy.ndarray类转为list类
        # random.sample(seq, n) 从序列seq中选择n个随机且独立的元素
        points = [] # 起始点列表
        for index in indexes: # 遍历所有的下标
            points.append(ndarray[index].tolist()) # 起始点列表中添加上面选到的随机起始点
        return np.array(points) # 返回np.array类型的points列表

x = np.random.rand(100, 8)
kmeans = KMeansClusterer(x, 10)
result, centers, distances, inner_distance = kmeans.cluster()
print(result)
print(centers)
print(distances)
print(inner_distance)