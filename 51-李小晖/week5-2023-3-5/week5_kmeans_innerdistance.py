import numpy as np
import random
import sys
import matplotlib.pyplot as plt
'''
Kmeans算法实现
原文链接：https://blog.csdn.net/qingchedeyongqi/article/details/116806277
'''

class KMeansClusterer:  # k均值聚类
    def __init__(self, ndarray, cluster_num):
        self.ndarray = ndarray
        self.cluster_num = cluster_num
        self.points = self.__pick_start_point(ndarray, cluster_num)

    def cluster(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
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
            return result, self.points, sum
        self.points = np.array(new_center)
        return self.cluster()

    def cluster_sort_distance(self):
        result = []
        for i in range(self.cluster_num):
            result.append([])
        for item in self.ndarray:
            distance_min = sys.maxsize
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
            avg_diastance = self.__inner_distance(self.points, result)
            return result, self.points, sum, avg_diastance
        self.points = np.array(new_center)
        return self.cluster_sort_distance()

    def __inner_distance(self, centers, points):
        assert len(centers) == len(points)
        avg_diastance = []
        clusters = len(centers)
        for i in range(clusters):
            sum_distance = 0.0
            point_counts = len(points[i])
            for j in range(point_counts):
                sum_distance += self.__distance(points[i][j], centers[i])
            avg_diastance.append(sum_distance / point_counts)
        return avg_diastance

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

    def __pick_start_point(self, ndarray, cluster_num):
        if cluster_num < 0 or cluster_num > ndarray.shape[0]:
            raise Exception("簇数设置有误")
        # 取点的下标
        indexes = random.sample(np.arange(0, ndarray.shape[0], step=1).tolist(), cluster_num)
        points = []
        for index in indexes:
            points.append(ndarray[index].tolist())
        return np.array(points)

classes = 4
x = np.random.randint(low=0, high=100, size=(20, 2))
kmeans = KMeansClusterer(x, classes)
result, centers, distances, avg_diastance = kmeans.cluster_sort_distance()
#进行距离排序
sorted_distance = np.argsort(avg_diastance)

print("类内平均距离为：", avg_diastance)
print("类内距离升序排序之后的序号为：", sorted_distance)
#输出和绘制图形
plt.figure()
style = ["ro", "g+", "bx", "y*"]
handlesq = []

print("==============经过kmeans之后的结果为==============")
print("总距离：", distances)
print("按照每一类类内平均距离升序排序结果为：")

labels = []
for i in range(classes):
    print("==================")
    print("第{}类在原始数据中的序号为:{}, 类内平均距离为：{}".format(i + 1, sorted_distance[i] + 1, avg_diastance[sorted_distance[i]]))
    print("第{}类所有数据的中心为：{}".format(i + 1, centers[sorted_distance[i]]))
    data = result[sorted_distance[i]]
    print("第{}类所有数据为：{}".format(i + 1, data))
    #print(data)
    x_data = [item[0] for item in data]
    #print(x_data)
    y_data = [item[1] for item in data]
    #print(y_data)
    print("==================")
    handle = plt.plot(x_data, y_data, style[i], label = "kmeans")
    handlesq.append(handle)
    #绘制中心点
    #x_data = centers[i][0]
    #print(x_data)
    #y_data = centers[i][1]
    #print(y_data)
    #plt.plot(x_data, y_data, "b*")
    #plt.legend(handles=[ x for x in handlesq], labels=["1", "2", "3", "4"])
    label = r"{} Distance {}".format(sorted_distance[i] + 1, avg_diastance[sorted_distance[i]])
    labels.append(label)
plt.legend(labels)
plt.show()