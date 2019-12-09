from collections import defaultdict
from math import sqrt
import pandas as pd

# 使用ARI进行K-means聚类性能评估
from sklearn import metrics


# 0:0 1 8 15 26
# 1:11 24 28 35 4
# 2:5 25 37 41 48
# 3:14 47 59 71 73
# 4:3 16 19 39 46
# 5:6 7 13 22 65
# 6:4 27 33 45 54
# 7:2 10 17 20 30
# 8:9 18 21 29 49
# 9:12 23 34 36 38

# 读取文件
def readdata():
    dataset = []
    with open("D:\MyProject\Machine Learning\data\optdigits.tra", "r") as file:
        for line in file:
            if line == "\n":
                continue
            # map()是 Python 内置的高阶函数，它接收一个函数 f 和一个 list，
            # 并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回。
            dataset.append(list(map(int, line.split(",")[:-1])))
        file.close()
        return dataset


# 读取种子集（有标签数据）
def readseed(dataset, label):
    seedset = defaultdict(list)
    for i in range(10):
        for j in range(len(label[0])):
            seedset[i].append(dataset[label[i][j]])
    return seedset


# 输入：每一类的数据
# 输出：这一类新的质心
# 方法：求出这一类数据的总和求平均值
def point_avg(points):
    dimensions = len(points[0])
    new_center = []
    for dimension in range(dimensions):
        sum = 0
        # 求出每一列的数据总和
        for p in points:
            sum += p[dimension]
        new_center.append(float("%.8f" % (sum / float(len(points)))))
    return new_center


# 输入：数据集，质心数据，质心数量
# 输出：新的质心数据
# 功能：更新质心数据
def update_centers(data_set, assignments, k):
    # new_means，记录前一次学习的标签结果
    new_means = defaultdict(list)
    centers = []
    for assignment, point in zip(assignments, data_set):
        new_means[assignment].append(point)
    for i in range(k):
        # 逐一对每一类进行质心更新
        points = new_means[i]
        centers.append(point_avg(points))
    return centers


# 输入：数据集，质心
# 输出：数组，每一个单元代表对应数据的类别。如assignment[0]=0，代表第0个数据的类别是第0类
# 功能：求数据集每个单元数据的标签
def assign_points(data_points, centers, label):
    assignments = []
    index = 0
    for point in data_points:
        flag = 1
        # 有标签数据，直接填写对应标签
        for k in range(len(label)):
            if index in label[k]:
                index = index + 1
                assignments.append(k)
                flag = 0
                break
        if flag == 0:
            continue
        # 无标签数据求类别
        # float('inf')表示正无穷
        shortest = float("inf")
        shortest_index = 0
        # 取距离最近的质心的下标
        for i in range(len(centers)):
            value = distance(point, centers[i])
            if value < shortest:
                shortest = value
                shortest_index = i
        assignments.append(shortest_index)
        index = index + 1
    return assignments


# 求欧拉距离
def distance(a, b):
    dimention = len(a)
    sum = 0
    for i in range(dimention):
        sq = (a[i] - b[i]) ** 2
        sum += sq
    return sqrt(sum)


# 输入：数据集，质心个数
# 输出：质心数据
# 功能：用随机数的方法生成初始质心
def generate_k(seedset, k):
    centers = []
    for i in range(k):
        point = seedset[i]
        centers.append(point_avg(point))
    return centers


def k_means(dataset, seedset, k, Y_train, label):
    # 求初始质心
    k_points = generate_k(seedset, k)
    # 求标签
    assignments = assign_points(dataset, k_points, label)
    old_assignments = None

    # 直到标签没变，停止
    while assignments != old_assignments:
        # 更新质心数据
        new_centers = update_centers(dataset, assignments, k)
        old_assignments = assignments
        # 更新标签
        assignments = assign_points(dataset, new_centers, label)

    # 输出准确率
    precision_rate = metrics.adjusted_rand_score(Y_train, assignments)
    print(precision_rate)


def main():
    # 标签集
    label = [
        [0, 1, 8, 15, 26],
        [11, 24, 28, 35, 42],
        [5, 25, 37, 41, 48],
        [14, 47, 59, 71, 73],
        [3, 16, 19, 39, 46],
        [6, 7, 13, 22, 65],
        [4, 27, 33, 45, 54],
        [2, 10, 17, 20, 30],
        [9, 18, 21, 29, 49],
        [12, 23, 34, 36, 38],
    ]
    dataset = readdata()
    seedset = readseed(dataset, label)
    # 使用pandas分别读取训练数据与测试数据
    digits_train = pd.read_csv(
        r"D:\MyProject\Machine Learning\data\optdigits.tra", header=None
    )
    Y_train = digits_train[64]
    k_means(dataset, seedset, 10, Y_train, label)


if __name__ == "__main__":
    main()
