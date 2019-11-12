import math
import pylab as pl
import pandas as pd
import numpy as np

dataset = pd.read_csv(r"D:\MyProject\机器学习\data\watermelon_3b.csv")
dataset = np.array(dataset)
dataset = dataset.tolist()


# 计算欧几里得距离,a,b分别为两个元组
def dist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# dist_min
def dist_min(Ci, Cj):
    return min(dist(i, j) for i in Ci for j in Cj)


# dist_max
def dist_max(Ci, Cj):
    return max(dist(i, j) for i in Ci for j in Cj)


# dist_avg
def dist_avg(Ci, Cj):
    return sum(dist(i, j) for i in Ci for j in Cj) / (len(Ci) * len(Cj))


# 找到距离最小的下标
def find_Min(M):
    min = 1000
    x = 0
    y = 0
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i != j and M[i][j] < min:
                min = M[i][j]
                x = i
                y = j
    return (x, y, min)


# 算法模型：
def AGNES(dataset, dist, k):
    # 初始化C和M
    C = []
    M = []
    for i in dataset:
        Ci = []
        Ci.append(i)
        C.append(Ci)
    for i in C:
        Mi = []
        for j in C:
            Mi.append(dist(i, j))
        M.append(Mi)
    q = len(dataset)
    # 合并更新
    while q > k:
        x, y, min = find_Min(M)
        C[x].extend(C[y])
        C.remove(C[y])
        M = []
        for i in C:
            Mi = []
            for j in C:
                Mi.append(dist(i, j))
            M.append(Mi)
        q -= 1
    return C


# 画图
def draw(C):
    colValue = ["r", "y", "g", "b", "c", "k", "m"]
    for i in range(len(C)):
        coo_X = []  # x坐标列表
        coo_Y = []  # y坐标列表
        for j in range(len(C[i])):
            coo_X.append(C[i][j][0])
            coo_Y.append(C[i][j][1])
        pl.scatter(coo_X, coo_Y, marker="x", color=colValue[i % len(colValue)], label=i)

    pl.legend(loc="upper right")
    pl.show()


C = AGNES(dataset, dist_avg, 3)
draw(C)
