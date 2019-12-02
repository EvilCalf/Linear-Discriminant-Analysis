import math
import numpy as np
import pylab as pl
import pandas as pd


# 数据处理：得到训练数据集dataset
def get_dataset():
    # 西瓜数据集：每三个一组（编号，密度，含糖量）
    data = """
    1,0.697,0.46,2,0.774,0.376,3,0.634,0.264,4,0.608,0.318,5,0.556,0.215,
    6,0.403,0.237,7,0.481,0.149,8,0.437,0.211,9,0.666,0.091,10,0.243,0.267,
    11,0.245,0.057,12,0.343,0.099,13,0.639,0.161,14,0.657,0.198,15,0.36,0.37,
    16,0.593,0.042,17,0.719,0.103,18,0.359,0.188,19,0.339,0.241,20,0.282,0.257,
    21,0.748,0.232,22,0.714,0.346,23,0.483,0.312,24,0.478,0.437,25,0.525,0.369,
    26,0.751,0.489,27,0.532,0.472,28,0.473,0.376,29,0.725,0.445,30,0.446,0.459
    """
    a = data.split(",")
    return [(float(a[i]), float(a[i + 1])) for i in range(1, len(a) - 1, 3)]


# 计算欧几里得距离：a,b分别为两个元组
def dist(a, b):
    return math.sqrt(math.pow(a[0] - b[0], 2) + math.pow(a[1] - b[1], 2))


# 算法模型
# 参数：邻域参数e, Minpts，dist距离函数
def DBSCAN(D, e, Minpts, dist):
    T = set()  # 核心对象集合T
    k = 0  # 聚类个数k
    C = []  # 聚类集合C
    P = set(D)  # 未访问集合P
    # 计算核心对象
    for d in D:
        if len([i for i in D if dist(d, i) <= e]) >= Minpts:
            T.add(d)
    # 开始聚类
    while len(T):
        P_old = P
        o = list(T)[np.random.randint(0, len(T))]
        P = P - set(o)
        Q = []
        Q.append(o)
        while len(Q):
            q = Q[0]
            Nq = [i for i in D if dist(q, i) <= e]
            if len(Nq) >= Minpts:
                S = P & set(Nq)
                Q += list(S)
                P = P - S
            Q.remove(q)
        k += 1
        Ck = list(P_old - P)
        T = T - set(Ck)
        C.append(Ck)
    return C, k


# 训练结果可视化
def draw(C):
    color = ["r", "y", "g", "b", "c", "k", "m"]
    for i in range(len(C)):
        x = []  # x坐标列表
        y = []  # y坐标列表
        for j in range(len(C[i])):
            x.append(C[i][j][0])
            y.append(C[i][j][1])
        pl.scatter(x, y, marker="x", color=color[i % len(color)], label=i + 1)
    pl.legend(loc="upper left")
    pl.title("DBSCAN")
    pl.show()


if __name__ == "__main__":
    # 数据处理得到训练数据集
    dataset = get_dataset()
    # 设置邻域参数
    e = 0.11
    Minpts = 5
    # 密度聚类得到k个聚类簇：以dist函数作为距离度量
    C, k = DBSCAN(dataset, e, Minpts, dist)
    # 聚类结果展示
    draw(C)
