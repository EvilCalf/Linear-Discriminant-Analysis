import numpy as np


def loadDataSet(filename):
    """
    加载数据集
    """
    dataList = []
    with open(filename) as fr:
        for line in fr:
            curLine = line.strip().split("\t")
            fltLine = list(map(float, curLine))
            dataList.append(fltLine)
    return np.mat(dataList)


def distEclud(vecA, vecB):
    """
    计算两个向量间的欧氏距离
    Args:
        vecA: 向量A
        vecB: 向量B
    Return:
        距离
    """
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    """
    构建簇质心
    Args:
        dataSet: 数据集
        k: 簇个数
    Return:
        centroids: 簇质心
    """
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(dataSet[:, j])
        rangeJ = float(np.max(dataSet[:, j]) - minJ)
        # 保证随机生成的质心在整个数据集的边界之内
        centroids[:, j] = minJ + np.random.rand(k, 1) * rangeJ
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    """
    k均值聚类算法
    Args:
        dataSet: 数据集
        k：需要聚类的簇的个数
        distMeas=distEclud：距离度量函数
        createCent=randCent：随机生成簇质心函数
    Return:
        centroids: 最终簇质心
        clusterAssment: 最终点聚类结果
    """
    m = np.shape(dataSet)[0]
    # 第一个保存所属质心，第二个保存距离
    clusterAssment = np.mat(np.zeros((m, 2)))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 将样本分配到相应的簇中
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # 更新簇质心
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


dataMat = loadDataSet(r"D:\MyProject\Machine Learning\data\KmeansData.txt")
myCentroids, clusterAssing = kMeans(dataMat, 4)

import matplotlib
import matplotlib.pyplot as plt


def showPlt(datMat, alg=kMeans, numClust=4):
    myCentroids, clustAssing = alg(datMat, numClust)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ["s", "o", "^", "8", "p", "d", "v", "h", ">", "<"]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label="ax0", **axprops)
    ax1 = fig.add_axes(rect, label="ax1", frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(
            ptsInCurrCluster[:, 0].flatten().A,
            ptsInCurrCluster[:, 1].flatten().A,
            marker=markerStyle,
            s=90,
        )
    ax1.scatter(
        myCentroids[:, 0].flatten().A[0],
        myCentroids[:, 1].flatten().A[0],
        marker="+",
        s=300,
    )
    plt.show()


showPlt(dataMat)
