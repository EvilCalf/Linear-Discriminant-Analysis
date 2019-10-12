import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import operator
import pandas as pd

# 特征字典，后面用到了好多次，干脆当全局变量了
featureDic = {
    "色泽": ["浅白", "青绿", "乌黑"],
    "根蒂": ["硬挺", "蜷缩", "稍蜷"],
    "敲声": ["沉闷", "浊响", "清脆"],
    "纹理": ["清晰", "模糊", "稍糊"],
    "脐部": ["凹陷", "平坦", "稍凹"],
    "触感": ["硬滑", "软粘"],
}


def getDataSet():
    """
    get watermelon data set 3.0.
    :return: 编码好的数据集以及特征的字典。
    """
    dataSet = pd.read_csv("D:\MyProject\机器学习\data\watermelon_3.csv")
    dataSet = dataSet.values[:, 1:]

    features = ["色泽", "根蒂", "敲声", "纹理", "脐部", "触感", "密度", "含糖量"]
    numList = []  # [3, 3, 3, 3, 3, 2]
    for i in range(len(features) - 2):
        numList.append(len(featureDic[features[i]]))

    dataSet = np.array(dataSet)
    return dataSet, features


def AODE(dataSet, data, features):
    """
    AODE(Averaged One-Dependent Estimator)。意思为尝试将每个属性作为超父来构建SPODE。
    :param dataSet:
    :param data:
    :param features:
    :return:
    """
    m, n = dataSet.shape
    n = n - 3  # 特征不取连续值的属性，如密度和含糖量。
    pDir = {}  # 保存三个值。好瓜的可能性，坏瓜的可能性，和预测的值。
    for classLabel in ["好瓜", "坏瓜"]:
        P = 0.0
        if classLabel == "好瓜":
            sign = "1"
        else:
            sign = "0"
        extrDataSet = dataSet[dataSet[:, -1] == sign]  # 抽出类别为sign的数据
        for i in range(n):  # 对于第i个特征
            xi = data[i]
            # 计算classLabel类，第i个属性上取值为xi的样本对总数据集的占比
            Dcxi = extrDataSet[extrDataSet[:, i] == xi]  # 第i个属性上取值为xi的样本数
            Ni = len(featureDic[features[i]])  # 第i个属性可能的取值数
            Pcxi = (len(Dcxi) + 1) / float(m + 2 * Ni)
            # 计算类别为c且在第i和第j个属性上分别为xi和xj的样本，对于类别为c属性为xi的样本的占比
            mulPCond = 1
            for j in range(n):
                xj = data[j]
                Dcxij = Dcxi[Dcxi[:, j] == xj]
                Nj = len(featureDic[features[j]])
                PCond = (len(Dcxij) + 1) / float(len(Dcxi) + Nj)
                mulPCond *= PCond
            P += Pcxi * mulPCond
        pDir[classLabel] = P

    if pDir["好瓜"] > pDir["坏瓜"]:
        preClass = "好瓜"
    else:
        preClass = "坏瓜"

    return pDir["好瓜"], pDir["坏瓜"], preClass


def calcAccRate(dataSet, features):
    """
    计算准确率
    :param dataSet:
    :param features:
    :return:
    """
    cnt = 0
    for data in dataSet:
        _, _, pre = AODE(dataSet, data, features)
        if (pre == "好瓜" and data[-1] == "1") or (pre == "坏瓜" and data[-1] == "0"):
            cnt += 1
    return cnt / float(len(dataSet))


def main():
    dataSet, features = getDataSet()
    pG, pB, pre = AODE(dataSet, dataSet[0], features)
    print("pG = ", pG)
    print("pB = ", pB)
    print("pre = ", pre)
    print("real class = ", dataSet[0][-1])
    print(calcAccRate(dataSet, features))


if __name__ == "__main__":
    main()
