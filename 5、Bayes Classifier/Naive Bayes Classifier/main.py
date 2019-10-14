import numpy as np
import pandas as pd

dataset = pd.read_csv("D:\MyProject\机器学习\data\watermelon_3.csv", delimiter=",")
del dataset["编号"]
print(dataset)
X = dataset.values[:, :-1]
m, n = np.shape(X)
for i in range(m):
    X[i, n - 1] = round(X[i, n - 1], 3)
    X[i, n - 2] = round(X[i, n - 2], 3)
y = dataset.values[:, -1]
columnName = dataset.columns
colIndex = {}
for i in range(len(columnName)):
    colIndex[columnName[i]] = i

Pmap = {}  # memory the P to avoid the repeat computing
kindsOfAttribute = (
    {}
)  # kindsOfAttribute[0]=3 because there are 3 different types in '色泽'
# this map is for laplacian correction
for i in range(n):
    kindsOfAttribute[i] = len(set(X[:, i]))
continuousPara = (
    {}
)  # memory some parameters of the continuous data to avoid repeat computing

goodList = []
badList = []
for i in range(len(y)):
    if y[i] == "是":
        goodList.append(i)
    else:
        badList.append(i)

import math


def P(colID, attribute, C):  # P(colName=attribute|C) P(色泽=青绿|是)
    if (colID, attribute, C) in Pmap:
        return Pmap[(colID, attribute, C)]
    curJudgeList = []
    if C == "是":
        curJudgeList = goodList
    else:
        curJudgeList = badList
    ans = 0
    if colID >= 6:  # density or ratio which are double type data
        mean = 1
        std = 1
        if (colID, C) in continuousPara:
            curPara = continuousPara[(colID, C)]
            mean = curPara[0]
            std = curPara[1]
        else:
            curData = X[curJudgeList, colID]
            mean = curData.mean()
            std = curData.std()
            continuousPara[(colID, C)] = (mean, std)
        ans = (
            1
            / (math.sqrt(math.pi * 2) * std)
            * math.exp((-(attribute - mean) ** 2) / (2 * std * std))
        )
    else:
        for i in curJudgeList:
            """
            计算当前类别下的取值(列如颜色中的乌黑)的条件概率
            即求出当前(属性的个数+1)/(类别总数+可能的取值数)
            """
            if X[i, colID] == attribute:
                ans += 1
        ans = (ans + 1) / (len(curJudgeList) + kindsOfAttribute[colID])
    Pmap[(colID, attribute, C)] = ans
    return ans


def predictOne(single):
    """
    通过拉普拉斯修正，估计先验概率P(c)
    再为每个属性估计条件概率P(xi|c),求其联合分布概率(相乘)，对数相加
    得出正例反例的概率进行比较得大者结论
    """
    ansYes = math.log2((len(goodList) + 1) / (len(y) + 2))
    ansNo = math.log2((len(badList) + 1) / (len(y) + 2))
    for i in range(len(single)):
        ansYes += math.log2(P(i, single[i], "是"))
        ansNo += math.log2(P(i, single[i], "否"))
    if ansYes > ansNo:
        return "是"
    else:
        return "否"


def predictAll(iX):
    predictY = []
    for i in range(m):
        predictY.append(predictOne(iX[i]))
    return predictY


predictY = predictAll(X)
print(y)
print(np.array(predictAll(X)))

confusionMatrix = np.zeros((2, 2))
for i in range(len(y)):
    if predictY[i] == y[i]:
        if y[i] == "否":
            confusionMatrix[0, 0] += 1
        else:
            confusionMatrix[1, 1] += 1
    else:
        if y[i] == "否":
            confusionMatrix[0, 1] += 1
        else:
            confusionMatrix[1, 0] += 1
print(confusionMatrix)
