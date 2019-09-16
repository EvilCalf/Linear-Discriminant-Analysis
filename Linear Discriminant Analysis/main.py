import numpy as np
import matplotlib.pyplot as plt

data = [
    [0.697, 0.460, 1],
    [0.774, 0.376, 1],
    [0.634, 0.264, 1],
    [0.608, 0.318, 1],
    [0.556, 0.215, 1],
    [0.403, 0.237, 1],
    [0.481, 0.149, 1],
    [0.437, 0.211, 1],
    [0.666, 0.091, 0],
    [0.243, 0.267, 0],
    [0.245, 0.057, 0],
    [0.343, 0.099, 0],
    [0.639, 0.161, 0],
    [0.657, 0.198, 0],
    [0.360, 0.370, 0],
    [0.593, 0.042, 0],
    [0.719, 0.103, 0],
]  # 书中89页西瓜数据集
# 数据集按瓜好坏分类
data = np.array([i[:-1] for i in data])  # [:-1]去除最后一个字符
X0 = np.array(data[:8])  # 读到序列7截止
X1 = np.array(data[8:])  # 从序列8的元素开始读
# 求正反例均值
# axis = 0：压缩行，对各列求均值，返回 1* n 矩阵，reshape(-1,1)转换成1列：
μ0 = np.mean(X0, axis=0).reshape((-1, 1))
μ1 = np.mean(X1, axis=0).reshape((-1, 1))
# 求协方差
cov0 = np.cov(X0, rowvar=False)
cov1 = np.cov(X1, rowvar=False)
# 求出ω
S_w = np.mat(cov0 + cov1)  # 类内散度矩阵
ω = S_w.I * (μ0 - μ1)  #求出直线ω方向，用散度矩阵的逆矩阵乘上均值之差（均值肯定在直线上，肯定同方向）
# 画出点、直线
plt.scatter(X0[:, 0], X0[:, 1], c="b", label="+", marker="+")
plt.scatter(X1[:, 0], X1[:, 1], c="r", label="-", marker="_")
plt.plot([0, 1], [0, -ω[0] / ω[1]])
plt.xlabel("密度", fontproperties="SimHei", fontsize=15, color="green")
plt.ylabel("含糖率", fontproperties="SimHei", fontsize=15, color="green")
plt.title(r"线性判别分析", fontproperties="SimHei", fontsize=25)
plt.legend()
plt.show()
