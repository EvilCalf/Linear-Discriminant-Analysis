import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def sigmoid(x):  # Sigmoid函数
    """
    Sigmoid function.
    Input:
        x:np.array
    Return:
        y: the same shape with x
    """
    y = 1.0 / (1 + np.exp(-x))
    return y


def newton(X, y):  # 牛顿法
    """
    Input:
        X: np.array with shape [N, 3]. Input.
        y: np.array with shape [N, 1]. Label.
    Return:
        weight: np.array with shape [1, 3]. Optimal params with newton method
    """
    #N条数据
    N = X.shape[0]
    #初始权值
    weight = np.ones((1, 3))
    #训练集的数据与当前的权值矩阵相乘，得预测值
    PredictiveValue = X.dot(weight.T)
    #log-likehood
    old_l = 0
    new_l = np.sum(y * PredictiveValue + np.log(1 + np.exp(PredictiveValue)))
    iters = 0
    while (np.abs(old_l - new_l) > 1e-5):
        #y=1的概率，shape [N, 1]
        p1 = np.exp(PredictiveValue) / (1 + np.exp(PredictiveValue))
        #转化成对角矩阵，ln（y（1-y))=ω^Tx+b为线性
        p = np.diag((p1 * (1 - p1)).reshape(N))
        #一阶导数shape [1, 3]
        First_Derivative = -np.sum(X * (y - p1), 0, keepdims=True)
        #二阶导数shape [3, 3]
        Second_Derivative = X.T.dot(p).dot(X)

        #更新，即原先权值-一阶导数和二阶导数逆矩阵的乘积
        weight -= First_Derivative.dot(np.linalg.inv(Second_Derivative))
        PredictiveValue = X.dot(weight.T)
        old_l = new_l
        new_l = np.sum(y * PredictiveValue + np.log(1 + np.exp(PredictiveValue)))

        iters += 1
    print("iters: ", iters)
    print(new_l)
    return weight


def gradDescent(X, y):  # 梯度下降法
    """
    Input:
        X: np.array with shape [N, 3]. Input.
        y: np.array with shape [N, 1]. Label.
    Return:
        weight: np.array with shape [1, 3]. Optimal params with gradient descent method
    """
    #N条数据
    N = X.shape[0]
    #学习率
    learningRate = 0.05
    #初始化权值
    weight = np.ones((1, 3)) * 0.1
    #训练集的数据与当前的权值矩阵相乘，得预测值
    PredictiveValue = X.dot(weight.T)
    old_l = 0
    new_l = np.sum(y * PredictiveValue + np.log(1 + np.exp(PredictiveValue)))
    iters = 0
    while (np.abs(old_l - new_l) > 1e-5):
        #y=1的概率，shape [N, 1]
        p1 = np.exp(PredictiveValue) / (1 + np.exp(PredictiveValue))
        #转化成对角矩阵，ln（y（1-y))=ω^Tx+b为线性
        p = np.diag((p1 * (1 - p1)).reshape(N))
        #一阶导数shape [1, 3]
        First_Derivative = -np.sum(X * (y - p1), 0, keepdims=True)

        #更新，朝着梯度下降方向前进，并更新预测值
        weight -= First_Derivative * learningRate
        PredictiveValue = X.dot(weight.T)
        old_l = new_l
        new_l = np.sum(y * PredictiveValue + np.log(1 + np.exp(PredictiveValue)))
        iters += 1

    print("iters: ", iters)
    print(new_l)
    return weight


if __name__ == "__main__":

    #read data from csv file
    workbook = pd.read_csv("D:\MyProject\机器学习\data\watermelon_3a.csv",
                           header=None)
    #在序号三的位置扩展了一列，全为1，使得后续权值可以合并ω和b
    workbook.insert(3, "3", 1)
    X = workbook.values[:, 1:-1]  #截取从第二列到倒数第一列为止，不含最后一列
    y = workbook.values[:, 4].reshape(-1, 1)  #y为第5列

    #分别取出正例和反例
    positive_data = workbook.values[workbook.values[:, 4] == 1.0, :]
    negative_data = workbook.values[workbook.values[:, 4] == 0, :]

    plt.plot(positive_data[:, 1], positive_data[:, 2], 'bo')
    plt.plot(negative_data[:, 1], negative_data[:, 2], 'r+')

    #牛顿法绿色
    weight = newton(X, y)
    newton_left = -(weight[0, 0] * 0.1 + weight[0, 2]) / weight[0, 1]
    newton_right = -(weight[0, 0] * 0.9 + weight[0, 2]) / weight[0, 1]
    plt.plot([0.1, 0.9], [newton_left, newton_right], 'g-')

    #梯度下降黄色
    weight = gradDescent(X, y)
    grad_descent_left = -(weight[0, 0] * 0.1 + weight[0, 2]) / weight[0, 1]
    grad_descent_right = -(weight[0, 0] * 0.9 + weight[0, 2]) / weight[0, 1]
    plt.plot([0.1, 0.9], [grad_descent_left, grad_descent_right], 'y-')

    plt.xlabel('密度', fontproperties="SimHei")
    plt.ylabel('含糖率', fontproperties="SimHei")
    plt.title("対率回归结果ln(y(1-y))", fontproperties="SimHei")
    plt.show()
    """
    最终牛顿法和梯度下降法求得的loss几乎一样，但是梯度下降是牛顿法的1000倍
    当然牛顿法在高维度的矩阵求逆求导等等计算量过大
    所以这就是为啥梯度下降还是使用最多的原因吧！
    """