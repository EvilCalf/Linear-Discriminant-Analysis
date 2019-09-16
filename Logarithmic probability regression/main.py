# encoding: utf-8
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
        beta: np.array with shape [1, 3]. Optimal params with newton method
    """
    #N条数据
    N = X.shape[0]
    #初始权值
    beta = np.ones((1, 3))
    #训练集的数据与当前的权值矩阵相乘，得预测值
    z = X.dot(beta.T)
    #log-likehood
    old_l = 0
    new_l = np.sum(y * z + np.log(1 + np.exp(z)))
    iters = 0
    while (np.abs(old_l - new_l) > 1e-5):
        #y=1的概率，shape [N, 1]
        p1 = np.exp(z) / (1 + np.exp(z))
        #转化成对角矩阵，在ω和b条件下的概率
        p = np.diag((p1 * (1 - p1)).reshape(N))
        #一阶导数shape [1, 3]
        First_Derivative = -np.sum(X * (y - p1), 0, keepdims=True)
        #二阶导数shape [3, 3]
        Second_Derivative = X.T.dot(p).dot(X)

        #更新，即原先权值-一阶导数和二阶导数逆矩阵的乘积
        beta -= First_Derivative.dot(np.linalg.inv(Second_Derivative))
        z = X.dot(beta.T)
        old_l = new_l
        new_l = np.sum(y * z + np.log(1 + np.exp(z)))

        iters += 1
    print("iters: ", iters)
    print(new_l)
    return beta


def gradDescent(X, y):  # 梯度下降法
    """
    Input:
        X: np.array with shape [N, 3]. Input.
        y: np.array with shape [N, 1]. Label.
    Return:
        beta: np.array with shape [1, 3]. Optimal params with gradient descent method
    """
    #N条数据
    N = X.shape[0]
    #学习率
    lr = 0.05
    #初始化权值
    beta = np.ones((1, 3)) * 0.1
    #训练集的数据与当前的权值矩阵相乘，得预测值
    z = X.dot(beta.T)
    old_l = 0
    new_l = np.sum(y * z + np.log(1 + np.exp(z)))
    iters = 0
    while (np.abs(old_l - new_l) > 1e-5):
        #y=1的概率，shape [N, 1]
        p1 = np.exp(z) / (1 + np.exp(z))
        #转化成对角矩阵，在ω和b条件下的概率
        p = np.diag((p1 * (1 - p1)).reshape(N))
        #一阶导数shape [1, 3]
        First_Derivative = -np.sum(X * (y - p1), 0, keepdims=True)

        #更新，朝着梯度下降方向前进，并更新预测值
        beta -= First_Derivative * lr
        z = X.dot(beta.T)
        old_l = new_l
        new_l = np.sum(y * z + np.log(1 + np.exp(z)))
        iters += 1

    print("iters: ", iters)
    print(new_l)
    return beta


if __name__ == "__main__":

    #read data from csv file
    workbook = pd.read_csv("D:\MyProject\机器学习\data\watermelon_3a.csv", header=None)
    #在序号三的位置扩展了一列，全为1，使得后续权值可以合并ω和b
    workbook.insert(3, "3", 1) 
    X = workbook.values[:, 1:-1] #截取从第二列到倒数第一列为止，不含最后一列
    y = workbook.values[:, 4].reshape(-1, 1) #y为第5列

    #分别取出正例和反例
    positive_data = workbook.values[workbook.values[:, 4] == 1.0, :]
    negative_data = workbook.values[workbook.values[:, 4] == 0, :]

    
    plt.plot(positive_data[:, 1], positive_data[:, 2], 'bo')
    plt.plot(negative_data[:, 1], negative_data[:, 2], 'r+')

    #牛顿法绿色
    beta = newton(X, y)
    newton_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    newton_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [newton_left, newton_right], 'g-')

    #梯度下降黄色
    beta = gradDescent(X, y)
    grad_descent_left = -(beta[0, 0] * 0.1 + beta[0, 2]) / beta[0, 1]
    grad_descent_right = -(beta[0, 0] * 0.9 + beta[0, 2]) / beta[0, 1]
    plt.plot([0.1, 0.9], [grad_descent_left, grad_descent_right], 'y-')

    plt.xlabel('密度',fontproperties="SimHei")
    plt.ylabel('含糖率',fontproperties="SimHei")
    plt.title("対率回归结果",fontproperties="SimHei")
    plt.show()