import numpy as np

#learningRate学习率，Loopnum迭代次数


def liner_Regression(data_x, data_y, learningRate, Loopnum):
    Weight = np.ones(shape=(1, data_x.shape[1]))
    #全为1.0的array，和data_x.shape[1]相同大小当前是3
    baise = np.array([[1]])  #b初始为1
    for num in range(Loopnum):
        WXPlusB = np.dot(data_x, Weight.T) + baise
        # 求出当前权值下得出的预测值y
        loss = np.dot((data_y - WXPlusB).T, data_y - WXPlusB) / data_y.shape[0]
        #loss就是实际值与预测值的差的平均数
        """
        求矩阵导数，得到新的变化率w和b，b的求法也就是相当于矩阵乘全1的那一列
        """
        w_gradient = -(2 / data_x.shape[0]) * np.dot(
            (data_y - WXPlusB).T, data_x)
        baise_gradient = -2 * np.dot(
            (data_y - WXPlusB).T,
            np.ones(shape=[data_x.shape[0], 1])) / data_x.shape[0]
        #得到新的权值
        Weight = Weight - learningRate * w_gradient
        baise = baise - learningRate * baise_gradient
        if num % 50 == 0:
            print(loss)  #每迭代50次输出一次loss
    return (Weight, baise)


if __name__ == "__main__":
    data_x = np.random.normal(0, 10, [5, 3])  #3*5的矩阵 5个array
    """
    numpy.random.normal(loc=0,scale=1e-2,size=shape)
    参数loc(float)：正态分布的均值，对应着这个分布的中心。loc=0说明这一个以Y轴为对称轴的正态分布，
    参数scale(float)：正态分布的标准差，对应分布的宽度，scale越大，正态分布的曲线越矮胖，scale越小，曲线越高瘦。
    参数size(int 或者整数元组)：输出的值赋在shape里，默认为None。
    """
    Weights = np.array([[3, 4, 6]])  # array行向量
    data_y = np.dot(data_x, Weights.T) + 5
    res = liner_Regression(data_x, data_y, learningRate=0.001, Loopnum=5000)
    print(res[0], res[1])
