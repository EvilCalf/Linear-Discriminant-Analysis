import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class SOMNet():
    """
    参数：
    η0 学习效率初始值
    τ1是一个时间常数（拓扑邻域的大小随着时间收缩）
    τ2是另一个时间常数（学习效率的大小随着时间收缩）
    σ0为初始拓扑邻域的半径

    SOM算法过程总结如下：
    1、初始化 - 为初始权向量wj选择随机值。
    2、采样 - 从输入空间中抽取一个训练输入向量样本x。
    3、匹配 - 找到权重向量最接近输入向量的获胜神经元I(x)。
    4、更新 - 更新权重向量 Δweightji=η(t)⋅Tj,I(x)(t)⋅(xi−weightji)
    5、继续 - 继续回到步骤2，直到特征映射趋于稳定。
    """
    def __init__(self, input_dims, output_nums, σ0, rta0, τ1, τ2, iterations):
        self.input_dims = input_dims
        self.output_nums = output_nums
        self.σ0 = σ0
        self.η0 = η0
        self.τ1 = τ1
        self.τ2 = τ2
        self.iterations = iterations
        self.weights = np.random.rand(output_nums, input_dims)

    def distance(self, weight, train_x):
        """
        每个神经元计算样本和自身携带的权向量之间的距离
        Compute distance between array weight and vector train_x.
        Input:
            weight: np.array with shape [m, d]
            train_x: np.array with shape [1, d]
        Return:
            dist_square: np.array with shape [m, 1] when weight is array or float when weight is vector
        """
        m = weight.shape[0]
        if m==1:
            dist_square = np.sum((weight-train_x)**2)
        else:
            dist_square = np.sum((weight-train_x)**2, axis=1, keepdims=True)
        return dist_square

    def bmu(self, x):
        """
        计算最佳匹配单元
        Compute the best match unit(BMU) given input vector x.
        Input:
            x: np.array with shape [1, d].
        Return:
            index: the index of BMU 索引
        """
        dist_square = self.distance(self.weights, x)
        index = np.argmax(dist_square)
        return index

    def radius(self, iter):
        """
        计算拓扑邻域半径
        Computer neighborhood radius for current BMU.
        Input:
            iter: the current iteration.
        """
        σ = self.σ0 * np.exp(-iter / self.τ1)
        return σ

    def update(self, x, iter, σ):
        """
        Update weight vector for all output unit each iteration.
        Input:
            x: np.array with shape [1, d]. The current input vector.
            iter: int, the current iteration.
            σ: float, the current neighborhood function. 拓扑邻域半径
        """
        η = self.η0 * np.exp(-iter / self.τ2)
        neighbor_function = np.exp( - self.distance(self.weights, x) / (2*σ*σ) )
        self.weights = self.weights + η * neighbor_function * (x - self.weights)

    def train(self, train_X):
        """
        Learning the weight vectors of all output units.
        Input:
            train_X: list with lenght n and element is np.array with shape [1, d]. Training instances.
        """
        n = len(train_X)
        for iter in range(self.iterations):
            #step 2: choose instance from train set randomly
            x = train_X[random.randint(0, n-1)]

            #step 3: compute BMU for current instance
            bmu = self.bmu(x)

            #step 4: computer neighborhood radius
            σ = self.radius(iter)

            #step5: update weight vectors for all output unit
            self.update(x, iter, σ)
        print (σ)

    def eval(self, x):
        """
        Computer index of BMU given input vector.
        """
        return self.bmu(x)

if __name__=="__main__":

    #prepare train data
    train_X = []
    train_y = []
    DataSet = pd.read_csv(r"D:\MyProject\机器学习\data\watermelon_3a.csv")
    X1 = DataSet.values[:, 1] # 取序号为1的一列
    X2 = DataSet.values[:, 2] # 取序号为2的一列
    for i in range(len(X1)):
        train_X.append(np.array([[X1[i], X2[i]]])) #组成两个因素为一个array的序列
    train_y = DataSet.values[:, 3] # 取序号为3的一列即训练集结果

    #training SOM network
    output_nums = 4
    σ0 = 3
    η0 = 0.1
    τ1 = 1
    τ2 = 1
    iterations = 100
    som_net = SOMNet(2, output_nums, σ0, η0, τ1, τ2, iterations)
    som_net.train(train_X)

    #plot data in 2 dimension space
    left_top_count = 0
    left_bottom_count = 0
    right_top_count = 0
    right_bottom_count = 0
    for i in range(len(train_X)):
        bmu = som_net.eval(train_X[i])
        if train_y[i] == 1:
            style = "bo"
        else:
            style = "r+"
        print (bmu)
        if bmu == 0:
            plt.plot([1+left_top_count*0.03], [2], style)
            left_top_count += 1
        elif bmu == 1:
            plt.plot([2+right_top_count*0.03], [2], style)
            right_top_count += 1
        elif bmu == 2:
            plt.plot([1+left_bottom_count*0.03], [1], style)
            left_bottom_count += 1
        else:
            plt.plot([2+right_bottom_count*0.03], [1], style)
            right_bottom_count += 1

    plt.xlim([1, 3])
    plt.ylim([1, 3])
    plt.show()