import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

class SOMNet():
    """
    input_dims  输入维度
    output_nums  输出层单元数量
    sigam0
    eta0
    tau1
    tau2
    iterations  迭代次数
    weights  每一个输入层因素到输出层的权值array序列
    """
    def __init__(self, input_dims, output_nums, sigma0, rta0, tau1, tau2, iterations):
        self.input_dims = input_dims
        self.output_nums = output_nums
        self.sigma0 = sigma0
        self.eta0 = eta0
        self.tau1 = tau1
        self.tau2 = tau2
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
            index: the index of BMU
        """
        dist_square = self.distance(self.weights, x)
        index = np.argmax(dist_square)
        return index

    def radius(self, iter):
        """
        Computer neighborhood radius for current BMU.
        Input:
            iter: the current iteration.
        """
        sigma = self.sigma0 * np.exp(-iter / self.tau1)
        return sigma

    def update(self, x, iter, sigma):
        """
        Update weight vector for all output unit each iteration.
        Input:
            x: np.array with shape [1, d]. The current input vector.
            iter: int, the current iteration.
            sigma: float, the current neighborhood function.
        """
        eta = self.eta0 * np.exp(-iter / self.tau2)
        neighbor_function = np.exp( - self.distance(self.weights, x) / (2*sigma*sigma) )
        self.weights = self.weights + eta * neighbor_function * (x - self.weights)

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
            sigma = self.radius(iter)

            #step5: update weight vectors for all output unit
            self.update(x, iter, sigma)
        print (sigma)

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
    sigma0 = 3
    eta0 = 0.1
    tau1 = 1
    tau2 = 1
    iterations = 100
    som_net = SOMNet(2, output_nums, sigma0, eta0, tau1, tau2, iterations)
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