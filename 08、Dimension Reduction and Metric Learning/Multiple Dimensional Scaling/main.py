import numpy as np
import matplotlib.pyplot as plt


class MyMDS:
    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, data):
        m, n = data.shape
        dist = np.zeros((m, m))
        disti = np.zeros(m)
        distj = np.zeros(m)
        B = np.zeros((m, m))
        for i in range(m):
            dist[i] = np.sum(np.square(data[i] - data), axis=1).reshape(1, m)
        for i in range(m):
            disti[i] = np.mean(dist[i, :])
            distj[i] = np.mean(dist[:, i])
        distij = np.mean(dist)
        for i in range(m):
            for j in range(m):
                B[i, j] = -0.5 * (dist[i, j] - disti[i] - distj[j] + distij)
        lamda, V = np.linalg.eigh(B)
        index = np.argsort(-lamda)[: self.n_components]
        diag_lamda = np.sqrt(np.diag(-np.sort(-lamda)[: self.n_components]))
        V_selected = V[:, index]
        Z = V_selected.dot(diag_lamda)
        return Z


from sklearn.datasets import load_iris

iris = load_iris()

clf1 = MyMDS(2)
iris_t1 = clf1.fit(iris.data)
plt.scatter(iris_t1[:, 0], iris_t1[:, 1], c=iris.target)
plt.title("Using My MDS")
plt.show()

from sklearn.manifold import MDS

clf2 = MDS(2)
clf2.fit(iris.data)
iris_t2 = clf2.fit_transform(iris.data)
plt.scatter(iris_t2[:, 0], iris_t2[:, 1], c=iris.target)
plt.title("Using sklearn MDS")
plt.show()

