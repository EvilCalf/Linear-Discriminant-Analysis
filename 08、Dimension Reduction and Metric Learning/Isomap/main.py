import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, decomposition, manifold


def load_data():
    iris = datasets.load_iris()
    return iris.data, iris.target


def test_Isomap(*data):
    X, Y = data
    for n in [4, 3, 2, 1]:
        isomap = manifold.Isomap(n_components=n)
        isomap.fit(X)
        print(
            "reconstruction_error(n_components=%d):%s"
            % (n, isomap.reconstruction_error())
        )


def plot_Isomap_k(*data):
    X, Y = data
    Ks = [1, 5, 25, Y.size - 1]
    fig = plt.figure()
    #  colors=((1,0,0),(0,1,0),(0,0,1),(0.5,0.5,0),(0,0.5,0.5),(0.5,0,0.5),(0.4,0.6,0),(0.6,0.4,0),(0,0.6,0.4),(0.5,0.3,0.2),)
    for i, k in enumerate(Ks):
        isomap = manifold.Isomap(n_components=2, n_neighbors=k)
        X_r = isomap.fit_transform(X)
        ax = fig.add_subplot(2, 2, i + 1)
        colors = (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (0.5, 0.5, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.4, 0.6, 0),
            (0.6, 0.4, 0),
            (0, 0.6, 0.4),
            (0.5, 0.3, 0.2),
        )
        for label, color in zip(np.unique(Y), colors):
            position = Y == label
            ax.scatter(
                X_r[position, 0],
                X_r[position, 1],
                label="target=%d" % label,
                color=color,
            )
    ax.set_xlabel("X[0]")
    ax.set_ylabel("Y[0]")
    ax.legend(loc="best")
    ax.set_title("k=%d" % k)
    plt.suptitle("Isomap")
    plt.show()


X, Y = load_data()
test_Isomap(X, Y)
plot_Isomap_k(X, Y)

