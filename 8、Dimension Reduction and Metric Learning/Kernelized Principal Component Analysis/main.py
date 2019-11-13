from sklearn.datasets import make_moons
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
import numpy as np

x2, y2 = make_moons(n_samples=100, random_state=123)

if __name__ == "__main__":
    pca = PCA(n_components=2)
    x_spca = pca.fit_transform(x2)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 6))
    ax[0].scatter(x_spca[y2 == 0, 0], x_spca[y2 == 0, 1], color='red', marker='^', alpha=0.5)
    ax[0].scatter(x_spca[y2 == 1, 0], x_spca[y2 == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[1].scatter(x_spca[y2 == 0, 0], np.zeros((50, 1)) + 0.02, color='red', marker='^', alpha=0.5)
    ax[1].scatter(x_spca[y2 == 1, 0], np.zeros((50, 1)) + 0.02, color='blue', marker='o', alpha=0.5)
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')
    ax[1].set_ylim([-1, 1])
    ax[1].set_yticks([])
    ax[1].set_xlabel('PCA')

    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
    x_kpca = kpca.fit_transform(x2)

    ax[2].scatter(x_kpca[y2 == 0, 0], x_kpca[y2 == 0, 1], color='red', marker='^', alpha=0.5)
    ax[2].scatter(x_kpca[y2 == 1, 0], x_kpca[y2 == 1, 1], color='blue', marker='o', alpha=0.5)
    ax[2].set_xlabel('KPCA')

    plt.show()