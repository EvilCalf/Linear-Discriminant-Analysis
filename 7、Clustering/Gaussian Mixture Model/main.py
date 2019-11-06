import matplotlib.pyplot as plt
import numpy as np
import math

# 原始数据
x = [
    0.697,
    0.774,
    0.634,
    0.608,
    0.556,
    0.403,
    0.481,
    0.437,
    0.666,
    0.243,
    0.245,
    0.343,
    0.639,
    0.657,
    0.360,
    0.593,
    0.719,
    0.359,
    0.339,
    0.282,
    0.748,
    0.714,
    0.483,
    0.478,
    0.525,
    0.751,
    0.532,
    0.473,
    0.725,
    0.446,
]

y = [
    0.460,
    0.376,
    0.264,
    0.318,
    0.215,
    0.237,
    0.149,
    0.211,
    0.091,
    0.267,
    0.057,
    0.099,
    0.161,
    0.198,
    0.370,
    0.042,
    0.103,
    0.188,
    0.241,
    0.257,
    0.232,
    0.346,
    0.312,
    0.437,
    0.369,
    0.489,
    0.472,
    0.376,
    0.445,
    0.459,
]

# 矩阵测试
def test_matrix():
    Σ = np.mat([[0.2, 0.1], [0.0, 0.1]])
    Σ_Trans = Σ.T
    Σ_inverse = Σ.I
    print("Σ: {}".format(Σ))
    print("Σ Inverse: {}".format(Σ_inverse))
    print("Σ Transform: {}".format(Σ_Trans))


#
def gauss_density_probability(n, x, μ, Σ):
    """
	计算高斯概率密度。
	参数：
		n：数据维度
		x：原始数据
		μ：均值向量
		Σ：协方差矩阵
	返回：
		p：高斯概率
	"""
    Σ_det = np.linalg.det(Σ)
    divisor = pow(2 * np.pi, n / 2) * np.sqrt(Σ_det)
    exp = np.exp(-0.5 * (x - μ) * Σ.I * (x - μ).T)
    p = exp / divisor
    return p


# 后验概率测试
def test_posterior_probability():
    xx = np.mat([[x[0], y[0]]])
    μ_datasets = [
        np.mat([[x[5], y[5]]]),
        np.mat([[x[21], y[21]]]),
        np.mat([[x[26], y[26]]]),
    ]
    Σ = np.mat([[0.1, 0.0], [0.0, 0.1]])
    det = np.linalg.det(Σ)
    print("det: {}".format(det))
    p_all = []
    for μ in μ_datasets:
        p = gauss_density_probability(2, xx, μ, Σ)
        p = p / 3
        p_all.append(p)
    p_mean = []
    for p in p_all:
        p_sum = np.sum(p_all)
        p = p / p_sum
        p_mean.append(p)
    print("probability: {}".format(p_mean[0]))


def posterior_probability(k, steps):
    """
	高斯混合聚类+EM算法实现聚类。
	参数：
		k：簇类数
		steps：迭代次数
	返回：
		p_all：后验概率
		μ_datasets：均值矩阵
		Σ_datasets：协方差矩阵
		α_datasets：混合系数
		classification_cluster：簇分类
	"""
    α_datasets = [np.mat([1 / k]) for _ in range(k)]  # 初始每一簇先验概率为1/k
    xx = [np.mat([[x[i], y[i]]]) for i in range(len(x))]
    μ_rand = np.random.randint(0, 30, (1, k))
    # μ_datasets = [np.mat([[x[i], y[i]]]) for i in μ_rand[0]]
    μ_datasets = [
        np.mat([[x[5], y[5]]]),
        np.mat([[x[21], y[21]]]),
        np.mat([[x[26], y[26]]]),
    ]
    Σ_datasets = [np.mat([[0.1, 0.0], [0.0, 0.1]]) for _ in range(k)]
    # det = np.linalg.det(Σ_datasets[0])
    for step in range(steps):
        p_all = []
        # create cluster
        classification_temp = locals()
        for i in range(k):
            classification_temp["cluster_" + str(i)] = []
        # 后验概率分组
        for j in range(len(x)):
            p_group = []
            for i in range(k):
                μ = μ_datasets[i]
                p = gauss_density_probability(2, xx[j], μ, Σ_datasets[i])

                p = p * α_datasets[i].getA()[0][0]
                p_group.append(p)
            p_sum = np.sum(p_group)
            # 取最大后验概率
            max_p = max(p_group)
            max_index = p_group.index(max_p)
            # 最大后验概率聚类
            for i in range(k):
                if i == max_index:
                    classification_temp["cluster_" + str(max_index)].append(j)

            p_group = [p_group[i] / p_sum for i in range(len(p_group))]
            p_all.append(p_group)

        # 更新 μ, Σ, α
        μ_datasets = []
        Σ_datasets = []
        α_datasets = []

        for i in range(k):
            μ_temp_numerator = 0
            μ_temp_denominator = 0
            Σ_temp = 0
            α_temp = 0
            μ_numerator = [p_all[j][i] * xx[j] for j in range(len(x))]
            for mm in μ_numerator:
                μ_temp_numerator += mm

            μ_denominator = [p_all[j][i] for j in range(len(x))]
            for nn in μ_denominator:
                μ_temp_denominator += nn

            μ_dataset = μ_temp_numerator / μ_temp_denominator
            μ_datasets.append(μ_dataset)

            Σ = [
                p_all[j][i].getA()[0][0] * (xx[j] - μ_dataset).T * (xx[j] - μ_dataset)
                for j in range(len(x))
            ]
            for ss in Σ:
                Σ_temp += ss
            Σ_dataset = Σ_temp / μ_temp_denominator
            Σ_datasets.append(Σ_dataset)

            α_new = [p_all[j][i] for j in range(len(x))]
            for α_nn in α_new:
                α_temp += α_nn
            α_dataset = α_temp / len(x)
            α_datasets.append(α_dataset)
    return p_all, μ_datasets, Σ_datasets, α_datasets, classification_temp


def cluster_visiualization(k, steps):
    """
	可视化聚类结果。
	参数：
		k：簇类数
		steps：迭代次数
	返回：
		无
	"""
    post_probability, μ_datasets, Σ_datasets, α_datasets, classification_temp = posterior_probability(
        k, steps
    )
    plt.figure(figsize=(8, 8))
    markers = [".", "s", "^", "<", ">", "P"]
    plt.xlim(0.1, 0.9)
    plt.ylim(0, 0.9)
    plt.grid()
    plt.scatter(x, y, color="r")

    plt.figure(figsize=(8, 8))
    for i in range(k):
        # 依据聚类获取对应数据，并显示
        xx = [x[num] for num in classification_temp["cluster_" + str(i)]]
        yy = [y[num] for num in classification_temp["cluster_" + str(i)]]

        plt.xlim(0.1, 0.9)
        plt.ylim(0, 0.9)
        plt.grid()
        plt.scatter(xx, yy, marker=markers[i])
    plt.savefig(
        r"D:\MyProject\机器学习\7、Clustering\Gaussian Mixture Model\gauss_cluster.png",
        format="png",
    )


if __name__ == "__main__":
    cluster_visiualization(3, 100)

