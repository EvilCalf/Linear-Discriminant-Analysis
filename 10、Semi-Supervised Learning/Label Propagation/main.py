import time
import math
import numpy as np

# return k neighbors index
def navie_knn(dataSet, query, k):
    numSamples = dataSet.shape[0]

    ## step 1: calculate Euclidean distance
    diff = np.tile(query, (numSamples, 1)) - dataSet
    squaredDiff = diff ** 2
    squaredDist = np.sum(squaredDiff, axis=1)  # sum is performed by row

    ## step 2: sort the distance
    sortedDistIndices = np.argsort(squaredDist)
    if k > len(sortedDistIndices):
        k = len(sortedDistIndices)

    return sortedDistIndices[0:k]


# build a big graph (normalized weight matrix)
def buildGraph(MatX, kernel_type, rbf_sigma=None, knn_num_neighbors=None):
    num_samples = MatX.shape[0]
    affinity_matrix = np.zeros((num_samples, num_samples), np.float32)
    if kernel_type == "rbf":
        if rbf_sigma == None:
            raise ValueError("You should input a sigma of rbf kernel!")
        for i in range(num_samples):
            row_sum = 0.0
            for j in range(num_samples):
                diff = MatX[i, :] - MatX[j, :]
                affinity_matrix[i][j] = np.exp(sum(diff ** 2) / (-2.0 * rbf_sigma ** 2))
                row_sum += affinity_matrix[i][j]
            affinity_matrix[i][:] /= row_sum
    elif kernel_type == "knn":
        if knn_num_neighbors == None:
            raise ValueError("You should input a k of knn kernel!")
        for i in range(num_samples):
            k_neighbors = navie_knn(MatX, MatX[i, :], knn_num_neighbors)
            affinity_matrix[i][k_neighbors] = 1.0 / knn_num_neighbors
    else:
        raise NameError("Not support kernel type! You can use knn or rbf!")

    return affinity_matrix


# label propagation
def labelPropagation(
    Mat_Label,
    Mat_Unlabel,
    labels,
    kernel_type="rbf",
    rbf_sigma=1.5,
    knn_num_neighbors=10,
    max_iter=500,
    tol=1e-3,
):
    # initialize
    num_label_samples = Mat_Label.shape[0]
    num_unlabel_samples = Mat_Unlabel.shape[0]
    num_samples = num_label_samples + num_unlabel_samples
    labels_list = np.unique(labels)
    num_classes = len(labels_list)

    MatX = np.vstack((Mat_Label, Mat_Unlabel))
    clamp_data_label = np.zeros((num_label_samples, num_classes), np.float32)
    for i in range(num_label_samples):
        clamp_data_label[i][labels[i]] = 1.0

    label_function = np.zeros((num_samples, num_classes), np.float32)
    label_function[0:num_label_samples] = clamp_data_label
    label_function[num_label_samples:num_samples] = -1

    # graph construction
    affinity_matrix = buildGraph(MatX, kernel_type, rbf_sigma, knn_num_neighbors)

    # start to propagation
    iter = 0
    pre_label_function = np.zeros((num_samples, num_classes), np.float32)
    changed = np.abs(pre_label_function - label_function).sum()
    while iter < max_iter and changed > tol:
        if iter % 1 == 0:
            print("---> Iteration %d/%d, changed: %f", (iter, max_iter, changed))
        pre_label_function = label_function
        iter += 1

        # propagation
        label_function = np.dot(affinity_matrix, label_function)

        # clamp
        label_function[0:num_label_samples] = clamp_data_label

        # check converge
        changed = np.abs(pre_label_function - label_function).sum()

    # get terminate label of unlabeled data
    unlabel_data_labels = np.zeros(num_unlabel_samples)
    for i in range(num_unlabel_samples):
        unlabel_data_labels[i] = np.argmax(label_function[i + num_label_samples])

    return unlabel_data_labels


# show
def show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels):
    import matplotlib.pyplot as plt

    for i in range(Mat_Label.shape[0]):
        if int(labels[i]) == 0:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], "Dr")
        elif int(labels[i]) == 1:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], "Db")
        else:
            plt.plot(Mat_Label[i, 0], Mat_Label[i, 1], "Dy")

    for i in range(Mat_Unlabel.shape[0]):
        if int(unlabel_data_labels[i]) == 0:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], "or")
        elif int(unlabel_data_labels[i]) == 1:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], "ob")
        else:
            plt.plot(Mat_Unlabel[i, 0], Mat_Unlabel[i, 1], "oy")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.xlim(0.0, 12.0)
    plt.ylim(0.0, 12.0)
    plt.show()


def loadCircleData(num_data):
    center = np.array([5.0, 5.0])
    radiu_inner = 2
    radiu_outer = 4
    num_inner = num_data / 3
    num_outer = num_data - num_inner

    data = []
    theta = 0.0
    for i in range(int(num_inner)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_inner * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_inner * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 2

    theta = 0.0
    for i in range(int(num_inner)):
        pho = (theta % 360) * math.pi / 180
        tmp = np.zeros(2, np.float32)
        tmp[0] = radiu_outer * math.cos(pho) + np.random.rand(1) + center[0]
        tmp[1] = radiu_outer * math.sin(pho) + np.random.rand(1) + center[1]
        data.append(tmp)
        theta += 1

    Mat_Label = np.zeros((2, 2), np.float32)
    Mat_Label[0] = center + np.array([-radiu_inner + 0.5, 0])
    Mat_Label[1] = center + np.array([-radiu_outer + 0.5, 0])
    labels = [0, 1]
    Mat_Unlabel = np.vstack(data)
    return Mat_Label, labels, Mat_Unlabel


def loadBandData(num_unlabel_samples):
    # Mat_Label = np.array([[5.0, 2.], [5.0, 8.0]])
    # labels = [0, 1]
    # Mat_Unlabel = np.array([[5.1, 2.], [5.0, 8.1]])

    Mat_Label = np.array([[5.0, 2.0], [5.0, 8.0]])
    labels = [0, 1]
    num_dim = Mat_Label.shape[1]
    Mat_Unlabel = np.zeros((num_unlabel_samples, num_dim), np.float32)
    Mat_Unlabel[: num_unlabel_samples / 2, :] = (
        np.random.rand(num_unlabel_samples / 2, num_dim) - 0.5
    ) * np.array([3, 1]) + Mat_Label[0]
    Mat_Unlabel[num_unlabel_samples / 2 : num_unlabel_samples, :] = (
        np.random.rand(num_unlabel_samples / 2, num_dim) - 0.5
    ) * np.array([3, 1]) + Mat_Label[1]
    return Mat_Label, labels, Mat_Unlabel


# main function
if __name__ == "__main__":
    num_unlabel_samples = 800
    # Mat_Label, labels, Mat_Unlabel = loadBandData(num_unlabel_samples)
    Mat_Label, labels, Mat_Unlabel = loadCircleData(num_unlabel_samples)

    ## Notice: when use 'rbf' as our kernel, the choice of hyper parameter 'sigma' is very import! It should be
    ## chose according to your dataset, specific the distance of two data points. I think it should ensure that
    ## each point has about 10 knn or w_i,j is large enough. It also influence the speed of converge. So, may be
    ## 'knn' kernel is better!
    # unlabel_data_labels = labelPropagation(Mat_Label, Mat_Unlabel, labels, kernel_type = 'rbf', rbf_sigma = 0.2)
    unlabel_data_labels = labelPropagation(
        Mat_Label,
        Mat_Unlabel,
        labels,
        kernel_type="knn",
        knn_num_neighbors=10,
        max_iter=400,
    )
    show(Mat_Label, labels, Mat_Unlabel, unlabel_data_labels)
