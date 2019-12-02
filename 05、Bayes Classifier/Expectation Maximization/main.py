from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import random


def create_sample_data(m, n):
    mat_y = mat(zeros((m, n)))

    for i in range(m):
        for j in range(n):
            # 通过产生随机数，每一行表示一次实验结果
            mat_y[i, j] = random.randint(0, 1)
    return mat_t


# EM算法
def em(arr_y, theta, tol, iterator_num):
    PI = 0
    P = 0
    Q = 0
    m, n = shape(arr_y)
    mat_y = arr_y.getA()

    for i in range(iterator_num):
        miu = []
        PI = copy(theta[0])
        P = copy(theta[1])
        Q = copy(theta[2])
        for j in range(m):
            miu_value = (PI * (P ** mat_y[j]) * ((1 - P) ** (1 - mat_y[j]))) / (
                PI * (P ** mat_y[j]) * ((1 - P) ** (1 - mat_y[j]))
                + (1 - PI) * (Q ** mat_y[j]) * ((1 - Q) ** (1 - mat_y[j]))
            )
            miu.append(miu_value)

        sum1 = 0.0
        for j in range(m):
            sum1 += miu[j]
        theta[0] = sum1 / m

        sum1 = 0.0
        sum2 = 0.0
        for j in range(m):
            sum1 += miu[j] * mat_y[j]
            sum2 += miu[j]
        theta[1] = sum1 / sum2

        sum1 = 0.0
        sum2 = 0.0
        for j in range(m):
            sum1 += (1 - miu[j]) * mat_y[j]
            sum2 += 1 - miu[j]
        theta[2] = sum1 / sum2

        print("-------------------")
        print(theta)
        if (
            abs(theta[0] - PI) <= tol
            and abs(theta[1] - P) <= tol
            and abs(theta[2] - Q) <= tol
        ):
            print("break")
            break
    return PI, P, Q


def main():
    # mat_y = create_sample_data(100, 1)
    mat_y = mat(zeros((10, 1)))
    mat_y[0, 0] = 1
    mat_y[1, 0] = 1
    mat_y[2, 0] = 0
    mat_y[3, 0] = 1
    mat_y[4, 0] = 0
    mat_y[5, 0] = 0
    mat_y[6, 0] = 1
    mat_y[7, 0] = 0
    mat_y[8, 0] = 1
    mat_y[9, 0] = 1
    theta = [0.4, 0.6, 0.7]
    print(mat_y)
    PI, P, Q = em(mat_y, theta, 0.001, 100)
    print(PI, P, Q)


main()
