import pandas as pd
import matplotlib.pyplot as plt

# online loading
from urllib.request import urlopen

Dataset = pd.read_csv("D:\MyProject\机器学习\data\iris.csv")

X = Dataset.iloc[:, :4].get_values()

# label (generation after transform output to categorical variables)
Dataset.iloc[:, -1] = Dataset.iloc[:, -1].astype("category")
label = Dataset.iloc[:, 4].values.categories

# output 1 (generation after string categorical variables to numerical values)
# 把种类字符串换成数字
Dataset.iloc[:, 4].cat.categories = [0, 1, 2]
y = Dataset.iloc[:, 4].get_values()

# output 2 (generation after one hot encoding)
# 转化成独热编码
Y = pd.get_dummies(Dataset.iloc[:, 4]).get_values()
"""
split of train set and test set (using sklearn function)
"""
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y, train_Y, test_Y = train_test_split(
    X, y, Y, test_size=0.5, random_state=42
)
"""
construction of BP network
"""
from BPNetwork import *

bpn1 = BP_network()  # initial a BP network class
bpn1.CreateNN(4, 5, 3, actfun="Sigmoid", learningrate=0.05)  # build the network
"""
experiment of fixed learning rate
"""

# parameter training with fixed learning rate initial above
e = []
for i in range(1000):
    err, err_k = bpn1.TrainStandard(train_X, train_Y)
    e.append(err)

# draw the convergence curve of output error by each step of iteration
import matplotlib.pyplot as plt

f1 = plt.figure(1)
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0, 1)
plt.title("training error convergence curve with fixed learning rate")
plt.plot(e)

# get the test error in test set
pred = bpn1.PredLabel(test_X)
count = 0
for i in range(len(test_y)):
    if pred[i] == test_y[i]:
        count += 1

test_err = 1 - count / len(test_y)
print("test error rate: %.3f" % test_err)
"""
experiment of dynamic learning rate
"""

bpn2 = BP_network()  # initial a BP network class
bpn2.CreateNN(4, 5, 3, actfun="Sigmoid", learningrate=0.05)  # build the network

# parameter training with fixed learning rate initial above
e = []
for i in range(1000):
    err, err_k = bpn2.TrainStandard_Dynamic_Lr(train_X, train_Y)
    e.append(err)

# draw the convergence curve of output error by each step of iteration
# import matplotlib.pyplot as plt
f2 = plt.figure(2)
plt.xlabel("epochs")
plt.ylabel("error")
plt.ylim(0, 1)
plt.title("training error convergence curve with dynamic learning rate")
plt.plot(e)

# get the test error in test set
pred = bpn2.PredLabel(test_X)
count = 0
for i in range(len(test_y)):
    if pred[i] == test_y[i]:
        count += 1

test_err = 1 - count / len(test_y)
print("test error rate: %.3f" % test_err)

plt.show()

print("haha")

