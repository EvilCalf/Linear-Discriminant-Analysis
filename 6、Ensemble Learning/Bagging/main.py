from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# load data
from sklearn import datasets, model_selection


def load_data():
    iris = datasets.load_iris()  # scikit-learn 自带的 iris 数据集
    X_train = iris.data
    y_train = iris.target
    return model_selection.train_test_split(
        X_train, y_train, test_size=0.25, random_state=0, stratify=y_train
    )


bagging = BaggingClassifier(
    KNeighborsClassifier(), max_samples=0.1, max_features=0.5, random_state=1
)

X_train, X_test, y_train, y_test = load_data()

bagging.fit(X_train, y_train)
y_pre = bagging.predict(X_test)
print(accuracy_score(y_test, y_pre))

import matplotlib.pyplot as plt
import numpy as np

param_range = range(1, 11, 1)
sores_list = []
for i in param_range:
    baggingclf = BaggingClassifier(
        KNeighborsClassifier(),
        max_samples=i / 10,
        max_features=0.5,
        random_state=1000,
        oob_score=True,
    )
    baggingclf.fit(X_train, y_train)
    y_pre = baggingclf.predict(X_test)
    sores_list.append(accuracy_score(y_test, y_pre, normalize=True))

plt.plot(param_range, sores_list)
plt.show()

sores_list = []
param_range = range(1, X_train.shape[1] + 1)
for i in param_range:
    baggingclf_2 = BaggingClassifier(
        KNeighborsClassifier(),
        max_samples=0.5,
        max_features=i,
        random_state=100,
        oob_score=True,
    )  # 一般而言特征选择越少,方差越大
    baggingclf_2.fit(X_train, y_train)
    y_pre = baggingclf_2.predict(X_test)
    sores_list.append(accuracy_score(y_test, y_pre, normalize=True))

plt.plot(param_range, sores_list)
plt.show()

sores_list = []
param_range = range(0, 101)
for i in param_range:
    baggingclf_2 = BaggingClassifier(
        KNeighborsClassifier(), max_samples=0.8, max_features=0.8, random_state=i
    )
    baggingclf_2.fit(X_train, y_train)
    y_pre = baggingclf_2.predict(X_test)
    sores_list.append(accuracy_score(y_test, y_pre, normalize=True))

plt.plot(param_range, sores_list)
plt.show()
