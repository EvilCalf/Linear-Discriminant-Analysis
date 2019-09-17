# using pandas dataframe for .csv read which contains chinese char.
import pandas as pd
import random

data_file_encode = "utf-8"  # the watermelon_3.csv is file codec type
with open("D:\MyProject\机器学习\data\watermelon_3.csv",
          mode='r',
          encoding=data_file_encode) as data_file:
    DateSet = pd.read_csv(data_file)

import DT as decision_tree

root = decision_tree.TreeGenerate(DateSet)

accuracy_scores = []

# k-folds cross prediction k折交叉验证法

n = len(DateSet.index)
k = random.randint(4,6)
for i in range(k):
    m = int(n / k)
    test = []
    for j in range(i * m, i * m + m):
        test.append(j)

    DateSet_train = DateSet.drop(test)
    DateSet_test = DateSet.iloc[test]
    root = decision_tree.TreeGenerate(DateSet_train)  # generate the tree

    # test the accuracy
    pred_true = 0
    # 遍历测试集
    for i in DateSet_test.index:
        label = decision_tree.Predict(root, DateSet[DateSet.index == i])
        if label == DateSet_test[DateSet_test.columns[-1]][i]:
            pred_true += 1

    accuracy = pred_true / len(DateSet_test.index)
    accuracy_scores.append(accuracy)

# print the prediction accuracy result
accuracy_sum = 0
print("accuracy: ", end="")
for i in range(k):
    print("%.3f  " % accuracy_scores[i], end="")
    accuracy_sum += accuracy_scores[i]
print("\naverage accuracy: %.3f" % (accuracy_sum / k))

# dicision tree visualization using pydotplus.graphviz
root = decision_tree.TreeGenerate(DateSet)

decision_tree.DrawPNG(root, "Decision Tree/Decision Trees Based on Information Entropy/Decision Trees Based on Information Entropy.png")