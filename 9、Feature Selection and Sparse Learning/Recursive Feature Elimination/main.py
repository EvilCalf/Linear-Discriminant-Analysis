from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier


# ### 引入数据集


from sklearn.datasets import load_boston, load_iris


# ### 回归问题


dataset_boston = load_boston()
data_boston = dataset_boston.data
target_boston = dataset_boston.target


rfe = RFE(estimator=Lasso(), n_features_to_select=4)
rfe.fit(data_boston, target_boston)
rfe.support_
print(rfe.support_)
# 输出
# array([False, False, False, False, False,  True, False,  True, False,
#        False,  True, False,  True])

# ### 分类问题


dataset_iris = load_iris()
data_iris = dataset_iris.data
target_iris = dataset_iris.target


rfe = RFE(estimator=DecisionTreeClassifier(), n_features_to_select=2)
rfe.fit(data_iris, target_iris)
rfe.support_
print(rfe.support_)
# array([False, False,  True,  True])

# ### RFECV

rfecv = RFECV(estimator=DecisionTreeClassifier())
rfecv.fit(data_iris, target_iris)
rfecv.support_
print(rfecv.support_)
# array([False, False,  True,  True])