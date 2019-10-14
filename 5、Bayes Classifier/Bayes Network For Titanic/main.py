import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.estimators import (
    BayesianEstimator,
    BdeuScore,
    BicScore,
    HillClimbSearch,
    K2Score,
)
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from sklearn.cluster import KMeans

# import seaborn as sns


"""
PassengerId => 乘客ID
Pclass => 客舱等级(1/2/3等舱位)
Name => 乘客姓名
Sex => 性别 清洗成male=1 female=0
Age => 年龄 插补后分0,1,2 代表 幼年（0-15） 成年（15-55） 老年（55-）
SibSp => 兄弟姐妹数/配偶数
Parch => 父母数/子女数
Ticket => 船票编号
Fare => 船票价格 经聚类变0 1 2 代表少 多 很多
Cabin => 客舱号 清洗成有无此项，并发现有的生存率高
Embarked => 登船港口 清洗na,填S
"""
# combine train and test set.
train = pd.read_csv(r"D:\MyProject\机器学习\data\train.csv")
test = pd.read_csv(r"D:\MyProject\机器学习\data\test.csv")
full = pd.concat([train, test], ignore_index=True)
full["Embarked"].fillna("S", inplace=True)
full.Fare.fillna(full[full.Pclass == 3]["Fare"].median(), inplace=True)
full.loc[full.Cabin.notnull(), "Cabin"] = 1
full.loc[full.Cabin.isnull(), "Cabin"] = 0
full.loc[full["Sex"] == "male", "Sex"] = 1
full.loc[full["Sex"] == "female", "Sex"] = 0

full["Title"] = full["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
nn = {
    "Capt": "Rareman",
    "Col": "Rareman",
    "Don": "Rareman",
    "Dona": "Rarewoman",
    "Dr": "Rareman",
    "Jonkheer": "Rareman",
    "Lady": "Rarewoman",
    "Major": "Rareman",
    "Master": "Master",
    "Miss": "Miss",
    "Mlle": "Rarewoman",
    "Mme": "Rarewoman",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Ms": "Rarewoman",
    "Rev": "Mr",
    "Sir": "Rareman",
    "the Countess": "Rarewoman",
}
full.Title = full.Title.map(nn)
# assign the female 'Dr' to 'Rarewoman'
full.loc[full.PassengerId == 797, "Title"] = "Rarewoman"
full.Age.fillna(999, inplace=True)


def girl(aa):
    if (aa.Age != 999) & (aa.Title == "Miss") & (aa.Age <= 14):
        return "Girl"
    elif (aa.Age == 999) & (aa.Title == "Miss") & (aa.Parch != 0):
        return "Girl"
    else:
        return aa.Title


full["Title"] = full.apply(girl, axis=1)

Tit = ["Mr", "Miss", "Mrs", "Master", "Girl", "Rareman", "Rarewoman"]
for i in Tit:
    full.loc[(full.Age == 999) & (full.Title == i), "Age"] = full.loc[
        full.Title == i, "Age"
    ].median()

full.loc[full["Age"] <= 15, "Age"] = 0
full.loc[(full["Age"] > 15) & (full["Age"] < 55), "Age"] = 1
full.loc[full["Age"] >= 55, "Age"] = 2
full["Pclass"] = full["Pclass"] - 1

Fare = full["Fare"].values
Fare = Fare.reshape(-1, 1)
km = KMeans(n_clusters=3).fit(Fare)  # 将数据集分为2类
Fare = km.fit_predict(Fare)
full["Fare"] = Fare
full["Fare"] = full["Fare"].astype(int)
full["Age"] = full["Age"].astype(int)
full["Cabin"] = full["Cabin"].astype(int)
full["Pclass"] = full["Pclass"].astype(int)
full["Sex"] = full["Sex"].astype(int)
# full['Survived']=full['Survived'].astype(int)


dataset = full.drop(
    columns=["Embarked", "Name", "Parch", "PassengerId", "SibSp", "Ticket", "Title"]
)
dataset.dropna(inplace=True)
dataset["Survived"] = dataset["Survived"].astype(int)
# dataset=pd.concat([dataset, pd.DataFrame(columns=['Pri'])])
train = dataset[:800]
test = dataset[800:]
"""
最后保留如下项目：
Pclass => 客舱等级(1/2/3等舱位)
Sex => 性别 male=1 female=0
Age => 年龄 插补后分0,1,2 代表 幼年（0-15） 成年（15-55） 老年（55-）
Fare => 船票价格 经聚类变0 1 2 代表少 多 很多
Cabin => 客舱号 清洗成有无此项，并发现有的生存率高
"""


train.head()


# # 贝叶斯网络建模


# model = BayesianModel([('Age', 'Pri'), ('Sex', 'Pri'),('Pri','Survived'),('Fare','Pclass'),('Pclass','Survived'),('Cabin','Survived')])
model = BayesianModel(
    [
        ("Age", "Survived"),
        ("Sex", "Survived"),
        ("Fare", "Pclass"),
        ("Pclass", "Survived"),
        ("Cabin", "Survived"),
    ]
)
model.fit(
    train, estimator=BayesianEstimator, prior_type="BDeu"
)  # default equivalent_sample_size=5


# for cpd in model.get_cpds():
#    print(cpd)
predict_data = test.drop(columns=["Survived"], axis=1)
y_pred = model.predict(predict_data)


(y_pred["Survived"] == test["Survived"]).sum() / len(test)  # 测试集精度


model_infer = VariableElimination(model)
q = model_infer.query(variables=["Survived"], evidence={"Fare": 0})
print(q["Survived"])
q = model_infer.map_query(
    variables=["Fare", "Age", "Sex", "Pclass", "Cabin"], evidence={"Survived": 1}
)
print(q)


# # 用结构学习建立模型


hc = HillClimbSearch(train, scoring_method=BicScore(train))
best_model = hc.estimate()
print(best_model.edges())

best_model.fit(
    train, estimator=BayesianEstimator, prior_type="BDeu"
)  # default equivalent_sample_size=5
predict_data = test.drop(columns=["Survived"], axis=1)
y_pred = best_model.predict(predict_data)
(y_pred["Survived"] == test["Survived"]).sum() / len(test)  # 测试集精度


# # 预测原test集并保存csv


kaggle_test = full.drop(
    columns=["Embarked", "Name", "Parch", "PassengerId", "SibSp", "Ticket", "Title"]
)
kaggle_test = kaggle_test[kaggle_test["Survived"].isna()]
kaggle_test = kaggle_test.drop(columns=["Survived"], axis=1)
kaggle_test_pred = model.predict(kaggle_test)
# kaggle_test_pred = best_model.predict(kaggle_test)

kaggle_test_pred = kaggle_test_pred.reindex(columns=["PassengerId", "Survived"])
kaggle_test_pred["PassengerId"] = np.linspace(892, 1309, 1309 - 892 + 1, dtype=np.int)
kaggle_test_pred
kaggle_test_pred.to_csv("D:\MyProject\机器学习\data\kaggle_test_pred.csv", index=0)
