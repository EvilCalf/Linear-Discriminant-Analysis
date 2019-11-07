import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 颜色
color = sns.color_palette()
# 数据print精度
pd.set_option("precision", 3)

df = pd.read_csv("D:\MyProject\机器学习\data\winequality-red.csv", sep=";")
df.describe()

plt.style.use("ggplot")

colnm = df.columns.tolist()
fig = plt.figure(figsize=(10, 6))

for i in range(12):
    plt.subplot(2, 6, i + 1)
    sns.boxplot(df[colnm[i]], orient="v", width=0.5, color=color[0])
    plt.ylabel(colnm[i], fontsize=12)
# plt.subplots_adjust(left=0.2, wspace=0.8, top=0.9)

plt.tight_layout()
plt.show()

