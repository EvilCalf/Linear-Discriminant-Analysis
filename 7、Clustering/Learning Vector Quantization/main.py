import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
 
#首先读入数据
midu = []
hantanglv = []
mark = []
df = pd.read_csv (r'D:\MyProject\机器学习\data\watermelon_3b.csv')
for i in df.index.values:
    midu.append(df.ix[i].values[1])
    hantanglv.append(df.ix[i].values[2])
for i in range(8):
    mark.append(1)
for i in range(8, 20):
    mark.append(0)
for i in range(20,30):
    mark.append(1)
    
#按照书上示例，将原型向量个数定义为5 学习率定义为0.1
q = 5
learningRate = 0.1
 
#随机选出q个数据作为初始原型向量
qIndex = random.sample(range(0,len(midu)), q)
P = []
for i in qIndex:
    P.append([np.array([midu[i], hantanglv[i]]), mark[i]])
 
#下面开始迭代 假定迭代轮数为400轮
r = 400
i = 0
for i in range(400):
    #从样本集中随机选取一个样本
    j = random.randint(0, len(midu)-1)
    dis = [np.linalg.norm(np.array([midu[j], hantanglv[j]])- p[0]) for p in P]
    #找出最近的原型向量
    minDis = dis.index(min(dis))
    #更新原型向量
    if P[minDis][1] == mark[j]:
        p_ = P[minDis][0]+learningRate*(np.array([midu[j], hantanglv[j]])- P[minDis][0])
    else:
        p_ = P[minDis][0]-learningRate*(np.array([midu[j], hantanglv[j]])- P[minDis][0])
    P[minDis][0] = p_
    
#将结果可视化
co = ['r', 'g', 'b', 'm']
for i in range(q):
    mm = [j[0][0] for j in P]
    hh = [j[0][1] for j in P]
plt.scatter(mm, hh, marker='x')
mm = [midu[i] for i in range(len(mark)) if mark[i] == 1]
hh = [hantanglv[i] for i in range(len(mark)) if mark[i] == 1]
plt.scatter(mm, hh, marker='o')
mm = [midu[i] for i in range(len(mark)) if mark[i] == 0]
hh = [hantanglv[i] for i in range(len(mark)) if mark[i] == 0]
plt.scatter(mm, hh, marker='v')
plt.show()