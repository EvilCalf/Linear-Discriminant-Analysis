import pandas as pd
import numpy as np
import numpy.linalg as la
import random


class Filter:
    def __init__(self, data_df, sample_rate, t, k):
        """
        #
        :param data_df: 数据框（字段为特征，行为样本）
        :param sample_rate: 抽样比例
        :param t: 统计量分量阈值
        :param k: 选取的特征的个数
        """
        self.__data = data_df
        self.__feature = data_df.columns
        self.__sample_num = int(round(len(data_df) * sample_rate))
        self.__t = t
        self.__k = k

    # 数据处理（将离散型数据处理成连续型数据，比如字符到数值）
    def get_data(self):
        new_data = pd.DataFrame()
        for one in self.__feature[:-1]:
            col = self.__data[one]
            if (str(list(col)[0]).split(".")[0]).isdigit() or str(list(col)[0]).isdigit() or (str(list(col)[0]).split('-')[-1]).split(".")[-1].isdigit():
                new_data[one] = self.__data[one]
                # print '%s 是数值型' % one
            else:
                # print '%s 是离散型' % one
                keys = list(set(list(col)))
                values = list(range(len(keys)))
                new = dict(zip(keys, values))
                new_data[one] = self.__data[one].map(new)
        new_data[self.__feature[-1]] = self.__data[self.__feature[-1]]
        return new_data

    # 返回一个样本的猜中近邻和猜错近邻
    def get_neighbors(self, row):
        df = self.get_data()
        row_type = row[df.columns[-1]]
        right_df = df[df[df.columns[-1]] == row_type].drop(columns=[df.columns[-1]])
        wrong_df = df[df[df.columns[-1]] != row_type].drop(columns=[df.columns[-1]])
        aim = row.drop(df.columns[-1])
        f = lambda x: eulidSim(np.mat(x), np.mat(aim))
        right_sim = right_df.apply(f, axis=1)
        right_sim_two = right_sim.drop(right_sim.idxmin())
        # print right_sim_two
        # print right_sim.values.argmax()   # np.argmax(wrong_sim)
        wrong_sim = wrong_df.apply(f, axis=1)
        # print wrong_sim
        # print wrong_sim.values.argmax()
        # print right_sim_two.idxmin(), wrong_sim.idxmin()
        return right_sim_two.idxmin(), wrong_sim.idxmin()

    # 计算特征权重
    def get_weight(self, feature, index, NearHit, NearMiss):
        data = self.__data.drop(self.__feature[-1], axis=1)
        row = data.iloc[index]
        nearhit = data.iloc[NearHit]
        nearmiss = data.iloc[NearMiss]
        if (str(row[feature]).split(".")[0]).isdigit() or str(row[feature]).isdigit() or (str(row[feature]).split('-')[-1]).split(".")[-1].isdigit():
            max_feature = data[feature].max()
            min_feature = data[feature].min()
            right = pow(round(abs(row[feature] - nearhit[feature]) / (max_feature - min_feature), 2), 2)
            wrong = pow(round(abs(row[feature] - nearmiss[feature]) / (max_feature - min_feature), 2), 2)
            # w = wrong - right
        else:
            right = 0 if row[feature] == nearhit[feature] else 1
            wrong = 0 if row[feature] == nearmiss[feature] else 1
            # w = wrong - right
        w = wrong - right
        # print w
        return w

    # 过滤式特征选择
    def relief(self):
        sample = self.get_data()
        # print sample
        m, n = np.shape(self.__data)  # m为行数，n为列数
        score = []
        sample_index = random.sample(range(0, m), self.__sample_num)
        print ('采样样本索引为 %s ',sample_index)
        num = 1
        for i in sample_index:    # 采样次数
            one_score = dict()
            row = sample.iloc[i]
            NearHit, NearMiss = self.get_neighbors(row)
            print ('第 %s 次采样，样本index为 %s，其NearHit行索引为 %s ，NearMiss行索引为 %s' ,(num, i, NearHit, NearMiss))
            for f in self.__feature[0:-1]:
                w = self.get_weight(f, i, NearHit, NearMiss)
                one_score[f] = w
                print ('特征 %s 的权重为 %s.' % (f, w))
            score.append(one_score)
            num += 1
        f_w = pd.DataFrame(score)
        print ('采样各样本特征权重如下：')
        print (f_w)
        print ('平均特征权重如下：')
        print (f_w.mean())
        return f_w.mean()

    # 返回最终选取的特征
    def get_final(self):
        f_w = pd.DataFrame(self.relief(), columns=['weight'])
        final_feature_t = f_w[f_w['weight'] > self.__t]
        print (final_feature_t)
        final_feature_k = f_w.sort_values('weight').head(self.__k)
        print (final_feature_k)
        return final_feature_t, final_feature_k


# 几种距离求解
def eulidSim(vecA, vecB):
    return la.norm(vecA - vecB)


def cosSim(vecA, vecB):
    """
    :param vecA: 行向量
    :param vecB: 行向量
    :return: 返回余弦相似度（范围在0-1之间）
    """
    num = float(vecA * vecB.T)
    denom = la.norm(vecA) * la.norm(vecB)
    cosSim = 0.5 + 0.5 * (num / denom)
    return cosSim


def pearsSim(vecA, vecB):
    if len(vecA) < 3:
        return 1.0
    else:
        return 0.5 + 0.5 * np.corrcoef(vecA, vecB, rowvar=0)[0][1]


if __name__ == '__main__':
    data = pd.read_csv('D:\MyProject\机器学习\data\watermelon_3.csv')[['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖量', '好瓜']]
    print (data)
    f = Filter(data, 1, 0.8, 6)
    f.relief()
    # f.get_final()


