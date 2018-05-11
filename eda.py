import pandas as pd
import numpy as np

# read train data
train = pd.read_csv("atec_anti_fraud_train.csv")

# summary
train.head()  # 有空值
train.shape  # (994731, 300)
train.groupby("label").size()
"""
1是有风险,0是无风险,-1是无标签.
-1      4725
 0    977884
 1     12122
"""
len(train.id.unique())  # 994731
train.dtypes.nunique()  # object,int64,float64

# Nan情况
def featHasNan(featName):
    return train[featName].isnull().max()

for i in range(1, 298):
    if (~featHasNan("f%d" % (i))): print("f%d" % (i))
"""仅f1-f4, f6-f19无空值"""

# date(62天)
train.date.max()  # 20171105
train.date.min()  # 20170905

# f1-f4, f8-19: 均为0,1,2取值.
train.groupby("f12").size()

# f5 TODO
train.f5.describe()
train.f5.isnull().sum()  # 199825
train.f5.unique()  # 321

a = np.array([int(_ / 10000) for _ in train.f5 if _ > 0])
set(a)
b = np.array([int((_ % 10000) / 100) for _ in train.f5 if _ > 0])
set(b)
c = np.array([(_ % 100) for _ in train.f5 if _ > 0])
set(c)

# f6
train.f6.unique()  # 0-4取值

# f7
train.f7.unique()  # 0-7取值

# 都是　取值0,1的数量占绝对优势
"""缺失值数量都是207448"""
train.f20.unique()  # 0-32 (大概)
train.f21.unique()  # 0-63 (大概)
train.f22.unique()  # 0-177 (大概)
train.f23.unique()  # 0-291 (大概)
"""缺失值数量都是207585"""
train.f24.unique()  # 0-32 (大概)
train.f25.unique()  # 0-63 (大概)
train.f26.unique()  # 0-85 (大概)
train.f27.unique()  # 0-153 (大概)

train.groupby("f27").size()
train.f28.isnull().sum()
train.f27.max()