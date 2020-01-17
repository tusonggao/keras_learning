# 数据源自Kaggle比赛Quora Insincere Questions Classification: https://www.kaggle.com/c/quora-insincere-questions-classification
# 由于我们无法得到test.csv的标签，所以这里只使用train.csv文件
# 将train.csv按照0.7 0.3的比例切分为训练集和测试集,分别命名为train_new.csv和test_new.csv
import os
import sys
import time
import numpy as np 
import pandas as pd

df = pd.read_csv('./data/train.csv')
df = df.sample(frac=1.0, random_state=42) 
print('df.shpae', df.shape)

sum_target = df['target'].sum()
print('sum_target is ', sum_target, 'ratio is ', sum_target/ratio)

train_size = int(0.7*len(df))
df_train_new, df_test_new = df.iloc[:train_size], df.iloc[train_size:]

len_sum = len(df_train_new) + len(df_test_new)
print('len_sum is ', len_sum)






