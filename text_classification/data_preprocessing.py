# 数据源自Kaggle比赛Quora Insincere Questions Classification: https://www.kaggle.com/c/quora-insincere-questions-classification
# 由于我们无法得到test.csv的标签，所以这里只使用train.csv文件
# 将train.csv按照0.7vs0.3的比例切分为本项目会使用到的训练集和测试集,分别命名为train_new.csv和test_new.csv
import os
import sys
import time
import numpy as np 
import pandas as pd

df = pd.read_csv('./data/train.csv')
df = df.sample(frac=1.0, random_state=42) 
print('df.shpae', df.shape)

print('df sum_target is ', df.target.sum(), 'ratio is ', df.target.sum()/len(df))

train_size = int(0.7*len(df))
df_train, df_test = df.iloc[:train_size], df.iloc[train_size:]

print('df_train sum_target is ', df_train.target.sum(), 'ratio is ', df_train.target.sum()/len(df_train))
print('df_test sum_target is ', df_test.target.sum(), 'ratio is ', df_test.target.sum()/len(df_test))

len_sum = len(df_train) + len(df_test)
print('len_sum is ', len_sum)

df_train.to_csv('./data/train_new.csv', index=False)
df_test.to_csv('./data/test_new.csv', index=False)

print('prog ends here!')






