# https://www.kaggle.com/heesoo37/facebook-s-fasttext-algorithm
from __future__ import print_function, division, with_statement
import os
import sys
import time
import numpy as np
import pandas as pd
import fasttext
import re
import fasttext as ft
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter

# Some preprocesssing that will be common to all the text classification methods you will see. 
puncts = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', ]

def clean_text(x):
    x = str(x)
    x = x.replace('\n', ' ')  #去除掉所有的换行符
    for punct in puncts:
        x = x.replace(punct, f' {punct} ')
    return x

def create_fasttext_format_files():
    print('in create_fasttext_format_files')
    train_df = pd.read_csv("./data/train_new.csv")
    test_df = pd.read_csv("./data/test_new.csv")
    
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)

    train_df["question_text"] = train_df["question_text"].apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
    train_df["question_text"].fillna("_##_").values
    test_df["question_text"].fillna("_##_").values

    train_df['label'] = '__label__' + train_df.target.astype(str)
    train_df['labels_text'] = train_df.label.str.cat(train_df.question_text, sep=' ')
    test_df['label'] = '__label__' + test_df.target.astype(str)

    train_df, val_df = train_test_split(train_df, test_size=0.08, random_state=2018) # .08 since the datasize is large enough.
    train_df['labels_text'].to_csv('./data/train_new_fasttext_formated.csv', index=False, header=False)
    print('train_df.shape is ', train_df.shape, 'val_df.shape is ', val_df.shape, 'test_df.shape is ', test_df.shape)

    return  train_df, val_df, test_df

train_df, val_df, test_df = create_fasttext_format_files()
test_df = test_df.sample(30)  # 减少数据，便于观察

# Function to do K-fold CV across different fasttext parameter values
def tune(Y, X, YX, k, lr, wordNgrams, epoch):
    # Record results
    results = []
    for lr_val in lr:
        for wordNgrams_val in wordNgrams:
            for epoch_val in epoch:  
                # K-fold CV
                kf = KFold(n_splits=k, shuffle=True)
                fold_results = []
                # For each fold
                for train_index, test_index in kf.split(X):
                    # Write the training data for this CV fold
                    training_file = open('train_cv.txt','w')
                    training_file.writelines(YX[train_index] + '\n')
                    training_file.close()
                    # Fit model for this set of parameter values
                    model = ft.FastText.train_supervised('train_cv.txt',
                                          lr=lr_val,
                                          wordNgrams=wordNgrams_val,
                                          epoch=epoch_val)
                    # Predict the holdout sample
                    pred = model.predict(X[test_index].tolist())
                    pred = pd.Series(pred[0]).apply(lambda x: re.sub('__label__', '', x[0]))
                    # Compute accuracy for this CV fold
                    fold_results.append(accuracy_score(Y[test_index], pred.values))
                # Compute mean accuracy across 10 folds
                mean_acc = pd.Series(fold_results).mean()
                print([lr_val, wordNgrams_val, epoch_val, mean_acc])
    # Add current parameter values and mean accuracy to results table
    results.append([lr_val, wordNgrams_val, epoch_val, mean_acc])         
    # Return as a DataFrame 
    #results = pd.DataFrame(results)
    #results.columns = ['lr','wordNgrams','epoch','mean_acc']
    #return(results)
    
# Tune parameters using 10-fold CV
#results = tune(Y = df.cuisine,
#     X = df.ingredients,
#     YX = df.labels_text,
#     k = 10, 
#     lr = [0.05, 0.1, 0.2],
#     wordNgrams = [1,2,3],
#     epoch = [15,17,20])

# Train final classifiers
print('start fasttext training...')
#data/train_new_fasttext_formated.csv
classifier1 = ft.FastText.train_supervised('./data/train_new_fasttext_formated.csv', lr=0.1, wordNgrams=1, epoch=15)
#classifier2 = ft.FastText.train_supervised('train.txt', lr=0.1, wordNgrams=2, epoch=15)
#classifier3 = ft.FastText.train_supervised('train.txt', lr=0.1, wordNgrams=3, epoch=15)

# Predict test data
print('start fasttext predicting...')
predictions1 = classifier1.predict(test_df.question_text.tolist())
#predictions2 = classifier2.predict(df_test.ingredients.tolist())
#predictions3 = classifier3.predict(df_test.ingredients.tolist())

print('type of predictions1 is ', type(predictions1))
print('predictions1 is ', predictions1)
print('test_df.label is ', test_df.label)

sys.exit(0)



# Write submission file
submit = pd.DataFrame({'id': df_test.id, 'cuisine': pd.Series(majority_vote)})
submit.cuisine = submit.cuisine.apply(lambda x: re.sub('__label__', '', x))
submit.to_csv('submit.csv', index=False)


