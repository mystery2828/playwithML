# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 09:48:12 2020

@author: Ashwin Sharma P
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler


file_name='winequality.csv'
df = pd.read_csv(file_name)


#removing duplicate rows
def dupli(dataframe):
    dataframe.drop_duplicates(subset=None, keep='first', inplace=True)
    return dataframe


## Function to fill the NaN values
def fill_na(df):
    for col in df.columns:
        if df[col].isnull().count()*3 >= df.shape[0]:
            df = df.drop([col], axis=1)
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


X=df.iloc[:,0:-1]  ## independent features
y=df.iloc[:,-1]  ## dependent features


from sklearn.model_selection import train_test_split
## Function for splitting the dataset        
def splitdata(dataframe):
    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    return X_train, X_test, y_train, y_test


## Function for Scaling
def scale(X_train,X_test):
    sc = RobustScaler()   
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    sc1=StandardScaler()
    X_train=sc1.fit_transform(X_train)
    X_test=sc1.transform(X_test)
    return X_train, X_test

