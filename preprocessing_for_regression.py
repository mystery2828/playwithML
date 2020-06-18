# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 09:48:12 2020

@author: Ashwin Sharma P
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split


file_name='winequality.csv'
dataframe = pd.read_csv(file_name)


#removing duplicate rows
def dupli(dataframe):
    dataframe.drop_duplicates(subset=None, keep='first', inplace=True)
    return dataframe


## Function to fill the NaN values
def fill_na(dataframe):
    for col in dataframe.columns:
        if dataframe[col].isnull().count()*3 >= dataframe.shape[0]:
            dataframe = dataframe.drop([col], axis=1)
        else:
            dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
    return dataframe


## Function for splitting the dataset        
def splitdata(dataframe):
    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    return X_train, X_test, y_train, y_test


## Function for labelencoding
def encode(dataframe):
    for col in dataframe.columns:
        if dataframe[col].dtype.name == 'object':
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe


## Function for Scaling
def scale(X_train,X_test):
    sc = RobustScaler()   
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    sc1=StandardScaler()
    X_train=sc1.fit_transform(X_train)
    X_test=sc1.transform(X_test)
    return X_train, X_test

