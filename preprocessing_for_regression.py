<<<<<<< HEAD
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

filename = 'pulse.xlsx'
dataframe = pd.read_csv(filename)

=======
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 09:48:12 2020

@author: Ashwin Sharma P
"""

import pandas as pd
import numpy as np
#from sklearn.linear_model import Lasso,Ridge
from sklearn.preprocessing import RobustScaler,StandardScaler


file_name='winequality.csv'
df=pd.read_csv(file_name)

df

df.describe

#removing duplicate rows
def dupli(dataframe):
    dataframe.drop_duplicates(subset=None, keep='first', inplace=True)
    return dataframe

dupli(df)

total=df.shape[0]
total

df.info()

## Function to fill the NaN values
def fill_na(df):
    for col in df.columns:
        if df[col].isnull().count()*3 >= total:
            df = df.drop([col], axis=1)
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df

X=df.iloc[:,0:-1]  ## independent features
y=df.iloc[:,-1]  ## dependent features

X,y

from sklearn.model_selection import train_test_split
## Function for splitting the dataset        
def splitdata(dataframe):
    X = dataframe.iloc[:,:-1]
    y = dataframe.iloc[:,-1]
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
    return X_train, X_test, y_train, y_test

splitdata(df)


## Function for Scaling
def scale(X_train,X_test):
    sc = RobustScaler()   
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    sc1=StandardScaler()
    X_train=sc1.fit_transform(X_train)
    X_test=sc1.transform(X_test)
    return X_train, X_test

#def dummyvar(firstcol,secondcol):
#    df_firstcol = pd.get_dummies(df[firstcol])
#    df_secondcol = pd.get_dummies(df[secondcol])
#    #Concat new columns to original dataframe 
#    df_concat = pd.concat([df, df_firstcol, df_secondcol], axis=1)
#    print (df_concat.head())






>>>>>>> 30ba8b92c6b07b9d4d8a5aad7592e01ebe0a1fd0
