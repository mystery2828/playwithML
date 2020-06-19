# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 17:11:50 2020

@author: Ashwin Sharma P
"""

#from playwithML import preprocessing_for_regression as pfr
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.preprocessing import RobustScaler,StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE,RandomOverSampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor

file_name='winequality.csv'
dataframe=pd.read_csv(file_name)
num_rec = dataframe.shape[0]


#removing duplicate rows
def dupli(dataframe):
    dataframe.drop_duplicates(subset=None, keep='first', inplace=True)
    return dataframe


def fill_na(dataframe):

    for col in dataframe.columns:
        if dataframe[col].dtype.name != 'object':
            if (dataframe[col].isnull().sum())*2 >= num_rec:
                dataframe = dataframe.drop([col], axis=1)
            else:
                dataframe[col] = dataframe[col].fillna(dataframe[col].mean())
    return dataframe

## Function for splitting the dataset        
def splitdata(X,y):
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
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

## Function for oversampling
def oversample(X,y):
    
    smote = random.choice([SMOTE(),RandomOverSampler()])
    X,y = smote.fit_resample(X,y)
    return X,y

## Function for labelencoding
def encode(dataframe):
    
    for col in dataframe.columns:
        if dataframe[col].dtype.name == 'object':
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe


#Linear Regression
def linearregressor(X_train,X_test,y_train,y_test):
    regressor=LinearRegression()
   # parameters=[{'beta':[None]}]
   # regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=10)
    regressor.fit(X_train,y_train)
    print ("r2/variance for linear regressor is: ", regressor.score(X_test,y_test))
    print ("Residual sum of squares for linear regressor is: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
    return regressor.predict(X_test)

#Ridge Regression
def ridgeregressor(X_train,X_test,y_train,y_test):
    regressor=Ridge()
    a=[]
    for i in range(1, 10):
        a.append(random.uniform(0, 100))
    parameters={'alpha':a}
    regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=10)
    regressor.fit(X_train,y_train)
    print ("r2/variance for ridge regressor is:",regressor.score(X_test,y_test))
    print ("Residual sum of squares for ridge regressor is: %.2f"%np.mean((regressor.predict(X_test) - y_test) ** 2))
    print(regressor.best_estimator_)
    return regressor.predict(X_test)
    

#Lasso Regression
def lassoregressor(X_train,X_test,y_train,y_test):
    regressor=Lasso()
    a = []
    for i in range(1,10):
        a.append(random.uniform(0, 100))
        parameters={'alpha':a}
    regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=10)
    regressor.fit(X_train,y_train)
    print ("r2/variance for lasso regressor is: ", regressor.score(X_test,y_test))
    print ("Residual sum of squares for lasso regressor is: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
    print(regressor.best_estimator_)
    return regressor.predict(X_test)

#Decision Tree Regression
def decisiontreeregressor(X,y):
    regressor = DecisionTreeRegressor()
    cross_val_score(regressor, X, y,scoring='r2', cv=10)
    regressor.fit(X,y)
    print ("r2/variance for decision tree regressor is: ", regressor.score(X_test,y_test))
    print ("Residual sum of squares for decision tree regressor is: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
    return regressor.predict(X_test)



def gradientboostingregressor(X_train,X_test,y_train,y_test):
    regressor = GradientBoostingRegressor()
    parameters ={'max_features':[None]
                 }
    regressor = GridSearchCV(estimator=regressor, param_grid = parameters, cv = 2, n_jobs=-1)
    regressor.fit(X_train,y_train)
    print ("r2/variance for gradient boosting regressor is: ", regressor.score(X_test,y_test))
    print ("Residual sum of squares for gradient boosting regressor is: %.2f" % np.mean((regressor.predict(X_test) - y_test) ** 2))
    print(regressor.best_estimator_)
    return regressor.predict(X_test)





### Calling out the functions
    
## EDA functions
dataframe=dupli(dataframe)
dataframe = fill_na(dataframe)

dataframe = encode(dataframe)


X = dataframe.iloc[:,:-1]
print(X.shape)
y = dataframe.iloc[:,-1]
print(y.shape)

X,y = oversample(X,y)

splitreturn = splitdata(X,y)
X_train,X_test,y_train,y_test = splitreturn[0],splitreturn[1],splitreturn[2],splitreturn[3]

scalereturn = scale(X_train,X_test)
X_train,X_test = scalereturn[0],scalereturn[1]

## Model functions

list_of_options = ['linearregressor','ridgeregressor','lassoregressor',
                   'decisiontreeregressor','gradientboostingregressor']
for i in range(5):
    selected_option = list_of_options[i]

    if selected_option == 'linearregressor':
        regressor_output = linearregressor(X_train,X_test,y_train,y_test)
        
    elif selected_option == 'ridgeregressor':
        regressor_output = ridgeregressor(X_train,X_test,y_train,y_test)
    
    elif selected_option == 'lassoregressor':
        regressor_output = lassoregressor(X_train,X_test,y_train,y_test)
    
    elif selected_option == 'decisiontreeregressor':
        regressor_output = decisiontreeregressor(X,y)
        
    elif selected_option == 'gradientboostingregressor':
        regressor_output = gradientboostingregressor(X_train,X_test,y_train,y_test)
    
    
    print('')

