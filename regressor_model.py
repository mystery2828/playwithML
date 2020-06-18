# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 20:36:17 2020

@author: Ashwin Sharma P
"""

from playwithML import preprocessing_for_regression as pfr
from sklearn.pipeline import Pipeline
import pandas as pd
import random
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Ridge,Lasso


#Linear Regression
def linearregressor(X_train,y_train,X_test,y_test):
    lin_regressor=LinearRegression()
    lin_regressor.fit(X_train,y_train)
    prediction_linear=lin_regressor.predict(X_test)
    print ("r2/variance : ", lin_regressor.score(X_test,y_test))
    print ("Residual sum of squares: %.2f" % np.mean((lin_regressor.predict(X_test) - y_test) ** 2))
    return prediction_linear

#Ridge Regression
def ridgeregressor(X_train,y_train,X_test,y_test):
    ridge_regressor=Ridge()
    ridge_regressor.fit(X_train,y_train)
    prediction_ridge=ridge_regressor.predict(X_test)
    return prediction_ridge

#Lasso Regression
def lassoregressor(X_train,y_train,X_test,y_test):
    lasso_regressor=Lasso()
    lasso_regressor.fit(X_train,y_train)
    prediction_lasso=lasso_regressor.predict(X_test)
    return prediction_lasso

#Decision Tree Regression
def decisiontreeregressor(X,y):
    dt_regressor = DecisionTreeRegressor()
    cross_val_score(dt_regressor, X, y, cv=10)
    dt_regressor.fit(X,y)
    prediction_dt=regressor.predict(X_test)
    return prediction_dt



