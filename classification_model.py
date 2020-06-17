# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 13:51:12 2020

@author: Akash
"""
from playwithML import preprocessing_for_classification as pfc
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import jaccard_similarity_score, precision_score,\
    recall_score, accuracy_score, f1_score

## Function for RandomForestClassifier
def randomforestclassifier(X_train,X_test,y_train,y_test):
    
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)
    return classifier.predict(X_test)
    

## Function for DecisionTreeCLassifier
def decisiontreeclassifier(X_train,X_test,y_train,y_test):
    
    classifier = DecisionTreeClassifier()
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test)


## Function for SVC
def svc(X_train,X_test,y_train,y_test):
    
    classifier = SVC()
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test)


## Function for xgboost
def xgboostclassifier(X_train,X_test,y_train,y_test):
    
    classifier = xgboost.XGBClassifier()
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test)


## Function for sgdclassifier
def sgdclassifier(X_train,X_test,y_train,y_test):
    
    classifier = SGDClassifier(loss='hinge', penalty='l2')
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test)


## Function for gradientboostingclassifier
def gradientboostingclassifier(X_train,X_test,y_train,y_test):
    
    classifier = GradientBoostingClassifier()
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test)



print('Accuracy score is: {}'.format(accuracy_score(y_test,cmp)))
print('f1 score is: {}'.format(f1_score(y_test,cmp,average='weighted')))
print('Recall score is: {}'.format(recall_score(y_test,cmp,average='weighted')))
print('Precision score is: {}'.format(precision_score(y_test,cmp,average='weighted')))