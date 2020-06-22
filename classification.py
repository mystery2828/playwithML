# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:37:03 2020

@author: Akash
"""

## Lets import the libraries
import random
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE,RandomOverSampler

file_name = 'pulse.xlsx'
dataframe = pd.read_excel(file_name)
num_rec = dataframe.shape[0]


## Function to fill the NaN values
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
    
    sc = StandardScaler()   
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


## Function for oversampling
def oversample(X,y):
    
    smote = random.choice([SMOTE(),RandomOverSampler()])
    X,y = smote.fit_resample(X,y)
    return X,y



### Classification part of code


import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


## Function for RandomForestClassifier
def randomforestclassifier(X_train,X_test,y_train,y_test):
    
    classifier = RandomForestClassifier()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'max_depth':[None]}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_
    

## Function for DecisionTreeCLassifier
def decisiontreeclassifier(X_train,X_test,y_train,y_test):
    
    classifier = DecisionTreeClassifier()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'splitter':['best']}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_


## Function for SVC
def svc(X_train,X_test,y_train,y_test):
    
    classifier = SVC()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'gamma':['auto']}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_


## Function for xgboost
def xgboostclassifier(X_train,X_test,y_train,y_test):
    
    classifier = xgboost.XGBClassifier()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'eta':[.2]}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_


## Function for sgdclassifier
def sgdclassifier(X_train,X_test,y_train,y_test):
    
    classifier = SGDClassifier()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'penalty':['l2']}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_


## Function for gradientboostingclassifier
def gradientboostingclassifier(X_train,X_test,y_train,y_test):
    
    classifier = GradientBoostingClassifier()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'max_features':[None]}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_


## Function for adaboost
def adaboostclassifier(X_train,X_test,y_train,y_test):
    
    classifier = AdaBoostClassifier()
    clffit = classifier.fit(X_train,y_train)
    parameters = [{'base_estimator':[None]}]
    gs = GridSearchCV(estimator = clffit,
                      param_grid = parameters,
                      n_jobs = -1,
                      scoring = 'accuracy',
                      cv = 2)
    gs.fit(X_train, y_train)
    classifier = gs.best_estimator_
    classifier.fit(X_train,y_train)
    return classifier.predict(X_test), gs.best_estimator_
    

### Calling out the functions
    
## EDA functions
dataframe = fill_na(dataframe)

dataframe = encode(dataframe)

X = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]

X,y = oversample(X,y)

splitreturn = splitdata(X,y)
X_train,X_test,y_train,y_test = splitreturn[0],splitreturn[1],splitreturn[2],splitreturn[3]

scalereturn = scale(X_train,X_test)
X_train,X_test = scalereturn[0],scalereturn[1]

## Model functions

list_of_options = ['randomforestclassifier1','decisiontreeclassifier1','svc1',
                   'xgboostclassifier1','sgdclassifier1','gradientboostingclassifier1',
                   'adaboostclassifier1']
for i in range(7):
    selected_option = list_of_options[i]

    if selected_option == 'randomforestclassifier1':
        classifier_output = randomforestclassifier(X_train,X_test,y_train,y_test)
        
    elif selected_option == 'decisiontreeclassifier1':
        classifier_output = decisiontreeclassifier(X_train,X_test,y_train,y_test)
    
    elif selected_option == 'svc1':
        classifier_output = svc(X_train,X_test,y_train,y_test)
    
    elif selected_option == 'xgboostclassifier1':
        classifier_output = xgboostclassifier(X_train, X_test, y_train, y_test)
    
    elif selected_option == 'sgdclassifier1':
        classifier_output = sgdclassifier(X_train,X_test,y_train,y_test)
    
    elif selected_option == 'gradientboostingclassifier1':
        classifier_output = gradientboostingclassifier(X_train,X_test,y_train,y_test)
    
    elif selected_option == 'adaboostclassifier1':
        classifier_output = adaboostclassifier(X_train,X_test,y_train,y_test)


    ### Time for printingout the result
    asd = classifier_output[1]
    # print('Accuracy score of {} is: {}'.format(selected_option,accuracy_score(y_test,classifier_output[0])))
    # print('f1 score of {} is: {}'.format(selected_option,f1_score(y_test,classifier_output[0],average='weighted')))
    # print('Recall score of {} is: {}'.format(selected_option,recall_score(y_test,classifier_output[0],average='weighted')))
    # print('Precision score of {} is: {}'.format(selected_option,precision_score(y_test,classifier_output[0],average='weighted')))
    # print('Selected parameters are \n{}'.format(classifier_output[1]))
    print(type(asd))