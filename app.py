## Do the streamlit part here not in webapp.py
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:37:03 2020

@author: Akash
"""
## Lets import the libraries

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import emoji
import random
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, mean_squared_error, plot_confusion_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier




html_temp = """
    <head>
    <style>
    .heading {
    font-family:"Times New Roman", Times, serif;
    font-size: large;
    }
    </style>
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    </head>
    <div style="background-color:blue;padding:10px">
    <h2 class="heading" style="color:white;text-align:center;font-family:"Times New Roman", Times, serif; font-size:large;">PLAY WITH ML <i class="material-icons" style="font-size:36px;">computer</i></h2>
    <p style="text-align:right;">Find us on LinkedIn <a style="color:white;" target="_blank" href='https://www.linkedin.com/in/akash-c-3a0468148/'>Akash</a>, <a style="color:white;" target="_blank" href='https://www.linkedin.com/in/ashwinsharmap'>Ashwin</a></p></div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
# st.subheader("Made with {} by Akash and Ashwin".format(emoji.emojize(":heart:")))
st.subheader("Upload a dataset {}".format(emoji.emojize(":cloud:")))
file_name = st.file_uploader("Please upload a small dataset(.csv or .xlsx) as the app is still in development stage :)", type=["csv","xlsx"])
st.text('*Try to avoid columns with nan values for categorical features')

option = st.selectbox("PLease select the format of your file",
    ["Select one",
     ".xlsx",
     ".csv"])

## EDA functions
            
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

if file_name is not None and option != "Select one":
    
    if option == '.xlsx':
        dataframe = pd.read_excel(file_name)
    elif option == '.csv':
        dataframe = pd.read_csv(file_name)
        
    num_rec = dataframe.shape[0]
    st.subheader("Glimpse of dataset {}".format(emoji.emojize(":blush:")))
    st.dataframe(dataframe.head(10))     
    activities = ["Select one","CLASSIFICATION","REGRESSION"]	
    st.subheader("Select the type of model {}".format(emoji.emojize(":smiley:")))
    choice = st.selectbox('',activities)
    
    
    
    if choice == 'CLASSIFICATION':
        
        classification_activities = ["Select one",'Random Forest Classifier','Decision Tree Classifier','SVC',
                           'SGD Classifier','Gradient Boosting Classifier',
                           'Adaboost Classifier']
        st.subheader("Select a Classification model to train your Dataset on {}".format(emoji.emojize(":eyes:")))
        classifier_choice = st.selectbox("",classification_activities)

        submit = st.button('TRAIN')
        
        if submit:
            
            dataframe = fill_na(dataframe)
        
            dataframe = encode(dataframe)
            
            X = dataframe.iloc[:,:-1]
            y = dataframe.iloc[:,-1]
            
            X,y = oversample(X,y)
            
            splitreturn = splitdata(X,y)
            X_train,X_test,y_train,y_test = splitreturn[0],splitreturn[1],splitreturn[2],splitreturn[3]
            
            scalereturn = scale(X_train,X_test)
            X_train,X_test = scalereturn[0],scalereturn[1]
            
            st.write("Give us some {} to build your project".format(emoji.emojize(":watch:")))
            
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
                c = classifier.fit(X_train,y_train)
                return classifier.predict(X_test), gs.best_params_, c
                
            
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
                c = classifier.fit(X_train,y_train)
                return classifier.predict(X_test), gs.best_params_, c
            
            
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
                return classifier.predict(X_test), gs.best_params_, classifier.fit(X_train,y_train)
            
            
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
                c = classifier.fit(X_train,y_train)
                return classifier.predict(X_test), gs.best_params_, c
            
            
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
                c = classifier.fit(X_train,y_train)
                return classifier.predict(X_test), gs.best_params_, c
            
            
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
                c = classifier.fit(X_train,y_train)
                return classifier.predict(X_test), gs.best_params_, c
    
    
            ## Model functions
            
            if classifier_choice == 'Random Forest Classifier':
                classifier_output = randomforestclassifier(X_train,X_test,y_train,y_test)
                
            elif classifier_choice == 'Decision Tree Classifier':
                classifier_output = decisiontreeclassifier(X_train,X_test,y_train,y_test)
            
            elif classifier_choice == 'SVC':
                classifier_output = svc(X_train,X_test,y_train,y_test)
            
            elif classifier_choice == 'SGD Classifier':
                classifier_output = sgdclassifier(X_train,X_test,y_train,y_test)
            
            elif classifier_choice == 'Gradient Boosting Classifier':
                classifier_output = gradientboostingclassifier(X_train,X_test,y_train,y_test)
            
            elif classifier_choice == 'Adaboost Classifier':
                classifier_output = adaboostclassifier(X_train,X_test,y_train,y_test)
                
                
             ### Time for printingout the result
            
            st.write('My system caught on {} training your model to get the output for you {}'.format(emoji.emojize(':fire:'), emoji.emojize(':satisfied:')))
            time.sleep(1.5)
            st.write('Be safe, wear a mask{}'.format(emoji.emojize(':mask:')))
            time.sleep(1.5)
            st.write('Your scores are here {}'.format(emoji.emojize(':raised_hands:')))
            time.sleep(1.5)
            st.write("\n")
            st.success('Accuracy score of {} is: {}'.format(classifier_choice,accuracy_score(y_test,classifier_output[0])))
            st.success('f1 score of {} is: {}'.format(classifier_choice,f1_score(y_test,classifier_output[0],average='weighted')))
            st.success('Recall score of {} is: {}'.format(classifier_choice,recall_score(y_test,classifier_output[0],average='weighted')))
            st.success('Precision score of {} is: {}'.format(classifier_choice,precision_score(y_test,classifier_output[0],average='weighted')))
            st.write('Selected parameters are: ',classifier_output[1])
            st.subheader("Code")
            file = open('codes_to_display/'+classifier_choice+' Code.txt','r')
            classifier_code = file.read()
            st.code(classifier_code, language='python')
            file.close()
            
            
            
    elif choice == 'REGRESSION':
        
        regression_activities = ["Select one",'Linear Regressor','Ridge Regressor','Lasso Regressor',
                   'DecisionTree Regressor','Gradient Boosting Regressor']
        st.subheader("Select a Regression model to train your Dataset on {}".format(emoji.emojize(":eyes:")))
        regressor_choice = st.selectbox("",regression_activities)
    
        submit = st.button('TRAIN')
        
        if submit:
            
            dataframe = fill_na(dataframe)
        
            dataframe = encode(dataframe)
            
            X = dataframe.iloc[:,:-1]
            y = dataframe.iloc[:,-1]
            
            splitreturn = splitdata(X,y)
            X_train,X_test,y_train,y_test = splitreturn[0],splitreturn[1],splitreturn[2],splitreturn[3]
            
            scalereturn = scale(X_train,X_test)
            X_train,X_test = scalereturn[0],scalereturn[1]
            
            #Linear Regression
            def linearregressor(X_train,X_test,y_train,y_test):
                regressor=LinearRegression()
                parameters=[{'n_jobs':[None]}]
                regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=2)
                regressor.fit(X_train,y_train)
                
                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
            
            #Ridge Regression
            def ridgeregressor(X_train,X_test,y_train,y_test):
                regressor=Ridge()
                parameters=[{'random_state':[None]}]
                regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=2)
                regressor.fit(X_train,y_train)
                
                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
                
            #Lasso Regression
            def lassoregressor(X_train,X_test,y_train,y_test):
                regressor=Lasso()
                parameters=[{'random_state':[None]}]
                regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=2)
                regressor.fit(X_train,y_train)
                
                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
            
            #Decision Tree Regression
            def decisiontreeregressor(X_train,X_test,y_train,y_test):
                regressor = DecisionTreeRegressor()
                parameters=[{'max_depth':[None]}]
                regressor=GridSearchCV(regressor,parameters,scoring='r2',cv=2)
                regressor.fit(X,y)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)
            
            #Gradient Boosting Regression
            def gradientboostingregressor(X_train,X_test,y_train,y_test):
                regressor = GradientBoostingRegressor()
                parameters ={'max_features':[None]}
                regressor = GridSearchCV(regressor,parameters,scoring='r2', cv=2)
                regressor.fit(X_train,y_train)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test,y_test)

           
            ## Model functions
            
            if regressor_choice == 'Linear Regressor':
                regressor_output = linearregressor(X_train,X_test,y_train,y_test)
                
            elif regressor_choice == 'Ridge Regressor':
                regressor_output = ridgeregressor(X_train,X_test,y_train,y_test)
            
            elif regressor_choice == 'Lasso Regressor':
                regressor_output = lassoregressor(X_train,X_test,y_train,y_test)
            
            elif regressor_choice == 'DecisionTree Regressor':
                regressor_output = decisiontreeregressor(X_train,X_test,y_train,y_test)
            
            elif regressor_choice == 'Gradient Boosting Regressor':
                regressor_output = gradientboostingregressor(X_train,X_test,y_train,y_test)

             ### Time for printingout the result
                
            st.write('My system caught on {} training your model to get the output for you {}'.format(emoji.emojize(':fire:'), emoji.emojize(':satisfied:')))
            time.sleep(1.5)
            st.write('Be safe, wear a mask{}'.format(emoji.emojize(':mask:')))
            time.sleep(1.5)
            st.write('Your scores are here {}'.format(emoji.emojize(':raised_hands:')))
            time.sleep(1.5)
            st.write("\n")
            st.success("r2/variance for {} is: {}".format(regressor_choice, regressor_output[2]))
            st.success("Residual sum of squares is: {}".format(np.mean((regressor_output[0] - y_test) ** 2)))
            st.success("Mean Squared Error for {} is: {}".format(regressor_choice, mean_squared_error(y_test, regressor_output[0])))
            st.write('Selected parameters are: ',regressor_output[1])
            st.subheader("Code")
            file = open('codes_to_display/'+regressor_choice+' Code.txt','r')
            regressor_code = file.read()
            st.code(regressor_code, language='python')
            file.close()