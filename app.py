# Do the streamlit part here not in webapp.py
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 11:37:03 2020

@author: Akash
"""
# Lets import the libraries

from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, mean_squared_error, plot_confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import time
import random
import emoji
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


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
    <p style="text-align:right;color:tomato;">Find us on LinkedIn. Lead Author: <a style="color:white;" target="_blank" href='https://www.linkedin.com/in/akash-c-3a0468148/'>Akash</a> Co-Author: <a style="color:white;" target="_blank" href='https://www.linkedin.com/in/ashwinsharmap'>Ashwin</a></p></div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
# st.subheader("Made with {} by Akash and Ashwin".format(emoji.emojize(":heart:")))
st.subheader("Upload a dataset {}".format(emoji.emojize(":cloud:")))
file_name = st.file_uploader(
    "Please upload a small dataset(.csv or .xlsx) as the app is still in development stage :)", type=["csv", "xlsx"])
st.text('*Try to avoid columns with nan values for categorical features \n*Please remove columns with data such as "DATE","NAME","ID" for better accuracy\n*We currently dont process Time series data.')

option = st.selectbox("PLease select the format of your file",
                      ["Select one",
                       ".xlsx",
                       ".csv"])

# EDA functions

# Function to fill the NaN values


def fill_na(dataframe):

    for col in dataframe.columns:
        if dataframe[col].dtype.name != 'object':
            if (dataframe[col].isnull().sum())*2 >= num_rec:
                dataframe = dataframe.drop([col], axis=1)
            else:
                dataframe[col] = dataframe[col].fillna(dataframe[col].mean())

    return dataframe


# Function for splitting the dataset
def splitdata(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    return X_train, X_test, y_train, y_test


# Function for labelencoding
def encode(dataframe):

    for col in dataframe.columns:
        if dataframe[col].dtype.name == 'object':
            le = LabelEncoder()
            dataframe[col] = le.fit_transform(dataframe[col])
    return dataframe


# Function for Scaling
def scale(X_train, X_test):

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


# Function for oversampling
def oversample(X, y):

    smote = random.choice([SMOTE(), RandomOverSampler()])
    X, y = smote.fit_resample(X, y)
    return X, y

# Function forcomputing confusion matrix


def confusion_matrix(classifier, X_test, y_test):

    return plot_confusion_matrix(classifier, X_test, y_test)


if file_name is not None and option != "Select one":

    if option == '.xlsx':
        dataframe = pd.read_excel(file_name)
    elif option == '.csv':
        dataframe = pd.read_csv(file_name)

    num_rec = dataframe.shape[0]
    st.subheader("Glimpse of dataset {}".format(emoji.emojize(":blush:")))
    st.dataframe(dataframe.head(10))
    all_columns_names = dataframe.columns.tolist()
    plot_type = st.selectbox("Select Type of Plot", [
                             "area", "bar", "line", "hist"])
    selected_columns_names = st.multiselect(
        "Select Columns To Plot", all_columns_names)

    # Plot By Streamlit
    if plot_type == 'area':
        req_data = dataframe[selected_columns_names]
        st.area_chart(req_data)

    elif plot_type == 'bar':
        req_data = dataframe[selected_columns_names]
        st.bar_chart(req_data)

    elif plot_type == 'line':
        req_data = dataframe[selected_columns_names]
        st.line_chart(req_data)

    # Custom Plot
    elif plot_type:
        cust_plot = dataframe[selected_columns_names].plot(kind=plot_type)
        st.write(cust_plot)
        st.pyplot()

    activities = ["Select one", "CLASSIFICATION", "REGRESSION"]
    st.subheader("Select the type of model {}".format(
        emoji.emojize(":smiley:")))
    choice = st.selectbox('', activities)

    if choice == 'CLASSIFICATION':

        classification_activities = ["Select one", 'Random Forest Classifier', 'Decision Tree Classifier', 'SVC',
                                     'SGD Classifier', 'Gradient Boosting Classifier',
                                     'Adaboost Classifier']
        st.subheader("Select a Classification model to train your Dataset on {}".format(
            emoji.emojize(":eyes:")))
        classifier_choice = st.selectbox("", classification_activities)
        st.subheader(
            "Select the hyper-parameter {}".format(emoji.emojize(":smiley:")))
        if classifier_choice == 'Random Forest Classifier':
            max_depth = st.slider(
                label='max_depth', min_value=0, max_value=16, step=1)
            n_estimators = st.slider(
                label='n_estimators', min_value=100, max_value=800, step=50)
            min_samples_split = st.slider(
                label='min_samples_split', min_value=0, max_value=16, step=1)

        elif classifier_choice == 'Decision Tree Classifier':
            max_depth = st.slider(
                label='max_depth', min_value=0, max_value=16, step=1)
            min_samples_split = st.slider(
                label='min_samples_split', min_value=0, max_value=16, step=1)
            criterion = st.selectbox('criterion', ['gini', 'entropy'])

        elif classifier_choice == 'SVC':
            c = st.slider(label='C', min_value=0, max_value=16, step=1)
            degree = st.slider(label='degree', min_value=1,
                               max_value=10, step=1)
            kernel = st.selectbox(
                'kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

        elif classifier_choice == 'SGD Classifier':
            loss = st.selectbox(
                'loss', ['hinge', 'log', 'modified_huber', 'squared_hinge'])
            penalty = st.selectbox('penalty', ['l1', 'l2', 'elasticnet'])
            alpha = st.text_input(
                label='alpha(enter a value between 0.0001 to 0.001)', value='0.0001')
            alpha = float(alpha)

        elif classifier_choice == 'Gradient Boosting Classifier':
            n_estimators = st.slider(
                label='n_estimators', min_value=100, max_value=800, step=50)
            learning_rate = st.slider(
                label='learning_rate', min_value=0.1, max_value=1.0, step=0.05)
            criterion = st.selectbox(
                'criterion', ['friedman_mse', 'mse', 'mae'])

        elif classifier_choice == 'Adaboost Classifier':
            n_estimators = st.slider(
                label='n_estimators', min_value=50, max_value=800, step=50)
            learning_rate = st.slider(
                label='learning_rate', min_value=0.5, max_value=3.0, step=0.05)

        cross_validation = st.slider(
            label='cross validation (higher the number more the time taken to train)', min_value=1, max_value=10, step=1)
        submit = st.button('TRAIN')

        if submit:

            dataframe = fill_na(dataframe)

            dataframe = encode(dataframe)

            X = dataframe.iloc[:, :-1]
            y = dataframe.iloc[:, -1]

            X, y = oversample(X, y)

            splitreturn = splitdata(X, y)
            X_train, X_test, y_train, y_test = splitreturn[
                0], splitreturn[1], splitreturn[2], splitreturn[3]

            scalereturn = scale(X_train, X_test)
            X_train, X_test = scalereturn[0], scalereturn[1]

            st.write("Give us some {} to build your project".format(
                emoji.emojize(":watch:")))

            classifier = None

            # Function for RandomForestClassifier
            def randomforestclassifier(X_train, X_test, y_train, y_test, max_depth=None, n_estimators=100, min_samples_split=2, cross_validation=2):

                global classifier
                classifier = RandomForestClassifier()
                clffit = classifier.fit(X_train, y_train)
                parameters = [{'max_depth': [max_depth], 'n_estimators':[
                    n_estimators], 'min_samples_split':[min_samples_split]}]
                gs = GridSearchCV(estimator=clffit,
                                  param_grid=parameters,
                                  n_jobs=-1,
                                  scoring='accuracy',
                                  cv=cross_validation)
                gs.fit(X_train, y_train)
                classifier = gs.best_estimator_
                c = classifier.fit(X_train, y_train)
                return classifier.predict(X_test), gs.best_params_, c

            # Function for DecisionTreeCLassifier
            def decisiontreeclassifier(X_train, X_test, y_train, y_test, max_depth=None, min_samples_split=2, criterion='gini', cross_validation=2):

                global classifier
                classifier = DecisionTreeClassifier()
                clffit = classifier.fit(X_train, y_train)
                parameters = [{'max_depth': [max_depth], 'min_samples_split':[
                    min_samples_split], 'criterion':[criterion]}]
                gs = GridSearchCV(estimator=clffit,
                                  param_grid=parameters,
                                  n_jobs=-1,
                                  scoring='accuracy',
                                  cv=cross_validation)
                gs.fit(X_train, y_train)
                classifier = gs.best_estimator_
                c = classifier.fit(X_train, y_train)
                return classifier.predict(X_test), gs.best_params_, c

            # Function for SVC
            def svc(X_train, X_test, y_train, y_test, c=1.0, degree=3, kernel='rbf', cross_validation=2):

                global classifier
                classifier = SVC()
                clffit = classifier.fit(X_train, y_train)
                parameters = [{'C': [c], 'degree':[degree], 'kernel':[kernel]}]
                gs = GridSearchCV(estimator=clffit,
                                  param_grid=parameters,
                                  n_jobs=-1,
                                  scoring='accuracy',
                                  cv=cross_validation)
                gs.fit(X_train, y_train)
                classifier = gs.best_estimator_
                classifier.fit(X_train, y_train)
                return classifier.predict(X_test), gs.best_params_, classifier.fit(X_train, y_train)

            # Function for sgdclassifier
            def sgdclassifier(X_train, X_test, y_train, y_test, loss='hinge', penalty='l2', alpha=0.0001, cross_validation=2):

                global classifier
                classifier = SGDClassifier()
                clffit = classifier.fit(X_train, y_train)
                parameters = [
                    {'penalty': ['l2'], 'loss':[loss], 'alpha':[alpha]}]
                gs = GridSearchCV(estimator=clffit,
                                  param_grid=parameters,
                                  n_jobs=-1,
                                  scoring='accuracy',
                                  cv=cross_validation)
                gs.fit(X_train, y_train)
                classifier = gs.best_estimator_
                c = classifier.fit(X_train, y_train)
                return classifier.predict(X_test), gs.best_params_, c

            # Function for gradientboostingclassifier
            def gradientboostingclassifier(X_train, X_test, y_train, y_test, n_estimators=100, learning_rate=0.1, criterion='friedman_mse', cross_validation=2):

                global classifier
                classifier = GradientBoostingClassifier()
                clffit = classifier.fit(X_train, y_train)
                parameters = [{'n_estimators': [n_estimators], 'learning_rate':[
                    learning_rate], 'criterion':[criterion]}]
                gs = GridSearchCV(estimator=clffit,
                                  param_grid=parameters,
                                  n_jobs=-1,
                                  scoring='accuracy',
                                  cv=cross_validation)
                gs.fit(X_train, y_train)
                classifier = gs.best_estimator_
                c = classifier.fit(X_train, y_train)
                return classifier.predict(X_test), gs.best_params_, c

            # Function for adaboost
            def adaboostclassifier(X_train, X_test, y_train, y_test, n_estimators=50, learning_rate=1, cross_validation=2):

                global classifier
                classifier = AdaBoostClassifier()
                clffit = classifier.fit(X_train, y_train)
                parameters = [{'n_estimators': [n_estimators],
                               'learning_rate':[learning_rate]}]
                gs = GridSearchCV(estimator=clffit,
                                  param_grid=parameters,
                                  n_jobs=-1,
                                  scoring='accuracy',
                                  cv=cross_validation)
                gs.fit(X_train, y_train)
                classifier = gs.best_estimator_
                c = classifier.fit(X_train, y_train)
                return classifier.predict(X_test), gs.best_params_, c

            # Model functions

            if classifier_choice == 'Random Forest Classifier':
                # max_depth= st.slider(label='max_depth', min_value=0.0, max_value=16.0, step=0.5)
                # n_estimators= st.slider(label='n_estimators', min_value=0.0, max_value=16.0, step=0.5)
                # min_samples_split= st.slider(label='min_samples_split', min_value=0.0, max_value=16.0, step=0.5)
                classifier_output = randomforestclassifier(
                    X_train, X_test, y_train, y_test, max_depth, n_estimators, min_samples_split)

            elif classifier_choice == 'Decision Tree Classifier':
                classifier_output = decisiontreeclassifier(
                    X_train, X_test, y_train, y_test, max_depth, min_samples_split, criterion)

            elif classifier_choice == 'SVC':
                classifier_output = svc(
                    X_train, X_test, y_train, y_test, c, degree, kernel)

            elif classifier_choice == 'SGD Classifier':
                classifier_output = sgdclassifier(
                    X_train, X_test, y_train, y_test, loss, penalty, alpha)

            elif classifier_choice == 'Gradient Boosting Classifier':
                classifier_output = gradientboostingclassifier(
                    X_train, X_test, y_train, y_test, n_estimators, learning_rate, criterion)

            elif classifier_choice == 'Adaboost Classifier':
                classifier_output = adaboostclassifier(
                    X_train, X_test, y_train, y_test)

             # Time for printingout the result

            st.write('My system caught on {} training your model to get the output for you {}'.format(
                emoji.emojize(':fire:'), emoji.emojize(':satisfied:')))
            time.sleep(1.5)
            st.write('Be safe, wear a mask{}'.format(emoji.emojize(':mask:')))
            time.sleep(1.5)
            st.write('Your scores are here {}'.format(
                emoji.emojize(':raised_hands:')))
            time.sleep(1.5)
            st.write("\n")
            st.success('Accuracy score of {} is: {}'.format(
                classifier_choice, accuracy_score(y_test, classifier_output[0])))
            st.success('f1 score of {} is: {}'.format(classifier_choice, f1_score(
                y_test, classifier_output[0], average='weighted')))
            st.success('Recall score of {} is: {}'.format(classifier_choice, recall_score(
                y_test, classifier_output[0], average='weighted')))
            st.success('Precision score of {} is: {}'.format(
                classifier_choice, precision_score(y_test, classifier_output[0], average='weighted')))
            st.write('Selected parameters are: ', classifier_output[1])
            st.write('Confusion Matrix:', confusion_matrix(
                classifier, X_test, y_test))
            st.pyplot()

            st.subheader("Code (adjust hyperparameters manually)")
            file = open('codes_to_display/'+classifier_choice+' Code.txt', 'r')
            classifier_code = file.read()
            st.code(classifier_code, language='python')
            file.close()

            st.subheader("Report")
            know = open('knowledge_to_display/' +
                        classifier_choice+' Report.txt', 'rb')
            classifier_report = know.read().decode(errors='replace')
            st.code(classifier_report)
            know.close()

    elif choice == 'REGRESSION':

        regression_activities = ["Select one", 'Linear Regressor', 'Ridge Regressor', 'Lasso Regressor',
                                 'DecisionTree Regressor', 'Gradient Boosting Regressor']
        st.subheader("Select a Regression model to train your Dataset on {}".format(
            emoji.emojize(":eyes:")))
        regressor_choice = st.selectbox("", regression_activities)
        st.subheader(
            "Select the hyper-parameter {}".format(emoji.emojize(":smiley:")))

        if regressor_choice == 'Linear Regressor':
            normalize = st.selectbox('normalize', ['True', 'False'])

        elif regressor_choice == 'Ridge Regressor':
            max_iter = st.slider(
                label='max_iter', min_value=100, max_value=1500, step=100)
            solver = st.selectbox(
                'solver', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'])
            alpha = st.text_input(
                label='alpha(enter a value between 0.01 to 10.0)', value='1')
            alpha = float(alpha)

        elif regressor_choice == 'Lasso Regressor':
            max_iter = st.slider(
                label='max_iter', min_value=100, max_value=1500, step=100)
            selection = st.selectbox('selection', ['cyclic', 'random'])
            alpha = st.text_input(
                label='alpha(enter a value between 0.01 to 10.0)', value='1')
            alpha = float(alpha)

        elif regressor_choice == 'DecisionTree Regressor':
            min_samples_split = st.slider(
                label='min_samples_split', min_value=0, max_value=16, step=1)
            max_depth = st.slider(
                label='max_depth', min_value=0, max_value=16, step=1)
            criterion = st.selectbox(
                'criterion', ['mse', 'friedman_mse', 'mae'])

        elif regressor_choice == 'Gradient Boosting Regressor':
            loss = st.selectbox('loss', ['ls', 'lad', 'huber', 'quantile'])
            n_estimators = st.slider(
                label='n_estimators', min_value=100, max_value=800, step=50)
            learning_rate = st.text_input(
                label='learning_rate(enter a value between 0.001 to 1.0)', value='0.1')
            learning_rate = float(learning_rate)

        cross_validation = st.slider(
            label='cross validation (higher the number more the time taken to train)', min_value=1, max_value=10, step=1)
        submit = st.button('TRAIN')

        if submit:

            dataframe = fill_na(dataframe)

            dataframe = encode(dataframe)

            X = dataframe.iloc[:, :-1]
            y = dataframe.iloc[:, -1]

            splitreturn = splitdata(X, y)
            X_train, X_test, y_train, y_test = splitreturn[
                0], splitreturn[1], splitreturn[2], splitreturn[3]

            scalereturn = scale(X_train, X_test)
            X_train, X_test = scalereturn[0], scalereturn[1]

            st.write("Give us some {} to build your project".format(
                emoji.emojize(":watch:")))

            # Linear Regression
            def linearregressor(X_train, X_test, y_train, y_test, normalize=False, cross_validation=2):
                regressor = LinearRegression()
                parameters = [{'normalize': [normalize]}]
                regressor = GridSearchCV(
                    regressor, parameters, scoring='r2', cv=cross_validation)
                regressor.fit(X_train, y_train)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test, y_test)

            # Ridge Regression
            def ridgeregressor(X_train, X_test, y_train, y_test, max_iter=100, solver='auto', alpha=1.0, cross_validation=2):
                regressor = Ridge()
                parameters = [{'max_iter': [max_iter],
                               'solver':[solver], 'alpha':[alpha]}]
                regressor = GridSearchCV(
                    regressor, parameters, scoring='r2', cv=cross_validation)
                regressor.fit(X_train, y_train)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test, y_test)

            # Lasso Regression
            def lassoregressor(X_train, X_test, y_train, y_test, max_iter=100, selection='cyclic', alpha=1.0, cross_validation=2):
                regressor = Lasso()
                parameters = [{'max_iter': [max_iter],
                               'selection':[selection], 'alpha':[alpha]}]
                regressor = GridSearchCV(
                    regressor, parameters, scoring='r2', cv=cross_validation)
                regressor.fit(X_train, y_train)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test, y_test)

            # Decision Tree Regression
            def decisiontreeregressor(X_train, X_test, y_train, y_test, min_samples_split=2, max_depth=6, criterion='friedman_mse', cross_validation=2):
                regressor = DecisionTreeRegressor()
                parameters = [{'max_depth': [max_depth], 'min_samples_split':[
                    min_samples_split], 'criterion':[criterion]}]
                regressor = GridSearchCV(
                    regressor, parameters, scoring='r2', cv=cross_validation)
                regressor.fit(X, y)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test, y_test)

            # Gradient Boosting Regression
            def gradientboostingregressor(X_train, X_test, y_train, y_test, loss='ls', n_estimators=100, learning_rate=0.1, cross_validation=2):
                regressor = GradientBoostingRegressor()
                parameters = {'n_estimators': [n_estimators], 'loss': [
                    loss], 'learning_rate': [learning_rate]}
                regressor = GridSearchCV(
                    regressor, parameters, scoring='r2', cv=cross_validation)
                regressor.fit(X_train, y_train)

                return regressor.predict(X_test), regressor.best_params_, regressor.score(X_test, y_test)

            # Model functions

            if regressor_choice == 'Linear Regressor':
                regressor_output = linearregressor(
                    X_train, X_test, y_train, y_test, normalize)

            elif regressor_choice == 'Ridge Regressor':
                regressor_output = ridgeregressor(
                    X_train, X_test, y_train, y_test, max_iter, solver, alpha)

            elif regressor_choice == 'Lasso Regressor':
                regressor_output = lassoregressor(
                    X_train, X_test, y_train, y_test, max_iter, selection, alpha)

            elif regressor_choice == 'DecisionTree Regressor':
                regressor_output = decisiontreeregressor(
                    X_train, X_test, y_train, y_test, min_samples_split, max_depth, criterion)

            elif regressor_choice == 'Gradient Boosting Regressor':
                regressor_output = gradientboostingregressor(
                    X_train, X_test, y_train, y_test, loss, n_estimators, learning_rate)

             # Time for printing out the result

            st.write('My system caught on {} training your model to get the output for you {}'.format(
                emoji.emojize(':fire:'), emoji.emojize(':satisfied:')))
            time.sleep(1.5)
            st.write('Be safe, wear a mask{}'.format(emoji.emojize(':mask:')))
            time.sleep(1.5)
            st.write('Your scores are here {}'.format(
                emoji.emojize(':raised_hands:')))
            time.sleep(1.5)
            st.write("\n")
            st.success(
                "r2/variance for {} is: {}".format(regressor_choice, regressor_output[2]))
            st.success("Mean Squared Error for {} is: {}".format(
                regressor_choice, mean_squared_error(y_test, regressor_output[0])))
            st.write('Selected parameters are: ', regressor_output[1])

            st.subheader("Code (adjust hyperparameters manually)")
            file = open('codes_to_display/'+regressor_choice+' Code.txt', 'r')
            regressor_code = file.read()
            st.code(regressor_code, language='python')
            file.close()

            st.subheader("Report")
            know = open('knowledge_to_display/' +
                        regressor_choice+' Report.txt', 'rb')
            regressor_report = know.read().decode(errors='replace')
            st.code(regressor_report)
            know.close()
