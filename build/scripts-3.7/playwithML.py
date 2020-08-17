# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:00:34 2020

@author: Akash C
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier, \
    GradientBoostingRegressor
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import ExtraTreesClassifier

__Author__ = 'Akash C'
__coAuthor__ = 'Ashwin Sharma'
__package__ = 'playwithml'
print('Thank you for downloading our package')
print('You can use the web version of this project on https://playwithml.herokuapp.com')
print('Contributors {} and {}'.format(__Author__, __coAuthor__))
print()


class predictor:

    def __init__(self, df):
        try:
            self.df = pd.read_csv(df, header=0, skip_blank_lines=True,
                                  skipinitialspace=True, encoding='latin-1')
        except:
            self.df = pd.read_excel(df, header=0)
        # self.df = pd.read_csv(df)
        self.dataframe = self.df
        print("Your dataset looks like this")
        print('\n', self.df.head(10))
        # print("Your dataset's info is here:")
        # print(self.df.info())
        delete = list(map(int, input("Enter the column names you want to delete from the datset in space separated "
                                     "manner: ").split()))
        self.df = self.df.drop(self.df.iloc[:, delete], axis=1)
        print('')
        print("After deleting the selected columns your dataset looks like this")
        print(self.df.head())
        num_rec = self.df.shape[0]
        self.dataframe = self.df

        for col in self.dataframe.columns:
            if self.dataframe[col].dtype.name != 'object':
                if (self.dataframe[col].isnull().sum()) * 2 >= num_rec:
                    self.dataframe = self.dataframe.drop([col], axis=1)
                else:
                    self.dataframe[col] = self.dataframe[col].fillna(self.dataframe[col].mean())
        self.dataframe = self.df

        for col in self.df.columns:
            if self.df[col].dtype.name == 'object':
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col])

        self.X, self.y = self.df.iloc[:, :-1], self.df.iloc[:, -1]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.25)
        self.dataframe = self.df

        sc = StandardScaler()
        self.X_train = sc.fit_transform(self.X_train)
        self.X_test = sc.transform(self.X_test)
        self.dataframe = self.df

    def randomforestclassifier(self, X_train, X_test, y_train, y_test):

        classifier = RandomForestClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    def decisiontreeclassifier(self, X_train, X_test, y_train, y_test):

        classifier = DecisionTreeClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    def svc(self, X_train, X_test, y_train, y_test):

        classifier = SVC()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    def sgdclassifier(self, X_train, X_test, y_train, y_test):

        classifier = SGDClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    def extratreeclassifier(self, X_train, X_test, y_train, y_test):

        classifier = ExtraTreesClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    ## Function for gradientboostingclassifier
    def gradientboostingclassifier(self, X_train, X_test, y_train, y_test):

        classifier = GradientBoostingClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    ## Function for adaboost
    def adaboostclassifier(self, X_train, X_test, y_train, y_test):

        classifier = AdaBoostClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    def passiveclassifier(self, X_train, X_test, y_train, y_test):

        classifier = PassiveAggressiveClassifier()
        clffit = classifier.fit(self.X_train, self.y_train)
        return classifier.predict(self.X_test)

    def highlight_max(self, s):
        '''
        highlight the maximum in a Series yellow.
        '''
        is_max = s == s.max()
        return ['background-color: yellow' if v else '' for v in is_max]

    def do_all(self, c=False, r=False):
        if c:
            self.ans = {'Random Forest Classifier': '', 'Decision Tree Classifier': '', 'SVC': '',
                        'SGD Classifier': '', 'Gradient Boosting Classifier': '',
                        'Adaboost Classifier': '', 'Extra Tree Classifier': '', 'Passive Aggresive Classifier': ''}

            self.res = {'Random Forest Classifier': '', 'Decision Tree Classifier': '', 'SVC': '',
                        'SGD Classifier': '', 'Gradient Boosting Classifier': '',
                        'Adaboost Classifier': '', 'Extra Tree Classifier': '', 'Passive Aggresive Classifier': ''}

            self.ans['Random Forest Classifier'] = (
                self.randomforestclassifier(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['Decision Tree Classifier'] = (
                self.decisiontreeclassifier(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['SVC'] = (self.svc(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['SGD Classifier'] = (self.sgdclassifier(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['Gradient Boosting Classifier'] = (
                self.gradientboostingclassifier(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['Adaboost Classifier'] = (
                self.adaboostclassifier(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['Extra Tree Classifier'] = (
                self.extratreeclassifier(self.X_train, self.X_test, self.y_train, self.y_test))
            self.ans['Passive Aggresive Classifier'] = (
                self.passiveclassifier(self.X_train, self.X_test, self.y_train, self.y_test))

            for ele in self.ans:
                self.res[ele] = []
                self.res[ele].append(accuracy_score(self.y_test, self.ans[ele]))
                self.res[ele].append(precision_score(self.y_test, self.ans[ele], average='weighted'))
                self.res[ele].append(recall_score(self.y_test, self.ans[ele], average='weighted'))
                self.res[ele].append(f1_score(self.y_test, self.ans[ele], average='weighted'))

            self.output_res = pd.DataFrame.from_dict(self.res, orient='index',
                                                     columns=['Accuracy Score', 'Precision Score', 'Recall Score',
                                                              'F1 Score'])

            return self.output_res.style.apply(self.highlight_max)
