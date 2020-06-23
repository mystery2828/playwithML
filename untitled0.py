# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:25:35 2020

@author: user
"""


regression_activities = ['Linear Regressor','Ridge Regressor','Lasso Regressor',
                   'DecisionTree Regressor','Gradient Boosting Regressor']
classification_activities = ['Random Forest Classifier','Decision Tree Classifier','SVC',
                           'SGD Classifier','Gradient Boosting Classifier',
                           'Adaboost Classifier']

for ele in classification_activities:
    file = open('knowledge_to_display/'+ele+' Report.txt','a')
    file.close()