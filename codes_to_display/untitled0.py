# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 23:07:48 2020

@author: user
"""

s = '''

'''



list_of_options1 = ['Random Forest Classifier','Decision Tree Classifier','SVC',
                           'SGD Classifier','Gradient Boosting Classifier',
                           'Adaboost Classifier']
regression_activities = ['Linear Regressor','Ridge Regressor','Lasso Regressor',
                   'DecisionTree Regressor','Gradient Boosting Regressor']

for ele in regression_activities:
    file = open(ele+' Code.txt','a')
    file.write(s)
    file.close()
    
