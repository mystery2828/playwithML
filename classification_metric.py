# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 15:55:15 2020

@author: Akash
"""

from sklearn.metrics import jaccard_similarity_score, precision_score,\
    recall_score, accuracy_score, f1_score
    
from playwithML import classification_model as cmp

y_test = 

print('Accuracy score is: {}'.format(accuracy_score(y_test,cmp)))
print('f1 score is: {}'.format(f1_score(y_test,cmp,average='weighted')))
print('Recall score is: {}'.format(recall_score(y_test,cmp,average='weighted')))
print('Precision score is: {}'.format(precision_score(y_test,cmp,average='weighted')))