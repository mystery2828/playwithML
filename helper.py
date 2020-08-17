# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 15:04:40 2020

@author: user
"""


from playwithml import predictor as p

P = p('datasets/iris.csv')
print(P.do_all(c=True))
