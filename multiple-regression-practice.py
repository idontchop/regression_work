# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:09:27 2020

@author: micro
"""

import numpy
import pandas
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import platform

filename = 'real_estate.csv'

if ( platform.system() == 'Windows'):
    csvFile = filename
else:
    csvFile = '~/PythonWorkspace/' + filename
    
data = pandas.read_csv(csvFile)

print (data.describe())

# GPA = b0 + b1(SAT) + b2(Rand)

y = data['price']
x1 = data[['size','year']]

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()

print("x:\n")
print(x)
print(results.summary())