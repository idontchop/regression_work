#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:09:57 2020

@author: nathan
"""


import numpy
import pandas
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data2 = pandas.read_excel('~/PythonWorkspace/pa-regression.xlsx')

data2.describe()



plt.figure(1)
plt.scatter(data2['No. of Incidents'], data2['Officers at Scene'])
plt.xlabel("# Incidents")
plt.ylabel("# Officers")

plt.show()

plt.figure(2)

data2 = data2.drop([9])
x1 = data2['No. of Incidents']
y = data2['Officers at Scene']

x = sm.add_constant(x1)
results = sm.OLS(y,x).fit()



print("\nParams ----------\n")
print(results.params)
print("\nTvalues ---------\n")
print(results.tvalues)

yhat = 1.3663 * x + 30.4001

sns.set(style="darkgrid")
plt.scatter(x1,y)
plt.xlabel("# Incidents")
plt.ylabel("# Officers")

fig = plt.plot(x1, yhat, lw=1, c='red', label='regression')

print(yhat)


