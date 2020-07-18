# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy
import pandas
import scipy
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data = pandas.read_csv('~/PythonWorkspace/simple-linear.csv')
datadata = pandas.read_csv('~/PythonWorkspace/simple-linear.csv')


y = data['GPA']
x1 = data['SAT']

plt.figure(1)
plt.scatter(x1,y)
plt.xlabel('SAT', fontsize=20)
plt.ylabel('GPA', fontsize=20)
plt.show()

x = sm.add_constant(x1)
x
results = sm.OLS(y,x).fit()
results.summary()

plt.scatter(x1,y)
yhat = .0017 * x1 + .275
fig = plt.plot(x1, yhat, lw=1, c='red', label='regression')
plt.xlabel('SAT')
plt.ylabel('GPA')


plt.figure(2)
sns.set(style="darkgrid")
sns.residplot(x1, y, lowess=True, color="green")

plt.figure(3)
sns.set(style = "darkgrid")
sns.residplot(x1, y, label="GPA Regression Residual", color ="brown")
