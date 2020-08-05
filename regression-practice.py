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
import platform
from sklearn.linear_model import LinearRegression

if ( platform.system() == 'Windows'):
    csvFile = "simple-linear.csv"
else:
    csvFile = '~/PythonWorkspace/simple-linear.csv'
    
data = pandas.read_csv(csvFile)
datadata = pandas.read_csv(csvFile)



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


# sklearn
reg = LinearRegression()

reg.fit(x1.values.reshape(84,1),y)
x_mat = x1.values.reshape(-1,1)

print(reg.score(x_mat,y))

new_data = pandas.DataFrame( data= [1900,2400], columns=['SAT'])
print(reg.predict(new_data))

