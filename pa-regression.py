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

data2 = data2.drop([9])

plt.figure(1)
plt.scatter(data2['No. of Incidents'], data2['Officers at Scene'])
plt.xlabel("# Incidents")
plt.ylabel("# Officers")




