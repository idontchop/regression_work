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

sns.set( style="darkgrid")

if ( platform.system() == 'Windows'):
    csvFile = "Clean Data.xlsx"
else:
    csvFile = '~/PythonWorkspace/Clean Data.xlsx'
    
data = pandas.read_excel(csvFile)


# Column names
incident = 'Event Clearance Group'
officers = 'OFFICERS_AT_SCENE'
district = 'District/Sector'
date = 'Event Clearance Data'

#print(data.describe())
#print(data['District/Sector'].count())

# group by district and sum officers
districtCount = data.\
    groupby(district)\
    .agg( {incident: 'count', officers: 'sum'} )\
    .rename(columns={incident: '# Incidents', officers: '# officers'})


# create plot with all data      
x = districtCount['# Incidents']
x_matrix = x.values.reshape(-1,1)
y = districtCount['# officers']


# sklearn
reg = LinearRegression()

reg.fit(x_matrix,y)

print(reg.score(x_matrix,y), "\n", reg.coef_)
predictValues = pandas.DataFrame(data=[165,20])
print(reg.predict(predictValues))


