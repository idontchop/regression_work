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
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from random import randrange

scaler = StandardScaler()
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

# Create a rand 1,2,3 column
numbers_list = []
for i in range(len(districtCount)):
    numbers_list.append(randrange(3)+1)
    
#districtCount['Rand'] = numbers_list;

# create plot with all data      
x = districtCount[['# Incidents']]
x_matrix = x
y = districtCount['# officers']


    

# sklearn
reg = LinearRegression()

scaler.fit(x_matrix)
x_scaled = scaler.transform(x_matrix)
#print(x_scaled, numpy.mean(x_matrix))

reg.fit(x_scaled,y)

p_values = f_regression(x_scaled,y)[1]
print(p_values.round(3))

print(reg.score(x_matrix,y), "\n", reg.coef_)

print(districtCount.describe())

## Summary
reg_summary = pandas.DataFrame([['Bias'],['# Incidents']], columns = ['Features'])
reg_summary['Weights'] = reg.intercept_, reg.coef_[0]

print(reg_summary)

new_data = pandas.DataFrame(data=[[32],[135]],columns=['# Incidents'])
new_data_scaled = scaler.transform(new_data)
print("--\n",new_data_scaled,"\n--")
print(reg.predict(new_data_scaled))