# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 03:23:45 2020

@author: Bhavin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv("income.csv")
data.info()
data.isnull().sum()
data.describe()
data.columns
sns.heatmap(data.corr())
data['JobType'].value_counts()
data['occupation'].value_counts()

pd.set_option('display.max_columns',500)
data['JobType'].unique()

data1=pd.read_csv("income.csv", na_values=[" ?" ])
data1.isnull().sum()
missing=data1[data1.isnull().any(axis=1)]
data2=data1.dropna(axis=0)

data2.corr()


gender=pd.crosstab(index= data2["gender"],
                   columns='count',normalize=True)
print(gender)

sal_stat=pd.crosstab(index= data2['gender'],
                     columns=data2['SalStat'],
                     margins=True,
                     normalize='index')
sal_stat


sal=sns.countplot(data2['SalStat'])
sal

sns.distplot(data2['age'],bins=10,kde=False) #kernel density estimation

sns.boxplot('SalStat','age',data=data2)


data2.groupby('SalStat')['age'].median()






data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000': 0,' 
     greater than 50,000': 1})
print(data2['SalStat'])

####### employee working 40-50 hours those who got >500000 ########

sns.boxplot('SalStat','hoursperweek',data=data2)
######### 95% capitalloss is 0 ##########
sns.distplot(data2['capitalloss'],bins=10,kde=False)


##### 92% of capital gain is 0  ############

sns.distplot(data2['capitalgain'],bins=10,kde=False)

#####  managerial and professional speciolity has greater salary #####
occupation=pd.crosstab(index= data2['occupation'],
                     columns=data2['SalStat'],
                     margins=True,
                     normalize='index')

occupation

######managerial and professional speciolity has greater salary####
plt.figure(figsize=(30,5))
sal_occupation=sns.countplot(data2['occupation'], hue=data2['SalStat'])
sal_occupation

####### education type : Doctorate, masters, prof-school have high salary
Edtype_salary=pd.crosstab(index= data2['EdType'],
                     columns=data2['SalStat'],
                     margins=True,
                     normalize='index')

Edtype_salary

#########education type : Doctorate, masters, prof-school have high salary
plt.figure(figsize=(20,5))
sal_Edtype=sns.countplot(data2['EdType'], hue=data2['SalStat'])
sal_Edtype

######### self employed people earning more 55% ######
JobType_salary=pd.crosstab(index= data2['JobType'],
                     columns=data2['SalStat'],
                     margins=True,
                     normalize='index')

JobType_salary

##########self employed people earning more 55% #33

plt.figure(figsize=(20,5))
sal_JobType=sns.countplot(data2['JobType'], hue=data2['SalStat'])
sal_JobType

#data3=data2.iloc[:,12:13]
#data4=data3['SalStat'].map({' less than or equal to 50,000': 0,' greater than 50,000': 1})

new_data=pd.get_dummies(data2,drop_first=True)
column_list=list(new_data.columns)
features=list(set(column_list)-set(['SalStat']))
print(features)

X=new_data[features].values
y=new_data['SalStat'].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)

##LogisticRegression algorithm
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
classifier.intercept_
classifier.coef_

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

print("misclassified samples= %d" %(y_test!=y_pred).sum())
 
##############KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

print("misclassified samples= %d" %(y_test!=y_pred).sum())
#######################Random Forest algorithm

from sklearn.ensemble import RandomForestClassifier
classifier4=RandomForestClassifier()
classifier4.fit(X_train,y_train)

y_pred1=classifier4.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred1)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred1))

print("misclassified samples= %d" %(y_test!=y_pred1).sum())


####Bagging classifier
from sklearn.ensemble import BaggingClassifier
classifier5=BaggingClassifier()
classifier5.fit(X_train,y_train)

y_pred2=classifier5.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred2)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred2))

print("misclassified samples= %d" %(y_test!=y_pred2).sum())

##### Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
classifier6=GradientBoostingClassifier()
classifier6.fit(X_train,y_train)

y_pred3=classifier6.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred3)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred3))

print("misclassified samples= %d" %(y_test!=y_pred3).sum())
############ Support Vector Machine

from sklearn.svm import SVC
classifier7=SVC()
classifier7.fit(X_train,y_train)

y_pred4=classifier7.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred4)
cm

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred4))

print("misclassified samples= %d" %(y_test!=y_pred4).sum())







from sklearn import svm

svm.






from sklearn import ensemble

ensemble.






from sklearn import ensemble 
ensemble.RandomForestClassifier
ensemble.BaggingClassifier
ensemble.GradientBoostingClassifier