# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 16:50:20 2022

@author: Muqeet
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


data=pd.read_csv('titanic_train.csv')

len(data)


data.head()

data.index

data.columns

data.info()

data.describe()


#dataanalysis
#using seaborn library

sns.countplot(x='Survived',data=data,hue='Sex')

#find null values

data.isnull()

data.isnull().sum()

sns.heatmap(data.isnull())

(data['Age'].isnull().sum()/len(data['Age']))*100



(data['Cabin'].isnull().sum()/len(data['Cabin']))*100

sns.displot(x="Age",data=data)


#data cleaning

data['Age'].fillna(data['Age'].mean(),inplace=True)

data['Age'].isnull().sum()

sns.heatmap(data.isnull())

#drop column

data.drop('Cabin',axis=1,inplace=True)

sns.heatmap(data.isnull())

data.dtypes

#convert sex column to numerical values

gender=pd.get_dummies(data['Sex'],drop_first=True)

data['Gender']=gender

data.columns

data.dtypes

data.drop(['Name','Sex','Ticket','Embarked'],axis=1,inplace=True)

data.columns

x=data[['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
       'Gender']]

y=data['Survived']


#data modeling

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=42)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()

lr.fit(x_train,y_train)

predict=lr.predict(x_test)


#confusion matrix

from sklearn.metrics import confusion_matrix

pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted no','Predicted yes'],index=["actual no",'actual yes'])


