# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:37:44 2018

@author: soanand

Problem Statetment
We will be using a decision tree to make predictions about the Titanic data set from
Kaggle. This data set provides information on the Titanic passengers and can be used to
predict whether a passenger survived or not.
"""

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn
from pandas import Series, DataFrame
from pylab import rcParams
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report

url="https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
titanic = pd.read_csv(url)
titanic.columns = ['PassengerId','Survived','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked']

"""
You use only Pclass, Sex, Age, SibSp (Siblings aboard), Parch (Parents/children aboard),
and Fare to predict whether a passenger survived.
"""
X = pd.DataFrame(data=titanic, columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
y = pd.DataFrame(titanic.Survived)

"""
function to preprocess data
"""
def preprocess_data(data):
    data['Age'] = data.Age.fillna(data.Age.mean())
    data['Sex'] = pd.Series([1 if s == 'male' else 0 for s in data.Sex], name = 'Sex')
    return data

# Preprocess the dataset
X = preprocess_data(X)

# Spliting datatset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#create a decision tree model
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

#predict the result
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cm=confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
clas = classification_report(y_test, y_pred)


#cross validation
from sklearn.cross_validation import KFold
cv = KFold(n=len(X), n_folds=10, shuffle=True, random_state=0)

X_fold = X
y_fold = y

fold_accuracy=[]
for train_fold, valid_fold in cv:
    train = X_fold.loc[train_fold]      #extract train data with cv indices
    valid = X_fold.loc[valid_fold]      #extract valid adata with cv indices

    train_y = y_fold.loc[train_fold]
    valid_y = y_fold.loc[valid_fold]
    
    model=classifier.fit(train, train_y)
    valid_acc=model.score(valid, valid_y)
    fold_accuracy.append(valid_acc)
    
print("Accuracy per fold: ", fold_accuracy, "\n")
print("Average accuracy: ",sum(fold_accuracy)/len(fold_accuracy))
