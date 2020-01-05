# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 23:55:14 2019

@author: Sampritha H M
"""

import pandas as pd
#from matplotlib.colors import ListedColormap

#from sklearn import neighbors
hazelnut = pd.read_csv('./hazelnut.csv')

print("hazelnut data before separating target\n")
print(hazelnut.head())

X = hazelnut.drop(['variety','sample_id'],axis=1)
print("hazelnut data after separating target\n",X.head(),"\n")

y = hazelnut['variety'].values
print("Top 5 values in Target (variety)\n")
print(y[:5],"\n")


from sklearn.model_selection import train_test_split

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)
print("Y pred:",y_pred)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# select kernel as linear with penalty parameter C 
clf = SVC(kernel='linear', C=1)
# train model with cv of 10
cv_scores = cross_val_score(clf, X, y, cv=10)
# plot the graph for each fold in knn
cv_range = (1,2,3,4,5,6,7,8,9,10)
plt.plot(cv_range,cv_scores)
plt.xlabel('Number of folds')
plt.ylabel('10-fold cross validation accuracy')

print("Cross Validation scores: \n",cv_scores)
mean = cv_scores.mean()
mean = float(mean) * int(100)
print("Accuracy: %0.2f" %mean,"%")                             
