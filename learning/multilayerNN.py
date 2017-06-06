#!/usr/bin/env python2.7

'''
Simple example of multilayer neural network 
'''

## get data
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X = cancer['data']
y = cancer['target']

## obtain train and test datasets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

## nomalize data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

## training
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(30, 30, 30)) #30 because data has the shape(590, 30)
mlp.fit(X_train, y_train)

## predict
predictions = mlp.predict(X_test)

## evaluate
from sklearn.metrics import classification_report, confusion_matrix

print "Confusion matrix results:"
print (confusion_matrix(y_test, predictions))
print "Classification report:"
print (classification_report(y_test, predictions))
