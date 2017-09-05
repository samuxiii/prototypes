#!/usr/bin/env python2.7

from sklearn import tree, neighbors, svm
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

#Train data
train = [[181,80,43], [168,60,39], [171,77,41], [155,59,40], [154,60,36], [191,99,45], [168,70,42], [165, 61,39]]
train = np.array(train).reshape(8, 3)

#labels associated
label = ['male','female','male','female','female','male','male','female']
le = preprocessing.LabelEncoder().fit(label)
print(le.transform(label))

#Test sample
test = np.array([160, 48, 39]).reshape(1, -1)

#Decision Tree calssifier Test
clf1 = tree.DecisionTreeClassifier()
clf1 = clf1.fit(train, label)
prediction1 = clf1.predict(test)

print('Decision Tree Classifier predicts: {}'.format(str(prediction1)))

#Kneighbors Classifier Test
clf2 = neighbors.KNeighborsClassifier()
clf2 = clf2.fit(train,label)
prediction2 = clf2.predict(test)

print('K-nearest neighbor classifier predicts: {}'.format(str(prediction2)))

#Support Vector Classifier
clf3 = svm.SVC()
clf3 = clf3.fit(train, label)
prediction3 = clf3.predict(test)

print('SVC Classifier predicts: {}'.format(str(prediction3)))


plt.figure(1)
plt.plot(train[:,0], le.transform(label), 'ro')
plt.plot(train[:,0], le.transform(clf1.predict(train)), 'bx')
plt.plot(train[:,0], le.transform(clf2.predict(train)), 'gx')
plt.plot(train[:,0], le.transform(clf3.predict(train)), 'yx')
plt.show()
