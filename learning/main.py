#!/bin/python

#from sklearn import datasets
from sklearn.linear_model import RidgeCV
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname = 'data.txt', delimiter = ',')
X, y = data[:,:1], data[:,6]

m = X.shape[0] #number of samples

X_test = X[1,:]
y_test = y[1]

clf = RidgeCV(alphas = [0.1, 1.0, 10.0])
clf.fit(X, y)

prediction = clf.predict(X_test);
print 'X_test is: %f' % X_test
print 'Expected is: %f' % y_test
print 'Prediction is: %f' % prediction

accuracy = 0
for i in range(m):
    if y[i] == clf.predict(X[i,:]):
       accuracy+=1

print "Accuracy: %f" % accuracy
print "Score: %f" % clf.score(X, y)

plt.figure(1)
plt.plot(X, y, 'ro')
plt.plot(X, clf.predict(X), 'g-')
plt.ylabel('Ys')
plt.xlabel('Xs')
plt.show()
