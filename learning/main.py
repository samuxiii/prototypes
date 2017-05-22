#!/bin/python

#from sklearn import datasets
from sklearn.linear_model import RidgeCV
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname = 'data.txt', delimiter = ',')
X, y = data[:,:5], data[:,5]

print X
print y

m = X.shape[0] #number of samples

X_test = X[0,:]
y_test = y[0]

clf = RidgeCV(alphas = [0.1, 1.0, 10.0], normalize=True)
clf.fit(X, y)

prediction = clf.predict(X_test);
print 'X_test is: %f'
print X_test
print 'Expected is: %f' % y_test
print 'Prediction is: %f' % prediction

accuracy = 0
for i in range(m):
    if y[i] == clf.predict(X[i,:]):
       accuracy+=1

print "Accuracy: %f" % accuracy
print "Score: %f" % clf.score(X, y)

plt.figure(1)
plt.plot(X[:,0], y, 'bo')
plt.plot(X[:,0], clf.predict(X), 'gx')
plt.ylabel('Ys')
plt.xlabel('Xs')
plt.show()
