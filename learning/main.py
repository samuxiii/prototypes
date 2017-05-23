#!/bin/python

#from sklearn import datasets
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt(fname = 'data.txt', delimiter = ',')
X, y = data[:,:5], data[:,5]

print X
print y

m = X.shape[0] #number of samples

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RidgeCV(alphas = [0.1, 1.0, 10.0], normalize=True)
clf.fit(X_train, y_train)

prediction = clf.predict(X_test);
print 'X_test is: %f'
print X_test
print 'Expected is: %f' % y_test[0]
print 'Prediction is: %f' % prediction[0]
print "Score: %f" % clf.score(X, y)
print "Alpha: %f" % clf.alpha_

plt.figure(1)
plt.plot(X[:,0], y, 'bo')
plt.plot(X[:,0], clf.predict(X), 'gx')
plt.ylabel('Ys')
plt.xlabel('Xs')
plt.show()
