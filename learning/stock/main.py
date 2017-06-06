#!/usr/bin/env python2.7

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import numpy as np
import matplotlib.pyplot as plt
import os

data = np.loadtxt(fname = 'data.txt', delimiter = ',')
X, y = data[:,:5], data[:,5]

print X
print y

m = X.shape[0] #number of samples

#training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = RidgeCV(alphas = [0.1, 1.0, 10.0], normalize=True)
clf.fit(X_train, y_train)

#make parsistent the model or retrieve if exists
filename = 'model.pkl'

if (os.path.isfile(filename)):
   joblib.load(filename)
   print "loaded existing model"
else:
   joblib.dump(clf, filename)
   print "created new persistent model"

#predict
prediction = clf.predict(X_test);
print 'X_test is: %f'
print X_test
print 'Expected is: %f' % y_test[0]
print 'Prediction is: %f' % prediction[0]
print "Score: %f" % clf.score(X, y)
print "Alpha: %f" % clf.alpha_

#plotting all data
plt.figure(1)
real, = plt.plot(np.arange(m), y, 'b-', label='real')
predicted, = plt.plot(np.arange(m), clf.predict(X), 'r-', label='predicted')
plt.ylabel('Stock')
plt.xlabel('Time')
plt.legend([real, predicted], ['Real', 'Predicted'])
plt.show()

#pltting only test
mtest = X_test.shape[0]
real, = plt.plot(np.arange(mtest), y_test, 'b-', label='real')
test, = plt.plot(np.arange(mtest), clf.predict(X_test), 'go', label='test')
plt.ylabel('Stock')
plt.xlabel('Time')
plt.legend([real, test], ['Real', 'Test'])
plt.show()
