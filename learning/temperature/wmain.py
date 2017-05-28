#!/bin/python

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from datetime import datetime

def datestr2num(s):
    return date2num(datetime.strptime(s, '%Y-%m-%d'))

data = np.loadtxt(fname = 'weather.txt', delimiter = ',', converters = {0: datestr2num})
X, y = data[:,1:4], data[:,4]

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
real, = plt.plot(np.arange(m), y, 'b-', label='real')
predicted, = plt.plot(np.arange(m), clf.predict(X), 'r-', label='predicted')
plt.ylabel('Temperature')
plt.xlabel('Date')

plt.legend([real, predicted], ['Real', 'Predicted'])

plt.show()

#predict the number of the user input
t = input('Today\'s temperature:')
h = input('Today\'s humidity:')
w = input('Today\'s wind speed:')
user_input = np.array([t, h, w]).reshape(1, -1)
print "Tomorrow's temperature: %f" % clf.predict(user_input)