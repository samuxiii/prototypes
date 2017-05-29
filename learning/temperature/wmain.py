#!/bin/python

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.dates import date2num
import matplotlib.dates as mdates
from datetime import datetime

def datestr2num(s):
    return date2num(datetime.strptime(s, '%Y-%m-%d'))

data = np.loadtxt(fname = 'weather.txt', delimiter = ',', converters = {0: mdates.strpdate2num('%Y-%m-%d')})
print data
X0, X, y = data[:,0], data[:,0:4], data[:,4]

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

#plotting
plt.figure(1)
real, = plt.plot(X0, y, 'b-', label='real')
predicted, = plt.plot(X0, clf.predict(X), 'r-', label='predicted')

#gca = get current axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.axhline(y=0, color='g')
#gcf = get current figure
plt.gcf().autofmt_xdate()

#legend
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
