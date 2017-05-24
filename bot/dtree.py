from sklearn import tree, neighbors, svm
from sklearn import preprocessing
import numpy as np

def predictGender(height, weight, foot):
        train = [[181,80,43], [168,60,39], [171,77,41], [155,59,40], [154,60,36], [191,99,45], [168,70,42], [165, 61,39]]
        train = np.array(train).reshape(8, 3)

        label = ['man','woman','man','woman','woman','man','man','woman']

        test = np.array([height, weight, foot]).reshape(1, -1)

        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train, label)
        prediction = clf.predict(test)

        return prediction[0]
