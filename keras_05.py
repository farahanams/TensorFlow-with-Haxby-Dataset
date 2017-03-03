#-------------------------------
#   Encoding with sklearn
#-------------------------------

import numpy as np
import matplotlib.pyplot as plt
import load_data
plt.ion()

data_x = np.recfromtxt('encoded_test.csv', delimiter= ' ')
data_y = np.recfromtxt('label_test.csv', delimiter=' ')

print (' Size of data_x is:', data_x.shape, 'size of data_y is:', data_y.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

cv = StratifiedKFold(shuffle=True, random_state=10, n_splits=10)

def fmriClassifier(clf, X, Y, cv):
    fitter = clf.fit(X, Y)
    return cross_val_score(fitter, X, Y, cv=cv)

cv_score_1 = fmriClassifier(LogisticRegression(), data_x, data_y, cv)


print(cv_score_1)

