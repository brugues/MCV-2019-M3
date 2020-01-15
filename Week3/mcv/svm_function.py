import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def applySVM(features, labels):
    C_range = 10. ** np.arange(-3,8)
    gamma_range = 10. ** np.arange(-5,4)

    param_grid = dict(gamma=gamma_range, C=C_range)

    grid = GridSearchCV(SVC(kernel='rbf'), param_grid = param_grid, cv=3)
    grid.fit(features,labels)

    return grid
