"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
#from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from sklearn.tree import DecisionTreeClassifier as DecisionTree
from sklearn.utils import shuffle
#from linearRegression.linearRegression import LinearRegression
from sklearn.datasets import load_iris
np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'entropy'
tree = DecisionTree(criterion=criteria, max_depth = 1)

Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)

#[fig1, fig2] = Classifier_AB.plot(X,y)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))



##### AdaBoostClassifier on Iris data set using the entire data set with sepal width and petal width as the two features
iris_data=datasets.load_iris()
df_iris= pd.DataFrame(iris_data.data)
X=df_iris
#print(type(X))
y=iris_data.target
X = pd.DataFrame(data=X)
y = pd.Series(data=y, dtype="category")

X, y = shuffle(X, y, random_state=42) ###Shuffling data with random seed=42

a=int(0.7*len(X))
Xtrain=X[:a]
Xtest=X[a:]
ytrain=pd.Series(y[:a],dtype=y.dtype)
ytest=pd.Series(y[a:],dtype=y.dtype)

###############Implementing on Iris Data Set
for criteria in ["information_gain",'gini_index']:
    tree = DecisionTree(criterion=criteria, max_depth=3)
    
    tree.fit(Xtrain, ytrain)
    
    y_hat = tree.predict(Xtrain)
    y_test_hat = tree.predict(Xtest)
  #  tree.plot()
    print('Criteria :', criteria)
    print('Train Accuracy: ', accuracy(y_hat, ytrain))
    print('Test Accuracy: ', accuracy(y_test_hat, ytest))
    
    for cls in y.unique():
        print("Class =",cls)
        print('Precision: ', precision(y_test_hat, ytest, cls))
        print('Recall: ', recall(y_test_hat, ytest, cls))

###############################################
######Decision Stump#############
##################################
Decision_stump = DecisionTree(criterion=criteria, max_depth =2)
Decision_stump.fit(train_X,train_y)
iris_stump = pd.Series(Decision_stump.predict(test_X))

print("Decision Stump")
print('Criteria :', criteria)
print('Accuracy: ', accuracy(iris_stump, test_y))
for cls in np.unique(y):
    print('Precision: ', precision(iris_stump, test_y, cls))
    print('Recall: ', recall(iris_stump, test_y, cls))