import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn import datasets
from sklearn.utils import shuffle
np.random.seed(42)

# Read IRIS data set
# ...
# 
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

