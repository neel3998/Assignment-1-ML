import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor
from sklearn import datasets
from sklearn.utils import shuffle
###Write code here
############
# Setting random seed
############
np.random.seed(42)
#####################
# Importing Iris data
#####################
iris_data=datasets.load_iris()
df_iris= pd.DataFrame(iris_data.data)

X=df_iris
#print(type(X))
y=iris_data.target
X = pd.DataFrame(data=X)
y = pd.Series(data=y, dtype="category")
####################
#print(X)
#print(y)
###################
X, y = shuffle(X, y, random_state=42) ###Shuffling data with random seed=42

a=int(0.6*len(X))
Xtrain=X[:a]
Xtest=X[a:]
ytrain=y[:a]
ytest=pd.Series(y[a:])

randforest=RandomForestClassifier(3,criterion='entropy',max_depth=4) #No.of estimators=3
randforest.fit(Xtrain,ytrain)
y_hat=randforest.predict(Xtrain)
y_test_hat=randforest.predict(Xtest)

print('Train Accuracy: ', accuracy(y_hat, ytrain))
print('Test Accuracy: ', accuracy(y_test_hat, ytest))
    
for cls in y.unique():
    print("Class =",cls)
    print('Precision: ', precision(y_test_hat, ytest, cls))
    print('Recall: ', recall(y_test_hat, ytest, cls))
#randforest.plot()