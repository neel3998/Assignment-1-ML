
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn import tree
np.random.seed(42)
maxdepth = 5
# Read real-estate realestatedata set
# ...
# 
realestatedata = pd.read_excel("Real estate valuation data set.xlsx")
realestatedata = realestatedata.drop(["No"], axis=1) #No need of first column in the realestatedataset


#Now let us split the realestatedata set into train and test realestatedatasets
#and let the ratio be 80:20
size=realestatedata.shape
no_of_samples=size[0]
train_sample=int(0.8*no_of_samples)


X_train = realestatedata.iloc[:train_sample, :-1]
y_train = realestatedata.iloc[:train_sample, -1]
X_test = realestatedata.iloc[train_sample:, :-1]
y_test = realestatedata.iloc[train_sample:, -1]


def my_model(criteria,X_train,y_train,X_test,y_test):


    tree = DecisionTree(criterion=criteria, max_depth=maxdepth) 
    tree.fit(X_train, y_train) 
    
    y_hat = tree.predict(X_train)
    print("My Model results:")
    
    print("RMSE and MAE Scores on trainset:"+"\n")
    print('RMSE: '+ str(rmse(y_hat, y_train))+"\n")
    print('MAE: '+ str(mae(y_hat, y_train))+"\n")

    y_hat_test= tree.predict(X_test)
    print("RMSE and MAE Scores on Testset:"+"\n")
    print('RMSE: '+ str(rmse(y_hat_test, y_test))+"\n")
    print('MAE: '+ str(mae(y_hat_test, y_test))+"\n")
    return 0

###################################################################################


####################################################################################
def sklearnmodel(criteria,X_train,y_train,X_test,y_test):

   
    treesk = tree.DecisionTreeRegressor(max_depth=maxdepth)
    treesk = treesk.fit(X_train,y_train)
    y_hat = pd.Series(treesk.predict(X_train))
   
    print("Sklearn Model resuls:")
    print("RMSE and MAE Scores on trainset:"+"\n")
    print('RMSE: '+ str(rmse(y_hat, y_train))+"\n")
    print('MAE: '+ str(mae(y_hat, y_train))+"\n")


    y_hat_test = pd.Series(treesk.predict(X_test))
    print("RMSE and MAE Scores on testset:")
    print('RMSE: '+ str(rmse(y_hat_test, y_test))+"\n")
    print('MAE: '+ str(mae(y_hat_test, y_test))+"\n")
    return 0
criteria = 'information_gain'
my_model(criteria,X_train,y_train,X_test,y_test)
sklearnmodel(criteria,X_train,y_train,X_test,y_test)