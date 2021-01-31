from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.tree import export_graphviz
from subprocess import call
from copy import deepcopy as dc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
from sklearn.decomposition import PCA

def max_count(a):
    counts = [[a.count(i),i] for i in a]
    counts.sort(key=lambda x:x[0]) 
    return counts[-1][1]
class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion=None, max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
     #   print(X)
      #  print(y)
        models = []
        no_features = X.shape[1]
        feature_no=[]
        X_samp = []
        X_samp=[]
        a = int(np.sqrt(no_features)) #Generally we take features=sqrt(no_of_features) in one model 
        for i in range(self.n_estimators):
            model = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
            j = np.random.choice(range(no_features), size=a, replace=True)
            X_sub = X[j] #Taking those columns only whic were randomly selected in the above step
            X_samp.append(dc(X_sub)) #Storing the X[id]
            feature_no.append(j)
            model.fit(X_sub, y)
            models.append(model)

        self.feature_no= feature_no     
        self.models= models 
        self.X_samp=X_samp
        self.X = X
        self.y = y
        return models

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred=[]
        for i in X.index.values:
            X_test = X[X.index==i]
            #print(X_test)
            preds=[]
            for j in range(self.n_estimators):
                X_1 = X_test[self.feature_no[j]]
                pred = self.models[j].predict(X_1)
               # print(pred)
                preds.append(pred[0])
            y_pred.append(max_count(preds))
        #print(y_pred)
        return pd.Series(y_pred)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
       pass
       
        



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion=None, max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        models = []
        no_features = X.shape[1]
        feature_no=[]
        X_samp = []
        X_samp=[]
        a = int(np.sqrt(no_features)) #Generally we take features=sqrt(no_of_features) in one model 
        for i in range(self.n_estimators):
            model = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth)
            j = np.random.choice(range(no_features), size=a, replace=True)
            X_sub = X[j] #Taking those columns only whic were randomly selected in the above step
            X_samp.append(dc(X_sub)) #Storing the X[id]
            feature_no.append(j)
            model.fit(X_sub, y)
            models.append(model)

        self.feature_no= feature_no     
        self.models= models 
        self.X_samp=X_samp
        self.X = X
        self.y = y
        return models

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred=[]
        for i in X.index.values:
            X_test = X[X.index==i]
            #print(X_test)
            preds=[]
            for j in range(self.n_estimators):
                X_1 = X_test[self.feature_no[j]]
                pred = self.models[j].predict(X_1)
               # print(pred)
                preds.append(pred[0])
            y_pred.append(mean(preds))
        #print(y_pred)
        return pd.Series(y_pred)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
