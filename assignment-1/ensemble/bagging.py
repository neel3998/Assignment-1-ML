import random
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from copy import deepcopy as dc
from matplotlib.colors import ListedColormap
from statistics import mode


class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.tree = base_estimator # The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
        self.no_of_estimators = n_estimators #The number of estimators/models in ensemble.

    def fit(self, X, y):
        models = []
        X_samp = []
        y_samp = []
        X_values = X.values
       # print(X_values) #each row
        y_values = y.values
        #print(y_values) #each column
        
        a=0
        """
        Function to train and cimport copyonstruct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        while a < self.no_of_estimators:
            temp_X = []
            temp_y = []
            for j in range(len(X)):
                number = random.randint(0,len(X)-1) #0 to len(X)-1 because X index varies from 0 to len(X)-1
                s1,s2 = X_values[number], y_values[number]
                temp_X.append(s1)# adding the sample into temp_X 
                temp_y.append(s2)
            y_samp.append(temp_y)
            X_samp.append(temp_X)
            model = self.tree.fit(temp_X,temp_y) #Training those 30 points which were choosen randomly in ith estimator
            models.append(dc(model)) #storing model
            a+=1
        
        self.X_samp=X_samp
        self.y_samp=y_samp
        self.X = X
        self.y = y
        self.models = models
      #  print(models)
        return models
        

    def predict(self,X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        i = 0
        y_hat = []
        length  = len(X)
        while i< length:
            y_pred = []
            tmp = X[X.index == i] # taking each samples one by one
            for each in self.models:
                predict = each.predict(tmp) #this is the predict built in function of sklearn and it gives the output corresponding to models which we trained earlier.
               # print(predict)
                y_pred.extend(predict) #storing values that were predicted
            y_hat.append(mode(y_pred))  #Taking the highest occuring value
            i+=1
        y_hat = pd.Series(y_hat) 
        
        return y_hat

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        h = .02  # step size in the mesh
        figure1 = plt.figure(figsize=(40, 9))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        for i in range(self.no_of_estimators):
            plots = plt.subplot(1,self.no_of_estimators,i+1)
            temp_X = self.X_samp[i]
            temp_y = self.y_samp[i]
            temp_X = pd.DataFrame(temp_X)
            temp_y = pd.Series(temp_y)
            X_cord = temp_X.columns[0]
            y_cord = temp_X.columns[1]
            x_min = temp_X[X_cord].min() - .5
            x_max = temp_X[X_cord].max() + .5
            y_min = temp_X[y_cord].min() - .5
            y_max = temp_X[y_cord].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = np.array(self.models[i].predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],columns= temp_X.columns)))
            Z = Z.reshape(xx.shape)
            plots.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plots.scatter(temp_X[X_cord], temp_X[y_cord], c = temp_y, cmap=cm_bright,edgecolors='k', alpha=0.6)
            plots.set_xlim(xx.min(), xx.max())
            plots.set_ylim(yy.min(), yy.max())
            plt.title("Estimator "+ str(i+1))

            
        figure2 = plt.figure(figsize=(9,9))
        temp_X = self.X
        temp_y = self.y
        plots = plt.subplot(111)
        X_cord = temp_X.columns[0]
        y_cord = temp_X.columns[1]
        x_min = temp_X[X_cord].min() - .5
        x_max = temp_X[X_cord].max() + .5
        y_min = temp_X[y_cord].min() - .5
        y_max = temp_X[y_cord].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = np.array(self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],columns=temp_X.columns)))
        Z = Z.reshape(xx.shape)
        plots.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        plots.scatter(temp_X[X_cord], temp_X[y_cord],c = temp_y, cmap=cm_bright,edgecolors='k', alpha = 0.6)
        plots.set_xlim(xx.min(), xx.max())
        plots.set_ylim(yy.min(), yy.max())
        plt.title("Final Decision Tree Bagging Graph")
        plt.tight_layout()
        plt.show()
        return[figure1,figure2]


      #  Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

            

