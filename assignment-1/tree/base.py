"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)


# creating class for a tree node
class Node():
    def __init__(self):
        self.value = None
        self.isleaf = False
        self.attr_no = None
        self.splitvalue = None
        self.isAttr = False
        self.children = {}


class DecisionTree():
    def __init__(self, criterion, max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.head = None

    
    def helper_fit(self,X,y,depth):

        currentNode = Node()   # Creating a new Node

        currentNode.attr_no = -1
        split_value = None
        final_value = None #Initializing some variables
                           
        # Classification Problems
        if(y.dtype.name=="category"): #discrete output
            classes = np.unique(y) #No. of different outputs

            if(X.shape[1]==0): #Zero features or attributes
                currentNode.isleaf = True
                currentNode.isAttr = True
                currentNode.value = y.value_counts().idxmax() #Output is the max number of output in y
                return currentNode

            if(classes.size==1): #If there is only one type of output
                currentNode.isleaf = True #Only one node is required 
                currentNode.isAttr = True #Assign that node to be attr category
                currentNode.value = classes[0] # Output is only one
                return currentNode

            if(self.max_depth!=None): #If some max_depth is given
                if(self.max_depth==depth): 
                    currentNode.isleaf = True 
                    currentNode.isAttr = True
                    currentNode.value = y.value_counts().idxmax() #Output is the max number of output in y
                    return currentNode
           
            for i in X: # i is column name in X dataframe
                
                x_column = X[i] #x_column is column X[i]
                
                # Discreate Input and Discreate Output
                if(x_column.dtype.name=="category"): #discrete input
                    measured_value = None
                    if(self.criterion=="gini_index"):        #Gini Index    
                        classes1 = np.unique(x_column)
                        s1 = 0
                        for j in classes1:
                            y_sub = pd.Series([y[k] for k in range(y.size) if x_column[k]==j])
                            s1 += (y_sub.size)*gini_index(y_sub)
                        measured_value = -1*(s1/x_column.size) #Multiplying with negative sign because lesser the gini, the more likely to split according to it.
                        
                    else:       #Information Gain               
                        measured_value = information_gain(y,x_column)
                    if(final_value==None):
                            attr_no = i
                            final_value = measured_value
                            split_value = None
                    else:
                        if(final_value<measured_value):#Choosing the feature with max info gain
                            attr_no = i
                            final_value = measured_value
                            split_value = None
                
                # Real Input and Discreate Output
                else:
                    x_column_sorted = x_column.sort_values()
                    for j in range(len(x_column_sorted)-1):
                        index = x_column_sorted.index[j]
                        next_index = x_column_sorted.index[j+1]

                        if(y[index]!=y[next_index]):
                            measured_value = None
                            splitvalue = (x_column[index]+x_column[next_index])/2
                            
                            if(self.criterion=="information_gain"):         # Information gain
                                helper_attr = pd.Series(x_column<=splitvalue)
                                measured_value = information_gain(y,helper_attr)
                                
                            else:                                  # Gini index             
                                y_dash1=[]
                                for k in range(y.size):
                                    if x_column[k]<=splitvalue:
                                        y_dash1.append(y[k])
                                y_dash2=[]
                                for k in range(y.size):
                                    if x_column[k]>splitvalue:
                                        y_dash2.append(y[k])
                                y_sub1 = pd.Series(y_dash1)
                                y_sub2 = pd.Series(y_dash2)
                                measured_value = y_sub1.size*gini_index(y_sub1) + y_sub2.size*gini_index(y_sub2)
                                measured_value =  -1*(measured_value/y.size)
                            if(final_value==None):
                                attr_no = i
                                final_value = measured_value
                                split_value = splitvalue
                            else:
                                if(final_value<measured_value):
                                    attr_no = i
                                    final_value = measured_value
                                    split_value = splitvalue
            
        
        #Regression
        else:
            classes=np.unique(y)
            if(X.shape[1]==0):
                currentNode.isleaf = True
                currentNode.value = y.mean()
                return currentNode

            if(self.max_depth!=None):
                if(depth==self.max_depth):
                    currentNode.isleaf = True
                    currentNode.value = y.mean()
                    return currentNode
            
            if(classes.size==1): #No use as output is real but then also let's keep it
                currentNode.isleaf = True
                currentNode.value = classes[0]
                return currentNode

            for i in X:
                x_column = X[i]

                # Discreate Input Real Output
                if(x_column.dtype.name=="category"):
                    classes1 = np.unique(x_column)
                    measured_value = 0
                    for j in classes1:
                        y_dash=[]
                        for k in range(y.size):
                            if x_column[k]==j:
                                y_dash.append(y[k])
                        y_sub = pd.Series(y_dash)
                        measured_value += (y_sub.size)*np.var(y_sub)
                    if(final_value==None):
                        final_value = measured_value
                        attr_no = i
                        split_value = None
                    else:
                        if(final_value>measured_value):
                            final_value = measured_value
                            attr_no = i
                            split_value = None
                
                # Real Input Real Output
                else:
                    x_column_sorted = x_column.sort_values()
                    for j in range(y.size-1):
                        index = x_column_sorted.index[j]
                        next_index = x_column_sorted.index[j+1]
                        splitvalue = (x_column[index]+x_column[next_index])/2
                        y_dash1=[]
                        y_dash2=[]
                        for k in range(y.size):
                            if x_column[k]<=splitvalue:
                                y_dash1.append(y[k])
                        for h in range(y.size):
                            if x_column[h]>splitvalue:
                                y_dash2.append(y[h])
                        y_sub1 = pd.Series(y_dash1)
                        y_sub2 = pd.Series(y_dash2)
                        measured_value = y_sub1.size*np.var(y_sub1) + y_sub2.size*np.var(y_sub2)
                        
                        if(final_value==None):
                            attr_no = i
                            final_value = measured_value
                            split_value = splitvalue
                        else:
                            if(measured_value<final_value):
                                attr_no = i
                                final_value = measured_value
                                split_value = splitvalue
        

        # currentnode==category based
        if(split_value==None):
            currentNode.attr_no = attr_no
            currentNode.isAttr = True
            classes = np.unique(X[attr_no])
            for j in classes:
                ynew=[]
                for k in range(y.size):
                    if X[attr_no][k]==j:
                        ynew.append(y[k])
                y_new = pd.Series(ynew, dtype=y.dtype)
                X_new = X[X[attr_no]==j]
                X_new=X_new.reset_index()
                X_new=X_new.drop(['index',attr_no],axis=1)
                currentNode.children[j] = self.helper_fit(X_new, y_new, depth+1)
        # currentnode==split based
        else:
            currentNode.attr_no = attr_no
            currentNode.splitvalue = split_value
            y_new1=[]
            y_new2=[]
            for k in range(y.size):
                if X[attr_no][k]<=split_value:
                    y_new1.append(y[k])
            for h in range(y.size):
                if X[attr_no][h]>split_value:
                    y_new2.append(y[h])
            y_new1 = pd.Series(y_new1, dtype=y.dtype)
            X_new1 = X[X[attr_no]<=split_value]
            X_new1=X_new1.reset_index()
            X_new1=X_new1.drop(['index'],axis=1)
            y_new2 = pd.Series(y_new2, dtype=y.dtype)
            X_new2 = X[X[attr_no]>split_value]
            X_new2=X_new2.reset_index()
            X_new2=X_new2.drop(['index'],axis=1)
            currentNode.children["lessThan"] = self.helper_fit(X_new1, y_new1, depth+1)
            currentNode.children["greaterThan"] = self.helper_fit(X_new2, y_new2, depth+1)
        
        return currentNode


    def fit(self, X, y):
        
        assert(X.shape[0]==y.size)
        assert(y.size>0)
        self.head = self.helper_fit(X,y,0)


    def predict(self, X):
        y_hat = []                  

        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]    #each row
            node = self.head
            while(not node.isleaf):                            #node is not leaf
                if(node.isAttr):                               #category based
                    node = node.children[xrow[node.attr_no]]
                else:                                       #split based
                    if(xrow[node.attr_no]>node.splitvalue):
                        node = node.children["greaterThan"]
                    else:
                        node = node.children["lessThan"]
            
            y_hat.append(node.value)                           
        
        y_hat = pd.Series(y_hat)
        print(pd.Series(y_hat))
        return y_hat
           

    def plot(self):
        pass
