
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)


# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

import seaborn as sns
import matplotlib.pyplot as plt

#
def Fakedata(N,M,case):  #N is no. of samples and M is the number of columns(features)
    if(case==1): # Real Input and Real Output
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    elif(case==2): # Discrete Input and Real Output
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randn(N))
    elif(case==3): # Discrete Input and Discrete Output
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
    else: # Real Input and Discrete Output
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(5)})
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
    
    return X, y

def fitpredicttime(N,M,case):
    X, y = Fakedata(N,M,case)
    tree = DecisionTree(criterion="information_gain", max_depth=4)
    startTime1 = time.time()
    tree.fit(X,y)
    endTime1 = time.time()
    startTime2 = time.time()
    y_hat = tree.predict(X)
    endTime2 = time.time()    
    return (endTime1-startTime1,endTime2-startTime2)

def Timeanalysis(case):
    fittingTimes = {'N':[], 'M':[], 'Time':[]}
    predictingTimes = {'N':[], 'M':[], 'Time':[]}
    for N in range(100,110):
        for M in range(2,12):
            Fittime,Predicttime=fitpredicttime(N,M,case)
            fittingTimes['N'].append(N)
            fittingTimes['M'].append(M)
            fittingTimes['Time'].append(Fittime)

            predictingTimes['N'].append(N)
            predictingTimes['M'].append(M)
            predictingTimes['Time'].append(Predicttime)
    df1 = pd.DataFrame(data=fittingTimes)
    heatmap_data1 = pd.pivot_table(df1, values='Time', index=['N'], columns='M')
    plotgraph(heatmap_data1)

    df2 = pd.DataFrame(data=predictingTimes)
    heatmap_data2 = pd.pivot_table(df2, values='Time', index=['N'], columns='M')
    plotgraph(heatmap_data2)
    
def plotgraph(heatmap_data):
    sns.heatmap(heatmap_data, cmap="Blues")
    plt.show()


Timeanalysis(1)
Timeanalysis(2)
Timeanalysis(3)
Timeanalysis(4)