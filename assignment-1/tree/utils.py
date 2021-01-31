import pandas as pd
import numpy as np
def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """
    Size=Y.size
    DfClass={}
    for i in range(Size):

        if(Y.iat[i] not in DfClass):
            DfClass[Y.iat[i]] = 1
        else:
            DfClass[Y.iat[i]] +=1
    
    entropy = 0
    for j in DfClass.keys():
        p_j = DfClass[j]/Size
        entropy -= (p_j*np.log2(p_j))
    
    return entropy
    

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    Size=Y.size
    DfClass={}
    for i in range(Size):

        if(Y.iat[i] not in DfClass):
            DfClass[Y.iat[i]] = 1
        else:
            DfClass[Y.iat[i]] +=1
    gini=1
    for j in DfClass.keys():
        p_j = DfClass[j]/Size
        gini -= np.square(p_j)
    
    return gini
    

    

def information_gain(Y, attr):
    assert(Y.size==attr.size)
    DfClass={}
    for i in range(attr.size):
        if(attr.iat[i]  not in DfClass):
            DfClass[attr.iat[i]] = [Y.iat[i]]
        else:
            DfClass[attr.iat[i]].append(Y.iat[i])
    
    info_gain = entropy(Y)

    for j in DfClass.keys():
        p_j = float(len(DfClass[j]))/attr.size
        info_gain -= (p_j*entropy(pd.Series(DfClass[j])))
    
    return info_gain
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
