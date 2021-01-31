import numpy as np
def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    
    
    size=y_hat.size
    a=0
    ans=0
    while(a<size):
        if y.iat[a]==y_hat.iat[a]:
            ans+=1
        a+=1
    return (ans/size)
    

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    TP=0
    TP_FP=0
    for i in range(y.size):
        if y_hat.iat[i]==cls:
            if y_hat.iat[i]==y.iat[i]:
                TP+=1
            TP_FP+=1
    return TP/TP_FP
    

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    TP=0
    TP_TN=0
    for i in range(y.size):
        if y.iat[i]==cls:
            if y_hat.iat[i]==y.iat[i]:
                TP+=1
            TP_TN+=1
    return TP/TP_TN

    

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    se=0
    mse=0
    rmsef=0
    for i in range(y.size):
        se+=(y_hat.iloc[i]-y.iloc[i])**2
    mse=se/y.size
    rmsef=np.sqrt(mse)
    return rmsef

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    ans=0
    for i in range(y_hat.size):
        ans+=abs(y_hat.iloc[i]-y.iloc[i])
    fans=0
    fans=ans/y.size
    return fans
