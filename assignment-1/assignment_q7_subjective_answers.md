# Q7 a)
Here are the results of the q7_RandomForest.py:

Random Forest Classifier:

Criteria : entropy
Accuracy:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0

Criteria : gini
Accuracy:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0
Precision:  1.0
Recall:  1.0

Random Forest Regressor:
Criteria : mse
RMSE:  0.0
MAE:  0.0

Here we can see that the results are 100% correct because in the question there was no max_depth given. Also, the number of estimators were also high. So, we got this type of results.

# Q7 b)
Below are the results of implementing our model on iris data set:
In this case, as we have put num_of_estimators as 3 and max_depth=4, we can see that the results are not coming 100% accurate.
Train Accuracy:  0.9777777777777777
Test Accuracy:  0.9
Class = 1
Precision:  0.8571428571428571
Recall:  0.8571428571428571
Class = 0
Precision:  1.0
Recall:  1.0
Class = 2
Precision:  0.875
Recall:  0.875
