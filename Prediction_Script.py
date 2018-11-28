# Importing the libraries
import numpy as np
import pandas as pd
import csv as csv
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# processing train data
# read the data 
dataset = pd.read_csv('train_luc.csv', header=0)

# function used to generate derived column - 'Hour'
def hour_of_day(dt):
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").time().hour

dataset['hour'] = dataset['datetime'].map(hour_of_day)

# using features from 1 to 9 for training(includes all the sampes of train_luc.cs)
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9]].values # array of independent features
y = dataset.iloc[:, 11].values # array of dependent features

# Using logistic regression
################################################################################
logistic = linear_model.LogisticRegression()

# Create regularization penalty space
penalty = ['l1', 'l2']

# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)

# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)

# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0, n_jobs=4)

# Fit grid search
best_model = clf.fit(X, y)

# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])

# processing test data
# read the data 
dataset_test = pd.read_csv('test_luc.csv', header=0)

# function used to generate derived column - 'Hour'
def hour_of_day(dt):
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").time().hour

dataset_test['hour'] = dataset_test['datetime'].map(hour_of_day)

# using features from 1 to 9 for training(includes all the sampes of train_luc.cs)
X_test = dataset_test.iloc[:, [1,2,3,4,5,6,7,8,9]].values # array of independent features

# prediction on test data using the best model 
y_pred = best_model.predict(X_test)

# calculate the rms for training prediction
rms_score = np.sqrt(mean_squared_error(y_test,y_pred))
###############################################################################

# Using 

