# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 17:32:17 2018

@author: Rohit
"""

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
# using features from 1 to 9 and 11th feature for training(includes all the sampes of train_luc.cs)
X = dataset.iloc[:, [1,2,3,4,5,6,7,8,12]].values # array of independent features
y = dataset.iloc[:, 11].values # array of dependent features

# =============================================================================
# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# =============================================================================

# processing test data 
dataset_test = pd.read_csv('test_luc.csv', header=0)
# function used to generate derived column - 'Hour'
def hour_of_day(dt):
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").time().hour
dataset_test['hour'] = dataset_test['datetime'].map(hour_of_day)
# using features from 1 to 9 for training(includes all the sampes of train_luc.cs)
X_test = dataset_test.iloc[:, [1,2,3,4,5,6,7,8,9]].values # array of independent features

# Feature Scaling
# =============================================================================
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)
# X_test = sc.transform(X_test)
# =============================================================================

################################################################################
# logistic regression
################################################################################
logistic = linear_model.LogisticRegression()
# Create regularization penalty space
penalty = ['l1', 'l2']
# Create regularization hyperparameter space
C = np.logspace(0, 4, 10)
# Create hyperparameter options
hyperparameters = dict(C=C, penalty=penalty)
# Create grid search using 5-fold cross validation
clf = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0, n_jobs=-1)
# Fit grid search
best_model = clf.fit(X, y)
# View best hyperparameters
print('Best Penalty:', best_model.best_estimator_.get_params()['penalty'])
print('Best C:', best_model.best_estimator_.get_params()['C'])
# prediction on test data using the best model 
# =============================================================================
# y_pred = best_model.predict(X_test)
# =============================================================================

################################################################################
# random forest regression
################################################################################
# =============================================================================
# from sklearn.ensemble import RandomForestRegressor
# regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
# regressor.fit(X, y)
# # prediction on test data using the best model 
# y_pred = regressor.predict(X_test)
# 
# # Grid Search for Random Forest
# from sklearn.model_selection import GridSearchCV
# # Create the parameter grid based on the results of random search 
# param_grid = {
#     'bootstrap': [True],
#     'max_depth': [80, 90, 100, 110],
#     'max_features': [2, 3],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]
# }
# # Create a based model
# rf = RandomForestRegressor()
# # Instantiate the grid search model
# grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
#                           cv = 3, n_jobs = -1, verbose = 2)
# =============================================================================
################################################################################
# Simple Linear Regression
################################################################################
# Fitting Simple Linear Regression to the Training set
# =============================================================================
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X, y)
# 
# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
# 
# 
# # results evaluation
# # estimate the rms for training prediction by comparing with ground truth information
# # loading the ground truth predictions
# dataset_ground_truth = pd.read_csv('sample_prediction.csv', header=0)
# y_ground_truth = dataset_ground_truth.iloc[:, 1].values
# rms_score = np.sqrt(mean_squared_error(y_ground_truth,y_pred))
# 
# # generate output csv for kaggle prediction
# dataset_test['count'] = y_pred
# # save the predicted count as a csv with a header column and datetime row
# dataset_test_final = dataset_test[['datetime', 'count']].to_csv('my_prediction.csv',
#                                                 index=False, header=True)
# =============================================================================














# =============================================================================
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(C=464.15888336127773, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=-1, penalty='l1', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)
# 
# classifier.fit(X, y)
# y_pred = classifier.predict()
# rms_score = np.sqrt(mean_squared_error(y_test,y_pred))
# =============================================================================

