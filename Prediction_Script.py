# Importing the libraries
import numpy as np
import pandas as pd
# =============================================================================
# import csv as csv
# import matplotlib.pyplot as plt
# =============================================================================
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

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# processing test data
# read the data 
dataset_test = pd.read_csv('test_luc.csv', header=0)

# function used to generate derived column - 'Hour'
def hour_of_day(dt):
    return datetime.strptime(dt, "%Y-%m-%d %H:%M:%S").time().hour

dataset_test['hour'] = dataset_test['datetime'].map(hour_of_day)

# using features from 1 to 9 for training(includes all the sampes of train_luc.cs)
X_test = dataset_test.iloc[:, [1,2,3,4,5,6,7,8,9]].values # array of independent features

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

# prediction on test data using the best model 
y_pred = best_model.predict(X_test)

# calculate the rms for training prediction by comparing with ground truth information
# loading the ground truth predictions

dataset_ground_truth = pd.read_csv('sample_prediction.csv', header=0)
y_ground_truth = dataset_ground_truth.iloc[:, 1].values

rms_score = np.sqrt(mean_squared_error(y_ground_truth,y_pred))
###############################################################################
# Using Simple Linear Regression

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, y)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# calculate the rms for training prediction by comparing with ground truth information
# loading the ground truth predictions

dataset_ground_truth = pd.read_csv('sample_prediction.csv', header=0)
y_ground_truth = dataset_ground_truth.iloc[:, 1].values

rms_score = np.sqrt(mean_squared_error(y_ground_truth,y_pred))
###############################################################################
# =============================================================================
# # Fitting Polynomial Regression to the dataset
# 
# from sklearn.preprocessing import PolynomialFeatures
# poly_reg = PolynomialFeatures(degree = 4)
# X_poly = poly_reg.fit_transform(X_train)
# # poly_reg.fit(X_poly, y)
# lin_reg_2 = LinearRegression()
# lin_reg_2.fit(X_poly, y_train)
# 
# # Predicting the Test set results
# y_pred = lin_reg_2.predict(X_test)
# =============================================================================
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
y_train = y_train.reshape(-1, 1)
y_train = sc_y.fit_transform(y_train)

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, y_train)

# Predicting result
y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)

rms_score = np.sqrt(mean_squared_error(y_test,y_pred))
###############################################################################
# Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
rms_score = np.sqrt(mean_squared_error(y_test,y_pred))
###############################################################################
# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)
rms_score = np.sqrt(mean_squared_error(y_test,y_pred))
###############################################################################
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(C=21.544346900318832, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1, penalty='l1', random_state=0, solver='liblinear', tol=0.0001, verbose=0, warm_start=False)

classifier.fit(X, y)
y_pred = classifier.predict()
rms_score = np.sqrt(mean_squared_error(y_test,y_pred))

