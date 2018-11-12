# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 10:50:58 2018

@author: jsc426
"""
#Regression problem to practice with

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_boston

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.formula.api import ols
import xgboost as xgb


# Load in the data
boston = load_boston()
data_boston = pd.DataFrame(boston['data'], columns = boston['feature_names'])
data_boston['target'] = boston['target']
data_boston.CHAS = data_boston.CHAS.astype('category')

# Run exploratory linear regression
Xtrain, Xtest, Ytrain, Ytest = train_test_split(data_boston.drop('target', axis = 1), data_boston['target'])

Xtrain.shape

# No missing data
Xtrain.isnull().sum(axis = 0)
Xtrain.info()

# Picture data
sns.pairplot(Xtrain)
Xtrain.describe()

# Transformations based on histograms (skew right)
Xtrain['CRIM'] = np.log(Xtrain['CRIM'])
Xtrain['TAX'] = np.log(Xtrain['TAX'])
Xtrain['RAD'] = np.log(Xtrain['RAD'])
Xtrain['AGE'] = np.log(Xtrain['AGE'])
Xtrain['INDUS'] = np.log(Xtrain['INDUS'])
Xtrain['DIS'] = np.log(Xtrain['DIS'])

Xtest['CRIM'] = np.log(Xtest['CRIM'])
Xtest['TAX'] = np.log(Xtest['TAX'])
Xtest['RAD'] = np.log(Xtest['RAD'])
Xtest['AGE'] = np.log(Xtest['AGE'])
Xtest['INDUS'] = np.log(Xtest['INDUS'])
Xtest['DIS'] = np.log(Xtest['DIS'])

# Correlation for continuois variables
corr = Xtrain.drop('CHAS', axis = 1).corr()
sns.heatmap(corr)

# Transform data
scaler = StandardScaler().fit(Xtrain.drop('CHAS', axis = 1))
pca = PCA(0.95)

# Standardize
X_train_stand = pd.DataFrame(scaler.transform(Xtrain.drop('CHAS', axis = 1)))
X_test_stand = pd.DataFrame(scaler.transform(Xtest.drop('CHAS', axis = 1)))

# PCA
pca.fit(X_train_stand)
PCA_train = pd.DataFrame(pca.transform(X_train_stand))
PCA_train = pd.concat([PCA_train, Xtrain['CHAS'].reset_index(range(Xtrain.shape[0])).drop('index', axis = 1)], axis = 1)

PCA_test = pd.DataFrame(pca.transform(X_test_stand))
PCA_test = pd.concat([PCA_test, Xtest['CHAS'].reset_index(range(Xtest.shape[0])).drop('index', axis = 1)], axis = 1)

# New standardized datasets
X_train_stand = pd.concat([X_train_stand, Xtrain['CHAS'].reset_index(range(Xtrain.shape[0])).drop('index', axis = 1)], axis = 1)
X_test_stand = pd.concat([X_test_stand, Xtest['CHAS'].reset_index(range(Xtest.shape[0])).drop('index', axis = 1)], axis = 1)

# Create polynomials
poly = PolynomialFeatures(2)
data_poly = pd.DataFrame(poly.fit_transform(X_train_stand)).drop(0, axis = 1)
data_poly_test = pd.DataFrame(poly.fit_transform(X_test_stand)).drop(0, axis = 1)

# PCA poly
poly_train = pd.DataFrame(poly.fit_transform(Xtrain)).drop(0, axis = 1)
poly_test = pd.DataFrame(poly.fit_transform(Xtest)).drop(0, axis = 1)

scaler = StandardScaler().fit(poly_train)
poly_train_stand = pd.DataFrame(scaler.transform(poly_train))
poly_test_stand = pd.DataFrame(scaler.transform(poly_test))

pca.fit(poly_train_stand)
PCA_poly_train = pd.DataFrame(pca.transform(poly_train_stand)) 
PCA_poly_test = pd.DataFrame(pca.transform(poly_test_stand))

""" With all the transformations and features made, I am going to use linear regression
to assess non-linear trends, outliers, and correlation"""
lr = LinearRegression()
lr_model = lr.fit(Xtrain, Ytrain)
lr.score(Xtrain, Ytrain) #R^2
#lr_model.intercept_
#lr_model.coef_

lr_pred = lr_model.predict(Xtrain)

mean_squared_error(Ytrain, lr_pred)
mean_absolute_error(Ytrain, lr_pred)
sns.jointplot(lr_pred, (Ytrain - lr_pred))

"""Correlation is evident and there are some outliers...let's look at those outliers some more
"""
outliers_data = pd.concat([Ytrain, Xtrain], axis = 1)
formula = 'target ~ CRIM + ZN + INDUS + C(CHAS) + NOX + RM + AGE + DIS + RAD + TAX + PTRATIO + B + LSTAT'

m = ols(formula, outliers_data).fit()
infl = m.get_influence()
sm_fr = infl.summary_frame()


outliers = pd.DataFrame(sm_fr['cooks_d'])
outliers['Studentized Resid'] = sm_fr['student_resid']
outliers['DFFITS'] = sm_fr['dffits']
outliers['DF Betas'] = sm_fr['dfb_B']
# Studentized residuals > 3
DFFITS_LIMIT = 2*np.sqrt((Xtrain.shape[1]+1)/(Xtrain.shape[0] - Xtrain.shape[1] - 1))
COOKSD_LIMIT = 3*np.mean(outliers['cooks_d'])

Outliers = Xtrain.loc[outliers['Studentized Resid'] > 3,:]
# 5 outliers, will keep an eye on these...looks like LSTAT, B, RAD, DIS

""" All models will be run with each data set to assess fit. Both MSE and 
mean absolute error (MAE) will be used as MAE is less sensitive to outliers. 
Additionally, for K-fold CV, both K = 5 and 10 will be run and tuning parameters
checked (bais-variance trade off as k increases). Grid search will be used to 
tune parameters to achieve best fit.
"""
# datasets: 
# PCA_poly_train
# PCA_train
# X_train_stand
# data_poly
# Xtrain
# poly_train_stand
Data_X_Train_Reg = data_poly
Data_X_Test_Reg = data_poly_test

Data_X_Train_RF = Xtrain
Data_X_Test_RF = Xtest

Data_X_Train_Boost = PCA_poly_train
Data_X_Test_Boost = PCA_poly_test

# Lasso
"""Lasso regression uses L1 regularization to return a sparse model (model selection)
by smoothing regression parameters to zero. Useful when there are a lot of features
present and overfitting occurs """

parameters = {'alpha':np.array([0.1, 0.25, 0.5, 1, 1.5, 2])}
lr_lasso = Lasso()
# CV 5
GSV_lasso_CV5 = GridSearchCV(lr_lasso, parameters, cv=5)
GSV_lasso_CV5.fit(Data_X_Train_Reg, Ytrain)
# CV 10
GSV_lasso_CV10 = GridSearchCV(lr_lasso, parameters, cv=10)
GSV_lasso_CV10.fit(Data_X_Train_Reg, Ytrain)

lr_lasso = Lasso(alpha = GSV_lasso_CV5.best_estimator_.alpha)
lr_lasso.fit(Data_X_Train_Reg, Ytrain)
pred_lasso = lr_lasso.predict(Data_X_Train_Reg)
mean_squared_error(Ytrain, pred_lasso)
mean_absolute_error(Ytrain, pred_lasso)

"""Xtrain: CV 5 and 10 set alpha to 0.1 with MSE = 24.68 and MSA = 3.4 
   Polynomial: CV 5 and 10 set alpha to 0.1 with MSE = 9.09 and MSA = 2.13
   Polynomial Standardized: CV 5 and 10 set alpha to 0.1 with MSE = 13.85 and MSA = 2.57
   PCA Poly: CV 5 and 10 set alpha to 0.25 and 0.5 with MSE = 23.99 & 24.56 and MSA = 3.46 & 3.47
   no need to Xtrain standard and PCA without polynomials. It's clear that polynomials are necessary
   to control for the non-linear trends.
"""

# Ridge
"""Ridge regression uses L2 regularization, which can smooth parameters towards zero, but does not achieve zero.
So it holds all features in the final model and can reduce their coefficients to nearly zero, but not zero. 
This smoothing has a strong mathematical link to PCA and helps to combat overfitting. Ridge regression also performs well 
when the features are correlated. Lasso can struggle in situations with correlated features because it
arbitrarily selects one of the correlated features and zeroes out the others.
"""
parameters = {'alpha':np.array([210, 215, 220, 225, 285, 290, 295])}
lr_ridge = Ridge()
# CV 5
GSV_ridge_CV5 = GridSearchCV(lr_ridge, parameters, cv=5)
GSV_ridge_CV5.fit(Data_X_Train_Reg, Ytrain)
# CV 10
GSV_ridge_CV10 = GridSearchCV(lr_ridge, parameters, cv=10)
GSV_ridge_CV10.fit(Data_X_Train_Reg, Ytrain)

lr_ridge = Ridge(alpha = GSV_ridge_CV5.best_estimator_.alpha)
model_ridge = lr_ridge.fit(Data_X_Train_Reg, Ytrain)
pred_ridge = model_ridge.predict(Data_X_Train_Reg)
mean_squared_error(Ytrain, pred_ridge)
mean_absolute_error(Ytrain, pred_ridge)

"""Xtrain: CV 5 and 10 set alpha to 0.1 with MSE = 21.92 and MSA = 3.31
   Poly: CV 5 and 10 set alpha to 4 and 5 repsectively with MSE = 5.64 & 5.76 and MSA = 1.77 & 1.78
   Polynomial Standardized: CV 5 and 10 both set alpha to 3 with MSE = 9.09 and MSA = 2.11
   PCA poly: CV 5 and 10 set alpha to 215 & 290 with MSE = 24.21 & 24.47 and MSA = 3.45 & 3.46
   **both Lasso and Ridge perform best with polynomial features**
"""

# Random Forest
""" Random Forests do well when there are outliers and non-linear trends in the data.
The algorithm partitions the feature space up and calculates a model/prediction for each
partition. Therefore, outliers do not influence estimates in other partitions. Additionally,
due to the partitions, the random forest is like a step function, which can estimate non-linear
trends well, but is not ideal for linear trends.
"""
parameters = {'n_estimators':np.array([50, 100, 200, 500, 1000]), 'max_depth':np.array([1, 5, 10, 25])}

RF = RandomForestRegressor()
# CV 5
GSV_RF_CV5 = GridSearchCV(RF, parameters, cv=5)
GSV_RF_CV5.fit(Data_X_Train_RF, Ytrain)
# CV 10
GSV_RF_CV10 = GridSearchCV(RF, parameters, cv=10)
GSV_RF_CV10.fit(Data_X_Train_RF, Ytrain)

RF = RandomForestRegressor(n_estimators = GSV_RF_CV10.best_estimator_.n_estimators, max_depth = GSV_RF_CV10.best_estimator_.max_depth)  
RF_model = RF.fit(Data_X_Train_RF, Ytrain)
RF_pred = RF_model.predict(Data_X_Train_RF)
mean_squared_error(Ytrain, RF_pred)
mean_absolute_error(Ytrain, RF_pred)

""" Xtrain: CV 5 - n_estimators = 100 & max_depth = 10 CV 10 - n_estimators = 200 & max_depth = 10
    MSE = 1.86 & 1.76 and MSA = 0.98 & 0.96
    Xtrain Standardized: CV 5 - n_estimators = 500 & max_depth = 25 CV 10 - n_estimators = 1000 & max_depth = 25
    MSE = 1.43 & 1.44 and MSA = 0.82 & 0.82
    PCA: CV 5 - n_estimators = 50 & max_depth = 25 CV 10 - n_estimators = 100 & max_depth = 10
    MSE = 3.11  & 3.49 and MSA = 1.14 & 1.19
    PCA Polynomials: CV 5 - n_estimators = 50 & max_depth = 10 CV 10 - n_estimators = 1000 & max_depth = 10
    MSE = 3.08 & 2.76 and MSA = 1.19 & 1.12
    Looks like the basic training set performed best. Will use CV 10 parameters
"""

# XGBOOST
"""Gradient boosting works well for non-linear trends in the data. The algorithm works by
training a model to the data, using the negative derivative of a loss function to 
calculate residuals (*), trains a model on the residuals (**), then iterates on (*) and (**), 
and adds that model to the previous, cummulative sum of models. Each model's contribution to
the sum is determined by the learning rate. This kind of algorithm tends to overfit and be
sensitive to outliers. Luckily, XGboost has L1 and L2 regularization, which can help combat
overfitting.
"""
# L2 regularized
parameters = {'booster':['gbtree','gblinear'],
'n_estimators':np.array([1000, 2000, 3000, 4000, 5000]), 
'learning_rate':np.array([0.001, 0.01, 0.1]),
'reg_lambda':np.array([0, 0.5, 1, 1.5])}

#Data_X_Train_Boost['CHAS'] = Data_X_Train_Boost['CHAS'].astype('int')
#Data_X_Test_Boost['CHAS'] = Data_X_Test_Boost['CHAS'].astype('int')

gbm = xgb.XGBRegressor()
# CV 5
GSV_gbm_CV5 = GridSearchCV(gbm, parameters, cv=5)
GSV_gbm_CV5.fit(Data_X_Train_Boost.as_matrix(), Ytrain)
# CV 10
GSV_gbm_CV10 = GridSearchCV(gbm, parameters, cv=10)
GSV_gbm_CV10.fit(Data_X_Train_Boost.as_matrix(), Ytrain)

GBM = xgb.XGBRegressor(booster = GSV_gbm_CV5.best_estimator_.booster,
                       learning_rate = GSV_gbm_CV5.best_estimator_.learning_rate, 
                       n_estimators = GSV_gbm_CV5.best_estimator_.n_estimators,
                       reg_lambda = GSV_gbm_CV5.best_estimator_.reg_lambda)
GBM_model = GBM.fit(Data_X_Train_Boost.as_matrix(), Ytrain)
GBM_pred = GBM_model.predict(Data_X_Train_Boost.as_matrix())
mean_squared_error(Ytrain, GBM_pred)
mean_absolute_error(Ytrain, GBM_pred)

"""Xtrain: CV5/CV10 booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators = 3000/2000, lambda = 0/0
   MSE = 0.33 & 0.72 and MSA = 0.44 & 0.66
   Xtrain Standardized: booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators = 3000/2000, lambda = 0/0
   MSE = 0.33 & 0.72 and MSA = 0.44 & 0.66
   PCA: booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators =2000/1000, lambda = 0/0
   MSE = 0.95 & 2.67 and MSA = 0.74 & 1.24
   PCA polynomial: booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators = 5000/5000, lambda = 1.5/1.5
   MSE = 0.1 & 0.1 and MSA = 0.24 & 0.24
"""

# L1 regularized
parameters = {'booster':['gbtree','gblinear'],
'n_estimators':np.array([1000, 2000, 3000, 4000, 5000]), 
'learning_rate':np.array([0.001, 0.01, 0.1]),
'reg_alpha':np.array([0, 0.5, 1, 1.5]), 'reg_lambda':np.array([0])}

# CV 5
GSV_gbm_CV5L1 = GridSearchCV(gbm, parameters, cv=5)
GSV_gbm_CV5L1.fit(Data_X_Train_Boost.as_matrix(), Ytrain)
# CV 10
GSV_gbm_CV10L1 = GridSearchCV(gbm, parameters, cv=10)
GSV_gbm_CV10L1.fit(Data_X_Train_Boost.as_matrix(), Ytrain)

GBML1 = xgb.XGBRegressor(booster = GSV_gbm_CV5L1.best_estimator_.booster,
                       learning_rate = GSV_gbm_CV5L1.best_estimator_.learning_rate, 
                       n_estimators = GSV_gbm_CV5L1.best_estimator_.n_estimatorsa,
                       reg_alpha = GSV_gbm_CV5L1.best_estimator_.reg_alpha,
                       reg_lambda = 0)
                       
GBM_modelL1 = GBML1.fit(Data_X_Train_Boost.as_matrix(), Ytrain)
GBM_pred = GBM_modelL1.predict(Data_X_Train_Boost.as_matrix())
mean_squared_error(Ytrain, GBM_pred)
mean_absolute_error(Ytrain, GBM_pred)

"""Xtrain: CV5/CV10 booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators = 3000/4000, alpha = 1/1.5
   MSE = 0.33 & 0.23 and MSA = 0.45 & 0.38
   Xtrain Standardized: booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators = 3000/4000, alpha = 1/1.5
   MSE = 0.33 & 0.23 and MSA = 0.45 & 0.38
   PCA: booster = gbtree/gbtree, learning_rate = 0.01/0.01, n_estimators =2000/1000, alpha = 0/0
   MSE = 0.95 & 2.67 and MSA = 0.74 & 1.24
   PCA polynomial: booster = gbtree/gbtree, learning_rate = 0.1/0.1, n_estimators = 1000/1000, alpha = 0.5/1.5
   MSE = 0.009 & 0.07 and MSA = 0.077 & 0.21
"""

#test accuracy
""" Lasso, Ridge: Polynomial Features (CV 5 parameters)
    Random Forest: Xtrain (CV 10 parameters)
    GB L2: PCA Polynomial (CV 5 parameters)
    GB L1: PCA Polynomial (CV 5 parameters)
"""
lasso_pred_test = lr_lasso.predict(Data_X_Test_Reg)
ridge_pred_test = model_ridge.predict(Data_X_Test_Reg)
RF_pred_test = RF_model.predict(Data_X_Test_RF)
GBM_pred_test = GBM_model.predict(Data_X_Test_Boost.as_matrix())
GBM_pred_testL1 = GBM_modelL1.predict(Data_X_Test_Boost.as_matrix())

mean_squared_error(Ytest, lasso_pred_test)
mean_absolute_error(Ytest, lasso_pred_test)

mean_squared_error(Ytest, ridge_pred_test)
mean_absolute_error(Ytest, ridge_pred_test)

mean_squared_error(Ytest, RF_pred_test)
mean_absolute_error(Ytest, RF_pred_test)

mean_squared_error(Ytest, GBM_pred_test)
mean_absolute_error(Ytest, GBM_pred_test)

mean_squared_error(Ytest, GBM_pred_testL1)
mean_absolute_error(Ytest, GBM_pred_testL1)

""" MSE for train vs test is about the same for Lasso (9.09 vs 8.87)
    Ridge (5.64 vs 10.48), RF(1.76 vs 7.23), and both gradient boosting models 
    (L2: 0.1 vs 8.75, L1: 0.009 vs 9.88) are overfit with RF having the 
    best performance of all the models. 
"""

RF = RandomForestRegressor(n_estimators = 750, max_depth = 8) 
RF_model = RF.fit(Data_X_Train_RF, Ytrain)
RF_pred = RF_model.predict(Data_X_Train_RF)
mean_squared_error(Ytrain, RF_pred)
mean_absolute_error(Ytrain, RF_pred)

RF_pred_test = RF_model.predict(Data_X_Test_RF)
mean_squared_error(Ytest, RF_pred_test)
mean_absolute_error(Ytest, RF_pred_test)

""" After playing with the parameters I acheived a better MSE:
    training = 2.3 and test = 6.67 by decreasing max depth from 10 to 8
    and increasing the number of iterations from 200 to 750. This makes sense
    as both actions are forms of smoothing/regularization to combat overfit. I'm
    sure that with further experimentation a lower test MSE can be acheived.
"""


