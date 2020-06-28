# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Block kernel lsqr solver for multi-class classification."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import scipy.linalg as linalg
from scipy.sparse.linalg import spsolve
from sklearn import metrics
from utils.regression import RegressionFactory
from sklearn.preprocessing import MinMaxScaler
from utils.combination_generator import CombinationGeneratorFactory

class UncertainEnsembleRegression(object):
  def __init__(self,
        random_state=1, 
        n_shuffle=1000,
        alpha=0.1, gamma=0.1,
        cv=3, n_times=3,
        score_method="kr", search_param=False, # # GaussianProcess
        verbose=False):
    self.alpha = alpha
    self.gamma = gamma
    self.score_method = score_method
    self.search_param = search_param
    self.kernel = 'rbf'
    self.coef_ = None
    self.verbose = verbose
    self.gamma = gamma
    self.cv = cv
    self.n_times = n_times
    self.n_shuffle = n_shuffle
    self.random_state = random_state


  def fit(self, X_train, y_train, sample_weight=None):
    # # in fit function
    # # just return estimator with best param with X_train, y_train
    np.random.seed(self.random_state)
    n_features = X_train.shape[1]

    if self.gamma is None:
      self.gamma = 1./n_features

    self.X_train = X_train
    self.y_train = y_train

    estimator = RegressionFactory.get_regression(method=self.score_method, 
        kernel='rbf', alpha=self.alpha, gamma=self.gamma, 
        search_param=self.search_param, X=X_train, y=y_train,  
        cv=self.cv, n_times=self.n_times)
    self.estimator = estimator
    return estimator

  def predict(self, X_val, get_pred_vals=False):
    indices = list(range(len(self.y_train)))
    multiple_sub_indices = CombinationGeneratorFactory.get_generator(
        method="bagging", items=indices, 
        sample_size=0.6, n_shuffle=self.n_shuffle
    )

    X_train_copy = copy.copy(self.X_train)
    y_train_copy = copy.copy(self.y_train)
    estimator_copy = copy.copy(self.estimator)

    y_val_preds = []
    for sub_indices in multiple_sub_indices:
      X_context = X_train_copy[sub_indices, :]
      y_context = y_train_copy[sub_indices]
      
      #######################
      # Fit the context model
      estimator_copy.fit(X_context, y_context)
      y_val_pred = estimator_copy.predict(X_val)
      y_val_preds.append(y_val_pred)
      # r2 = r2_score(y_context, y_context_pred)
      # mae = mean_absolute_error(y_context, y_context_pred)
        #######################
    if get_pred_vals:
      return y_val_preds
    else:
      return np.mean(y_val_preds, axis=0)

  def score(self, X_val, y_val):
    y_pred = self.predict(X_val, get_variance=False)
    val_acc = metrics.accuracy_score(y_val, y_pred)
    return val_acc

  def predict_proba(self, X):
    # # large variance -> probability to be observed small
    # # small variance -> probability to be observed large 
    y_val_preds = self.predict(X, get_variance=True)

    # # normalize variance to 0-1
    var = np.var(y_val_preds, axis=1)
    var_norm = MinMaxScaler.fit_transform(var)
    prob = 1 / var_norm
    return prob


class UncertainGaussianProcess(object):
  def __init__(self,
        random_state=1,
        cv=3, n_times=3, search_param=False,
        verbose=False, ):

    self.search_param = search_param
    self.kernel = 'rbf'
    self.verbose = verbose
    self.cv = cv
    self.n_times = n_times
    self.estimator = None

    self.random_state = random_state


  def fit(self, X_train, y_train, sample_weight=None):
    # # in fit function
    # # just return estimator with best param with X_train, y_train
    np.random.seed(self.random_state)
    n_features = X_train.shape[1]


    self.X_train = X_train
    self.y_train = y_train

    if self.estimator is None:
      estimator = RegressionFactory.get_regression(method="gp", 
          kernel='rbf', alpha=None, gamma=None, 
          search_param=self.search_param, X=X_train, y=y_train,  
          cv=self.cv, n_times=self.n_times)
      self.estimator = estimator
    self.estimator.fit(X_train, y_train)

    return self.estimator

  def predict(self, X_val, get_variance=False):
    y_val_pred, y_val_pred_std = self.estimator.predict(X_val, return_std=True, return_cov=False)
    if get_variance:
      return y_val_pred, y_val_pred_std
    else:
      return y_val_pred

  def score(self, X_val, y_val):
    y_pred, y_val_pred_std = self.predict(X_val, get_variance=False)
    val_acc = metrics.accuracy_score(y_val, y_pred)
    return val_acc

  def predict_proba(self, X):
    # # large variance -> probability to be observed small
    # # small variance -> probability to be observed large 
    y_val_preds, y_val_pred_std = self.predict(X, get_variance=True)

    # # normalize variance to 0-1
    var_norm = MinMaxScaler().fit_transform(X=y_val_pred_std.reshape(-1, 1))
    prob = 1 / var_norm
    return prob







