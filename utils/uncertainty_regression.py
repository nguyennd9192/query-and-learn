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
from utils.regression import RegressionFactory, CV_predict_score
from sklearn.preprocessing import MinMaxScaler
from utils.combination_generator import CombinationGeneratorFactory
from scipy import stats
from scipy.signal import find_peaks, peak_prominences

from sklearn import neighbors
import metric_learn as mkl

from sklearn.metrics import pairwise_distances

# from statsmodels.distributions.empirical_distribution import ECDF
 
class UncertainEnsembleRegression(object):
  def __init__(self,
        random_state=1, 
        n_shuffle=10000,
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

    # # currently, anytime we fit the estimator with X_train, y_train
    # # we researching for parameter
    estimator, GridSearchCV = RegressionFactory.get_regression(method=self.score_method, 
        kernel='rbf', alpha=self.alpha, gamma=self.gamma, 
        search_param=self.search_param, X=X_train, y=y_train,  
        cv=self.cv, n_times=self.n_times)
    self.GridSearchCV = GridSearchCV
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
      # print("len_test_set:", len(y_val_pred))
      y_val_preds.append(y_val_pred)
      # r2 = r2_score(y_context, y_context_pred)
      # mae = mean_absolute_error(y_context, y_context_pred)
        #######################
    if get_pred_vals:
      return y_val_preds
    else:
      return np.mean(y_val_preds, axis=0)

  def score(self, X_val, y_val):
    y_pred = self.predict(X_val, get_pred_vals=False)
    val_acc = metrics.r2_score(y_val, y_pred)
    return val_acc

  def predict_proba(self, X, is_norm=True):
    # # large variance -> probability to be observed small -> sorting descending take first
    # # small variance -> probability to be observed large 
    y_val_preds = self.predict(X, get_pred_vals=True)
    # print("y_val_preds.shape", np.array(y_val_preds).shape)
    # # normalize variance to 0-1

    # # canonical method
    var = np.var(np.array(y_val_preds), axis=0).reshape(-1, 1)

    # # fitting with mixture gaussian, find cummulative
    # ecdf = ECDF(sample)
    # var = []
    # y_val_preds_T = np.array(y_val_preds).T
    # for y_val_pred in y_val_preds_T:
    #   # print (y_val_pred, len(y_val_pred))

    #   kernel = stats.gaussian_kde(y_val_pred)
    #   yref = np.linspace(min(y_val_pred),max(y_val_pred),100)
    #   pdf = kernel(yref).T

    #   # # find peaks
    #   peaks, _ = find_peaks(pdf)
    #   pr = peak_prominences(pdf, peaks)[0]
    #   argmax1, argmax2 = np.argsort(pr)[-2:] # # two largest prominence
    #   y_pred_peak1, y_pred_peak2 = yref[argmax1], yref[argmax2]

    #   # # variance between two largest peaks
    #   v = y_pred_peak2 - y_pred_peak1
    #   var.append([v])
    # var = np.array(var)

    var_norm = MinMaxScaler().fit_transform(X=var).ravel()

    # var_norm = var.reshape(-1, 1)
    # prob = 1 / (var_norm)
    if is_norm:
      return var_norm
    else:
      return var

  def best_score_(self):
    # # some conflict meaning between best_score_ for GridSearchCV object and this attribute:
    # # GridSearchCV.best_score_ returns cv score of best parameter
    # # this UncertainGaussianProcess.best_score_returns cv score of given params
    if self.GridSearchCV is None:
      r2, r2_std, mae, mae_std = CV_predict_score(self.estimator, self.X_train, self.y_train, 
                n_folds=3, n_times=3, score_type='r2')
      result = r2
    else:
      result = self.GridSearchCV.best_score_
    return result


class UncertainGaussianProcess(object):
  def __init__(self, 
        random_state=1, 
        cv=3, n_times=3, search_param=False,
        verbose=False, mt_kernel=None):

    self.search_param = search_param
    self.kernel = 'rbf'
    self.verbose = verbose
    self.cv = cv
    self.n_times = n_times
    self.estimator = None
    self.random_state = random_state
    self.mt_kernel = mt_kernel


  def fit(self, X_train, y_train, sample_weight=None):
    # # in fit function
    # # just return estimator with best param with X_train, y_train
    np.random.seed(self.random_state)
    n_features = X_train.shape[1]


    self.X_train = X_train
    self.y_train = y_train

    if self.estimator is None: # # for not always search parameters:
    # if self.estimator is None or self.search_param: # # either self.estimator is None or search_param is True-> require search
      estimator, GridSearchCV = RegressionFactory.get_regression(method="gp", 
          kernel='cosine', alpha=None, gamma=None, # # rbf
          search_param=self.search_param, X=X_train, y=y_train,  
          cv=self.cv, n_times=self.n_times, mt_kernel=self.mt_kernel) # mt_kernel=self.mt_kernel
      self.estimator = estimator
      self.GridSearchCV = GridSearchCV
    self.estimator.fit(X_train, y_train)
    return self.estimator

  def predict(self, X_val, get_variance=False):
    y_val_pred, y_val_pred_std = self.estimator.predict(X_val, return_std=True, return_cov=False)
    if get_variance:
      return y_val_pred, y_val_pred_std
    else:
      return y_val_pred

  def score(self, X_val, y_val):
    y_pred = self.predict(X_val, get_variance=False)    
    val_acc = metrics.r2_score(y_val, y_pred)
    return val_acc

  def predict_proba(self, X, is_norm=True):
    # # large variance -> probability to be observed small -> sorting descending take first
    # # small variance -> probability to be observed large 
    y_val_preds, y_val_pred_std = self.predict(X, get_variance=True)

    # # normalize variance to 0-1
    var_norm = MinMaxScaler().fit_transform(X=y_val_pred_std.reshape(-1, 1))
    # var_norm = y_val_pred_std.reshape(-1, 1)
    # prob = 1 / var_norm
    if is_norm:
      return var_norm.ravel()
    else:
      return y_val_pred_std.reshape(-1, 1)


  def best_score_(self):
    # # some conflict meaning between best_score_ for GridSearchCV object and this attribute:
    # # GridSearchCV.best_score_ returns cv score of best parameter
    # # this UncertainGaussianProcess.best_score_returns cv score of given params
    if self.GridSearchCV is None:
      r2, r2_std, mae, mae_std = CV_predict_score(self.estimator, self.X_train, self.y_train, 
                n_folds=3, n_times=3, score_type='r2')
      result = r2
    else:
      result = self.GridSearchCV.best_score_
    return result




class UncertainMetricLearningRegression(object):
  def __init__(self, 
        random_state=1, 
        cv=3, n_times=3, search_param=False,
        verbose=False):

    self.search_param = search_param
    self.kernel = 'rbf'
    self.verbose = verbose
    self.cv = cv
    self.n_times = n_times
    self.estimator = None
    self.random_state = random_state

    # learn_metric = mkl.ITML_Supervised()
    # learn_metric = mkl.SDML_Supervised(sparsity_param=0.1, balance_param=0.0015,
    #           prior='covariance')
    # learn_metric = mkl.LMNN(k=3, learn_rate=0.1) # 
    learn_metric = mkl.MLKR(n_components=2, init="auto")

    # learn_metric = mkl.LFDA(n_components=2, 
    #   k=10, embedding_type="orthonormalized") # weighted, orthonormalized
    self.learn_metric = learn_metric



  def fit(self, X_train, y_train, sample_weight=None):
    # # in fit function
    # # just return estimator with best param with X_train, y_train
    np.random.seed(self.random_state)
    n_features = X_train.shape[1]


    self.X_train = X_train
    self.y_train = y_train

    self.learn_metric.fit(X_train, y_train)
    X_train_embedded = self.learn_metric.transform(X_train)
    self.X_train_embedded = X_train_embedded

    if self.estimator is None: # # for not always search parameters:
      estimator, GridSearchCV = RegressionFactory.get_regression(
          method="gp", kernel='rbf', alpha=None, gamma=None, # # rbf, cosine
          search_param=self.search_param, X=X_train_embedded, y=y_train,  
          cv=self.cv, n_times=self.n_times) # mt_kernel=self.mt_kernel
      # self.estimator = neighbors.KNeighborsRegressor(10, weights="distance") # 'uniform', 'distance'
      self.estimator = estimator
    self.estimator.fit(X_train_embedded, y_train)
    return self.estimator

  def transform(self, X_val):
    X_val_transform = self.learn_metric.transform(X_val)
    
    return X_val_transform

  def predict(self, X_val, get_variance=False):

    X_val_transform = self.learn_metric.transform(X_val)

    y_val_pred = self.estimator.predict(X_val_transform)

    if get_variance:
      # nbs = [2, 5, 10, 20, 30]
      # y_preds = []
      # for nb in nbs:
      #   estimator = neighbors.KNeighborsRegressor(nb, weights="distance") # 'uniform', 'distance'
      #   estimator.fit(self.X_train_embedded, self.y_train)
        
      #   y_pred = estimator.predict(X_val_transform)
      #   y_preds.append(y_pred)
      # y_preds = np.array(y_preds)
      # return np.mean(y_preds, axis=0), # np.var(y_preds, axis=0)
      distances = pairwise_distances(X_val, self.X_train)
      print ("distances.shape:", distances.shape)
      max_distances = np.max(distances, axis=1)
      return y_val_pred, max_distances

    else:
      return y_val_pred

  def score(self, X_val, y_val):
    y_pred = self.predict(X_val, get_variance=False)    
    val_acc = metrics.r2_score(y_val, y_pred)
    return val_acc

  def predict_proba(self, X, is_norm=True):
    # # large variance -> probability to be observed small -> sorting descending take first
    # # small variance -> probability to be observed large 
    y_val_preds, y_val_pred_std = self.predict(X, get_variance=True)

    # # normalize variance to 0-1
    var_norm = MinMaxScaler().fit_transform(X=y_val_pred_std.reshape(-1, 1))
    # var_norm = y_val_pred_std.reshape(-1, 1)
    # prob = 1 / var_norm
    if is_norm:
      return var_norm.ravel()
    else:
      return y_val_pred_std.reshape(-1, 1)


  def best_score_(self):
    # # some conflict meaning between best_score_ for GridSearchCV object and this attribute:
    # # GridSearchCV.best_score_ returns cv score of best parameter
    # # this UncertainGaussianProcess.best_score_returns cv score of given params
    if self.GridSearchCV is None:
      r2, r2_std, mae, mae_std = CV_predict_score(self.estimator, self.X_train, self.y_train, 
                n_folds=3, n_times=3, score_type='r2')
      result = r2
    else:
      result = self.GridSearchCV.best_score_
    return result




