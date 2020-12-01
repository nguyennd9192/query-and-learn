from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
from sklearn import metrics
from utils.regression import RegressionFactory, CV_predict_score
from sklearn.preprocessing import MinMaxScaler
from scipy import stats

from sklearn import neighbors
import metric_learn as mkl

from sklearn.metrics import pairwise_distances



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