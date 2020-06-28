import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
# from least_square_fit import LeastSquareFit
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, scale
from sklearn.gaussian_process import GaussianProcessRegressor

class RegressionFactory(object):
    
  @staticmethod
  def get_regression(method, kernel='rbf', alpha=1, gamma=1, 
      search_param=False, X=None, y=None, cv=3, n_times=3):
    method = method.strip().lower()
    if method == "kr":
        if search_param:
            alpha, gamma, scores_mean, scores_std = RegressionFactory.kernel_ridge_parameter_search(
                X=X, y_obs=y, kernel=kernel, n_folds=cv, n_times=n_times)
        return KernelRidge(
            kernel = kernel,
            alpha = alpha, 
            gamma = gamma
        )
    elif method == "gp":
        if search_param:
            best_gpr = RegressionFactory.gaussian_process_cv_with_noise(
                X=X, y_obs=y, cv=cv, n_random=n_times)
        return best_gpr
    elif method == "lr":
        return LinearRegression()


  @staticmethod
  def CV_predict(model, X, y, n_folds=3, n_times=3, is_gp=False):
  
      if (n_folds <= 0) or (n_folds > len(y)):
          n_folds = len(y)
          n_times = 1

      y_predicts = []
      for i in range(n_times):
          indexes = np.random.permutation(range(len(y)))

          kf = KFold(n_splits=n_folds)

          y_cv_predict = []
          cv_test_indexes = []
          cv_train_indexes = []
          for train, test in kf.split(indexes):
              # cv_train_indexes += list(indexes[train])
              # print(train, test)
              cv_test_indexes += list(indexes[test])

              X_train, X_test = X[indexes[train]], X[indexes[test]]
              y_train, Y_test = y[indexes[train]], y[indexes[test]]

              model.fit(X_train, y_train)

              # y_train_predict = model.predict(X_train)
              if is_gp:
                  y_test_predict, y_test_pred_prob = model.predict(X_test)
              else:
                  y_test_predict = model.predict(X_test)
              y_cv_predict += list(y_test_predict)

          cv_test_indexes = np.array(cv_test_indexes)
          # print(cv_test_indexes)
          rev_indexes = np.argsort(cv_test_indexes)

          y_cv_predict = np.array(y_cv_predict)

          y_predicts += [y_cv_predict[rev_indexes]]

      y_predicts = np.array(y_predicts)

      return y_predicts

  @staticmethod
  def kernel_ridge_parameter_search(X, y_obs, kernel='rbf',
                                n_folds=3, n_times=3):
      # parameter initialize
    gamma_log_lb = -2.0
    gamma_log_ub = 2.0
    alpha_log_lb = -4.0
    alpha_log_ub = 1.0
    n_steps = 10
    n_rounds = 4
    alpha = 1
    gamma = 1
    lb = 0.8
    ub = 1.2
    n_instance = len(y_obs)

    if (n_folds <= 0) or (n_folds > n_instance):
        n_folds = n_instance
        n_times = 1

    # Start
    for i in range(n_rounds):
        scores_mean = []
        scores_std = []
        gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
        for gamma in gammas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = RegressionFactory.CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            cv_scores = list(map(lambda y_predict: r2_score(y_obs, y_predict), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        gamma = gammas[best_index]
        gamma_log_lb = np.log10(gamma * lb)
        gamma_log_ub = np.log10(gamma * ub)
        scores_mean = []
        scores_std = []
        alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
        for alpha in alphas:
            k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
            y_predicts = RegressionFactory.CV_predict(
                k_ridge, X, y_obs, n_folds=n_folds, n_times=n_times)
            cv_scores = list(map(lambda y_predict: r2_score(y_obs, y_predict), y_predicts))

            scores_mean += [np.mean(cv_scores)]
            scores_std += [np.std(cv_scores)]

        best_index = np.argmax(scores_mean)
        alpha = alphas[best_index]
        alpha_log_lb = np.log10(alpha * lb)
        alpha_log_ub = np.log10(alpha * ub)

    return alpha, gamma, scores_mean[best_index], scores_std[best_index]

  @staticmethod
  def gaussian_process_cv_with_noise(X, y_obs, cv=10, n_random=10):
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.model_selection import GridSearchCV
    n_steps = 5
    rbf_length_lb = -2
    rbf_length_ub = 1
    rbf_lengths = np.logspace(rbf_length_lb, rbf_length_ub, n_steps)

    const_lb = -1
    const_ub = 2
    consts = np.logspace(const_lb, const_ub, n_steps)

    noise_lb = -2
    noise_ub = 1
    noises = np.logspace(rbf_length_lb, rbf_length_ub, n_steps)

    alpha_lb = -2
    alpha_ub = 1
    alphas = np.logspace(rbf_length_lb, rbf_length_ub, n_steps)
    # param_grid = {'alpha':  
    # 'kernel__k1__k1__constant_value': np.logspace(-2, 2, 3), 
    # 'kernel__k1__k2__length_scale': np.logspace(-2, 2, 3), 
    # 'kernel__k2__noise_level':  np.logspace(-2, 1, 3), 
    # }

    # best_gpr = GridSearchCV(gp,cv=3,param_grid=param_grid,n_jobs=2)
    param_grid = {"alpha": alphas,
          "kernel": [ConstantKernel(constant_value=c)*RBF(length_scale=l) + WhiteKernel(noise_level=n)  # noise terms
                for c in consts for l in rbf_lengths for n in noises]}
    GridSearch = GridSearchCV(GaussianProcessRegressor(),param_grid=param_grid,
                cv=cv,n_jobs=4)
    GridSearch.fit(X, y_obs)
    best_gpr = GridSearch.best_estimator_
    best_gpr.fit(X, y_obs)
    print("best_gpr params:", best_gpr.get_params())
    return best_gpr


# class Regression(object):
#     def __init__(self):
#         pass
    
#     def fit(self, X, y):
#         self.__estimator.fit(X, y)

#     def predict(self, y):
#         self.__estimator.predict(y)

# class KernelRidgeRegression(Regression):

#     def __init__(self, kernel='rbf', alpha=1, gamma=1):
#         self.__kernel = kernel
#         self.__alpha = alpha
#         self.__gamma = gamma
#         self.__estimator = KernelRidge(
#             kernel = regression_configuration.kernel,
#             alpha = data_configuration.best_alpha, 
#             gamma = data_configuration.best_gamma
#         )

    