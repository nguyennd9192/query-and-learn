import pandas as pd
import numpy as np
import warnings

from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from least_square_fit import LeastSquareFit
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, scale


class RegressionFactory(object):
    
    @staticmethod
    def get_regression(method, kernel='rbf', alpha=1, gamma=1, theta0=0.1, nugget=0.1,
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
                nugget, theta0, best_score, best_score_std = RegressionFactory.gaussian_process_cv_with_noise(
                    X=X, y_obs=y, cv=cv, n_random=n_times)
            return GaussianProcess(
                theta0=theta0, nugget=nugget, random_start=10)
        elif method == "lr":
            return LinearRegression()
        elif method == "lsf":
            return LeastSquareFit()

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
                    y_test_predict_ y_test_pred_prob = model.predict(X_test)
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
    def kernel_ridge_parameter_search_boost(X, y_obs, kernel='rbf', n_folds=3,
                                        n_times=3, n_dsp=160, n_spt=5):
        """
        """
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
        if (n_dsp > n_instance) or (n_dsp <= 0):
            n_dsp = n_instance
            n_spt = 1

        if (n_folds <= 0) or (n_folds > n_instance):
            n_folds = n_instance
            n_times = 1

        for i in range(n_rounds):
            # Searching for Gamma
            gammas = np.logspace(gamma_log_lb, gamma_log_ub, num=n_steps)
            best_gammas = []
            for _ in range(n_spt):
                scores_mean = []
                scores_std = []

                indexes = np.random.permutation(range(n_instance))
                X_sample = X[indexes[:n_dsp]]
                y_obs_sample = y_obs[indexes[:n_dsp]]

                for gamma in gammas:
                    k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
                    y_sample_predict = RegressionFactory.CV_predict(k_ridge, X_sample, y_obs_sample,
                                                n_folds=n_folds, n_times=n_times)
                    cv_scores = map(lambda y_sample_predict: r2_score(
                        y_obs_sample, y_sample_predict), y_sample_predict)

                    scores_mean += [np.mean(cv_scores)]
                    scores_std += [np.std(cv_scores)]

                best_index = np.argmax(scores_mean)
                gamma = gammas[best_index]

                best_gammas += [gamma]

            best_gammas = np.array(best_gammas)
            gamma = np.mean(best_gammas)

            gamma_log_lb = np.log10(gamma * lb)
            gamma_log_ub = np.log10(gamma * ub)

            # Searching for Alpha
            alphas = np.logspace(alpha_log_lb, alpha_log_ub, num=n_steps)
            best_alphas = []
            for _ in range(n_spt):
                scores_mean = []
                scores_std = []

                indexes = np.random.permutation(range(n_instance))
                X_sample = X[indexes[:n_dsp]]
                y_obs_sample = y_obs[indexes[:n_dsp]]

                for alpha in alphas:
                    k_ridge = KernelRidge(alpha=alpha, gamma=gamma, kernel=kernel)
                    y_sample_predict = RegressionFactory.CV_predict(k_ridge, X_sample, y_obs_sample,
                                                n_folds=n_folds, n_times=n_times)
                    cv_scores = map(lambda y_sample_predict: r2_score(
                        y_obs_sample, y_sample_predict), y_sample_predict)

                    scores_mean += [np.mean(cv_scores)]
                    scores_std += [np.std(cv_scores)]

                best_index = np.argmax(scores_mean)
                alpha = alphas[best_index]

                best_alphas += [alpha]

            best_alphas = np.array(best_alphas)
            alpha = np.mean(best_alphas)

            alpha_log_lb = np.log10(alpha * lb)
            alpha_log_ub = np.log10(alpha * ub)

        return alpha, gamma, scores_mean[best_index], scores_std[best_index]

    @staticmethod
    def gaussian_process_cv_with_noise(X, y_obs, cv=10, n_random=10):
        """Validate Gaussian Process model using Cross-Validation

        :data: original data table
        :predicting_variables: List of explanatory of variables
        :target_variable: target variable
        :cv: number of folds for cross-validation
        :param n_random: the number of randomization
        """

        best_nugget = 0.1
        best_theta0 = 0.1
        nugget_lmax = -1.5
        nugget_lmin = -3.0
        theta0_lmax = -0.5
        theta0_lmin = -3.0
        num_grid = 10

        scores = []
        scores_std = []
        for _ in range(2):
            ##
            # Search for appropriate nugget value
            # nugget: in logspace(nugget_lmin, nugget_lmax, num_grid)
            ##
            nuggets = np.logspace(nugget_lmin, nugget_lmax, num_grid)
            scores = []
            scores_std = []

            for nugget in nuggets:
                gp = GaussianProcess(theta0=best_theta0, nugget=nugget, random_start=10)
                # this_scores, this_scores_std = n_times_cv(gp, X, y_obs, cv=cv, n_random=n_random)
                
                # # use built in function
                y_predict = RegressionFactory.CV_predict(gp, X, y_obs,
                            n_folds=n_folds, n_times=n_times, is_gp=True)
                cv_scores = map(lambda y_predict: r2_score(
                    y_obs, y_predict), y_predict)
                scores += [np.mean(cv_scores)]
                scores_std += [np.std(cv_scores)]

            idx_max = np.argmax(np.array(scores))
            best_nugget = nuggets[idx_max]
            gp.nugget = best_nugget

            ##
            # Search for appropriate theta0 value
            # theta0: in logspace(theta0_lmin, theta0_lmax, num_grid)
            ##
            theta0s = np.logspace(theta0_lmin, theta0_lmax, num_grid)
            scores = []
            scores_std = []
            for theta0 in theta0s:
                gp = GaussianProcess(theta0=theta0, nugget=best_nugget, random_start=10)
                # this_scores, this_scores_std = n_times_cv(gp, X, y_obs, cv=cv)
                # scores += [this_scores]
                # scores_std += [this_scores_std]
                y_predict = RegressionFactory.CV_predict(gp, X, y_obs,
                            n_folds=n_folds, n_times=n_times, is_gp=True)
                cv_scores = map(lambda y_predict: r2_score(
                    y_obs, y_predict), y_predict)
                scores += [np.mean(cv_scores)]
                scores_std += [np.std(cv_scores)]

            idx_max = np.argmax(np.array(scores))
            best_theta0 = theta0s[idx_max]
            gp.theta0 = best_theta0
            num_grid *= 2

        idx_max = np.argmax(np.array(scores))
        best_score = scores[idx_max]
        best_score_std = scores_std[idx_max]

        return best_nugget, best_theta0, best_score, best_score_std


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

    