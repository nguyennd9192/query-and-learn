from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise 
from sklearn.mixture import GaussianMixture
import pandas as pd
import random

def get_best_gmm(X_matrix, n_components, n_sampling=20, means_init=None):
  n_points = len(X_matrix)
  
  score_df = pd.DataFrame(index=range(n_sampling), columns=["AIC", "BIC"])

  for i in range(n_sampling):
      gmm = GaussianMixture(n_components=n_components,
          # reg_covar=0.0000001
          # covariance_type='full',
                          means_init=means_init,
      # #weights_init = [0.1, 0.33, 0.26, 0.1]
      # init_params='random'
                            )
      sample = random.sample(range(n_points), int(0.8*n_points))
      X_rand = X_matrix[sample]
      # X_rand = X_matrix

      # np.random.shuffle(X_matrix)
      gmm.fit(X=X_rand)
      this_AIC = gmm.aic(X=X_rand)
      this_BIC = gmm.bic(X=X_rand)
      score_df.loc[i, "AIC"] = this_AIC
      score_df.loc[i, "BIC"] = this_BIC

      if i == 0:
          best_AIC = this_AIC
          best_gmm = gmm
      else:
          if this_AIC < best_AIC:
              best_AIC = this_AIC
              best_gmm = gmm

  return best_gmm, score_df



class MarginExplSpace(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'MarginExplSpace'

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest margin/highest uncertainty.

    For binary classification, can just take the absolute distance to decision
    boundary for each point.
    For regression, must impl

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """

    # distance_matrix = pairwise.euclidean_distances(self.X)
    # distances = np.sum(distance_matrix, axis=0)
    clustering_model, score_df = get_best_gmm(X_matrix=self.X,  
            n_components=10, n_sampling=20, means_init=None)

    try:
      distances = clustering_model.decision_function(self.X)
    except:
      distances = clustering_model.predict_proba(self.X)

    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      sort_distances = np.sort(distances, 1)[:, -2:] # # 1: sorting follows each samples, get two most likely classes 
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    rank_ind = np.argsort(min_margin)[::-1] # # [::-1] sorting in descending order
    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N] # # get instances with smallest distances
    
    return active_samples, min_margin
    












