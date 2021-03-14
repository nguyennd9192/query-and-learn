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
import copy


class MaxEmbeddDir(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'MaxEmbeddDir'

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

    try:
      distances = model.decision_function(self.X)
    except:
      distances = model.predict_proba(self.X)

    # print("distances", distances)
    if len(distances.shape) < 2:
      min_margin = abs(distances)
    else:
      sort_distances = np.sort(distances, 1)[:, -2:] # # 1: sorting follows each samples, get two most likely classes 
      min_margin = sort_distances[:, 1] - sort_distances[:, 0]
    rank_ind = np.argsort(min_margin)[::-1] # # [::-1] sorting in descending order


    embedding_model = kwargs["embedding_model"]
    X_org = kwargs["X_org"]

    A = embedding_model.learn_metric.components_.T
    ft_coef = A[0]
    ft_idx_coef_sort = np.argsort(ft_coef)[::-1]

    rev_rank_ind = []
    for ft_idx in ft_idx_coef_sort:
      ft = X_org[:, ft_idx]
      ignore = np.where(ft!=0)[0]
      already_selected_more = np.concatenate((already_selected, ignore))
      
      this_ft_selected = [i for i in rank_ind if i not in already_selected_more]
      rev_rank_ind.extend(this_ft_selected)
      if len(rev_rank_ind) > N:
        break

    active_samples = rev_rank_ind[0:N] # # get instances with smallest distances
    
    
    return active_samples, min_margin
    












