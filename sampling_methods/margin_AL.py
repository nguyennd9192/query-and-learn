from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod
from sklearn.preprocessing import normalize


class MarginAL(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'margin'

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
    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N] # # get instances with smallest distances
    
    return active_samples, min_margin
    


class MarginAL_invert(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'margin_invert'

  def select_batch_(self, model, already_selected, N, **kwargs):
    """Returns batch of datapoints with smallest margin/highest uncertainty.

    For binary classification, can just take the absolute distance to decision
    boundary for each point.
    For multiclass classification, must consider the margin between distance for
    top two most likely classes.

    Args:
      model: scikit learn model with decision_function implemented
      already_selected: index of datapoints already selected
      N: batch size

    Returns:
      indices of points selected to add using margin active learner
    """

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
    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N] # # get instances with smallest distances
    
    return active_samples, min_margin





