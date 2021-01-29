from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise 

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

    distance_matrix = pairwise.euclidean_distances(self.X)
    distances = np.sum(distance_matrix, axis=0)
    print ("distances:", distances)

    rank_ind = np.argsort(distances)[::-1] # # [::-1] sorting in descending order
    
    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N] # # get instances with smallest distances
    print ("active_samples:", active_samples)
    
    return active_samples, distances
    












