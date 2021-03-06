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

"""Margin based AL method.

Samples in batches based on margin scores.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sampling_methods.sampling_def import SamplingMethod
from sklearn.preprocessing import normalize
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

class ExpectedImprovement(SamplingMethod):
  def __init__(self, X, y, seed):
    self.X = X
    self.y = y
    self.name = 'ExpectedImprovement'

  def select_batch_(self, model, already_selected, N, y_star, **kwargs):

    pred_values = model.predict(self.X)
    variances = model.predict_proba(self.X, is_norm=False)

    # # y_star: current minimal value
    acq_vals = np.array([st.norm(mu, var).cdf(y_star) for mu, var in zip(pred_values, variances)])

    acq_vals = MinMaxScaler().fit_transform(X=acq_vals).ravel()

    rank_ind = np.argsort(acq_vals)[::-1] # # sorting in descending order
    rank_ind = [i for i in rank_ind if i not in already_selected]
    active_samples = rank_ind[0:N] # # get instances with smallest distances
    
    return active_samples, acq_vals
    

