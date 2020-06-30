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

"""Run active learner on classification tasks.

Supported datasets include mnist, letter, cifar10, newsgroup20, rcv1,
wikipedia attack, and select classification datasets from mldata.
See utils/create_data.py for all available datasets.

For binary classification, mnist_4_9 indicates mnist filtered down to just 4 and
9.
By default uses logistic regression but can also train using kernel SVM.
2 fold cv is used to tune regularization parameter over a exponential grid.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
from time import gmtime
from time import strftime

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler

from absl import app
from tensorflow.io import gfile

from sampling_methods.constants import AL_MAPPING
from sampling_methods.constants import get_AL_sampler
from sampling_methods.constants import get_wrapper_AL_mapping
from utils import utils
from params import *

import pandas as pd


def get_savefile():
  if not FLAGS.is_test_separate:
    s_at =str( "results_score_" + FLAGS.score_method + 
                "_select_" + FLAGS.select_method +
                "_norm_" + (FLAGS.normalize_data) +
                "_is_search_params_" + str(FLAGS.is_search_params) +
                "_stand_" + FLAGS.standardize_data)
  else:
    s_at =str( "results_score_" + FLAGS.score_method + 
                "_select_" + FLAGS.select_method +
                "_norm_" + (FLAGS.normalize_data) +
                "_is_search_params_" + str(FLAGS.is_search_params) +
                "_stand_" + FLAGS.standardize_data +
                "_test_" + FLAGS.test_prefix)
  return s_at

def get_savedir():
  s_dir = str(os.path.join(
      FLAGS.save_dir,
      FLAGS.dataset + "_" + FLAGS.sampling_method))
  return s_dir

get_wrapper_AL_mapping()


def generate_one_curve(X, y,
                       X_sept_test, y_sept_test,
                       sampler,
                       score_model,
                       seed,
                       warmstart_size,
                       batch_size,
                       select_model=None,
                       confusion=0.1,
                       active_p=1.0,
                       max_points=None,
                       standardize_data=False,
                       norm_data=False,
                       train_horizon=0.5):
  """Creates one learning curve for both active and passive learning.

  Will calculate accuracy on validation set as the number of training data
  points increases for both PL and AL.
  Caveats: training method used is sensitive to sorting of the data so we
    resort all intermediate datasets

  Args:
    X: training data
    y: training labels
    sampler: sampling class from sampling_methods, assumes reference
      passed in and sampler not yet instantiated.
    score_model: model used to score the samplers.  Expects fit and predict
      methods to be implemented.
    seed: seed used for data shuffle and other sources of randomness in sampler
      or model training
    warmstart_size: float or int.  float indicates percentage of train data
      to use for initial model
    batch_size: float or int.  float indicates batch size as a percent of
      training data
    select_model: defaults to None, in which case the score model will be
      used to select new datapoints to label.  Model must implement fit, predict
      and depending on AL method may also need decision_function.
    confusion: percentage of labels of one class to flip to the other
    active_p: percent of batch to allocate to active learning
    max_points: limit dataset size for preliminary
    standardize_data: wheter to standardize the data to 0 mean unit variance
    norm_data: whether to normalize the data.  Default is False for logistic
      regression.
    train_horizon: how long to draw the curve for.  Percent of training data.

  Returns:
    results: dictionary of results for all samplers
    sampler_states: dictionary of sampler objects for debugging
  """
  # TODO(lishal): add option to find best hyperparameter setting first on
  # full dataset and fix the hyperparameter for the rest of the routine
  # This will save computation and also lead to more stable behavior for the
  # test accuracy

  # TODO(lishal): remove mixture parameter and have the mixture be specified as
  # a mixture of samplers strategy
  def select_batch(sampler, uniform_sampler, mixture, N, already_selected,
                   **kwargs):
    n_active = int(mixture * N)
    n_passive = N - n_active
    kwargs["N"] = n_active
    kwargs["already_selected"] = already_selected
    batch_AL = sampler.select_batch(**kwargs)
    already_selected = already_selected + batch_AL
    kwargs["N"] = n_passive
    kwargs["already_selected"] = already_selected

    # kwargs_copy = copy.copy(kwargs)
    # if "return_best_sim" in kwargs_copy.keys():
    #   del kwargs_copy[key]
    # batch_PL = uniform_sampler.select_batch(**kwargs_copy)
    batch_PL = uniform_sampler.select_batch(**kwargs)

    return batch_AL + batch_PL

  np.random.seed(seed)
  data_splits = [2./3, 1./6, 1./6]

  # 2/3 of data for training
  if max_points is None:
    max_points = len(y)
  train_size = int(min(max_points, len(y)) * data_splits[0])
  if batch_size < 1:
    batch_size = int(batch_size * train_size)
  else:
    batch_size = int(batch_size)
  if warmstart_size < 1:
    # Set seed batch to provide enough samples to get at least 4 per class
    # TODO(lishal): switch to sklearn stratified sampler
    seed_batch = int(warmstart_size * train_size)
  else:
    seed_batch = int(warmstart_size)

  if len(np.unique(y)) > 10 and FLAGS.is_clf:
    # # if performing classification but y is continuous variable
    print("Warning!!! Reconsidering is_clf tag. Number of classes is larger than 10.")

  if FLAGS.is_clf:
    # # check classification or not
    seed_batch = max(seed_batch, 6 * len(np.unique(y)))
  

  # # Nguyen, for regression
  if FLAGS.score_method == "krr":
    seed_batch = 5000
  # print ("score_model", score_model)
  if not FLAGS.is_test_separate:
    indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
        utils.get_train_val_test_splits(X,y,max_points,seed,confusion,
                                        seed_batch, split=data_splits, is_clf=FLAGS.is_clf))
  else:
      indices, X_train, y_train, X_val, y_val, X_test, y_test, y_noise = (
        utils.get_sept_train_val_test(X, y, X_sept_test, y_sept_test, 
          max_points, seed, confusion, seed_batch, split=[2./3, 1./3], is_clf=FLAGS.is_clf) 
      )
  print("Done splitting train, val, test")
  
  # Preprocess data
  if norm_data:
    print("Normalizing data")
    X_train = normalize(X_train)
    X_val = normalize(X_val)
    X_test = normalize(X_test)
  if standardize_data:
    print("Standardizing data")
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
  print("X shape: " + str(X.shape) + "y shape: " + str(y.shape) + " " +
    "train_size: " + str(train_size) + "active percentage: " + str(active_p) + " " +
    "warmstart batch: " + str(seed_batch) + "batch size: " + str(batch_size) + " " +
    "confusion: " + str(confusion) + "seed: " + str(seed))

  # Initialize samplers
  uniform_sampler = AL_MAPPING["uniform"](X_train, y_train, seed)
  sampler = sampler(X_train, y_train, seed)

  results = {}
  data_sizes = []
  accuracy = []
  selected_inds = list(range(seed_batch))
  cv_train_model = []
  save_model = []

  # If select model is None, use score_model
  same_score_select = False
  if select_model is None:
    select_model = score_model
    same_score_select = True
  n_batches = int(np.ceil((train_horizon * train_size - seed_batch) *
                          1.0 / batch_size)) + 1
  print("n_batches: ", n_batches)
  print("seed_batch: ", seed_batch)

  for b in range(n_batches):
    n_train = seed_batch + min(train_size - seed_batch, b * batch_size)
    print("Training model on " + str(n_train) + " datapoints")

    assert n_train == len(selected_inds)
    data_sizes.append(n_train)

    # Sort active_ind so that the end results matches that of uniform sampling
    partial_X = X_train[sorted(selected_inds)]
    partial_y = y_train[sorted(selected_inds)]
    print("prepare to fit")
    score_model.fit(partial_X, partial_y)
    print("done fitting")

    if not same_score_select:
      select_model.fit(partial_X, partial_y)
    acc = score_model.score(X_test, y_test)
    cv_train_model.append(score_model.best_score_)
    save_model.append(select_model)


    accuracy.append(acc)
    print("Sampler: %s, Accuracy: %.2f%%" % (sampler.name, accuracy[-1]*100))

    n_sample = min(batch_size, train_size - len(selected_inds))
    select_batch_inputs = {
        "model": select_model,
        "labeled": dict(zip(selected_inds, y_train[selected_inds])),
        "eval_acc": accuracy[-1],
        "X_test": X_val,
        "y_test": y_val,
        "y": y_train
    }
    new_batch = select_batch(sampler, uniform_sampler, active_p, n_sample,
                             list(selected_inds), **select_batch_inputs)
    selected_inds.extend(new_batch)
    print('Requested: %d, Selected: %d' % (n_sample, len(new_batch)))
    assert len(new_batch) == n_sample
    # print (selected_inds)
    assert len(list(set(selected_inds))) == len(selected_inds)

  # Check that the returned indice are correct and will allow mapping to
  # training set from original data
  assert all(y_noise[indices[selected_inds]] == y_train[selected_inds])
  results["org_data_size"] = X.shape
  results["save_model"] = save_model
  results["n_test"] = len(y_test)
  results["cv_train_model"] = cv_train_model
  results["accuracy"] = accuracy
  results["selected_inds"] = selected_inds
  results["data_sizes"] = data_sizes
  results["indices"] = indices
  results["noisy_targets"] = y_noise
  results["is_test_separate"] = FLAGS.is_test_separate
  results["test_prefix"] = FLAGS.test_prefix
  results["y_train_info"] = pd.DataFrame(y_train).describe()
  results["y_val_info"] = pd.DataFrame(y_val).describe()
  results["y_test_info"] = pd.DataFrame(y_test).describe()
  return results, sampler


def run():
  if not gfile.exists(FLAGS.save_dir):
    try:
      gfile.mkdir(FLAGS.save_dir)
    except:
      print(('WARNING: error creating save directory, '
             'directory most likely already created.'))

  save_dir = get_savedir()
  do_save = FLAGS.do_save == "True"

  if do_save:
    if not gfile.exists(save_dir):
      try:
        gfile.mkdir(save_dir)
      except:
        print(('WARNING: error creating save directory, '
               'directory most likely already created.'))
    # Set up logging
    filename = os.path.join(
        save_dir, "log-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + ".txt")
    sys.stdout = utils.Logger(filename)

  confusions = [float(t) for t in FLAGS.confusions.split(" ")]
  mixtures = [float(t) for t in FLAGS.active_sampling_percentage.split(" ")]
  all_results = {}
  max_dataset_size = None if FLAGS.max_dataset_size == "0" else int(
      FLAGS.max_dataset_size)
  normalize_data = FLAGS.normalize_data == "True"
  standardize_data = FLAGS.standardize_data == "True"
  if not FLAGS.is_test_separate:
    X, y = utils.get_mldata(FLAGS.data_dir, FLAGS.dataset)
    X_sept_test, y_sept_test = None, None
  else:
    X, y = utils.get_mldata(FLAGS.data_dir, FLAGS.dataset+"/train_"+FLAGS.test_prefix)
    X_sept_test, y_sept_test = utils.get_mldata(FLAGS.data_dir, FLAGS.dataset+"/test_"+FLAGS.test_prefix)
    print("Success in reading separated test set.")


  starting_seed = FLAGS.seed

  for c in confusions:
    for m in mixtures:
      for seed in range(starting_seed, starting_seed + FLAGS.trials):
        sampler = get_AL_sampler(FLAGS.sampling_method)
        score_model = utils.get_model(FLAGS.score_method, seed, FLAGS.is_search_params)
        if (FLAGS.select_method == "None" or
            FLAGS.select_method == FLAGS.score_method):
          select_model = None
        else:
          select_model = utils.get_model(FLAGS.select_method, seed)
        print("Done preparing select_model..")
        results, sampler_state = generate_one_curve(
            X, y, X_sept_test, y_sept_test, sampler, score_model, seed, FLAGS.warmstart_size,
            FLAGS.batch_size, select_model, c, m, max_dataset_size,
            standardize_data, normalize_data, FLAGS.train_horizon)
        key = (FLAGS.dataset, FLAGS.sampling_method, FLAGS.score_method,
               FLAGS.select_method, m, FLAGS.warmstart_size, FLAGS.batch_size,
               c, standardize_data, normalize_data, seed)
        sampler_output = sampler_state.to_dict()
        results["sampler_output"] = sampler_output
        all_results[key] = results
  fields = [
      "dataset", "sampler", "score_method", "select_method",
      "active percentage", "warmstart size", "batch size", "confusion",
      "standardize", "normalize", "seed", "cv_train_model", "n_test", "save_model"
  ]
  all_results["tuple_keys"] = fields

  if do_save:
    filename = get_savefile()
    existing_files = gfile.glob(os.path.join(save_dir, filename + "*.pkl"))
    extend_save_idx = str(1000+len(existing_files))[1:]
    filename = os.path.join(save_dir,
                            filename + "_" + extend_save_idx + ".pkl")    
    # filename = os.path.join(save_dir,
    #                         filename + "_" + "000" + ".pkl")
    pickle.dump(all_results, gfile.GFile(filename, "w"))
    sys.stdout.flush_file()
    return extend_save_idx

if __name__ == "__main__":
  # app.run(run)
  main()

