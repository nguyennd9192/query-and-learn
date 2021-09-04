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

"""Make datasets and save specified directory.

Downloads datasets using scikit datasets and can also parse csv file
to save into pickle format.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from io import BytesIO
import os, glob, sys
import pickle
from io import StringIO
import tarfile
import urllib

# import keras.backend as K
# from keras.datasets import cifar10
# from keras.datasets import cifar100
# from keras.datasets import mnist
from tensorflow.io import gfile


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

sc_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master"
for ld, subdirs, files in os.walk(sc_dir):
  if os.path.isdir(ld) and ld not in sys.path:
    sys.path.append(ld)

from absl import app
from absl import flags
from utils.general_lib import *
from features import OFMFeature

from preprocess_data import query_db

# flags.DEFINE_string('save_dir', '/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data',
#                     'Where to save outputs')
# flags.DEFINE_string('datasets', '',
#                     'Which datasets to download, comma separated.')


FLAGS = flags.FLAGS


class Dataset(object):

  def __init__(self, X, y, index=None):
    self.data = X
    self.target = y
    # if index is not None:
    self.index=index

def get_csv_data(filename):
  """Parse csv and return Dataset object with data and targets.

  Create pickle data from csv, assumes the first column contains the targets
  Args:
    filename: complete path of the csv file
  Returns:
    Dataset object
  """
  f = gfile.GFile(filename, 'r')
  mat = []
  for l in f: 
    row = l.strip()
    row = row.replace('"', '')
    row = row.split(',')
    print (row)
    row = [float(x) for x in row]
    mat.append(row)
  mat = np.array(mat)
  y = mat[:, 0]
  X = mat[:, 1:]
  data = Dataset(X, y)
  return data

def get_pv_tv(df, pv, tv, rmvs):
  if pv is None:
    pv = df.columns.to_list()
    if tv in pv:
      pv.remove(tv)
    for rmv in rmvs:
      if rmv in pv:
        pv.remove(rmv)
  return pv, tv

def get_ofm_data(filename, pv, tv, rmvs):
  df = pd.read_csv(filename, index_col=0)
  df = df.dropna()
  pv, tv = get_pv_tv(df, pv, tv, rmvs)
  X = df[pv].values
  fe = df[tv].values
  # # revise here
  stb_thres = 0.0603 # # found in 200318_note1
  # y = np.array(['stable' if i < stb_thres else 'unstable' for i in fe])
  # print ("N stable", len(np.where(fe =="stable")[0]))
  # print ("N unstable", len(np.where(fe =="unstable")[0]))
  data = Dataset(X=X, y=fe, index=df.index)
  return data


def merge_df(filenames, pv, tv, rmvs):
  dfs = []

  for ith, filename in enumerate(filenames):
    df = pd.read_csv(filename, index_col=0)
    this_pv, tv = get_pv_tv(df, pv, tv, rmvs)
    if ith == 0:
      save_pv = this_pv
    assert save_pv == this_pv
    print(np.concatenate([pv, tv]))
    dfs.append(df[np.concatenate([pv, tv])])

  df_final = reduce(lambda left,right: pd.merge(left,right,on='name'), dfs)


def get_separate_test_set(filename, pv, tv, rmvs, test_cond):
  df = pd.read_csv(filename, index_col=0)
  df = df.dropna()
  pv, tv = get_pv_tv(df, pv, tv, rmvs)

  # idx = [k for k in df.index if test_cond in k]
  idx = [k for k in df.index if "Fe10" in k or "Fe22" in k]
  print(idx)
  test_idx = df.index.isin(idx)
  # fe = df[tv].values
  # stb_thres = 0.0603 # # found in 200318_note1
  # y_cat = np.array(['stable' if i < stb_thres else 'unstable' for i in fe])
  # df[tv] = fe

  X_train = df.loc[~test_idx, pv]
  y_train = df.loc[~test_idx, tv]

  X_test = df.loc[test_idx, pv].values
  y_test = df.loc[test_idx, tv].values
  # # revise here 
  print ("N stable train", len(np.where(y_train =="stable")[0]))
  print ("N unstable train", len(np.where(y_train =="unstable")[0]))
  print ("N stable test", len(np.where(y_test =="stable")[0]))
  print ("N unstable test", len(np.where(y_test =="unstable")[0]))
  print ("5 train idx", y_train.index[:5])
  print ("5 test idx", idx[:5])


  data_train = Dataset(X_train.values, y_train.values, y_train.index)
  data_test = Dataset(X_test, y_test, idx)
  print(data_test.index)

  return data_train, data_test


def get_SmFe12_data(tv, rmvs, unlbl_data_dir, 
      saveat, ft_type="ofm1_no_d"):

  # # point to datadir of Volume6TB then get all unlnl data
  test_data = []
  unlbl_jobs = get_subdirs(unlbl_data_dir)
  for job in unlbl_jobs:
    listdir = glob.glob("{0}/{1}/*.*".format(job, ft_type)) # os.listdir(current_dir)    
    test_data = np.concatenate((test_data, listdir))

  # with open(struct_dir_file) as file:
  #   struct_reprs = file.read().splitlines()

  data = []
  for struct_repr in test_data:
    with open(struct_repr, "rb") as tmp_f:
      struct_repr_obj = pickle.load(tmp_f,encoding='latin1') #
      data.append(struct_repr_obj)

  # # get X
  feature = np.array([file.__getattribute__('feature') for file in data])
  feature_names = np.array(data[0].__getattribute__('feature_name'))

  # # get vasp calc data
  std_results, coarse_results, fine_results = query_db(db_flfd="SmFe12/210829")
  print (std_results.shape)
  print (coarse_results.shape)
  print (coarse_results.shape)
  
  print (test_data)

  # # map index to database
  data_map = map(functools.partial(id_qr_to_database, std_results=std_results,
    coarse_results=coarse_results, fine_results=fine_results, tv=tv), test_data) 
  data_map = np.array(list(data_map))

  print ("data_map.shape:", data_map.shape)
  # id_qr, y=unlbl_y, index=unlbl_index
  # # get y
  y_obs = data_map[:, -1]
  test_index = data_map[:, 1]

  # df.nunique(dropna=False)
  full_df = pd.DataFrame(feature, columns=feature_names, index=test_index)
  full_df[tv] = y_obs
  full_df = full_df.dropna()

  # # for train data only, to remove constant cols, finally get pv
  pv = copy.copy(feature_names)

  X_filter = full_df[pv].values 
  y_filter = full_df[tv].values 
  index_filter = full_df.index

  # # save data as pv, tv for all train and test
  data = pd.DataFrame(X_filter, columns=pv, index=index_filter)
  data[tv] = y_filter
  data.dropna()
  data.to_csv(saveat+'.csv')

  print("Save at:", saveat, y_filter.shape[0], y_obs.shape[0])


def get_wikipedia_talk_data():
  """Get wikipedia talk dataset.

  See here for more information about the dataset:
  https://figshare.com/articles/Wikipedia_Detox_Data/4054689
  Downloads annotated comments and annotations.
  """

  ANNOTATED_COMMENTS_URL = 'https://ndownloader.figshare.com/files/7554634'
  ANNOTATIONS_URL = 'https://ndownloader.figshare.com/files/7554637'

  def download_file(url):
    req = urllib.request(url)
    response = urllib.urlopen(req)
    return response

  # Process comments
  comments = pd.read_table(
      download_file(ANNOTATED_COMMENTS_URL), index_col=0, sep='\t')
  # remove newline and tab tokens
  comments['comment'] = comments['comment'].apply(
      lambda x: x.replace('NEWLINE_TOKEN', ' '))
  comments['comment'] = comments['comment'].apply(
      lambda x: x.replace('TAB_TOKEN', ' '))

  # Process labels
  annotations = pd.read_table(download_file(ANNOTATIONS_URL), sep='\t')
  # labels a comment as an atack if the majority of annoatators did so
  labels = annotations.groupby('rev_id')['attack'].mean() > 0.5

  # Perform data preprocessing, should probably tune these hyperparameters
  vect = CountVectorizer(max_features=30000, ngram_range=(1, 2))
  tfidf = TfidfTransformer(norm='l2')
  X = tfidf.fit_transform(vect.fit_transform(comments['comment']))
  y = np.array(labels)
  data = Dataset(X, y)
  return data


def get_keras_data(dataname):
  """Get datasets using keras API and return as a Dataset object."""
  if dataname == 'cifar10_keras':
    train, test = cifar10.load_data()
  elif dataname == 'cifar100_coarse_keras':
    train, test = cifar100.load_data('coarse')
  elif dataname == 'cifar100_keras':
    train, test = cifar100.load_data()
  elif dataname == 'mnist_keras':
    train, test = mnist.load_data()
  else:
    raise NotImplementedError('dataset not supported')

  X = np.concatenate((train[0], test[0]))
  y = np.concatenate((train[1], test[1]))

  if dataname == 'mnist_keras':
    # Add extra dimension for channel
    num_rows = X.shape[1]
    num_cols = X.shape[2]
    X = X.reshape(X.shape[0], 1, num_rows, num_cols)
    if K.image_data_format() == 'channels_last':
      X = X.transpose(0, 2, 3, 1)

  y = y.flatten()
  data = Dataset(X, y)
  return data


# TODO(lishal): remove regular cifar10 dataset and only use dataset downloaded
# from keras to maintain image dims to create tensor for tf models
# Requires adding handling in run_experiment.py for handling of different
# training methods that require either 2d or tensor data.
def get_cifar10():
  """Get CIFAR-10 dataset from source dir.

  Slightly redundant with keras function to get cifar10 but this returns
  in flat format instead of keras numpy image tensor.
  """
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
  def download_file(url):
    # req = urllib.request(url)
    response = urllib.request.urlopen(url)
    return response
  response = download_file(url)
  tmpfile = BytesIO()
  while True:
    # Download a piece of the file from the connection
    s = response.read(16384)
    # Once the entire file has been downloaded, tarfile returns b''
    # (the empty bytes) which is a falsey value
    if not s:
      break
    # Otherwise, write the piece of the file to the temporary file.
    tmpfile.write(s)
  response.close()

  tmpfile.seek(0)
  tar_dir = tarfile.open(mode='r:gz', fileobj=tmpfile)
  X = None
  y = None
  for member in tar_dir.getnames():
    if '_batch' in member:
      filestream = tar_dir.extractfile(member).read()
      # batch = pickle.load(StringIO.StringIO(filestream))
      # print (filestream)
      batch = pickle.load(BytesIO(filestream))

      # batch = pickle.load(filestream)

      if X is None:
        X = np.array(batch['data'], dtype=np.uint8)
        y = np.array(batch['labels'])
      else:
        X = np.concatenate((X, np.array(batch['data'], dtype=np.uint8)))
        y = np.concatenate((y, np.array(batch['labels'])))
  data = Dataset(X, y)
  return data


def dump_data(data, filename):
  X = data.data
  y = data.target
  index = data.index
  if X.shape[0] != y.shape[0]:
    X = np.transpose(X)
  assert X.shape[0] == y.shape[0]

  data = {'data': X, 'target': y, "index":index}
  # print (data)
  print (len(np.where(y == 'stable')))
  makedirs(filename)
  pickle.dump(data, gfile.GFile(filename, 'w'))
  print ("Save at: ", filename)


def get_mldata(dataset, is_test_separate=False, prefix=None):
  # Use scikit to grab datasets and save them save_dir.
  save_dir = ALdir+"/data"

  # if not gfile.exists(save_dir):
  #   gfile.mkdir(save_dir)

  if not os.path.exists(save_dir):
    os.makedirs(save_dir)

  # # revise here, do we need to remove existed dataset first
  if is_test_separate:
    data_train, data_test = get_separate_test_set(filename=dataset[0], 
      pv=None, tv=dataset[2], rmvs=dataset[3], test_cond=prefix)
    filedir = os.path.join(save_dir, dataset[1])
    
    prefix=prefix.replace("/","_")
    dump_data(data_train, filedir+"/train_"+prefix+'.pkl')
    dump_data(data_test, filedir+"/test_"+prefix+'.pkl')
    return

  filename = os.path.join(save_dir, dataset[1]+'.pkl')
  if not gfile.exists(filename):
    if "ofm" in dataset[0]:
      # # read ofm data:
      # print ("Process here")
      data = get_ofm_data(filename=dataset[0], 
        pv=None, tv=dataset[2], rmvs=dataset[3])

    else:
      if dataset[0][-3:] == 'csv':
        data = get_csv_data(dataset[0]) 
      elif dataset[0] == 'breast_cancer':
        data = load_breast_cancer()
      elif dataset[0] == 'iris':
        data = load_iris()
      elif dataset[0] == 'newsgroup':
        # Removing header information to make sure that no newsgroup identifying
        # information is included in data
        data = fetch_20newsgroups_vectorized(subset='all', remove=('headers'))
        tfidf = TfidfTransformer(norm='l2')
        X = tfidf.fit_transform(data.data)
        data.data = X
      elif dataset[0] == 'rcv1':
        sklearn.datasets.rcv1.URL = (
          'http://www.ai.mit.edu/projects/jmlr/papers/'
          'volume5/lewis04a/a13-vector-files/lyrl2004_vectors')
        sklearn.datasets.rcv1.URL_topics = (
          'http://www.ai.mit.edu/projects/jmlr/papers/'
          'volume5/lewis04a/a08-topic-qrels/rcv1-v2.topics.qrels.gz')
        data = sklearn.datasets.fetch_rcv1(
            data_home='/tmp')
      elif dataset[0] == 'wikipedia_attack':
        data = get_wikipedia_talk_data()
      elif dataset[0] == 'cifar10':
        data = get_cifar10()
      elif 'keras' in dataset[0]:
        data = get_keras_data(dataset[0])
      else:
        if len(dataset) == 3:
          print ("here")
          data = fetch_openml(data_id=dataset[-1]) 
        else:
          try:
            data = fetch_openml(dataset[0]) 
          except Exception as e:
            raise e
            raise Exception('ERROR: failed to fetch data from mldata.org')
            pass
          # return
    dump_data(data, filename)
    return
  else:
    print ("The following file EXISTED: {0} ".format(filename))
    return


def main():
  # First entry of tuple is mldata.org name, second is the name that we'll use
  # to reference the data.
  input_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/"
  # pv = ['p1-ofcenter', 'd6-ofcenter', 'd10-ofcenter', 'f6-ofcenter', 's2-s2', 
  #   'p1-s2', 'd6-s2', 'd10-s2', 'f6-s2', 's2-p1', 'p1-p1', 'd6-p1', 'd10-p1', 
  #   'f6-p1', 's2-d6', 'p1-d6', 'd6-d6', 'd10-d6', 'f6-d6', 's2-d10', 'p1-d10', 
  #   'd6-d10', 'd10-d10', 'f6-d10', 's2-f6', 'p1-f6', 'd6-f6', 'd10-f6']
  # tv = "energy_substance_pa"
  datasets = [
              # ('mnist_784', 'mnist'), ('australian', 'australian'),
              # ('heart', 'heart'), ('breast_cancer', 'breast_cancer'),
              # ('iris', 'iris'), 

              # ('vehicle', 'vehicle', 54), 
              # ('wine', 'wine', 973),
              # ('waveform ida', 'waveform', 60), 
              # ('german ida', 'german', 31),
              # ('splice ida', 'splice', 46), 
              # ('ringnorm ida', 'ringnorm', 1496),
              # ('twonorm ida', 'twonorm', 1507), 
              # ('diabetes_scale', 'diabetes', 37),
              # ('mushrooms', 'mushrooms', 24), ('letter', 'letter', 6), 
              # ('dna', 'dna', 40670),
              # ('banana-ida', 'banana', 1460), 
              # ('newsgroup', 'newsgroup', 1457), 

              # # not create yet
              # ('cifar10', 'cifar10'),
              # ('cifar10_keras', 'cifar10_keras'),
              # ('cifar100_keras', 'cifar100_keras'),
              # ('cifar100_coarse_keras', 'cifar100_coarse_keras'),
              # ('mnist_keras', 'mnist_keras'),
              # ('wikipedia_attack', 'wikipedia_attack'),
              # ('rcv1', 'rcv1'),

              # # # for SmFe12
              # # for energy
              ('energy_substance_pa', ["atoms", "magmom_pa"]),
              # # for magmom 
              ('magmom_pa', ["atoms", "energy_substance_pa"]),
              ]

  is_Sm12_test = True

  for d in datasets:
    if is_Sm12_test:
      tv = d[0]
      rmvs = d[-1]
      # # for train
      train_data = input_dir+"/SmFe12/train_{}".format(tv)
      pv = get_SmFe12_data(tv=tv, rmvs=rmvs, 
        unlbl_data_dir="/Volumes/Nguyen_6TB/work/SmFe12_screening/input/feature/single", 
        saveat=train_data)

      # train_file = input_dir+"/SmFe12/init_{}.csv".format(tv)
      # df = pd.read_csv(train_file, index_col=0)
      # pv = list(df.columns)
      # pv.remove(tv)
      # # for test
      test_data = input_dir+"/SmFe12/test_{}".format(tv)
      get_SmFe12_data(tv=tv, rmvs=rmvs, 
        unlbl_data_dir="/Volumes/Nguyen_6TB/work/SmFe12_screening/input/feature/mix", # mix_2-24, mix
        saveat=test_data)

      train_df = pd.read_csv(train_data+".csv", index_col=0)
      test_df = pd.read_csv(test_data+".csv", index_col=0)
      
      n_train = len(train_df)

      frames = [train_df, test_df]
      merge_df = pd.concat(frames)
      all_cols = merge_df.columns
      for cc in all_cols:
        nuq = len(np.unique(merge_df[cc].values))
        if nuq <= 1:
          merge_df = merge_df.drop([cc], axis=1)

      pv = list(merge_df.columns)
      pv.remove(tv)

      all_indexes = np.array(merge_df.index)
      train_df_revise = merge_df.loc[all_indexes[:n_train], :]
      test_df_revise = merge_df.loc[all_indexes[n_train:], :]


      train_revise_dir = train_data.replace("_full", "")
      test_revise_dir = test_data.replace("_full", "")
      train_df_revise.to_csv(train_revise_dir+".csv")
      test_df_revise.to_csv(test_revise_dir+".csv")

      train_ = Dataset(X=train_df_revise[pv].values, y=train_df_revise[tv], index=train_df_revise.index)
      dump_data(train_, train_revise_dir+'.pkl')

      test_ = Dataset(X=test_df_revise[pv].values, y=test_df_revise[tv], index=test_df_revise.index)
      dump_data(test_, test_revise_dir+'.pkl')

if __name__ == '__main__':
  main()

  # std_results, coarse_results, fine_results = query_db(db_flfd="SmFe12_side")


