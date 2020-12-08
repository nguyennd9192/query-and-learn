# # latbx_ofm1, ofm_subs_Ga123, letter, latbx_ofm1_fe
# # sampling_method: uniform, exploitation, margin, bandit_discrete, simulate_batch_best_sim
# # simulate_batch_mixture (not work yet), 
from absl import flags 
import pandas as pd
import ntpath, os
# from utils.general_lib import *
def get_basename(filename):
    head, tail = ntpath.split(filename)
    basename = os.path.splitext(tail)[0]
    return tail

ALdir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master"
# ALdir = "/home/nguyen/work/active-learning"

database_dir = ALdir + "/data/standard"
coarse_db_dir = ALdir + "/data/coarse_relax"
fine_db_dir = ALdir + "/data/fine_relax"

std_file = database_dir+"/summary.csv"
coarse_file = coarse_db_dir+"/summary.csv"
fine_file = fine_db_dir+"/summary.csv"

# # # data base storage
database_jobs = [
  "mix/query_1.csv",  "mix/supp_2.csv", "mix/supp_3.csv", "mix/supp_4.csv",  
  "mix/supp_5.csv", "mix/supp_6.csv", "mix/supp_7.csv", "mix/supp_8.csv",
  "mix/supp_9.csv", "mix/supp_10.csv",
          # "mix_2-24/query_1.csv"
          ]
database_results = [database_dir+"/"+k for k in database_jobs]
fine_db_rst = [fine_db_dir+"/"+k for k in database_jobs]
coarse_db_rst = [coarse_db_dir+"/"+k for k in database_jobs]

if os.path.isfile(std_file) and os.path.isfile(coarse_file) and os.path.isfile(fine_file):
  db_results = pd.read_csv(std_file, index_col="index_reduce")
  crs_db_results = pd.read_csv(coarse_file, index_col="index_reduce")
  fine_db_results = pd.read_csv(fine_file, index_col="index_reduce")
  print ("Done reading database.")
else:
  # # standard result
  frames = [pd.read_csv(k, index_col=0) for k in database_results]
  db_results = pd.concat(frames)
  index_reduce = [get_basename(k) for k in db_results.index]
  db_results["index_reduce"] = index_reduce
  db_results.set_index('index_reduce', inplace=True)

  # # coarse, fine db
  crs_frames = [pd.read_csv(k, index_col=0) for k in coarse_db_rst]
  crs_db_results = pd.concat(crs_frames)
  crs_db_results = crs_db_results.dropna()
  index_reduce = [get_basename(k) for k in crs_db_results.index]
  crs_db_results["index_reduce"] = index_reduce
  crs_db_results.set_index('index_reduce', inplace=True)


  fine_frames = [pd.read_csv(k, index_col=0) for k in fine_db_rst]
  fine_db_results = pd.concat(fine_frames)
  fine_db_results = fine_db_results.dropna()
  index_reduce = [get_basename(k) for k in fine_db_results.index]
  fine_db_results["index_reduce"] = index_reduce
  fine_db_results.set_index('index_reduce', inplace=True)

  db_results.to_csv(std_file)
  crs_db_results.to_csv(coarse_file)
  fine_db_results.to_csv(fine_file)

print (len(crs_db_results))
print (len(fine_db_results))
print (len(db_results))



result_dropbox_dir = ALdir + "/results"
color_codes = dict({"DQ":"firebrick", "OS":"forestgreen", "RND":"darkblue", "DQ_to_RND":"orange"})
pos_codes = dict({"DQ":0, "OS":1, "RND":2, "DQ_to_RND":3})


# python rank_unlbl.py§
flags.DEFINE_string("dataset", "SmFe12_init", "Dataset name")  # 11*10*23-21_CuAlZnTiMoGa___ofm1_no_d
flags.DEFINE_string("sampling_method", "margin", 
                  # uniform, exploitation, margin, expected_improvement
                    ("Name of sampling method to use, can be any defined in "
                     "AL_MAPPING in sampling_methods.constants"))

flags.DEFINE_string(
    "score_method", "u_gp", # # u_gp, u_knn, e_krr
    "Method to use to calculate accuracy.")  
flags.DEFINE_string(
    "embedding_method", "MLKR", # # org_space, MLKR, LFDA, LMNN
    "Method to transform space.") 

flags.DEFINE_string(
    "mae_update_threshold", "0.3", # # 0.3, 1.0, update_all
    "mean absolute error to update dq to estimator") 


flags.DEFINE_boolean(
    "is_test_separate", False, # # True, False
    ("Whether or not the test file was prepared separately.")
)
flags.DEFINE_boolean(
    "is_clf", False,
    ("Performing classification or regression model")
)
flags.DEFINE_boolean(
    "is_search_params", True,
    ("search estimator or not")
)
flags.DEFINE_string(
    "test_prefix", "Fe10-Fe22", # # Ga, M3_Mo, Fe10-Fe22
    ("The prefix of train, test separating files")
)
flags.DEFINE_float(
    "warmstart_size", 0.1,
    ("Can be float or integer.  Float indicates percentage of training data "
     "to use in the initial warmstart model")
)
flags.DEFINE_float(
    "batch_size", 0.05,
    ("Can be float or integer.  Float indicates batch size as a percentage "
     "of mlkrning data size.") # # number of updated data points to the model
)
flags.DEFINE_integer("trials", 1,
                     "Number of curves to create using different seeds")
flags.DEFINE_integer("seed", 1, "Seed to use for rng and random state")
# TODO(lisha): add feature noise to simulate data outliers
flags.DEFINE_string("confusions", "0.1", 
  "Percentage of labels to randomize") 
flags.DEFINE_string("active_sampling_percentage", "0.1 0.3 0.5 0.7 0.9",
                    "Mixture weights on active sampling.")


flags.DEFINE_string(
    "select_method", "None",
    "Method to use for selecting points.")
flags.DEFINE_string("normalize_data", "True", "Whether to normalize the data.")
flags.DEFINE_string("standardize_data", "False",
                    "Whether to standardize the data.")
flags.DEFINE_string("save_dir", ALdir+"/results",
                    "Where to save outputs")
flags.DEFINE_string("data_dir", ALdir+"/data",
                    "Directory with predownloaded and saved datasets.")
flags.DEFINE_string("max_dataset_size", "15000",
                    ("maximum number of datapoints to include in data "
                     "zero indicates no limit"))
flags.DEFINE_float("train_horizon", "1.0",
                   "how far to extend learning curve as a percent of train")
flags.DEFINE_string("do_save", "True",
                    "whether to save log and results")

FLAGS = flags.FLAGS

