
from absl import flags 
import pandas as pd
import ntpath, os

def get_basename(filename):
    head, tail = ntpath.split(filename)
    basename = os.path.splitext(tail)[0]
    return tail
batch_size =  20
batch_outstand = 20 
batch_rand = 20 
n_run = int(3024 / (batch_size + batch_outstand + batch_rand))

ALdir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master"
# ALdir = "/home/nguyen/work/active-learning"


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
    "score_method", "u_knn", # # u_gp, u_knn, e_krr, u_knn
    ("Method to use to calculate accuracy.")
)
flags.DEFINE_string(
    "embedding_method", "MLKR", # # org_space, MLKR, LFDA, LMNN
    ("Method to transform space.")
) 
flags.DEFINE_string(
    "mae_update_threshold", "0.1", # # 0.0, 0.3, 1.0, update_all
    ("mean absolute error to update dq to estimator")
) 

flags.DEFINE_integer("batch_size", batch_size, 
    ("batch size of DQ")
)
flags.DEFINE_integer("batch_outstand", batch_outstand, 
    ("batch size of out-standing points")
)
flags.DEFINE_integer("batch_rand", batch_rand, 
    ("batch size of random")
)
flags.DEFINE_integer("n_run", n_run, 
    ("number of active query launch")
)
flags.DEFINE_boolean("is_search_params", True, 
  ("search estimator or not")
)


flags.DEFINE_boolean(
    "is_test_separate", False, # # True, False
    ("Whether or not the test file was prepared separately."))
flags.DEFINE_boolean(
    "is_clf", False,
    ("Performing classification or regression model")
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

