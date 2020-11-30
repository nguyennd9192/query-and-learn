# # latbx_ofm1, ofm_subs_Ga123, letter, latbx_ofm1_fe
# # sampling_method: uniform, exploitation, margin, bandit_discrete, simulate_batch_best_sim
# # simulate_batch_mixture (not work yet), 
from absl import flags 

localdir = "/Volumes/Nguyen_6TB/work/SmFe12_screening"
ALdir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master"
database_dir = localdir + "/result/standard"
coarse_db_dir = localdir + "/result/coarse_relax"
fine_db_dir = localdir + "/result/fine_relax"


database_jobs = [
  "mix/query_1.csv",  "mix/supp_2.csv", "mix/supp_3.csv", "mix/supp_4.csv",  
  "mix/supp_5.csv", "mix/supp_6.csv", "mix/supp_7.csv", "mix/supp_8.csv",
  "mix/supp_9.csv", "mix/supp_10.csv",
          # "mix_2-24/query_1.csv"
          ]
database_results = [database_dir+"/"+k for k in database_jobs]
fine_db_rst = [fine_db_dir+"/"+k for k in database_jobs]
coarse_db_rst = [coarse_db_dir+"/"+k for k in database_jobs]




result_dropbox_dir = ALdir + "/results"
color_codes = dict({"DQ":"firebrick", "OS":"forestgreen", "RND":"darkblue", "DQ_to_RND":"orange"})
pos_codes = dict({"DQ":0, "OS":1, "RND":2, "DQ_to_RND":3})




# python rank_unlbl.py§
flags.DEFINE_string("dataset", "11*10*23-21_CuAlZnTiMoGa___ofm1_no_d", "Dataset name") 
flags.DEFINE_string("sampling_method", "margin", 
                  # uniform, exploitation, margin, expected_improvement
                    ("Name of sampling method to use, can be any defined in "
                     "AL_MAPPING in sampling_methods.constants"))
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
    "score_method", "mlkr", # # e_krr, u_gp, u_gp_mt, mlkr, fully_connected, moe, LeNet
    "Method to use to calculate accuracy.")  
flags.DEFINE_string(
    "select_method", "None",
    "Method to use for selecting points.")
flags.DEFINE_string("normalize_data", "True", "Whether to normalize the data.")
flags.DEFINE_string("standardize_data", "False",
                    "Whether to standardize the data.")
flags.DEFINE_string("save_dir", "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results",
                    "Where to save outputs")
flags.DEFINE_string("data_dir", "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data",
                    "Directory with predownloaded and saved datasets.")
flags.DEFINE_string("max_dataset_size", "15000",
                    ("maximum number of datapoints to include in data "
                     "zero indicates no limit"))
flags.DEFINE_float("train_horizon", "1.0",
                   "how far to extend learning curve as a percent of train")
flags.DEFINE_string("do_save", "True",
                    "whether to save log and results")

FLAGS = flags.FLAGS

