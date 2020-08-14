

# # latbx_ofm1, ofm_subs_Ga123, letter, latbx_ofm1_fe
# # running: uniform, bandit_discrete, margin, simulate_batch_best_sim 
# # simulate_batch_mixture (not work yet), 
from absl import flags 

flags.DEFINE_string("dataset", "11*10*23-21_CuAlZnTiMoGa___ofm1_no_d", "Dataset name") 
flags.DEFINE_string("sampling_method", "margin", 
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
     "of training data size.") # # number of updated data points to the model
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
    "score_method", "u_gp", # # logistic, kernel_svm, e_krr, u_gp
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

