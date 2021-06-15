from params import *
from utils.plot import *
from general_lib import *

import sys, pickle, functools, json, copy
import matplotlib.pyplot as plt
from utils.utils import load_pickle
import numpy as np


axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
title_font = {'fontname': 'serif', 'size': 14}
# performance_codes = dict({"uniform":"red", "exploitation":"black", 
# 		"margin":"blue", "expected_improvement":"green"})

# performance_codes = dict({"org_space":"blue", "MLKR":"red"}) 
performance_codes = dict({
	"org_space|uniform":"blue", "org_space|margin":"purple", "org_space|exploitation":"navy", 
	"MLKR|uniform":"red", "MLKR|margin":"orange", "MLKR|exploitation":"tomato", 
	}) 

hatch_codes = dict({"uniform":"/", "exploitation":"*", 
				"margin":"o", "expected_improvement":"/", "MaxEmbeddDir":"."}) 
# # '-', '+', 'x', '\\', '*', 'o', 'O', '.'

	
def get_local(qid):

	job_savedir = get_savedir(ith_trial=FLAGS.ith_trial)
	X_train, y_train, index_train, unlbl_X, unlbl_y, unlbl_index, pv = load_data()

	
	savedir = job_savedir.replace("/trial_1/", "/trial_{}/".format(FLAGS.ith_trial))
		
	# # read load only rand query
	eval_file = savedir + "/query_{0}/eval_query_{0}.pkl".format(qid)
	data = load_pickle(eval_file)

			
	position_file = savedir + "/query_{0}/query_{0}ipl_plot.csv".format(qid)
	pos_df = pd.read_csv(position_file, index_col=0)


	ebd_dist_file = savedir + "/query_{0}/MLKR_dist.csv".format(qid)
	dist_df = pd.read_csv(ebd_dist_file, index_col=0)
	print (dist_df)


if __name__ == "__main__":
	FLAGS(sys.argv)
	pr_file = sys.argv[-1]
	kwargs = load_pickle(filename=pr_file)
	FLAGS.score_method = kwargs["score_method"]
	FLAGS.sampling_method =	kwargs["sampling_method"]
	FLAGS.embedding_method = kwargs["embedding_method"]
	FLAGS.active_p = kwargs["active_p"]
	FLAGS.ith_trial = kwargs["ith_trial"]

	get_local(qid=15)











		