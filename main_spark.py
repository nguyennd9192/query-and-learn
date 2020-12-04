import sys
from params import *

from utils.general_lib import *
from a_rank_unlbl import map_unlbl_data
from e_autism_modeling import show_trace, error_dist
import copy
from itertools import product
def main_spark():



	FLAGS.sampling_method = sm
	FLAGS.score_method = es

	map_unlbl_data(ith_trial="000")
	show_trace(ith_trial="000")
	error_dist(ith_trial="000")


def create_params_grid():

	sampling_methods = [
		"uniform", "exploitation", "margin", "expected_improvement"]
	score_methods = ["e_krr", "u_gp", "u_gp_mt",  
		"fully_connected",	"ml-gp", "ml-knn"]

	all_kwargs = list(product(sampling_methods, score_methods))
	n_tasks = len(all_kwargs)
	MainDir = copy.copy(ALdir)

	shrun_file = open(MainDir+"/code/batch_run.sh","w") 
	shrun_file.write("#!/bin/bash \n")
	shrun_file.write("#SBATCH --ntasks={0}\n".format(n_tasks))
	shrun_file.write("#SBATCH --output=./output.txt\n")
	
	shrun_file.write("#SBATCH --cpus-per-task=8\n")
	shrun_file.write("#SBATCH --mem-per-cpu=8000\n")


	for ith, kw in enumerate(all_kwargs):
		sampling_method, score_method = kw[0], kw[1]
		kwargs = dict({})

		# # update kwargs
		kwargs["score_method"] = score_method
		kwargs["sampling_method"] = sampling_method

		print ("==========================")

		param_file = MainDir +"/data/params_grid/{0}_{1}.pkl".format(sampling_method, score_method)
		makedirs(param_file)
		dump_pickle(data=kwargs, filename=param_file)

		sh_file = MainDir+"/code/sh/{0}{1}.sh".format(sampling_method, score_method)
		makedirs(sh_file)

		with open(sh_file, "w") as f:
			f.write("cd {0}\n".format(MainDir+"/code"))
			f.write("python a_rank_unlbl.py {0}\n".format(param_file))
			f.write("python e_autism_modeling.py {0}\n".format(param_file))

		shrun_file.write("srun --ntasks=1 --nodes=1 sh {0}\n".format(sh_file))

	shrun_file.close()


if __name__ == "__main__":
	# FLAGS(sys.)
	FLAGS(sys.argv)

	create_params_grid()



