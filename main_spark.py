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
	score_methods = ["u_gp_mt", "u_gp", #  "e_krr",
		# "fully_connected",	"ml-gp", "ml-knn"
		]

	embedding_methods = ["org_space", "MLKR", "LFDA", "LMNN"]

	all_kwargs = list(product(sampling_methods, score_methods, embedding_methods))
	n_tasks = len(all_kwargs)
	MainDir = copy.copy(ALdir)

	ncpus_reserve = 1
	ncores_per_cpu = 32
	cpus_per_task = 16
	max_cpus = ncpus_reserve*ncores_per_cpu # # ncpus take * ncores per cpu
	ntask_per_batch = int(max_cpus / cpus_per_task)

	nbatch = int(n_tasks/ntask_per_batch)
	makedirs(MainDir+"/code/batch_list/tmps.txt")

	for batch_ith in range(nbatch):

		shrun_file = open(MainDir+"/code/batch_list/batch_run_{0}.sh".format(batch_ith),"w") 
		shrun_file.write("#!/bin/bash \n")
		shrun_file.write("#SBATCH --ntasks={0}\n".format(ntask_per_batch))
		shrun_file.write("#SBATCH --output=./output_{0}.txt\n".format(batch_ith))
		
		shrun_file.write("#SBATCH --cpus-per-task={}\n".format(cpus_per_task))
		# shrun_file.write("#SBATCH --mem-per-cpu=8000\n")

		init_kw = batch_ith*ntask_per_batch
		last_kw = (batch_ith+1)*ntask_per_batch

		for kw in all_kwargs[init_kw:last_kw]:
			sampling_method, score_method, embedding_method = kw[0], kw[1], kw[2]
			kwargs = dict({})

			# # update kwargs
			kwargs["sampling_method"] = sampling_method
			kwargs["score_method"] = score_method
			kwargs["embedding_method"] = embedding_method

			print ("==========================")

			param_file = MainDir +"/data/params_grid/{0}_{1}_{2}.pkl".format(sampling_method, score_method, embedding_method)
			makedirs(param_file)
			dump_pickle(data=kwargs, filename=param_file)

			sh_file = MainDir+"/code/sh/{0}_{1}_{2}.sh".format(sampling_method, score_method, embedding_method)
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



