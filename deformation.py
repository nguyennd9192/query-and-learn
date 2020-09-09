# encoding: utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from params import *
import re
import pickle, io
from utils.utils import load_pickle
from run_experiment import get_savedir, get_savefile, get_data_from_flags, get_train_test, get_othere_cfg


def get_job_get_id(structure_dir):

	slashes = [m.start() for m in re.finditer('/', structure_dir)]
	structure_name = structure_dir[slashes[-1]+1:]

	begin_job_id = structure_dir.find("/origin_struct/")+len("/origin_struct/")
	end_job_id = slashes[-1]
	structure_job = structure_dir[begin_job_id: end_job_id]

	if structure_job == "check_Mo_2-22-2":
		structure_job = "Sm2-Fe22-M2"
	return structure_job, structure_name

def read_deformation(qr_indexes):
	result_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/structure_trajectory"
	jobs = ["Sm-Fe10-M2_wyckoff", 
				"Sm-Fe11-M_wyckoff_2",
				"Sm2-Fe22-M2", "Sm2-Fe23-M", "Sm2-Fe21-M3",
				"Sm-Fe11-Ga1", "Sm-Fe10-Ga2",
				"Sm2-Fe23-Ga1", "Sm2-Fe22-Ga2", "Sm2-Fe21-Ga3",
				"CuAlZnTi_Sm2-Fe21-M3"]

	print (qr_indexes)
		# "mix/Sm-Fe9-Ti1-Ga2", "mix/Sm-Fe9-Ti2-Ga1", 
		# "mix/Sm-Fe10-Mo1-Ga1", "mix/Sm-Fe10-Ti1-Ga1"

	# # results 
	deform_results = dict({})
	for job in jobs:
		jobfile = result_dir+"/"+job+".pkl"
		# result = load_pickle(jobfile)
		with open(jobfile, "rb") as f:
			result = pickle.load(f, fix_imports=True, encoding="latin")
			deform_results[job] = result

	for index in qr_indexes:
		print (index)

		structure_job, structure_name = get_job_get_id(structure_dir=index)
		print (structure_job, structure_name)

		this_result = deform_results[structure_job]
		print(this_result.keys())
		print(this_result["index"])
		print ("======")


	# 	x = result["struct_infom"]["sum_vasp"]["positions_ofm"]
	# 	print(x)
		# break
if __name__ == "__main__":
	FLAGS(sys.argv)
	X_trval_csv, y_trval_csv, index_trval_csv, X_test_csv, y_test_csv, test_idx_csv = get_data_from_flags()
	read_deformation(index_trval_csv)













