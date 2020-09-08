# encoding: utf-8
import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

import pickle, io
from utils.utils import load_pickle


def read_deformation():
	result_dir = "/Volumes/Nguyen_6TB/work/SmFe12_screening/result/coarse_relax/structure_trajectory"
	jobs = ["Sm-Fe10-M2_wyckoff", 
				"Sm-Fe11-M_wyckoff_2",
				"Sm2-Fe22-M2", "Sm2-Fe23-M", "Sm2-Fe21-M3",
				"Sm-Fe11-Ga1", "Sm-Fe10-Ga2",
				"Sm2-Fe23-Ga1", "Sm2-Fe22-Ga2", "Sm2-Fe21-Ga3",
				"CuAlZnTi_Sm2-Fe21-M3"]
		# "mix/Sm-Fe9-Ti1-Ga2", "mix/Sm-Fe9-Ti2-Ga1", 
		# "mix/Sm-Fe10-Mo1-Ga1", "mix/Sm-Fe10-Ti1-Ga1"
	for job in jobs:
		jobfile = result_dir+"/"+job+".pkl"
		# result = load_pickle(jobfile)
		with open(jobfile, "rb") as f:
			result = pickle.load(f, fix_imports=True, encoding="latin")
		print(result.keys())
		x = result["struct_infom"]["sum_vasp"]["positions_ofm"]
		print(x)
		break
if __name__ == "__main__":
	read_deformation()