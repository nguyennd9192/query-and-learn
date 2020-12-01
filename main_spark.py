import sys
from params import *

from a_rank_unlbl import map_unlbl_data

def main_spark():
	sampling_methods = ["uniform", "exploitation", "margin", "expected_improvement"]
	estimators = ["e_krr", 
				"u_gp", "u_gp_mt",  "fully_connected",
				"ml-gp", "ml-knn", ]

if __name__ == "__main__":
	FLAGS(sys.argv)