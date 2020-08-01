
import os, sys, glob, json, shutil, qmpy

sc_dir = "/media/nguyen/work/SmFe12_screening/source_code/"
for ld, subdirs, files in os.walk(sc_dir):
	if os.path.isdir(ld) and ld not in sys.path:
		sys.path.append(ld)


import numpy as np
from general_lib import get_subdirs, get_basename, makedirs



def get_feature_dir(feature_type, fulldir_jobs, saveat):
 
	all_dirs = []
	for current_dir in fulldir_jobs:
		print current_dir
		listdir = glob.glob("{}/*.*".format(current_dir)) # os.listdir(current_dir)
		all_dirs = np.concatenate((all_dirs, listdir))

	print all_dirs	
		# all_dirs = listdir

	makedirs(saveat)
	with open(saveat, "w") as f:
		for s in all_dirs:
			f.write(str(s) +"\n")

if __name__ == '__main__':

	feature_type = "ofm1_no_d"
	# ofm1_with_d,  ofm1_no_d, xrd
	# savename = "Sm2-Fe21-M3|{0}".format(feature_type)
	# jobs = ["Sm2-Fe21-M3"]

	# # CuAlZnTi_Sm2-Fe22-M2, CuAlZnTi_Sm2-Fe21-M3
    # # check_Mo_2-22-2
	jobs = [

			# # 1-12
			"Sm-Fe11-M_wyckoff_2", 
			"Sm-Fe10-M2_wyckoff", 

			# "Sm-Fe11-Si1", "Sm-Fe10-Si2",
			"Sm-Fe11-Ga1", "Sm-Fe10-Ga2",

			# # 2-24
			"Sm2-Fe23-M", 
			"check_Mo_2-22-2", "CuAlZnTi_Sm2-Fe22-M2", 
			"CuAlZnTi_Sm2-Fe21-M3", "Sm2-Fe21-M3_std6000",
			# "Sm2-Fe23-Si1", "Sm2-Fe22-Si2", "Sm2-Fe21-Si3", 

			"Sm2-Fe23-Ga1", "Sm2-Fe22-Ga2", "Sm2-Fe21-Ga3", 
			"Sm2-Fe20-Ga4",
            # "Sm2-Fe20-Ga4",
            
            # "Nd-Fe-B"
            # "oqmd_NdFeB"

			# "Sm2-Fe21-M3_std6000"
            
            # "Sm2-Fe24"
			]


    # # FOR INITIAL SUBSTITUTION STRUCTURES
	input_dir = "/media/nguyen/work/SmFe12_screening/input"

    # # FOR STANDARD SUBSTITUTION STRUCTURES
	# input_dir = "/media/nguyen/work/SmFe12_screening/result/standard/opt_struct"
	
	# # for single task
	if False:
		for job in jobs:
			savename = "{0}|{1}".format(job, feature_type)

			saveat = "{0}/rep_dirs/{1}.txt".format(input_dir, savename)
			fulldir_jobs = ["{0}/feature/{1}/{2}".format(input_dir, job, feature_type)]

			get_feature_dir(feature_type=feature_type, 
				fulldir_jobs=fulldir_jobs, saveat=saveat)
			print "save at", saveat


	# # for oqmd
	if False:
		# for job in jobs:

		# savename = "oqmd_Sm-Fe-Mo|with|{0}|{1}".format(job, feature_type)
		job = "oqmd_Sm-Fe-Mo"
		savename = "{0}|{1}".format(job, feature_type)


		saveat = "{0}/rep_dirs/{1}.txt".format(input_dir, savename)
		fulldir_jobs = ["{0}/feature/{1}/{2}".format(input_dir, job, feature_type),
			"{0}/feature/oqmd_Sm-Fe-Mo/{1}".format(input_dir, feature_type)]

		get_feature_dir(feature_type=feature_type, 
			fulldir_jobs=fulldir_jobs, saveat=saveat)


	if True:
		fulldir_jobs = []
		for job in jobs:
			savename = "{0}|{1}".format(job, feature_type)			
			fulldir_jobs += ["{0}/feature/{1}/{2}".format(input_dir, job, feature_type)]

		saveat = "{0}/rep_dirs/11*10*23-21_CuAlZnTiMoGa.txt".format(input_dir)
		get_feature_dir(feature_type=feature_type, 
			fulldir_jobs=fulldir_jobs, saveat=saveat)
		print "save at", saveat


