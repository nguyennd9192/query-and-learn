
import pandas as pd
from shutil import copyfile
from utils.plot import makedirs
from utils.general_lib import get_basename
import os, glob

existed_files = glob.glob("/Volumes/Nguyen_6TB/work/SmFe12_screening/input/origin_struct/queries/query_1/*.poscar")

def pickup(struct_list, savedir):
	convert_list = []
	
	AL_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/unlabeled_data/"
	struct_storage_dir = "/Volumes/Nguyen_6TB/work/SmFe12_screening/input"

	for k in struct_list:
		jobname = k.replace(struct_storage_dir, "").replace("/ofm1_no_d/", "/").replace(".ofm1_no_d", ".poscar")
		jobname = jobname.replace("/feature/", "")
		# get_basename(jobname)
		from_str = struct_storage_dir + "/origin_struct/"+jobname
		saveto = savedir + "/" + jobname.replace("/", "-_-")

		if saveto.replace("query_1_supp", "query_1") not in existed_files:
			print (from_str, saveto)

			makedirs(saveto)
			copyfile(from_str, saveto)

		# convert_list.append(cvt)




def query2vasp():
	indir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/"
	task = "mix_2-24/query_01" # mix/query_01, mix_2-24/query_01

	# ith_qr = 1
	savedir = "/Volumes/Nguyen_6TB/work/SmFe12_screening/input/origin_struct/queries/{0}".format(task)
	query_list = [
		# #. -12
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_exploitation/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task),
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_exploitation/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task),
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_margin/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task),
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_margin/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task),
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_uniform/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task),
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_uniform/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task),
		# "11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_expected_improvement/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/{0}/m0.1_c0.1.csv".format(task)
		]
	query_list = [indir+k for k in query_list]

	# # temporary
	for qr_file in query_list:
		df = pd.read_csv(qr_file, index_col=0)
		update_DQ_str, outstand_str, random_str = get_qrindex(df=df)

		# print (update_DQ_str)
		pickup(struct_list=update_DQ_str, savedir=savedir)
		pickup(struct_list=outstand_str, savedir=savedir)
		pickup(struct_list=random_str, savedir=savedir)

		save_config_csv = savedir+"_csv/"+qr_file.replace(indir, "")
		makedirs(save_config_csv)

		copyfile(qr_file, save_config_csv)

		# break

	print ("Expected query:", len(query_list)*30)
	total_dq = len(os.listdir(savedir))
	# total_os = len(os.listdir(savedir+"/os"))
	# total_rnd = len(os.listdir(savedir+"/rnd"))

	print ("In reality:", total_dq)



if __name__ == "__main__":
	query2vasp()
