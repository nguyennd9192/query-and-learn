
import pandas as pd
from shutil import copyfile
from utils.plot import makedirs

def pickup(struct_list, savedir):
	convert_list = []
	
	AL_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/unlabeled_data/"
	struct_storage_dir = "/Volumes/Nguyen_6TB/work/SmFe12_screening/input/origin_struct/"

	for k in struct_list:
		jobname = k.replace(AL_dir, "").replace("/ofm1_no_d/", "/").replace("ofm1_no_d", "poscar")

		from_str = struct_storage_dir + jobname
		saveto = savedir + jobname.replace("/", "-_-")

		print (k, from_str, saveto)
		makedirs(saveto)
		copyfile(from_str, saveto)

		# convert_list.append(cvt)


def query2vasp():
	indir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/"
	query_list = [
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_exploitation/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv",
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_exploitation/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv",
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_margin/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv",
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_margin/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv",
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_margin/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv",
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_uniform/results_score_e_krr_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv",
		"11*10*23-21_CuAlZnTiMoGa___ofm1_no_d_uniform/results_score_u_gp_select_None_norm_True_is_search_params_True_stand_False_000/query_01/m0.1_c0.1.csv"
		]
	query_list = [indir+k for k in query_list]

	savedir = "/Volumes/Nguyen_6TB/work/SmFe12_screening/input/origin_struct/queries/"
	for qr_file in query_list:
		df = pd.read_csv(qr_file, index_col=0)
		update_DQ_str = df.loc[df["query2update_DQ"]=="query2update_DQ", "unlbl_index"].to_list()
		outstand_str = df.loc[df["query_outstanding"]=="query_outstanding", "unlbl_index"].to_list()
		random_str = df.loc[df["query_random"]=="query_random", "unlbl_index"].to_list()
		# print (update_DQ_str)
		pickup(struct_list=update_DQ_str, savedir=savedir)
		# break


if __name__ == "__main__":
	query2vasp()
