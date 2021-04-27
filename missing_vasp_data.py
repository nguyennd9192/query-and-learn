import numpy as np
import pandas as pd
from shutil import copyfile
def get_org_struct(idx):
	# /mix/Sm-Fe9-Cu1-Mo2/ofm1_no_d/Mo_1-8___Cu_0.ofm1_no_d
	idx = idx.replace("/feature/", "/origin_struct/")
	idx = idx.replace("/ofm1_no_d/", "/")
	from_file = idx.replace(".ofm1_no_d", ".poscar")

	org_struct = "/Volumes/Nguyen_6TB/work/SmFe12_screening/input/origin_struct/"
	to_dir = org_struct + "queries/supp_11/"

	file_name = from_file.replace(org_struct, "").replace("/", "-_-")
	to_file = to_dir + file_name
	print (from_file)
	print (to_file)
	copyfile(from_file, to_file)

	print ("====")


	# /Volumes/Nguyen_6TB/work/SmFe12_screening/input/coarse_relax/queries/mix
	return idx

if __name__ == '__main__':
	data_file = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data/SmFe12/test_energy_substance_pa.csv"
	df = pd.read_csv(data_file, index_col=0)
	nan_index = df['energy_substance_pa'].index[df['energy_substance_pa'].apply(np.isnan)]
	print (nan_index)
	for k in nan_index:
		get_org_struct(k)