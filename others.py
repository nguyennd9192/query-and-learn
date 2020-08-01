import pandas as pd
import glob, shutil, os
from general_lib import *

def merge_df():
	maindir = "/media/nguyen/work/SmFe12_screening/result/coarse_relax/Fe-Ti_Co"

	name_dfs = [ "Co_10" , 
				"Co_11", "Co_12", "Co_13", "Co_14", 
				"Co_15", "Co_16", "Co_17", "Co_18", "Co_19"]

	dfs = []
	for n in name_dfs:
		filename = "{0}/{1}.csv".format(maindir, n)
		this_df = pd.read_csv(filename, index_col=0)
		dfs.append(this_df)
	df_merged = reduce(lambda  left, right: pd.merge(left, right,
						left_index=True, right_index=True,
						on=list(left.columns), how='outer'), dfs)

	saveat = "{0}/merge.csv".format(maindir)
	df_merged.to_csv(saveat)
	print ("Save at: ", saveat)

def vasp_viz_reorder_dir():
	vasp_tasks = ["coarse_relax", "fine_relax", "standard"]

	result_dir = "/media/nguyen/work/SmFe12_screening/result"
	jobs = ["NdFeB_187_norelax_results"] # NdFeB_187_norelax_results, NdFeB_0716reproduce
	for current_job in jobs:
		subs_dirs = glob.glob('{0}/{1}/*'.format(result_dir, current_job))
		for subs_dir in subs_dirs:
			basename = get_basename(subs_dir)
			for task in vasp_tasks:
				fromfld = subs_dir+"/"+task
				basename_revise = basename.replace("+", "___").replace("*","-")
				tofld = result_dir+"/"+task+"/"+current_job+"/"+basename_revise
				print "from fld", fromfld
				print "to fld", tofld
				makedirs(tofld)
				if os.path.exists(fromfld):
					shutil.move(fromfld, tofld)
				# break



if __name__ == "__main__":
	# merge_df()
	vasp_viz_reorder_dir()
	