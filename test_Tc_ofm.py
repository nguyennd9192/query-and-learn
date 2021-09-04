from utils.plot import *
import pandas as pd 



filename = "/Users/nguyennguyenduong/Dropbox/6-Nguyen_Quyet/data/Tc_superconductivity/TC_data_101_max_fix_ofm.csv"
reducename = "/Users/nguyennguyenduong/Dropbox/6-Nguyen_Quyet/data/Tc_superconductivity/TC_data_101_max_fix_ofm_reduce.csv"

savedir = "/Users/nguyennguyenduong/Dropbox/6-Nguyen_Quyet/data/Tc_superconductivity/plot"

df = pd.read_csv(filename, index_col=0)
for cc in df.columns:
	nuq = len(np.unique(df[cc].values))
	if nuq <= 1:
	  df = df.drop([cc], axis=1)


variables = list(df.columns)
print (variables)
print (len(variables))
df.to_csv(reducename)

for v in variables:
	if v == "s2-d1":
		x = df[v].values
		y = df["Tc"].values
		save_file = "{0}/{1}.pdf".format(savedir, v)
		scatter_plot(x, y, xvline=None, yhline=None, 
		    sigma=None, mode='scatter', lbl=None, name=df.index, 
		    x_label='x', y_label='y', 
		    save_file=save_file, interpolate=False, color='blue', 
		    linestyle='-.', marker='o', title=None)



