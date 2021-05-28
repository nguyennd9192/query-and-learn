

from absl import app
from proc_results import main_proc, video_for_tunning
import pandas as pd
from utils.plot import * 
from sklearn.preprocessing import MinMaxScaler

# from rank_unlbl  import rank_unlbl_data, map_unlbl_data

def model_selection(argv):
	extend_save_idx = run() 

	# # to process learning curve
	# extend_save_idx = "002"
	main_proc(extend_save_idx)

	# # to process video for tunning
	# video_for_tunning(ith_trial=extend_save_idx)

	# # rank unlabel data 
	# rank_unlbl_data(ith_trial=extend_save_idx)

	# map_unlbl_data(ith_trial=extend_save_idx)


if __name__ == "__main__":
	# app.run(model_selection) 

	fn = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/SmFe12/train_energy_substance_pa/margin/u_gp/MLKR/update_all/1.0/True/trial_1/DQ/autism/idp_yobs.csv"
	df = pd.read_csv(fn, index_col=0)

	X = df.values
	scaler = MinMaxScaler()
	X_scale = scaler.fit_transform(X)
	df[df.columns] = X_scale
	save_file = fn.replace(".csv", ".pdf")
	# plot_heatmap(matrix=df, vmin=None, vmax=None, 
	# 	save_file=save_file, 
	# 		cmap="jet", lines=None, title=None)

	c_dict = dict({
			"s1":"darkblue", "s2":"green",
			"p1":"purple", "d2":"cyan", "d5":"red", 
			"d10":"darkgreen", "d6":"gray", "d7":"brown", "f6":"orange",
			})

	marker_dict = dict({
		"s1":"^", "s2":"s",
			"p1":"<", "d2":">", "d5":"p", 
			"d10":"x", "d6":"+", "d7":"*", "f6":"D",
			"ofcenter":"o"
		})
	indexes = df.index
	save_file = fn.replace(".csv", "_line.pdf")
	fig, main_ax = plt.subplots(figsize=(11, 8), linewidth=1.0) # 

	norm = mpl.colors.Normalize(vmin=0, vmax=len(indexes)) # 
	cmap = cm.jet # gist_earth
	m = cm.ScalarMappable(norm=norm, cmap=cmap)

	ith = 0
	for idx, v in zip(indexes, X_scale):
		print (np.max(v), idx)
		if np.max(v) < 0.5:
			c="gray"
			label = None
			marker = "."
		else:
			center  = idx[:idx.find("-")]
			nb = idx[idx.find("-")+1:]

			marker = marker_dict[nb]
			c = c_dict[center]
			# c = m.to_rgba(ith)

			label = idx
	
		main_ax.plot(v, linestyle='-.',
			# s=80, 
			marker=marker, 
			alpha=0.8, 
			c=c, 
			label=label,
			)
		ith += 1

	plt.tight_layout(pad=1.1)
	main_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	plt.savefig(save_file, transparent=False, bbox_inches="tight")

	print ("Save at: ", save_file)
	release_mem(fig=fig)

















