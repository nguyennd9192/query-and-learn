

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

	fdir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/SmFe12/train_energy_substance_pa/uniform/u_gp/MLKR/update_all/1.0/True/trial_4/DQ/autism/"
	fn = fdir + "idp_yobs.csv"
	df = pd.read_csv(fn, index_col=0)

	X = df.values
	scaler = MinMaxScaler()
	X_scale = scaler.fit_transform(X)

	X_scale = copy.copy(X)

	df[df.columns] = X_scale
	save_file = fn.replace(".csv", ".pdf")
	# plot_heatmap(matrix=df, vmin=None, vmax=None, 
	# 	save_file=save_file, 
	# 		cmap="jet", lines=None, title=None)

	c_dict = dict({
			"s1":"darkblue", "s2":"green",
			"p1":"purple", "d2":"cyan", "d5":"red", 
			"d10":"darkgreen", "d6":"blue", "d7":"brown", "f6":"orange",
			})

	marker_dict = dict({
		"s1":"^", "s2":"s",
			"p1":"<", "d2":">", "d5":"p", 
			"d10":"x", "d6":"+", "d7":"*", "f6":"D",
			"ofcenter":"o"
		})
	indexes = df.index
	save_file = fn.replace(".csv", "_line.pdf")

	terms = [ 
		"of", "s1-", "s2-",
		"p1-", "d2-", "d5-", "d6-", "d7-", "d10-", "f6-",
		"d10-"
		]

	# terms = [ 
	# 	# "of", "-s1", "-s2",
	# 	# "-p1", "-d2", "-d5", "-d6", "-d7", "-d10", "-f6",
	# 	# "-d10"
	# 	# "d5-p1", "p1-d5"
	# 	]
	query_index = np.array(range(X_scale.shape[1])).reshape(-1,1)

	for term in terms:
		fig, main_ax = plt.subplots(figsize=(12, 8), linewidth=1.0) # 
		norm = mpl.colors.Normalize(vmin=0, vmax=len(indexes)) # 
		cmap = cm.jet # gist_earth
		m = cm.ScalarMappable(norm=norm, cmap=cmap)
		save_file = fdir +"terms/{0}.pdf".format(term) 
		ith = 0
		for idx, v in zip(indexes, X_scale):
			print (idx, v[0], v[-1])
			# if np.max(v) < 0.9:
			# if v[0] > v[-1]:
			if term not in idx:
				c="gray"
				label = None
				marker = "."
				main_ax.plot(v, linestyle='-', marker=marker, alpha=0.2, linewidth=1,
					c=c, label=label)

		for idx, v in zip(indexes, X_scale):
			print (idx, v[0], v[-1])
			# if np.max(v) < 0.9:
			# if v[0] > v[-1]:
			if term in idx:
				center  = idx[:idx.find("-")]
				nb = idx[idx.find("-")+1:]
				marker = marker_dict[nb]
				c = c_dict[center]
				# c = m.to_rgba(ith)

				label = idx
				# main_ax.plot(v, linestyle='-', marker=marker, alpha=1.0, linewidth=2,
				# 	c=c, label=label, markersize=10, mfc='none')
				main_ax = curve_fit(x=query_index, y=v, 
				ax=main_ax, label=label, marker=marker, c=c, is_fit=False)
		
			
			ith += 1

		plt.tight_layout(pad=1.1)
		main_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		makedirs(save_file)
		plt.savefig(save_file, transparent=False, bbox_inches="tight")

		print ("Save at: ", save_file)
		release_mem(fig=fig)


	# # for down trend
	fig, main_ax = plt.subplots(figsize=(12, 8), linewidth=1.0) # 
	norm = mpl.colors.Normalize(vmin=0, vmax=len(indexes)) # 
	cmap = cm.jet # gist_earth
	m = cm.ScalarMappable(norm=norm, cmap=cmap)
	save_file = fdir +"terms/up_trend.pdf" # down_trend
	ith = 0

	for idx, v in zip(indexes, X_scale):
		print (idx, v[0], v[-1])
		if v[0] > v[59]:
			c = "gray"
			label = None
			marker = "."
			main_ax.plot(v, linestyle='-', marker=marker, alpha=0.2, linewidth=1,
				c=c, label=label)
		else:
			center  = idx[:idx.find("-")]
			nb = idx[idx.find("-")+1:]
			marker = marker_dict[nb]
			c = c_dict[center]
			print ("Here", idx)
			print ("Here", v, len(v))

			main_ax = curve_fit(x=query_index, y=v, 
				ax=main_ax, label=label, marker=marker, c=c, is_fit=False)
			
			# main_ax.plot(v, linestyle='-', marker=marker, alpha=1.0, linewidth=2,
			# 		c=c, label=label, markersize=10, mfc='none')

	plt.tight_layout(pad=1.1)
	main_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
	makedirs(save_file)
	plt.savefig(save_file, transparent=False, bbox_inches="tight")

	print ("Save at: ", save_file)
	release_mem(fig=fig)
	print (X_scale.shape)













