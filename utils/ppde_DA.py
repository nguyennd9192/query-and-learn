def plot_ppde(term, pv, estimator, X_train, y_train,
				X_all, xy, savedir):
	# fig = plt.subplots(nrows=1,  sharey=True)
	fig = plt.figure(figsize=(16, 8))	
	norm = mpl.colors.Normalize(vmin=0, vmax=20) # 
	cmap = cm.jet # gist_earth
	m = cm.ScalarMappable(norm=norm, cmap=cmap)
	c_list = ["black", "blue", "green", "purple", "brown", "gray", "black", "cyan", "lime",
	"darkgreen", "navy", "magenta"]
	# i = 0
	# for v, ax in zip(pv, axes.ravel()):

	estimator = estimator.fit(X_train, y_train)
	display = plot_partial_dependence(estimator, X_train, [0, 1],
		kind='average', n_jobs=3, grid_resolution=20,
    	line_kw=dict({"color": "red", "label":"formation energy", "linestyle":"-."}))
	ax = display.axes_

	for i, v in enumerate(pv):
		z_values = X_all[:, i]

		if len(set(z_values)) >1 and term in v:
			# try:
				# scatter_plot_6(x=xy[:, 0], y=xy[:, 1], 
				# 	z_values=z_values,
				# 	list_cdict=list_cdict, 
				# 	xvlines=[0.0], yhlines=[0.0], 
				# 	sigma=None, mode='scatter', lbl=None, name=None, 
				# 	s=60, alphas=alphas, 
				# 	title=save_file.replace(ALdir, ""),
				# 	x_label=FLAGS.embedding_method + "_dim_1",
				# 	y_label=FLAGS.embedding_method + "_dim_2", 
				# 	interpolate=False, cmap="seismic",
				# 	save_file=save_file,
				# 	preset_ax=None, linestyle='-.', marker=marker_array,
				# 	vmin=None, vmax=None
				# 	)
			save_file= savedir + "{0}.pdf".format(term)
			tmp_estimator = copy.copy(estimator)
			tmp_estimator = tmp_estimator.fit(xy, z_values)

			display = plot_partial_dependence(tmp_estimator, xy, [0, 1],
				   kind='average', n_jobs=3, grid_resolution=20,
    				ax=ax, line_kw=dict({"color": c_list[i % (len(c_list))], "label":v})) # m.to_rgba(i)
			ax = display.axes_
			f = display.figure_
			# f.suptitle(v)
			for j, a in enumerate(ax.flatten()):
				# if j == 0:
				# 	a.set_ylabel(v, rotation=0)
				# else:
				# 	a.set_ylabel("")
				if j == 1:
					a.set_ylabel("")
					a.legend(loc='center left', bbox_to_anchor=(1, 0.5))
				else:
					a.get_legend().remove()
				a.set_ylim([-0.1, 0.7])
				# a.set_xlabel("")
				# a.set_xticklabels([])
				# a.set_yticklabels([])
			# plt.ylabel(str(v))
			# plt.legend()

			print (str(v))
			makedirs(save_file)
			plt.savefig(save_file, transparent=False, bbox_inches="tight")

				# except:
				# 	pass
				# i += 1
				# if i == n_plots:
				# 	break
	release_mem(fig=fig)