

import matplotlib.pyplot as plt

def release_mem(fig):
	fig.clf()
	plt.close()
	gc.collect()



def ax_setting():
	plt.style.use('default')
	plt.tick_params(axis='x', which='major', labelsize=13)
	plt.tick_params(axis='y', which='major', labelsize=13)


def makedirs(file):
	if not os.path.isdir(os.path.dirname(file)):
		os.makedirs(os.path.dirname(file))


def scatter_plot(x, y, xvline=None, yhline=None, 
	sigma=None, mode='scatter', lbl=None, name=None, 
	x_label='x', y_label='y', 
	save_file=None, interpolate=False, color='blue', 
	preset_ax=None, linestyle='-.', marker='o'):
	if preset_ax is not None:
		fig = plt.figure(figsize=(8, 8))

	if 'scatter' in mode:
		plt.scatter(x, y, s=80, alpha=0.8, 
		marker=marker, c=color, edgecolor="white") # brown

	if 'line' in mode:
		plt.plot(x, y,  marker=marker, linestyle=linestyle, color=color,
		 alpha=1.0, label=lbl, markersize=10, mfc='none')

	if xvline is not None:
		plt.axvline(x=xvline, linestyle='-.', color='black')
	if yhline is not None:
		plt.axhline(y=yhline, linestyle='-.', color='black')

	if name is not None:
		for i in range(len(x)):
			# only for lattice_constant problem, 1_Ag-H, 10_Ag-He
			# if tmp_check_name(name=name[i]):
			   # reduce_name = str(name[i]).split('_')[1]
			   # plt.annotate(reduce_name, xy=(x[i], y[i]), size=5)
			plt.annotate(name[i], xy=(x[i], y[i]), size=size_text)
		

	plt.ylabel(y_label, **axis_font)
	plt.xlabel(x_label, **axis_font)
	ax_setting()

	if preset_ax is not None:
		

		# min_y, max_y = np.min(y), np.max(y)
		# dy = max_y - min_y
		# min_y -= 0.2*dy
		# max_y += 0.2*max_y
		# plt.ylim([min_y, max_y])

		# plt.grid(linestyle='--', color="gray", alpha=0.8)
		plt.legend(prop={'size': 16})
		makedirs(save_file)
		plt.savefig(save_file)
		release_mem(fig=fig)


scatter_plot(x=x[plot_idx] + 1, y=acc_cv_train[plot_idx], xvline=None, yhline=None, 
				sigma=None, mode='scatter', lbl="m:{0}__c{1}".format(m, c), name=None, 
				x_label='n update', y_label='accuracy', 
				preset_ax=None, save_file=None, interpolate=False, linestyle='-.',
				 color=colors[m], marker=markers[c]
				)
print("data_sizes:", result_dict["data_sizes"])
print("cv", result_dict["cv_train_model"])

plt.xticks(x[plot_idx] + 1)
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc='lower left', 
	mode="expand", borderaxespad=0, ncol=3, fontsize=12)
saveat = result_dir + "/" + filename + "_" + ith_trial +  "/" + "accuracy.pdf"
print ("saveat:", saveat)
makedirs(saveat)

init_data_size = result_key_to_text["data_sizes"][0]
batch_data_size = result_key_to_text["data_sizes"][1] - init_data_size

final_data_size = result_key_to_text["data_sizes"][-1]


text = "\n".join(["{0}: {1}".format(k, v) for k, v in exp_params.items() if k != "m" and k != "c"])
text += "\nm: Mix weights on ActSamp\n"
text += "c: percent of rand labels\n"

# text += "\n".join(["{0}: {1}".format(k, v) for k, v in result_key_to_text.items() if k != "data_sizes"])

text += "batch data size: " + str(batch_data_size) + "\n"
text += "init train size: "  + str(init_data_size)  + " cv: " + str(round(result_key_to_text["cv_train_model"][0], 2)) + "\n"
text += "final train size: " + str(final_data_size) + " cv: " + str(round(result_key_to_text["cv_train_model"][-1], 2)) +  "\n"
text += "test data size: " + str(result_key_to_text["n_test"]) + "\n"
text += "is test separate: " + str(result_key_to_text["is_test_separate"]) + "\n"
text += "test_prefix: " + str(result_key_to_text["test_prefix"]) + "\n"

text += "y_train_info: " + str(result_key_to_text["y_train_info"]) + "\n"
text += "y_val_info: " + str(result_key_to_text["y_val_info"]) + "\n"
text += "y_test_info: " + str(result_key_to_text["y_test_info"]) + "\n"

# org_data_size

print(result_key_to_text["y_train_info"])
print(str(result_key_to_text["data_sizes"]))
# plt.text(35.5, 0.5, text, fontsize=14)

side_text = plt.figtext(0.91, 0.12, text, bbox=dict(facecolor='white'))
fig.subplots_adjust(top=0.8)
# plt.title(text, fontsize=14)
# plt.tight_layout(pad=1.2)
# fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(saveat, bbox_inches='tight')