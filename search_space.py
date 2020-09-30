import numpy as np
import operator as op
from functools import reduce
from utils.plot import makedirs
import matplotlib.pyplot as plt
import math
from matplotlib import ticker


def ncr(n, r):
	r = min(r, n-r)
	numer = reduce(op.mul, range(n, n-r, -1), 1)
	denom = reduce(op.mul, range(1, r+1), 1)
	return numer // denom  # or / in Python 2

def get_info(nFe):
	# # n_subs
	nsubs = list(range(1, nFe))
	nFe_rest = [nFe - k for k in nsubs]
	
	# # formula, n calc
	formula = [r"""$SmFe_{%s}M_{%s}$""" % (a, b) for a, b in zip(nFe_rest, nsubs)]
	if nFe == 24:
		formula = [r"""$Sm_2Fe_{%s}M_{%s}$""" % (a, b) for a, b in zip(nFe_rest, nsubs)]

	n_structures = [ncr(n=nFe, r=nsub) for nsub in nsubs]
	form_nck = ["C_{0}^{1}".format(nFe, nsub) for nsub in nsubs]

	return nsubs, nFe_rest, formula, n_structures, form_nck


def search_space():
	# # SmFe12
	info12 = get_info(nFe=12)
	info24 = get_info(nFe=24)
	# # nA, nB
	# nAs = [range(1, nsub) for nsub in nsubs]
	# nBs = [nsub - nA for nsub, nA in zip(nsubs, nAs)]
	# print (n_subs_mix)

	# fig = plt.figure(figsize=(10, 8))
	fig, ax = plt.subplots(figsize=(10, 8))

	plt.style.use('default')

	axis_font = {'fontname': 'serif', 'size': 20, 'labelpad': 10}
	title_font = {'fontname': 'serif', 'size': 20}
	size_text = 10
	alpha_point = 0.8
	size_point = 18
	c_12 = "blue"
	c_24 = "red"

	# # config 12
	nsubs_12, nFe_rest_12, formula_12, n_structures_12, form_nck_12 = info12
	print (nsubs_12)
	print (12, "n_structures_12", n_structures_12)
	print ("form_nck_12", form_nck_12)
	# ax.plot(nsubs_12, n_structures_12, # s=size_point, edgecolor="black",
	# 	alpha=alpha_point, markersize=size_point,
	# 	c=c_12, label=r"$REFe_{12-x}A_{x}$", marker="o", linestyle="-.")
	# for x, y, n in zip(nsubs_12, n_structures_12, formula_12):
	# 	ax.annotate(n, xy=(x, y), size=size_text)
	# 	if x < 4:
	# 		ax.scatter(x, y, alpha=alpha_point, s=size_point, c=c_12, edgecolor="black")

	# # # config 24
	nsubs_24, nFe_rest_24, formula_24, n_structures_24, form_nck_24 = info24
	# ax.plot(nsubs_24, n_structures_24, # s=size_point, edgecolor="black",
	# 	alpha=alpha_point, markersize=size_point,
	# 	c=c_24, label=r"$RE_{2}Fe_{24-x}A_{x}$", marker="o", linestyle="-.")
	# for x, y, n in zip(nsubs_24, n_structures_24, formula_24):
	# 	ax.annotate(n, xy=(x, y), size=size_text)


	# # mix 12
	n_structures_12_mix = [(2**nsub)*nstr for nsub, nstr in zip(nsubs_12, n_structures_12)] # np.math.factorial(nsub)*nck
	ax.plot(nsubs_12, n_structures_12_mix, # s=size_point, edgecolor="black",
		alpha=alpha_point, markersize=size_point, 
		c="black",# c_12 
		mfc=None, 
		label=r"$REFe_{12-x-y}A_{x}B_{y}$", marker="o", linestyle="-.")
	# for x, y, n in zip(nsubs_12, n_structures_12_mix, formula_12):
	# 	ax.annotate(n.replace("M", "AB"), xy=(x, y), size=size_text)
	# 	if x < 4:
			# ax.scatter(x, y, alpha=alpha_point, s=size_point, c=c_12, edgecolor="black")



	# # mix 24
	n_structures_24_mix = [(2**nsub)*nstr for nsub, nstr in zip(nsubs_24, n_structures_24)] # np.math.factorial(nsub)*nck
	ax.plot(nsubs_24, n_structures_24_mix, # s=size_point, edgecolor="black",
		alpha=alpha_point, markersize=size_point, 
		c="black", # c_24
		mfc=None, 

		label=r"$RE_{2}Fe_{24-x-y}A_{x}B_{y}$", marker="^", linestyle="-.")
	# for x, y, n in zip(nsubs_24[:len(n_structures_24_mix)], n_structures_24_mix, formula_24):
	# 	ax.annotate(n.replace("M", "AB"), xy=(x, y), size=size_text)
	# 	if x < 4:
	# 		ax.scatter(x, y, alpha=alpha_point, s=size_point, c=c_12, edgecolor="black")
	
	# plt.yscale("log") # , nonposy='clip'
	# subs = list(range(2, 10))
	# ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs))
	ax.set_yscale("log", nonposy='clip')

	# ax2 = ax.twinx()
	# ax2.plot(nsubs_24, n_structures_24_mix, # s=size_point, edgecolor="black",
	# 	alpha=alpha_point, 
	# 	c=c_24, label=r"$Sm_{2}Fe_{24-x}AB_{x}$", marker="p", linestyle="-.")
	# ax2.set_yscale("log", nonposy='clip')


	plt.xlabel(r'Number of substituted sites, x+y', **axis_font)
	plt.ylabel(r'Number of hypothetical structures', **axis_font)
	# plt.title(, **title_font)
	plt.legend(prop={'size': 18})
	plt.tick_params(axis='x', which='major', labelsize=18)
	plt.tick_params(axis='y', which='major', labelsize=18)

	plt.tight_layout(pad=1.1)
	plt.xticks(nsubs_24)
	plt.yticks([10**k for k in range(1, 12)])

	
	save_file = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/search_space_reduce.pdf"
	makedirs(save_file)
	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)



if __name__ == "__main__":
	search_space()






