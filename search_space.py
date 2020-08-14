import numpy as np
import operator as op
from functools import reduce
from utils.plot import makedirs
import matplotlib.pyplot as plt


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
	formula = [r"""SmFe_%sM_%s""" % (a, b) for a, b in zip(nFe_rest, nsubs)]
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

	fig = plt.figure(figsize=(10, 8))
	axis_font = {'fontname': 'serif', 'size': 14, 'labelpad': 10}
	title_font = {'fontname': 'serif', 'size': 14}
	size_text = 10
	alpha_point = 0.8
	size_point = 160
	c_12 = "blue"
	c_24 = "red"

	# # plot config 12
	nsubs_12, nFe_rest_12, formula_12, n_structures_12, form_nck_12 = info12
	plt.plot(nsubs_12, n_structures_12, # s=size_point, edgecolor="black",
		alpha=alpha_point, 
		c=c_12, label="1-12", marker="o", linestyle="-.")
	for x, y, n in zip(nsubs_12, n_structures_12, formula_12):
		plt.annotate(n, xy=(x, y), size=size_text)

	# # plot config 12
	nsubs_24, nFe_rest_24, formula_24, n_structures_24, form_nck_24 = info24
	plt.plot(nsubs_24, n_structures_24, # s=size_point, edgecolor="black",
		alpha=alpha_point, 
		c=c_24, label="1-12", marker="p", linestyle="-.")
	for x, y, n in zip(nsubs_24, n_structures_24, formula_24):
		plt.annotate(n, xy=(x, y), size=size_text)

	plt.yscale("log", nonposy='clip')

	# # plot mix 12
	n_structures_12_mix = [nsub for nsub, nck in zip(nsubs_12, form_nck_12) if nsub < 5] # np.math.factorial(nsub)*nck
	plt.plot(nsubs_12, n_structures_12_mix, # s=size_point, edgecolor="black",
		alpha=alpha_point, 
		c=c_12, label="1-12", marker="o", linestyle="-.")
	for x, y, n in zip(nsubs_12, n_structures_12_mix, formula_12):
		plt.annotate(n.replace("M", "AB"), xy=(x, y), size=size_text)

	plt.ylabel(r'Number of substituted sites', **axis_font)
	plt.xlabel(r'Number of hypothetical structures', **axis_font)
	# plt.title(, **title_font)
	plt.style.use('default')
	plt.tick_params(axis='x', which='major', labelsize=13)
	plt.tick_params(axis='y', which='major', labelsize=13)

	plt.tight_layout(pad=1.1)
	plt.xticks(nsubs_24)
	
	save_file = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/results/search_space.pdf"
	makedirs(save_file)
	plt.savefig(save_file, transparent=False)
	print ("Save at: ", save_file)



if __name__ == "__main__":
	search_space()






