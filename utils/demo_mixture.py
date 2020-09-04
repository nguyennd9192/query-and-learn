
# example of a bimodal data sample
import matplotlib.pyplot as plt 
from numpy.random import normal
from numpy import hstack
import numpy as np
from scipy import stats
from scipy.signal import find_peaks, peak_prominences

# generate a sample
sample1 = normal(loc=20, scale=5, size=300)
sample2 = normal(loc=40, scale=5, size=700)
sample = hstack((sample1, sample2))
kernel = stats.gaussian_kde(sample)
yref = np.linspace(min(sample),max(sample),100)
pdf = kernel(yref).T

# # find peaks
peaks, _ = find_peaks(pdf)
pr = peak_prominences(pdf, peaks)[0]
argmax1, argmax2 = np.argsort(pr)[-2:] # # two largest prominence
y_pred_peak1, y_pred_peak2 = yref[argmax1], yref[argmax2]
print (y_pred_peak1, y_pred_peak2)
# # variance between two largest peaks
v = y_pred_peak2 - y_pred_peak1
# plt.hist(sample, bins=50)

sample_normal = normal(loc=30, scale=5, size=700)
v_normal = np.var(sample_normal, axis=0)

# plt.hist(sample_normal, bins=50)
plt.plot(yref, pdf)
print(v, v_normal)
plt.show()
