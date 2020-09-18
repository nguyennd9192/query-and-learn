
import os, glob, ntpath

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


def get_subdirs(sdir):
	subdirs = glob.glob(sdir+"/*")
	return subdirs

def get_basename(filename):
    head, tail = ntpath.split(filename)
    basename = os.path.splitext(tail)[0]
    return tail

def merge_two_dicts(x, y):
    """Given two dictionaries, merge them into a new dict as a shallow copy."""
    z = x.copy()
    z.update(y)
    return z