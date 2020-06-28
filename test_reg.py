


from utils import utils
from utils.uncertainty_regression import UncertainGaussianProcess, UncertainEnsembleRegression

from sklearn.metrics import r2_score

def get_train_test(X, y):
	n_train = int(len(y) * 0.9)
	X_train = X[:n_train, :]
	X_test = X[n_train:, :]
	y_train = y[:n_train]
	y_test = y[n_train:]
	print("n_test:", len(y_test))
	return X_train, y_train, X_test, y_test

def test_gaussian_process(X, y):
	X_train, y_train, X_test, y_test = get_train_test(X, y)
	gp = UncertainGaussianProcess(random_state=1, cv=3, n_times=3,
				search_param=True, verbose=False)
	gp.fit(X_train, y_train)
	
	y_pred = gp.predict(X_test, get_variance=False)
	y_prob = gp.predict_proba(X_test)

	r2 = r2_score(y_pred, y_test)
	print("gp best params: ", gp.estimator.get_params())
	print("y_prob: ", y_prob)
	print("r2 gaussian: ", r2)

def test_ensemble(X, y):
	X_train, y_train, X_test, y_test = get_train_test(X, y)
	ens_reg = UncertainEnsembleRegression(score_method="kr",
		n_shuffle=1000,
		random_state=1, cv=3, n_times=3, search_param=True, verbose=False)

	ens_reg.fit(X_train, y_train)
	y_pred = ens_reg.predict(X_test, get_pred_vals=False)
	r2 = r2_score(y_pred, y_test)

	print("gp best params: ", gp.estimator.get_params())
	print("y_prob: ", y_prob)
	print("r2 ensemble krr: ", r2)

if __name__ == "__main__":
	data_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data"
	dataset = "latbx_ofm1_fe"
	X, y = utils.get_mldata(data_dir, dataset)
	test_ensemble(X, y)
	# test_gaussian_process(X, y)
