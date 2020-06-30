


from utils import utils
from utils.uncertainty_regression import UncertainGaussianProcess, UncertainEnsembleRegression

from sklearn.metrics import r2_score

def get_train_test(X, y):
	n_train = int(len(y) * 0.5)
	X_train = X[:n_train, :]
	X_test = X[n_train:, :]
	y_train = y[:n_train]
	y_test = y[n_train:]
	print("n_test:", len(y_test))
	return X_train, y_train, X_test, y_test

def test_gaussian_process(X, y):
	X_train, y_train, X_test, y_test = get_train_test(X, y)
	gp = UncertainGaussianProcess(random_state=1, cv=10, n_times=3,
				search_param=True, verbose=False)
	gp.fit(X_train, y_train)
	
	y_pred = gp.predict(X_test, get_variance=False)
	y_prob = gp.predict_proba(X_test)

	r2 = r2_score(y_pred, y_test)
	print("gp best params: ", gp.estimator.get_params())
	print("y_prob: ", y_prob)
	print("score on train: ", gp.score(X, y))
	print("score on test: ", r2)

def test_ensemble(X, y):
	X_train, y_train, X_test, y_test = get_train_test(X, y)
	ens_reg = UncertainEnsembleRegression(score_method="kr",
		n_shuffle=1000,
		random_state=1, cv=3, n_times=3, search_param=True, verbose=False)

	ens_reg.fit(X_train, y_train)
	y_pred = ens_reg.predict(X_test, get_pred_vals=False)
	r2 = r2_score(y_pred, y_test)

	print("ens_reg best params: ", ens_reg.estimator.get_params())
	print("score on train: ", ens_reg.estimator.score(X, y))
	print("score on test: ", r2)

if __name__ == "__main__":
	data_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data"
	test_prefix = "Fe10-Fe22"
	dataset = "11*10*23-21_CuAlZnTiMoGa___ofm1_no_d"+"/train_"+test_prefix
	X, y = utils.get_mldata(data_dir, dataset)
	test_ensemble(X, y)
	# test_gaussian_process(X, y)
