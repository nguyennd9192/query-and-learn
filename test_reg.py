


from utils import utils
from utils.uncertainty_regression import UncertainGaussianProcess, UncertainEnsembleRegression
from utils.mixture_of_experts import *
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error


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


def test_MoE(X, y):
	# X_train, y_train, X_test, y_test = get_train_test(X, y)

	# load the dataset
	dataset = CSVDataset(X.astype(float), y.astype(float))
	# calculate split
	train, test = dataset.get_splits(n_test=0.33)
	# print ("train_dl.shape:", train.shape)
	# print ("test_dl.shape:", test.shape)


	train_dl = DataLoader(train, batch_size=32, shuffle=True)
	test_dl = DataLoader(test, batch_size=10, shuffle=False)
	# load model
	# kwargs = dict({"dim":X_train.shape[1], "num_experts":3, "hidden_dim":"default"})
	# model = MixtureOfExperts(kwargs=kwargs)


	model = MLP(n_inputs=85)
	# train the model
	train_model(train_dl, model)
	# evaluate the model
	error, pred, actuals = evaluate_model(test_dl, model)
	print('Accuracy: ', error)

	var = predict_proba(model=model, test_dl=test_dl, T=10)
	print('var: ', var)

	# moe.fit(X_train, y_train)

	# y_pred = moe.predict(X_test, get_variance=False)
	# r2 = r2_score(y_pred, y_test)
	# print("moe best params: ", moe.estimator.get_params())
	# print("score on train: ", moe.estimator.score(X, y))
	# print("score on test: ", r2)


if __name__ == "__main__":
	data_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data"
	test_prefix = "Fe10-Fe22"
	dataset = "11*10*23-21_CuAlZnTiMoGa___ofm1_no_d"+"/train_"+test_prefix
	X, y, index = utils.get_mldata(data_dir, dataset)
	# test_ensemble(X, y)
	# test_gaussian_process(X, y)
	test_MoE(X, y)










