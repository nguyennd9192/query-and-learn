


from utils import utils
from utils.uncertainty_regression import UncertainGaussianProcess, UncertainEnsembleRegression, UncertainMetricLearningRegression
from utils.mixture_of_experts import *
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt


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
	X_train, y_train, X_test, y_test = get_train_test(X, y)

	# var = predict_proba(model=model, test_dl=test_dl, T=10)
	NN_kwargs = dict({
		"method": "moe",
		"n_epoches":10,
		"batch_size":10, "lr": 0.01, "momentum":0.9,
		# #
		"num_experts": 5,
		"hidden_dim": 15,

		})

	model = NN_estimator(NN_kwargs=NN_kwargs)
	# model = MixtureOfExperts_mpt()


	model.fit(X_train, y_train)
	print('X_test.shape: ', X_test.shape)

	y_pred, var = model.predict_proba(X_test)
	print('y_pred: ', len(y_pred))
	mae = mean_absolute_error(y_pred, y_test)
	print('mae: ', mae)
	print('var: ', var)

	# y_pred = moe.predict(X_test, get_variance=False)
	# r2 = r2_score(y_pred, y_test)
	# print("moe best params: ", moe.estimator.get_params())
	# print("score on train: ", moe.estimator.score(X, y))
	# print("score on test: ", r2)


def test_mlkr(X, y):
	X_train, y_train, X_test, y_test = get_train_test(X, y)
	model = UncertainMetricLearningRegression(random_state=1, cv=10, n_times=3,
				search_param=False, verbose=False)

	# y_train_lbl = np.array(list(map(str, np.around(y_train,1))))

	y_train_lbl = y_train
	# print (y_train_lbl)
	model.fit(X_train, y_train_lbl )
	
	X_train_embedded = model.transform(X_train)
	X_test_embedded = model.transform(X_test)

	xtrain_embedded, ytrain_embedded = X_train_embedded[:, 0], X_train_embedded[:, 1]
	xtest_embedded, ytest_embedded = X_test_embedded[:, 0], X_test_embedded[:, 1]

	print ("X_test_embedded", X_test_embedded)
	print ("X_test_embedded.shape", X_test_embedded.shape)
	print ("test min max", min(xtest_embedded), max(xtest_embedded), min(ytest_embedded), max(ytest_embedded))
	print ("train min max", min(xtrain_embedded), max(xtrain_embedded), min(ytrain_embedded), max(ytrain_embedded))


	fig = plt.figure(figsize=(8, 8))
	plt.scatter(xtrain_embedded, ytrain_embedded, 
		s=50, alpha=0.8, c="b", label="train",
		edgecolor="black")
	for a, b, c in zip(xtrain_embedded, ytrain_embedded, y_train):
		plt.text(a, b, round(c,2))


	plt.scatter(xtest_embedded, ytest_embedded, 
		s=50, alpha=0.8, c="r", label="test",
		edgecolor="black")
	for a, b, c in zip(xtest_embedded, ytest_embedded, y_test):
		plt.text(a, b, round(c,2))
	plt.legend()
	plt.savefig("test_mlkr.pdf", transparent=False)

	y_pred, _ = model.predict(X_test, get_variance=True)
	y_prob = model.predict_proba(X_test)

	r2 = r2_score(y_pred, y_test)
	mae = mean_absolute_error(y_pred, y_test)

	print("model best params: ", model.estimator.get_params())
	print("score on train: ", model.score(X, y))
	print("y_prob: ", y_prob)

	print("score on test: ", r2)
	print("mae on test: ", mae)

if __name__ == "__main__":
	data_dir = "/Users/nguyennguyenduong/Dropbox/My_code/active-learning-master/data"
	test_prefix = "Fe10-Fe22"
	dataset = "11*10*23-21_CuAlZnTiMoGa___ofm1_no_d"+"/train_"+test_prefix
	X, y, index = utils.get_mldata(data_dir, dataset)
	# test_ensemble(X, y)
	# test_gaussian_process(X, y)

	interested_cols = range(10, 13)
	X = X[:, interested_cols]
	test_mlkr(X, y)










