
import torch
from torch import nn
from mixture_of_experts import MoE
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
from torch.nn import Linear, Dropout
from torch.nn import ReLU
from torch.nn import Sigmoid
from torch.nn import Module

from torch.nn import MSELoss
from torch.optim import SGD
import torch
from numpy import vstack
from sklearn.metrics import r2_score, mean_absolute_error

from torch.nn.init import kaiming_uniform_
from torch.nn.init import xavier_uniform_



# dataset definition
class CSVDataset(Dataset):
	# load the dataset
	def __init__(self, X, y):
		# store the inputs and outputs
		self.X = X
		self.y = y

	# number of rows in the dataset
	def __len__(self):
		return len(self.X)

	# get a row at an index
	def __getitem__(self, idx):
		return [self.X[idx], self.y[idx]]

	# get indexes for train and test rows
	def get_splits(self, n_test=0.33):
		# determine sizes
		test_size = round(n_test * len(self.X))
		train_size = len(self.X) - test_size
		# calculate the split
		return random_split(self, [train_size, test_size])


class MixtureOfExperts(object):
	def __init__(self,	random_state=1, cv=3, n_times=3, 
		search_param=False,
		verbose=False, kwargs=None):

		self.search_param = search_param
		self.kernel = 'rbf'
		self.verbose = verbose
		self.cv = cv
		self.n_times = n_times

		if kwargs["hidden_dim"] == "default":
			hidden_dim = kwargs["num_experts"] * 4
		else:
			hidden_dim = kwargs["hidden_dim"]

		self.layer = Linear(n_inputs, 1)
		self.activation = Sigmoid()
		moe = MoE(
				dim = kwargs["dim"],
				num_experts = kwargs["num_experts"],               # increase the experts (# parameters) of your model without increasing computation
				hidden_dim = hidden_dim,           # size of hidden dimension in each expert, defaults to 4 * dimension
				activation = nn.LeakyReLU,      # use your preferred activation, will default to GELU
				second_policy_train = 'random', # in top_2 gating, policy for whether to use a second-place expert
				second_policy_eval = 'random',  # all (always) | none (never) | threshold (if gate value > the given threshold) | random (if gate value > threshold * random_uniform(0, 1))
				second_threshold_train = 0.2,
				second_threshold_eval = 0.2,
				capacity_factor_train = 1.25,   # experts have fixed capacity per batch. we need some extra capacity in case gating is not perfectly balanced.
				capacity_factor_eval = 2.,      # capacity_factor_* should be set to a value >=1
				loss_coef = 1e-2                # multiplier on the auxiliary expert balancing auxiliary loss
			)
		self.estimator = moe
		self.random_state = random_state


	def fit(self, X_train, y_train):
		# # in fit function
		# # just return estimator with best param with X_train, y_train
		np.random.seed(self.random_state)
		n_features = X_train.shape[1]

		X = self.layer(X)
		X = self.activation(X)

		criterion = MSELoss()
		optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
		# inputs = np.array([4, X_train]) # [X_train, y_train]

		out, aux_loss = self.estimator(inputs)
		# print (out)
		# print (aux_loss)
		# self.estimator.fit(X_train, y_train)
		return X


	def predict(self, X_val, get_variance=False):
		y_val_pred, y_val_pred_std = self.estimator.predict(X_val, return_std=True, return_cov=False)
		if get_variance:
		  return y_val_pred, y_val_pred_std
		else:
		  return y_val_pred

	def score(self, X_val, y_val):
		y_pred = self.predict(X_val, get_variance=False)    
		val_acc = metrics.r2_score(y_val, y_pred)
		return val_acc

	def predict_proba(self, X, is_norm=True):
		# # large variance -> probability to be observed small -> sorting descending take first
		# # small variance -> probability to be observed large 
		y_val_preds, y_val_pred_std = self.predict(X, get_variance=True)

		# # normalize variance to 0-1
		var_norm = MinMaxScaler().fit_transform(X=y_val_pred_std.reshape(-1, 1))
		# var_norm = y_val_pred_std.reshape(-1, 1)
		# prob = 1 / var_norm
		if is_norm:
		  return var_norm.ravel()
		else:
		  return y_val_pred_std.reshape(-1, 1)


	def best_score_(self):
		# # some conflict meaning between best_score_ for GridSearchCV object and this attribute:
		# # GridSearchCV.best_score_ returns cv score of best parameter
		# # this UncertainGaussianProcess.best_score_returns cv score of given params
		if self.GridSearchCV is None:
		  r2, r2_std, mae, mae_std = CV_predict_score(self.estimator, self.X_train, self.y_train, 
					n_folds=3, n_times=3, score_type='r2')
		  result = r2
		else:
		  result = self.GridSearchCV.best_score_
		return result



def enable_dropout(m):
	for each_module in m.modules():
		if each_module.__class__.__name__.startswith('Dropout'):
			# print ("Enable module:", each_module.__class__.__name__)
			each_module.train()
	return m

# train the model
def train_model(train_dl, model):
	# define the optimization
	criterion = MSELoss()
	optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
	# enumerate epochs
	for epoch in range(100):
		# enumerate mini batches
		for i, (inputs, targets) in enumerate(train_dl):
			# clear the gradients
			optimizer.zero_grad()
			# compute the model output
			yhat = model(inputs.float())

			# calculate loss
			yhat = torch.reshape(yhat, targets.shape)

			loss = criterion(yhat, targets.float())
			# credit assignment
			loss.backward()
			# update model weights
			optimizer.step()
 
# evaluate the model
def evaluate_model(test_dl, model):
	predictions, actuals = list(), list()
	for i, (inputs, targets) in enumerate(test_dl):
		# evaluate the model on the test set
		yhat = model(inputs)
		# retrieve numpy array
		yhat = yhat.detach().numpy()
		actual = targets.numpy()
		actual = actual.reshape((len(actual), 1))
		# round to class values
		yhat = yhat.round()
		# store
		predictions.append(yhat)
		actuals.append(actual)
	predictions, actuals = vstack(predictions), vstack(actuals)
	print ("predictions.shape:", predictions.shape)
	error = mean_absolute_error(predictions, actuals)
	# calculate accuracy
	# acc = accuracy_score(actuals, predictions)
	return error, predictions, actuals

def uncertainties(p):
	aleatoric = np.mean(p*(1-p), axis=0)
	epistemic = np.mean(p**2, axis=0) - np.mean(p, axis=0)**2
	return aleatoric, epistemic

#----------------------------PREDICT-------------------------------------------
def predict_proba(model, test_dl, T):
	# enable_dropout(model)
	preds = []
	# predict stochastic dropout model T times
	for t in range(T):
		model = enable_dropout(model) # STILL NOT WORKING WITH DROPOUT AT TEST-TIME
		error, pred, actuals = evaluate_model(test_dl, model)
		preds.append(pred) # P( c = 0 | image)
	   
	# mean prediction
	var = np.var(preds)
	p_hat_lists = [preds]
	epistemic, aleatoric = uncertainties(np.array(p_hat_lists))

	print (epistemic, aleatoric)
	# estimate uncertainties (eq. 4 )
	# eq.4 in https://openreview.net/pdf?id=Sk_P2Q9sG
	# see https://github.com/ykwon0407/UQ_BNN/issues/1
	# p_hat_lists = 
	# epistemic, aleatoric = uncertainties(np.array(p_hat_lists[label]))

	return var


# model definition
class MLP(Module):
	# define model elements
	def __init__(self, n_inputs):
		super(MLP, self).__init__()
		# input to first hidden layer
		hyperparams = {'l1_out': 5,  'l2_out': 5,
				  'l1_drop': 0.1,  'l2_drop': 0.1,
				  'batch_size': 32,  'epochs': 10}
		self.hidden1 = Linear(n_inputs, 10)
		kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
		self.act1 = ReLU()
		self.dropout1 = Dropout(hyperparams['l1_drop'])

		# second hidden layer
		self.hidden2 = Linear(10, 8)
		kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
		self.act2 = ReLU()
		self.dropout2 = Dropout(hyperparams['l2_drop'])

		# third hidden layer and output
		self.hidden3 = Linear(8, 4)
		xavier_uniform_(self.hidden3.weight)
		self.act3 = Linear(4, 1)
 
	# forward propagate input
	def forward(self, X):
		# input to first hidden layer
		X = self.hidden1(X.float())
		X = self.act1(X)
		# X = self.dropout1(X)

		 # second hidden layer
		X = self.hidden2(X)
		X = self.act2(X)
		# X = self.dropout2(X)

		# third hidden layer and output
		X = self.hidden3(X)
		X = self.act3(X)
		return X



