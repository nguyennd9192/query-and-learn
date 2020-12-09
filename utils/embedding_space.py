
from sklearn import neighbors
import metric_learn as mkl
import numpy as np
from sklearn.metrics import pairwise_distances


class EmbeddingSpace(object):
	def __init__(self, embedding_method):
		self.embedding_method = embedding_method
		if embedding_method == "MLKR":
			learn_metric = mkl.MLKR(n_components=2, init="auto")
		elif embedding_method == "LFDA":
			learn_metric = mkl.LFDA(n_components=2, 
		 		k=10, embedding_type="orthonormalized") # weighted, orthonormalized
		elif embedding_method == "LMNN":
			learn_metric = mkl.LMNN(k=3, learn_rate=0.1) # 
		self.learn_metric = learn_metric
	# learn_metric = mkl.ITML_Supervised()
	# learn_metric = mkl.SDML_Supervised(sparsity_param=0.1, balance_param=0.0015,
	#           prior='covariance')

	def fit(self, X_train, y_train):
		n_features = X_train.shape[1]

		self.learn_metric.fit(X_train, y_train)
		X_train_embedded = self.learn_metric.transform(X_train)

		self.X_train = X_train
		self.y_train = y_train
		self.X_train_embedded = X_train_embedded

	def transform(self, X_val, get_min_dist=False):
		X_val_transform = self.learn_metric.transform(X_val)
		if get_min_dist:
			# # return max distance of each X_val to self.X_train
			# nbs = [2, 5, 10, 20, 30]
			# y_preds = []
			# for nb in nbs:
			#   estimator = neighbors.KNeighborsRegressor(nb, weights="distance") # 'uniform', 'distance'
			#   estimator.fit(self.X_train_embedded, self.y_train)

			#   y_pred = estimator.predict(X_val_transform)
			#   y_preds.append(y_pred)
			# y_preds = np.array(y_preds)
			# return np.mean(y_preds, axis=0), # np.var(y_preds, axis=0)
			distances = pairwise_distances(X_val, self.X_train)
			max_distances = np.min(distances, axis=1)
			return X_val_transform, max_distances
		return X_val_transform

	def predict_proba(self, X, is_norm=True):
		# # large variance -> probability to be observed small -> sorting descending take first
		# # small variance -> probability to be observed large 
		X_val_transform, variance = self.transform(X, get_min_dist=True)

		# # normalize variance to 0-1
		var_norm = MinMaxScaler().fit_transform(X=variance.reshape(-1, 1))
		if is_norm:
		  return var_norm.ravel()
		else:
		  return variance.reshape(-1, 1)


