import numpy as np
from decision_tree import DecisionTree

class RandomForest:
	def __init__(self, n_estimators=100, max_depth=None, criterion="gini",
				 max_features="sqrt", bootstrap=True, random_state=None):
		# Hyperparameters
		self.n_estimators = n_estimators
		self.max_depth = max_depth
		self.criterion = criterion
		self.max_features = max_features
		self.bootstrap = bootstrap
		self.random_state = random_state

		# Model attributes
		self.trees = []
		self.oob_score_ = None

		# Algorithm information
		self.n_samples = None
		self.n_features = None
		self.n_classes = None

	def bootstrap_sample(self, n_samples, rng):
		# Sampling with replacement; the rest are out-of-bag samples
		indices = rng.integers(0, n_samples, size=n_samples)
		oob_mask = np.ones(n_samples, dtype=bool)
		oob_mask[indices] = False
		return indices, np.where(oob_mask)[0]

	def fit(self, X, y):
		X = np.asarray(X, dtype=float)
		y = np.asarray(y).astype(int)
		self.n_samples, self.n_features = X.shape
		self.n_classes = int(y.max()) + 1

		rng = np.random.default_rng(self.random_state)
		self.trees = []

		# Accumulate OOB votes: (n_samples, n_classes)
		oob_votes = np.zeros((self.n_samples, self.n_classes))

		for _ in range(self.n_estimators):
			if self.bootstrap:
				indices, oob_idx = self.bootstrap_sample(self.n_samples, rng)
			else:
				indices, oob_idx = np.arange(self.n_samples), np.array([], dtype=int)

			tree = DecisionTree(
				max_depth=self.max_depth,
				criterion=self.criterion,
				max_features=self.max_features,
				random_state=int(rng.integers(0, 2**31 - 1))
			)
			tree.fit(X[indices], y[indices])
			self.trees.append(tree)

			# Each tree votes only on samples it has not seen
			if len(oob_idx) > 0:
				oob_pred = np.asarray(tree.predict(X[oob_idx])).astype(int)
				oob_votes[oob_idx, oob_pred] += 1

		# OOB accuracy over samples that received at least one vote
		has_vote = oob_votes.sum(axis=1) > 0
		if has_vote.any():
			oob_pred = np.argmax(oob_votes[has_vote], axis=1)
			self.oob_score_ = np.mean(oob_pred == y[has_vote])

	def predict_proba(self, X):
		X = np.asarray(X, dtype=float)
		votes = np.zeros((X.shape[0], self.n_classes))

		for tree in self.trees:
			preds = np.asarray(tree.predict(X)).astype(int)
			votes[np.arange(X.shape[0]), preds] += 1

		return votes / len(self.trees)

	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)

	def model_information(self):
		return {
			"trees": len(self.trees),
			"oob_score": self.oob_score_,
			"samples": self.n_samples,
			"features": self.n_features
		}