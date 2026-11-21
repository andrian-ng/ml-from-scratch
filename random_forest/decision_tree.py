import numpy as np
from node import Node

class DecisionTree:
	def __init__(self, max_depth=None, criterion="gini", gain_ratio_=False,
				 max_features=None, random_state=None):
		# Hyperparameters
		self.max_depth = max_depth
		self.criterion = criterion
		self.gain_ratio_ = gain_ratio_
		self.max_features = max_features	# int, "sqrt", "log2" or None (use all)
		self.rng = np.random.default_rng(random_state)

		# Model attributes
		self.root = None

	def impurity_from_counts(self, counts, criterion):
		# counts: (n_splits, n_classes) -> impurity of every row at once
		totals = counts.sum(axis=1, keepdims=True)
		p = counts / np.maximum(totals, 1)
		if criterion == "gini":
			return 1 - np.sum(p ** 2, axis=1)
		if criterion == "entropy":
			return -np.sum(p * np.log2(p + 1e-9), axis=1)  # Add a small value to avoid log(0)
		raise ValueError("Criterion must be 'gini' or 'entropy'")

	def get_n_features_to_try(self, n_features):
		if self.max_features is None:
			return n_features
		if self.max_features == "sqrt":
			return max(1, int(np.sqrt(n_features)))
		if self.max_features == "log2":
			return max(1, int(np.log2(n_features)))
		return max(1, min(int(self.max_features), n_features))

	def best_split(self, X, y, criterion):
		n_samples, n_features = X.shape

		# Gain ratio is always computed on top of entropy
		criterion = "entropy" if self.gain_ratio_ else criterion

		# One-hot labels: cumulative sums give class counts for every threshold
		classes, y_idx = np.unique(y, return_inverse=True)
		onehot = np.eye(len(classes))[y_idx]
		total_counts = onehot.sum(axis=0)
		parent_impurity = self.impurity_from_counts(total_counts[None, :], criterion)[0]

		# Random subset of features at EVERY node (the "random" in Random Forest)
		k = self.get_n_features_to_try(n_features)
		if k == n_features:
			feature_indices = np.arange(n_features)
		else:
			feature_indices = self.rng.choice(n_features, size=k, replace=False)

		best_gain = float("-inf")
		best_feature = None
		best_threshold = None

		n_left = np.arange(1, n_samples)
		n_right = n_samples - n_left

		for feature in feature_indices:
			order = np.argsort(X[:, feature], kind="stable")
			x_sorted = X[order, feature]

			# Split after position i: left = first i+1 samples
			left_counts = np.cumsum(onehot[order], axis=0)[:-1]
			right_counts = total_counts - left_counts

			# Only split between two different values
			valid = x_sorted[:-1] != x_sorted[1:]
			if not valid.any():
				continue

			impurity_left = self.impurity_from_counts(left_counts, criterion)
			impurity_right = self.impurity_from_counts(right_counts, criterion)

			p_left = n_left / n_samples
			p_right = n_right / n_samples
			gain = parent_impurity - (p_left * impurity_left + p_right * impurity_right)

			if self.gain_ratio_:
				split_info = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))
				gain = gain / split_info

			gain = np.where(valid, gain, -np.inf)
			i = np.argmax(gain)

			if gain[i] > best_gain:
				best_gain = gain[i]
				best_feature = feature
				best_threshold = x_sorted[i]

		if best_feature is None or best_gain <= 0: return None, None
		return best_feature, best_threshold

	def majority_class(self, y):
		classes, counts = np.unique(y, return_counts=True)
		return classes[np.argmax(counts)]

	def build_tree(self, X, y, depth=0):
		if len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth) or len(y) <= 1:
			return Node(value=self.majority_class(y))

		feature, threshold = self.best_split(X, y, self.criterion)
		if feature is None:
			return Node(value=self.majority_class(y))

		left_mask = X[:, feature] <= threshold
		right_mask = X[:, feature] > threshold

		left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
		right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)
		return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

	def fit(self, X, y):
		X = np.array(X, dtype=float)
		y = np.array(y, dtype=float)
		self.root = self.build_tree(X, y)

	def predict_one(self, x, tree):
		if tree.value is not None:
			return tree.value
		if x[tree.feature] <= tree.threshold:
			return self.predict_one(x, tree.left)
		else:
			return self.predict_one(x, tree.right)

	def predict(self, X):
		X = np.array(X, dtype=float)
		return [self.predict_one(x, self.root) for x in X]