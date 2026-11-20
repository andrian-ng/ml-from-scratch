import numpy as np
from node import Node

class DecisionTree:
	def __init__(self, max_depth=None, criterion="gini", gain_ratio_=False):
		# Hyperparameters
		self.max_depth = max_depth
		self.criterion = criterion
		self.gain_ratio_ = gain_ratio_

		# Model attributes
		self.root = None

	def gini(self, y):
		classes, counts = np.unique(y, return_counts=True)
		probabilities = counts / len(y)
		return 1 - np.sum(probabilities ** 2)

	def entropy(self, y):
		classes, counts = np.unique(y, return_counts=True)
		probabilities = counts / len(y)
		return -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Add a small value to avoid log(0)

	def information_gain(self, y, y_left, y_right, criterion):
		if len(y) == 0:
			raise ValueError("Parent node cannot be empty")
		if len(y_left) + len(y_right) != len(y):
			raise ValueError("Invalid split: sample sizes do not match")
		if criterion == "gini":
			impurity = self.gini
		elif criterion == "entropy":
			impurity = self.entropy
		else:
			raise ValueError("Criterion must be 'gini' or 'entropy'")

		impurity_parent = impurity(y)
		impurity_left = impurity(y_left)
		impurity_right = impurity(y_right)

		n_samples = len(y)
		n_left = len(y_left)
		n_right = len(y_right)

		weighted_impurity = (n_left / n_samples) * impurity_left + (n_right / n_samples) * impurity_right
		return impurity_parent - weighted_impurity

	def gain_ratio(self, y, y_left, y_right, criterion="entropy"):
		# Calculate the information gain
		information_gain = self.information_gain(y, y_left, y_right, criterion)
		# Calculate the split information
		n_samples = len(y)
		n_left = len(y_left)
		n_right = len(y_right)

		p_left = n_left / n_samples
		p_right = n_right / n_samples

		split_info = -(p_left * np.log2(p_left) + p_right * np.log2(p_right)) if p_left > 0 and p_right > 0 else 0.0
		
		return information_gain / split_info if split_info != 0 else 0.0

	def best_split(self, X, y, criterion):
		best_gain = float("-inf")
		best_feature = None
		best_threshold = None

		n_samples, n_features = X.shape
		for feature in range(n_features):
			thresholds = np.unique(X[:, feature])
			for threshold in thresholds:
				left_mask = X[:, feature] <= threshold
				right_mask = X[:, feature] > threshold

				y_left = y[left_mask]
				y_right = y[right_mask]

				if len(y_left) == 0 or len(y_right) == 0:
					continue

				if self.gain_ratio_:
					gain = self.gain_ratio(y, y_left, y_right, "entropy")
				else:
					gain = self.information_gain(y, y_left, y_right, criterion)

				if gain > best_gain:
					best_gain = gain
					best_feature = feature
					best_threshold = threshold

		if best_gain <= 0: return None, None
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