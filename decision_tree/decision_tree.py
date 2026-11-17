import numpy as np
from node import Node

class DecisionTree:
	def __init__(self, max_depth=None):
		self.max_depth = max_depth
		self.root = None

	# Calculate the Gini impurity
	def gini(self, y):
		classes, counts = np.unique(y, return_counts=True)
		probabilities = counts / len(y)
		return 1 - np.sum(probabilities ** 2)

	# Find the best way to split the data based on Gini impurity
	def best_split(self, X, y):
		best_gini = float("inf")
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

				# Ignore splits that result in empty child nodes
				if len(y_left) == 0 or len(y_right) == 0:
					continue

				# Calculate Weighted Gini impurity
				gini_left = self.gini(y_left)
				gini_right = self.gini(y_right)

				weighted_gini = len(y_left) / n_samples * gini_left + len(y_right) / n_samples * gini_right

				# Save the best split if the weighted Gini impurity is lower than the best found so far
				if weighted_gini < best_gini:
					best_gini = weighted_gini
					best_feature = feature
					best_threshold = threshold

		return best_feature, best_threshold


	# Find the majority class in a set of labels
	def majority_class(self, y):
		classes, counts = np.unique(y, return_counts=True)
		return classes[np.argmax(counts)]

	# Build the decision tree recursively
	def build_tree(self, X, y, depth=0):

		# Stop conditions for recursion
		if len(np.unique(y)) == 1 or depth >= self.max_depth or len(y) <= 1:
			return Node(value=self.majority_class(y))

		# Find the best split for the current node
		feature, threshold = self.best_split(X, y)

		# Not able to find a valid split
		if feature is None:
			return Node(value=self.majority_class(y))

		# Split the data into left and right branches
		left_mask = X[:, feature] <= threshold
		right_mask = X[:, feature] > threshold

		# Build the left and right subtrees recursively
		left_subtree = self.build_tree(X[left_mask], y[left_mask], depth + 1)
		right_subtree = self.build_tree(X[right_mask], y[right_mask], depth + 1)

		return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

	# Fit the decision tree to the training data
	def fit(self, X, y):
		X = np.asarray(X)
		y = np.asarray(y)
		self.root = self.build_tree(X, y)

	# Predict the class label for a single sample
	def predict_one(self, x, node):

		# If the current node is a leaf, return its value
		if node.is_leaf():
			return node.value

		# Go to the left or right subtree
		if x[node.feature] <= node.threshold:
			return self.predict_one(x, node.left)
		else:
			return self.predict_one(x, node.right)

	# Predict the class labels for multiple samples
	def predict(self, X):
		X = np.asarray(X)
		return np.array([self.predict_one(x, self.root) for x in X])