import numpy as np

class SupportVectorMachine:
	def __init__(self, learning_rate=0.001, iterations=10000, lambda_=0.01, decay=0.0):
		# Hyperparameters
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.lambda_ = lambda_
		self.decay = decay

		# Model attributes
		self.weights = None
		self.bias = None

		# Algorithm information
		self.n_samples = None
		self.n_features = None

		self.loss_history = []
		
	def initialize_parameters(self, n_features):
		self.weights = np.zeros(n_features)
		self.bias = 0
		
	def forward_propagation(self, X):
		y_pred = np.dot(X, self.weights) + self.bias
		return y_pred
		
	def compute_loss(self, y_true, y_pred):
		# Hinge loss
		hinge_loss = np.mean(np.maximum(0, 1 - y_true * y_pred))
		# L2 Regularization
		l2 = (self.lambda_ / 2) * np.dot(self.weights, self.weights)
		return hinge_loss + l2
		
	def compute_gradients(self, X, y_true, y_pred):
		n_samples = X.shape[0]

		# Check the margin for violations
		condition = y_true * y_pred < 1

		# Compute gradients
		dw = self.lambda_ * self.weights
		dw -= np.sum(y_true[condition, None] * X[condition], axis=0) / n_samples
		db = -np.sum(y_true[condition]) / n_samples

		return dw, db

	def gradients_descent(self, X, y_true):
		self.n_samples, self.n_features = X.shape

		# Initialize parameters
		self.initialize_parameters(self.n_features)
		self.loss_history = []

		for epoch in range(self.iterations):
			# Forward propagation
			y_pred = self.forward_propagation(X)

			# Compute loss (optional, for monitoring)
			loss = self.compute_loss(y_true, y_pred)
			self.loss_history.append(loss)

			# Compute gradients
			dw, db = self.compute_gradients(X, y_true, y_pred)

			# Learning rate decay
			lr = self.learning_rate / (1 + self.decay * epoch)

			# Update parameters
			self.weights -= lr * dw
			self.bias -= lr * db

			if epoch % 1000 == 0:
				print(f"Iteration {epoch}: Loss = {loss}")
	
	def fit(self, X, y_true):
		X = np.asarray(X, dtype=float)
		y_true = np.asarray(y_true, dtype=float)
		y = np.where(y_true <= 0, -1, 1)
		self.gradients_descent(X, y)

	def decision_function(self, X):
		X = np.asarray(X, dtype=float)
		if self.weights is None:
			raise RuntimeError(
				"SupportVectorMachine is not fitted yet. Call 'fit' with appropriate data before using this method."
			)
		return self.forward_propagation(X)

	def predict(self, X):
		scores = self.decision_function(X)
		return np.where(scores >= 0, 1, 0)

	@staticmethod
	def accuracy(y_true, y_pred):
		return np.mean(y_true == y_pred)

	@staticmethod
	def precision(y_true, y_pred):
		tp = np.sum((y_true == 1) & (y_pred == 1))
		fp = np.sum((y_true == 0) & (y_pred == 1))
		return tp / (tp + fp) if (tp + fp) > 0 else 0

	@staticmethod
	def recall(y_true, y_pred):
		tp = np.sum((y_true == 1) & (y_pred == 1))
		fn = np.sum((y_true == 1) & (y_pred == 0))
		return tp / (tp + fn) if (tp + fn) > 0 else 0

	@staticmethod
	def f1_score(y_true, y_pred):
		precision = SupportVectorMachine.precision(y_true, y_pred)
		recall = SupportVectorMachine.recall(y_true, y_pred)
		return (
			2 * precision * recall / (precision + recall)
			if (precision + recall) > 0 else 0
		)

	def model_information(self):
		return {
			"weights": self.weights,
			"bias": self.bias,
			"samples": self.n_samples,
			"features": self.n_features
		}