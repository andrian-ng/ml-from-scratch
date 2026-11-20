import numpy as np

class LinearRegression:
	def __init__(self, learning_rate=0.001, iterations=10000, method="gradients_descent"):
		# Hyperparameters
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.method = method

		# Algorithm parameters
		self.weights = None
		self.bias = None

		# Algorithm information
		self.n_samples = None
		self.n_features = None

		self.loss_history = []

	def initialize_parameters(self, n_features):
		self.weights = np.random.rand(n_features)
		self.bias = 0

	def forward_propagation(self, X):
		return np.matmul(X, self.weights) + self.bias

	def compute_loss(self, y_true, y_pred):
		# Mean Squared Error loss function
		loss = np.mean((y_true - y_pred) ** 2)
		return loss

	def compute_gradients(self, X, y_true, y_pred):
		m = X.shape[0]
		error = y_pred - y_true
		dw = np.matmul(X.T, error) / m
		db = np.mean(error)
		return dw, db

	def gradients_descent(self, X, y_true):
		for epoch in range(self.iterations):
			# Forward propagation
			y_pred = self.forward_propagation(X)

			# Calculate loss
			loss = self.compute_loss(y_true, y_pred)
			self.loss_history.append(loss)

			# Compute gradients
			dw, db = self.compute_gradients(X, y_true, y_pred)

			# Update parameters
			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db

	def normal_equation(self, X, y_true):
		# Add a column of ones to X for the bias term
		X_b = np.c_[np.ones((X.shape[0], 1)), X]
		# Calculate weights using the normal equation
		theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_true)
		self.bias = theta_best[0]
		self.weights = theta_best[1:]
	
	def fit(self, X, y_true):
		X = np.asarray(X, dtype=float)
		y_true = np.asarray(y_true, dtype=float)

		if X.shape[0] != y_true.shape[0]:
			raise ValueError("X and y must have same number of samples")

		if X.ndim != 2:
			raise ValueError("X must be a 2D array")

		if X.shape[0] == 0 or X.shape[1] == 0:
			raise ValueError("X must not be empty")

		if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y_true)):
			raise ValueError("X and y must contain finite values")

		if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
			raise ValueError("Learning rate must be positive")

		if self.iterations <= 0:
			raise ValueError("n_iters must be positive")

		self.n_samples, self.n_features = X.shape

		# Initialize parameters
		self.initialize_parameters(self.n_features)

		# Fit the model using the specified method
		if self.method == "gradients_descent":
			self.gradients_descent(X, y_true)
		elif self.method == "normal_equation":
			self.normal_equation(X, y_true)
		else:
			raise ValueError("Invalid method. Choose 'gradients_descent' or 'normal_equation'.")

		return self

	def predict(self, X):
		X = np.asarray(X, dtype=float)

		if X.ndim == 1:
			if self.n_features == 1:
				X = X.reshape(-1, 1)
			else:
				X = X.reshape(1, -1)

		if X.shape[1] != self.n_features:
			raise ValueError("X must have the same number of features as the training data")

		y_pred = self.forward_propagation(X)
		return y_pred

	@staticmethod
	def mse(y_true, y_pred):
		return np.mean((y_true - y_pred) ** 2)

	@staticmethod
	def rmse(y_true, y_pred):
		return np.sqrt(np.mean((y_true - y_pred) ** 2))

	@staticmethod
	def mae(y_true, y_pred):
		return np.mean(np.abs(y_true - y_pred))

	@staticmethod
	def r2_score(y_true, y_pred):
		ss_res = np.sum((y_true - y_pred) ** 2)
		ss_tot = np.sum(y_true - np.mean(y_true) ** 2)
		if ss_tot == 0:
			return 1.0 if ss_res == 0 else 0.0
		return 1 - (ss_res / ss_tot)

	@staticmethod
	def huber_loss(y_true, y_pred, delta=1.0):
		error = np.abs(y_true - y_pred)
		loss = np.where(error <= delta, 0.5 * (error**2), delta * (error - 0.5 * delta))
		return np.mean(loss)

	def model_information(self):
		return {
				"weights": self.weights,
				"bias": self.bias,
				"samples": self.n_samples,
				"features": self.n_features
				}