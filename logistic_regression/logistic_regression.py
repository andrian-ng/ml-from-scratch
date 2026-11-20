import numpy as np

class LogisticRegression:
	def __init__(self, learning_rate=0.0001, iterations=10000, threshold = 0.5, lambda_=0.0):
		# Hyperparameters
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.threshold = threshold
		self.lambda_ = lambda_
		
		# Algorithm parameters
		self.weights = None
		self.bias = None
		
		# Algorithm informations
		self.n_samples = None
		self.n_features = None
		
		# Loss history array
		self.loss_history = []
	
	@staticmethod
	def sigmoid(z):
		return 1 / (1 + np.exp(-z))
	
	def initialize_parameters(self, n_features):
		self.weights = np.random.rand(n_features)
		self.bias = 0
	
	def forward_propagation(self, X):
		z = np.matmul(X, self.weights) + self.bias
		y_pred = LogisticRegression.sigmoid(z)
		return y_pred
	
	def compute_loss(self, y_true, y_pred):
		epsilon = 1e-15
		y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
		# Binary Cross Entropy loss function
		loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
		
		# L2 regularization
		l2 = (self.lambda_ / (2 * self.n_samples)) * np.sum(self.weights ** 2)
		
		return loss + l2
	
	def compute_gradients(self, X, y_true, y_pred):
		m = X.shape[0]
		error = y_pred - y_true
		dw = (np.matmul(X.T, error) / m) + (self.lambda_ / self.n_samples) * self.weights
		db = np.mean(error)
		return dw, db
	
	def gradients_descent(self, X, y_true):
		for epoch in range(self.iterations):
			# Forward propagation
			y_pred = self.forward_propagation(X)
			
			# Calculate loss
			loss = self.compute_loss(y_true, y_pred)
			self.loss_history.append(loss)
			
			# Update parameters
			dw, db = self.compute_gradients(X, y_true, y_pred)
			self.weights -= self.learning_rate * dw
			self.bias -= self.learning_rate * db
	
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

		if not np.all(np.isin(y_true, [0, 1])):
			raise ValueError("Labels must be 0 or 1")

		if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
			raise ValueError("Learning rate must be positive")

		if self.iterations <= 0:
			raise ValueError("n_iters must be positive")

		if not 0 < self.threshold < 1:
			raise ValueError("Threshold must be between 0 and 1")

		if self.lambda_ < 0:
			raise ValueError("_lambda must be non-negative")
		
		self.n_samples, self.n_features = X.shape
		
		# Initialize parameters
		self.initialize_parameters(self.n_features)
		
		# Gradient descents
		self.gradients_descent(X, y_true)
		
		return self
		   
	def predict_proba(self, X):
		if self.weights is None:
			raise ValueError("Model has not been trained yet")
		
		X = np.asarray(X, dtype=float)
		
		if X.ndim == 1:
			if self.n_features == 1:
				X = X.reshape(-1, 1)
			else:
				X = X.reshape(1, -1)

		if X.shape[1] != self.n_features:
			raise ValueError("X and y must have same number of samples")
		
		return self.forward_propagation(X)
			
	def predict(self, X):
		predict_proba= self.predict_proba(X)
		return (predict_proba > self.threshold).astype(int)
		   
	@staticmethod
	def accuracy(y_true, y_pred):
		return np.mean(y_true == y_pred)
	
	@staticmethod
	def precision(y_true, y_pred):
		true_positive = np.sum((y_true == 1) & (y_pred == 1))
		false_positive = np.sum((y_true == 0) & (y_pred == 1))
		return true_positive / (true_positive + false_positive) if true_positive + false_positive > 0 else 0.0
	
	@staticmethod
	def recall(y_true, y_pred):
		true_positive = np.sum((y_true == 1) & (y_pred == 1))
		false_negative = np.sum((y_true == 1) & (y_pred == 0))
		return true_positive / (true_positive + false_negative) if true_positive + false_negative > 0 else 0.0
	
	@staticmethod
	def f1_score(y_true, y_pred):
		p = LogisticRegression.precision(y_true, y_pred)
		r = LogisticRegression.recall(y_true, y_pred)
		return 2 * p * r / (p + r) if p + r > 0 else 0.0
	
	@staticmethod
	def false_positive_rate(y_true, y_pred):
		false_positive = np.sum((y_true == 0) & (y_pred == 1))
		true_negative = np.sum((y_true == 0) & (y_pred == 0))
		return false_positive / (true_negative + false_positive) if true_negative + false_positive > 0 else 0.0
	
	@staticmethod
	def true_positive_rate(y_true, y_pred):
		return LogisticRegression.recall(y_true, y_pred)
	
	@staticmethod
	def confusion_matrix(y_true, y_pred):
		# True Negative
		true_negative = np.sum((y_true == 0) & (y_pred == 0))
		# False Positive
		false_positive = np.sum((y_true == 0) & (y_pred == 1))
		# False Negative
		false_negative = np.sum((y_true == 1) & (y_pred == 0))
		# True Positive
		true_positive = np.sum((y_true == 1) & (y_pred == 1))
		
		return np.array([[true_negative, false_positive], [false_negative, true_positive]])
		
	def model_information(self):
		return {
				"weights": self.weights,
				"bias": self.bias,
				"samples": self.n_samples,
				"features": self.n_features
				}