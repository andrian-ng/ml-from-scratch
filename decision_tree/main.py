import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from decision_tree import DecisionTree

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	classification_report,
	confusion_matrix
)

from decision_tree import DecisionTree


# =====================================
# 1. Load dataset
# =====================================

data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset:", data.DESCR.split("\n")[0])
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Classes:", data.target_names)


# =====================================
# 2. Split train/test
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
	X,
	y,
	test_size=0.2,
	random_state=42,
	stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))


# =====================================
# 3. Train From Scratch Decision Tree
# =====================================

print("\n========== FROM SCRATCH ==========")

custom_model = DecisionTree(max_depth=3)

start = time.perf_counter()

custom_model.fit(X_train, y_train)

my_train_time = time.perf_counter() - start

start = time.perf_counter()

my_predictions = custom_model.predict(X_test)

my_predict_time = time.perf_counter() - start


# =====================================
# 4. Train Scikit-learn Decision Tree
# =====================================

print("\n========== SCIKIT-LEARN ==========")

sk_model = DecisionTreeClassifier(
	criterion="gini",
	max_depth=3,
	random_state=42
)

start = time.perf_counter()

sk_model.fit(X_train, y_train)

sk_train_time = time.perf_counter() - start

start = time.perf_counter()

sk_predictions = sk_model.predict(X_test)

sk_predict_time = time.perf_counter() - start


# =====================================
# 5. Evaluation function
# =====================================

def evaluate_model(name, y_true, y_pred):

	print(f"\n========== {name} ==========")

	print("Accuracy :", accuracy_score(y_true, y_pred))

	print(
		"Precision:",
		precision_score(y_true, y_pred, zero_division=0)
	)

	print(
		"Recall   :",
		recall_score(y_true, y_pred, zero_division=0)
	)

	print(
		"F1-score :",
		f1_score(y_true, y_pred, zero_division=0)
	)

	print("\nConfusion Matrix:")
	print(confusion_matrix(y_true, y_pred))


# =====================================
# 6. Visualization functions
# =====================================

def plot_performance_comparison(y_test, my_predictions, sk_predictions):

	metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

	my_scores = [
		accuracy_score(y_test, my_predictions),
		precision_score(y_test, my_predictions, zero_division=0),
		recall_score(y_test, my_predictions, zero_division=0),
		f1_score(y_test, my_predictions, zero_division=0)
	]

	sk_scores = [
		accuracy_score(y_test, sk_predictions),
		precision_score(y_test, sk_predictions, zero_division=0),
		recall_score(y_test, sk_predictions, zero_division=0),
		f1_score(y_test, sk_predictions, zero_division=0)
	]

	x = np.arange(len(metrics))
	width = 0.35

	plt.figure(figsize=(10, 6))

	plt.bar(x - width/2, my_scores, width, label="From Scratch")
	plt.bar(x + width/2, sk_scores, width, label="Scikit-learn")

	plt.xticks(x, metrics)
	plt.ylim(0, 1.1)

	plt.ylabel("Score")
	plt.title("Performance Comparison")
	plt.legend()
	plt.grid(axis="y", alpha=0.3)

	plt.tight_layout()
	plt.show()


def plot_confusion_matrices(y_test, my_predictions, sk_predictions):

	fig, axes = plt.subplots(1, 2, figsize=(12, 5))

	cm_my = confusion_matrix(y_test, my_predictions)
	cm_sk = confusion_matrix(y_test, sk_predictions)

	sns.heatmap(
		cm_my,
		annot=True,
		fmt="d",
		cmap="Blues",
		xticklabels=data.target_names,
		yticklabels=data.target_names,
		ax=axes[0]
	)

	axes[0].set_title("From Scratch")
	axes[0].set_xlabel("Predicted")
	axes[0].set_ylabel("Actual")

	sns.heatmap(
		cm_sk,
		annot=True,
		fmt="d",
		cmap="Greens",
		xticklabels=data.target_names,
		yticklabels=data.target_names,
		ax=axes[1]
	)

	axes[1].set_title("Scikit-learn")
	axes[1].set_xlabel("Predicted")
	axes[1].set_ylabel("Actual")

	plt.suptitle("Confusion Matrix Comparison")
	plt.tight_layout()
	plt.show()


def plot_execution_time(
	my_train_time,
	sk_train_time,
	my_predict_time,
	sk_predict_time
):

	categories = ["Training", "Prediction"]

	custom_times = [my_train_time, my_predict_time]
	sklearn_times = [sk_train_time, sk_predict_time]

	x = np.arange(len(categories))
	width = 0.35

	plt.figure(figsize=(9, 5))

	plt.bar(
		x - width/2,
		custom_times,
		width,
		label="From Scratch"
	)

	plt.bar(
		x + width/2,
		sklearn_times,
		width,
		label="Scikit-learn"
	)

	plt.xticks(x, categories)
	plt.ylabel("Time (seconds)")
	plt.title("Execution Time Comparison")
	plt.legend()
	plt.grid(axis="y", alpha=0.3)

	plt.tight_layout()
	plt.show()


def plot_prediction_agreement(my_predictions, sk_predictions):

	same = np.sum(my_predictions == sk_predictions)
	different = len(my_predictions) - same

	plt.figure(figsize=(7, 7))

	plt.pie(
		[same, different],
		labels=["Same Predictions", "Different Predictions"],
		autopct="%1.2f%%",
		colors=["#4CAF50", "#FF7043"],
		startangle=90,
		explode=(0.05, 0)
	)

	plt.title("Prediction Agreement")
	plt.tight_layout()
	plt.show()


def plot_feature_importance(sk_model, feature_names):

	importances = sk_model.feature_importances_

	indices = np.argsort(importances)[-10:]

	plt.figure(figsize=(10, 6))

	plt.barh(
		np.array(feature_names)[indices],
		importances[indices],
		color="steelblue"
	)

	plt.xlabel("Importance")
	plt.ylabel("Features")
	plt.title("Top 10 Feature Importances - Scikit-learn")

	plt.tight_layout()
	plt.show()


# =====================================
# 7. Compare execution time
# =====================================

print("\n========== TIME COMPARISON ==========")

print(f"From scratch training : {my_train_time:.6f} seconds")
print(f"Scikit-learn training : {sk_train_time:.6f} seconds")

print(f"From scratch predict  : {my_predict_time:.6f} seconds")
print(f"Scikit-learn predict  : {sk_predict_time:.6f} seconds")


# =====================================
# 8. Compare predictions
# =====================================

print("\n========== PREDICTION COMPARISON ==========")

same_predictions = np.sum(
	my_predictions == sk_predictions
)

total_predictions = len(y_test)

print(f"Same predictions: {same_predictions}/{total_predictions}")

print(
	f"Prediction agreement: "
	f"{same_predictions / total_predictions * 100:.2f}%"
)


# =====================================
# 9. Visualization
# =====================================

print("\n========== VISUALIZATION ==========")

# 1. Performance comparison
plot_performance_comparison(
	y_test,
	my_predictions,
	sk_predictions
)

# 2. Confusion matrices
plot_confusion_matrices(
	y_test,
	my_predictions,
	sk_predictions
)

# 3. Execution time
plot_execution_time(
	my_train_time,
	sk_train_time,
	my_predict_time,
	sk_predict_time
)

# 4. Prediction agreement
plot_prediction_agreement(
	my_predictions,
	sk_predictions
)

# 5. Feature importance
plot_feature_importance(
	sk_model,
	data.feature_names
)