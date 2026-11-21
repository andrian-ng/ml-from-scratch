import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler

from svm import SupportVectorMachine

from sklearn.metrics import (
	accuracy_score,
	precision_score,
	recall_score,
	f1_score,
	confusion_matrix,
	classification_report,
	roc_curve,
	roc_auc_score
	)

# =====================================
# 1. Load dataset
# =====================================

dataset = load_breast_cancer()

X = dataset.data
y = dataset.target

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)
print("Features:", dataset.feature_names)

print("\nClasses:", dataset.target_names)

# =====================================
# 2. Split dataset
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# =====================================
# 3. Scale data
# =====================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================
# 4. Train Custom SVM
# =====================================

lambda_ = 0.01

custom_model = SupportVectorMachine(
	learning_rate=0.01,
	iterations=10000,
	lambda_=lambda_,
	decay=0.001
)

custom_model.fit(X_train, y_train)

custom_predictions = custom_model.predict(X_test)

# Decision scores (signed distance to the hyperplane, used for ROC)
custom_scores = custom_model.decision_function(X_test)

# =====================================
# 5. Train Scikit-learn SVM
# =====================================

# Relationship between lambda and C: lambda = 1 / (C * n_samples)
C = 1 / (lambda_ * X_train.shape[0])

sklearn_model = LinearSVC(
	C=C,
	loss="hinge",
	max_iter=100000,
	random_state=42
)

sklearn_model.fit(X_train, y_train)

sklearn_predictions = sklearn_model.predict(X_test)

sklearn_scores = sklearn_model.decision_function(X_test)

# =====================================
# 6. Evaluation function
# =====================================

def evaluate_model(y_true, y_pred, y_score):
	return {
		"Accuracy": accuracy_score(y_true, y_pred),
		"Precision": precision_score(y_true, y_pred, zero_division=0),
		"Recall": recall_score(y_true, y_pred, zero_division=0),
		"F1 Score": f1_score(y_true, y_pred, zero_division=0),
		"ROC AUC": roc_auc_score(y_true, y_score)
	}

# =====================================
# 7. Compare performance
# =====================================

custom_results = evaluate_model(y_test, custom_predictions, custom_scores)

sklearn_results = evaluate_model(y_test, sklearn_predictions, sklearn_scores)

results = pd.DataFrame(
	[custom_results, sklearn_results],
	index=["Custom SVM", "Scikit-learn"]
)

print("\n===== MODEL COMPARISON =====")
print(results)

# =====================================
# 8. Compare model parameters
# =====================================

print("\n===== MODEL PARAMETERS =====")

print("Custom weights:")
print(custom_model.weights)

print("\nCustom bias:")
print(custom_model.bias)

print("\nScikit-learn coefficients:")
print(sklearn_model.coef_)

print("\nScikit-learn intercept:")
print(sklearn_model.intercept_)

print("\nCosine similarity of weights (1.0 = same direction):")
cos_sim = np.dot(custom_model.weights, sklearn_model.coef_[0]) / (
	np.linalg.norm(custom_model.weights) * np.linalg.norm(sklearn_model.coef_[0])
)
print(f"{cos_sim:.4f}")

# =====================================
# 9. Margin analysis
# =====================================

# Labels in {-1, +1} to measure the functional margin y * f(x)
y_train_signed = np.where(y_train <= 0, -1, 1)

custom_margins = y_train_signed * custom_model.decision_function(X_train)
sklearn_margins = y_train_signed * sklearn_model.decision_function(X_train)

print("\n===== MARGIN ANALYSIS (train set) =====")

print(f"Custom   - margin violations (y*f < 1): {np.sum(custom_margins < 1)}")
print(f"Sklearn  - margin violations (y*f < 1): {np.sum(sklearn_margins < 1)}")

print(f"Custom   - margin width (2/||w||): {2 / np.linalg.norm(custom_model.weights):.4f}")
print(f"Sklearn  - margin width (2/||w||): {2 / np.linalg.norm(sklearn_model.coef_[0]):.4f}")

# =====================================
# 10. Classification reports
# =====================================

print("\n===== CUSTOM CLASSIFICATION REPORT =====")

print(
	classification_report(
		y_test,
		custom_predictions,
		target_names=dataset.target_names
	)
)

print("\n===== SKLEARN CLASSIFICATION REPORT =====")

print(
	classification_report(
		y_test,
		sklearn_predictions,
		target_names=dataset.target_names
	)
)

# =====================================
# 11. Compare predictions
# =====================================

comparison = pd.DataFrame({
	"Actual": y_test,
	"Custom Prediction": custom_predictions,
	"Sklearn Prediction": sklearn_predictions,
	"Custom Score": custom_scores,
	"Sklearn Score": sklearn_scores
})

print("\n===== PREDICTION COMPARISON =====")
print(comparison.head(10))

print(
	"\nPrediction agreement:",
	f"{np.mean(custom_predictions == sklearn_predictions):.4f}"
)

# =====================================
# 12. Plot confusion matrices
# =====================================

custom_cm = confusion_matrix(y_test, custom_predictions)
sklearn_cm = confusion_matrix(y_test, sklearn_predictions)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, cm, title in zip(
	axes,
	[custom_cm, sklearn_cm],
	["Custom SVM", "Scikit-learn"]
):

	ax.imshow(cm, cmap="Blues")

	ax.set_title(title)
	ax.set_xlabel("Predicted Label")
	ax.set_ylabel("Actual Label")

	ax.set_xticks([0, 1])
	ax.set_yticks([0, 1])

	ax.set_xticklabels(dataset.target_names)
	ax.set_yticklabels(dataset.target_names)

	for i in range(2):
		for j in range(2):
			ax.text(
				j,
				i,
				cm[i, j],
				ha="center",
				va="center",
				color="black",
				fontsize=12
			)

plt.tight_layout()
plt.show()

# =====================================
# 13. Plot ROC Curve
# =====================================

custom_fpr, custom_tpr, _ = roc_curve(y_test, custom_scores)
sklearn_fpr, sklearn_tpr, _ = roc_curve(y_test, sklearn_scores)

custom_auc = roc_auc_score(y_test, custom_scores)
sklearn_auc = roc_auc_score(y_test, sklearn_scores)

plt.figure(figsize=(10, 6))

plt.plot(
	custom_fpr,
	custom_tpr,
	label=f"Custom SVM (AUC = {custom_auc:.4f})"
)

plt.plot(
	sklearn_fpr,
	sklearn_tpr,
	label=f"Scikit-learn (AUC = {sklearn_auc:.4f})"
)

plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(True)
plt.show()

# =====================================
# 14. Plot Decision Score Distribution
# =====================================

plt.figure(figsize=(10, 5))

plt.hist(custom_scores[y_test == 0], bins=30, alpha=0.6, label=dataset.target_names[0])
plt.hist(custom_scores[y_test == 1], bins=30, alpha=0.6, label=dataset.target_names[1])

plt.axvline(0, color="r", linestyle="--", label="Decision boundary")
plt.axvline(-1, color="gray", linestyle=":", label="Margin (-1, +1)")
plt.axvline(1, color="gray", linestyle=":")

plt.xlabel("Decision score")
plt.ylabel("Count")
plt.title("Custom SVM Decision Score Distribution (test set)")

plt.legend()
plt.grid(True)
plt.show()

# =====================================
# 15. Plot Training Loss Curve
# =====================================

plt.figure(figsize=(10, 5))

plt.plot(custom_model.loss_history, label="Custom SVM")

plt.xlabel("Epoch")
plt.ylabel("Hinge Loss + L2 Regularization")

plt.title("Training Loss Curve")

plt.legend()
plt.grid(True)
plt.show()