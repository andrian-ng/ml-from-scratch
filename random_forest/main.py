import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from decision_tree import DecisionTree
from random_forest import RandomForest

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

# Note: no feature scaling here. Trees split on thresholds, so they are
# invariant to the scale of the features.

# =====================================
# 3. Hyperparameters
# =====================================

N_ESTIMATORS = 50
MAX_DEPTH = 8
RANDOM_STATE = 42

# =====================================
# 4. Train Custom Random Forest
# =====================================

start = time.time()

custom_model = RandomForest(
	n_estimators=N_ESTIMATORS,
	max_depth=MAX_DEPTH,
	criterion="gini",
	max_features="sqrt",
	random_state=RANDOM_STATE
)

custom_model.fit(X_train, y_train)

custom_time = time.time() - start

custom_predictions = custom_model.predict(X_test)
custom_probabilities = custom_model.predict_proba(X_test)[:, 1]

# =====================================
# 5. Train Scikit-learn Random Forest
# =====================================

start = time.time()

sklearn_model = RandomForestClassifier(
	n_estimators=N_ESTIMATORS,
	max_depth=MAX_DEPTH,
	criterion="gini",
	max_features="sqrt",
	oob_score=True,
	random_state=RANDOM_STATE
)

sklearn_model.fit(X_train, y_train)

sklearn_time = time.time() - start

sklearn_predictions = sklearn_model.predict(X_test)
sklearn_probabilities = sklearn_model.predict_proba(X_test)[:, 1]

# =====================================
# 6. Train a single Decision Tree (baseline)
# =====================================

single_tree = DecisionTree(max_depth=MAX_DEPTH, criterion="gini")

single_tree.fit(X_train, y_train)

single_predictions = np.asarray(single_tree.predict(X_test)).astype(int)

# =====================================
# 7. Evaluation function
# =====================================

def evaluate_model(y_true, y_pred, y_prob=None):
	results = {
		"Accuracy": accuracy_score(y_true, y_pred),
		"Precision": precision_score(y_true, y_pred, zero_division=0),
		"Recall": recall_score(y_true, y_pred, zero_division=0),
		"F1 Score": f1_score(y_true, y_pred, zero_division=0),
	}
	results["ROC AUC"] = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
	return results

# =====================================
# 8. Compare performance
# =====================================

results = pd.DataFrame(
	[
		evaluate_model(y_test, custom_predictions, custom_probabilities),
		evaluate_model(y_test, sklearn_predictions, sklearn_probabilities),
		evaluate_model(y_test, single_predictions)
	],
	index=["Custom Random Forest", "Scikit-learn", "Single Decision Tree"]
)

print("\n===== MODEL COMPARISON =====")
print(results)

print("\n===== TRAINING TIME (seconds) =====")
print(f"Custom      : {custom_time:.2f}")
print(f"Scikit-learn: {sklearn_time:.2f}")

print("\n===== OUT-OF-BAG SCORE =====")
print(f"Custom      : {custom_model.oob_score_:.4f}")
print(f"Scikit-learn: {sklearn_model.oob_score_:.4f}")

print("\n===== CUSTOM MODEL INFORMATION =====")
print(custom_model.model_information())

# =====================================
# 9. Classification reports
# =====================================

print("\n===== CUSTOM CLASSIFICATION REPORT =====")

print(classification_report(y_test, custom_predictions, target_names=dataset.target_names))

print("\n===== SKLEARN CLASSIFICATION REPORT =====")

print(classification_report(y_test, sklearn_predictions, target_names=dataset.target_names))

# =====================================
# 10. Compare predictions
# =====================================

comparison = pd.DataFrame({
	"Actual": y_test,
	"Custom Prediction": custom_predictions,
	"Sklearn Prediction": sklearn_predictions,
	"Custom Probability": custom_probabilities,
	"Sklearn Probability": sklearn_probabilities
})

print("\n===== PREDICTION COMPARISON =====")
print(comparison.head(10))

print("\nPrediction agreement:", f"{np.mean(custom_predictions == sklearn_predictions):.4f}")

# =====================================
# 11. Plot confusion matrices
# =====================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for ax, preds, title in zip(
	axes,
	[custom_predictions, sklearn_predictions, single_predictions],
	["Custom Random Forest", "Scikit-learn", "Single Decision Tree"]
):
	cm = confusion_matrix(y_test, preds)

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
			ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

plt.tight_layout()
plt.show()

# =====================================
# 12. Plot ROC Curve
# =====================================

custom_fpr, custom_tpr, _ = roc_curve(y_test, custom_probabilities)
sklearn_fpr, sklearn_tpr, _ = roc_curve(y_test, sklearn_probabilities)

custom_auc = roc_auc_score(y_test, custom_probabilities)
sklearn_auc = roc_auc_score(y_test, sklearn_probabilities)

plt.figure(figsize=(10, 6))

plt.plot(custom_fpr, custom_tpr, label=f"Custom Random Forest (AUC = {custom_auc:.4f})")
plt.plot(sklearn_fpr, sklearn_tpr, label=f"Scikit-learn (AUC = {sklearn_auc:.4f})")
plt.plot([0, 1], [0, 1], "r--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.grid(True)
plt.show()

# =====================================
# 13. Plot accuracy vs number of trees
# =====================================

custom_tree_votes = np.array(
	[np.asarray(tree.predict(X_test)).astype(int) for tree in custom_model.trees]
)
sklearn_tree_probs = np.array(
	[tree.predict_proba(X_test)[:, 1] for tree in sklearn_model.estimators_]
)

counts = np.arange(1, N_ESTIMATORS + 1)

# Cumulative vote of the first k trees
custom_curve = [
	accuracy_score(y_test, (custom_tree_votes[:k].mean(axis=0) > 0.5).astype(int))
	for k in counts
]
sklearn_curve = [
	accuracy_score(y_test, (sklearn_tree_probs[:k].mean(axis=0) > 0.5).astype(int))
	for k in counts
]

plt.figure(figsize=(10, 5))

plt.plot(counts, custom_curve, label="Custom Random Forest")
plt.plot(counts, sklearn_curve, label="Scikit-learn")
plt.axhline(
	accuracy_score(y_test, single_predictions),
	color="r", linestyle="--", label="Single Decision Tree"
)

plt.xlabel("Number of trees")
plt.ylabel("Test accuracy")
plt.title("Accuracy vs Number of Trees")

plt.legend()
plt.grid(True)
plt.show()