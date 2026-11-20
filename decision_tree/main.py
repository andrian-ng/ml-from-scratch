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
# 3. Model configurations
# =====================================

models = {
    "Custom - Gini": {
        "custom": DecisionTree(
            max_depth=3,
            criterion="gini",
            gain_ratio_=False
        ),
        "sklearn": DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
            random_state=42
        )
    },

    "Custom - Entropy": {
        "custom": DecisionTree(
            max_depth=3,
            criterion="entropy",
            gain_ratio_=False
        ),
        "sklearn": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=3,
            random_state=42
        )
    },

    "Custom - Gain Ratio": {
        "custom": DecisionTree(
            max_depth=3,
            criterion="entropy",
            gain_ratio_=True
        ),
        "sklearn": None
    }
}


# =====================================
# 4. Evaluation function
# =====================================

def evaluate_model(name, y_true, y_pred):

    print(f"\n========== {name} ==========")

    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")

    print(
        f"Precision: "
        f"{precision_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"Recall   : "
        f"{recall_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print(
        f"F1-score : "
        f"{f1_score(y_true, y_pred, zero_division=0):.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=data.target_names,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


# =====================================
# 5. Visualization functions
# =====================================

def plot_performance_comparison(results):

    metrics = ["Accuracy", "Precision", "Recall", "F1-score"]

    x = np.arange(len(metrics))
    n_models = len(results)
    width = 0.8 / n_models

    plt.figure(figsize=(12, 6))

    for i, (name, result) in enumerate(results.items()):

        scores = [
            result["accuracy"],
            result["precision"],
            result["recall"],
            result["f1"]
        ]

        offset = (i - (n_models - 1) / 2) * width

        plt.bar(
            x + offset,
            scores,
            width,
            label=name
        )

    plt.xticks(x, metrics)
    plt.ylim(0, 1.1)

    plt.ylabel("Score")
    plt.title("Model Performance Comparison")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(results):

    n_models = len(results)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6 * n_models, 5)
    )

    if n_models == 1:
        axes = [axes]

    for ax, (name, result) in zip(axes, results.items()):

        sns.heatmap(
            result["confusion_matrix"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=data.target_names,
            yticklabels=data.target_names,
            ax=ax
        )

        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrix Comparison")
    plt.tight_layout()
    plt.show()


def plot_execution_time(results):

    names = list(results.keys())

    train_times = [
        results[name]["train_time"]
        for name in names
    ]

    predict_times = [
        results[name]["predict_time"]
        for name in names
    ]

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(12, 6))

    plt.bar(
        x - width / 2,
        train_times,
        width,
        label="Training"
    )

    plt.bar(
        x + width / 2,
        predict_times,
        width,
        label="Prediction"
    )

    plt.xticks(x, names, rotation=15)
    plt.ylabel("Time (seconds)")
    plt.title("Execution Time Comparison")

    plt.legend()
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_prediction_agreement(results):

    names = list(results.keys())

    # Compare all custom models against the first custom model
    reference_name = names[0]
    reference_predictions = results[reference_name]["predictions"]

    agreements = []

    for name in names:

        predictions = results[name]["predictions"]

        agreement = np.mean(
            predictions == reference_predictions
        ) * 100

        agreements.append(agreement)

    plt.figure(figsize=(10, 5))

    plt.bar(
        names,
        agreements,
        color="steelblue"
    )

    plt.ylim(0, 105)
    plt.ylabel("Agreement (%)")
    plt.title(
        f"Prediction Agreement vs {reference_name}"
    )

    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_feature_importance(sk_model, feature_names):

    if sk_model is None:
        return

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
# 6. Train and evaluate models
# =====================================

results = {}

for name, config in models.items():

    print(f"\n========== {name} ==========")

    # -----------------------------
    # Custom model
    # -----------------------------

    custom_model = config["custom"]

    start = time.perf_counter()

    custom_model.fit(X_train, y_train)

    train_time = time.perf_counter() - start

    start = time.perf_counter()

    predictions = np.array(
        custom_model.predict(X_test)
    )

    predict_time = time.perf_counter() - start

    results[name] = {
        "predictions": predictions,
        "train_time": train_time,
        "predict_time": predict_time,
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        )
    }

    evaluate_model(name, y_test, predictions)

    print(f"Training time  : {train_time:.6f} seconds")
    print(f"Prediction time: {predict_time:.6f} seconds")

    # -----------------------------
    # Scikit-learn model
    # -----------------------------

    sk_model = config["sklearn"]

    if sk_model is None:
        continue

    sk_name = name.replace("Custom", "Sklearn")

    start = time.perf_counter()

    sk_model.fit(X_train, y_train)

    sk_train_time = time.perf_counter() - start

    start = time.perf_counter()

    sk_predictions = sk_model.predict(X_test)

    sk_predict_time = time.perf_counter() - start

    results[sk_name] = {
        "predictions": sk_predictions,
        "train_time": sk_train_time,
        "predict_time": sk_predict_time,
        "accuracy": accuracy_score(y_test, sk_predictions),
        "precision": precision_score(
            y_test, sk_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, sk_predictions, zero_division=0
        ),
        "f1": f1_score(
            y_test, sk_predictions, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test, sk_predictions
        )
    }

    evaluate_model(sk_name, y_test, sk_predictions)

    print(f"Training time  : {sk_train_time:.6f} seconds")
    print(f"Prediction time: {sk_predict_time:.6f} seconds")

    # Prediction agreement
    agreement = np.mean(
        predictions == sk_predictions
    ) * 100

    print(f"\nPrediction agreement: {agreement:.2f}%")


# =====================================
# 7. Summary
# =====================================

print("\n========== FINAL SUMMARY ==========")

print(
    f"{'Model':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-score':<12}"
)

print("-" * 73)

for name, result in results.items():

    print(
        f"{name:<25}"
        f"{result['accuracy']:<12.4f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['f1']:<12.4f}"
    )


# =====================================
# 8. Visualization
# =====================================

print("\n========== VISUALIZATION ==========")

plot_performance_comparison(results)

plot_confusion_matrices(results)

plot_execution_time(results)

plot_prediction_agreement(results)

# Plot feature importance from sklearn Gini model
plot_feature_importance(
    models["Custom - Gini"]["sklearn"],
    data.feature_names
)