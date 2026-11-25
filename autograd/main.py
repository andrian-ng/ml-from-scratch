import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from value import Value
from multi_layer_perceptron import MultiLayerPerceptron

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

random.seed(42)
np.random.seed(42)

# =====================================
# 1. Derivative of a single-variable function
# =====================================

# f(x) = 3x^2 - 4x + 5  ->  f'(x) = 6x - 4

def f(x):
    return 3 * x**2 - 4 * x + 5

x = Value(2.0)
y = f(x)
y.backward()

h = 1e-6
numerical = (f(2.0 + h) - f(2.0 - h)) / (2 * h)

print("===== 1. DERIVATIVE OF f(x) = 3x^2 - 4x + 5 AT x = 2 =====")
print(f"f(2)                 : {y.data}")
print(f"Autograd  f'(2)      : {x.grad}")
print(f"Analytic  f'(2)=6x-4 : {6 * 2.0 - 4}")
print(f"Numerical f'(2)      : {numerical:.8f}")

# =====================================
# 2. Gradient check on several expressions
# =====================================

def check_gradients(name, build, inputs, h=1e-6):
    # Autograd gradients
    values = [Value(v) for v in inputs]
    out = build(values)
    out.backward()
    analytic = [v.grad for v in values]

    # Numerical gradients (central difference)
    numerical = []
    for i in range(len(inputs)):
        plus = list(inputs)
        minus = list(inputs)
        plus[i] += h
        minus[i] -= h

        f_plus = build([Value(v) for v in plus]).data
        f_minus = build([Value(v) for v in minus]).data
        numerical.append((f_plus - f_minus) / (2 * h))

    return {
        "Expression": name,
        "Output": out.data,
        "Max |autograd - numerical|": max(abs(a - n) for a, n in zip(analytic, numerical)),
        "Autograd": np.round(analytic, 6).tolist(),
        "Numerical": np.round(numerical, 6).tolist()
    }

# tanh written with exp() only, to test exp, subtraction and division
def tanh_from_exp(n):
    e = (2 * n).exp()
    return (e - 1) / (e + 1)

expressions = {
    "(a*b + c) * f": (
        lambda v: (v[0] * v[1] + v[2]) * v[3],
        [2.0, -3.0, 10.0, -2.0]
    ),
    "a*b + a + b  (a, b reused)": (
        lambda v: v[0] * v[1] + v[0] + v[1],
        [-2.0, 3.0]
    ),
    "a + a": (
        lambda v: v[0] + v[0],
        [3.0]
    ),
    "(a*b) * (a+b)": (
        lambda v: (v[0] * v[1]) * (v[0] + v[1]),
        [-2.0, 3.0]
    ),
    "a / b": (
        lambda v: v[0] / v[1],
        [2.0, 3.0]
    ),
    "a ** 3": (
        lambda v: v[0] ** 3,
        [1.5]
    ),
    "tanh(x1*w1 + x2*w2 + b)": (
        lambda v: (v[0] * v[1] + v[2] * v[3] + v[4]).tanh(),
        [2.0, -3.0, 0.0, 1.0, 6.8813735870195432]
    ),
    "tanh written with exp()": (
        lambda v: tanh_from_exp(v[0] * v[1] + v[2] * v[3] + v[4]),
        [2.0, -3.0, 0.0, 1.0, 6.8813735870195432]
    )
}

grad_check = pd.DataFrame([
    check_gradients(name, build, inputs)
    for name, (build, inputs) in expressions.items()
])

print("\n===== 2. GRADIENT CHECK (autograd vs numerical) =====")
print(grad_check[["Expression", "Output", "Max |autograd - numerical|"]].to_string(index=False))

print("\nPer-input gradients:")
print(grad_check[["Expression", "Autograd", "Numerical"]].to_string(index=False))

assert grad_check["Max |autograd - numerical|"].max() < 1e-5, "Gradient check failed"

# =====================================
# 3. Compare with PyTorch autograd
# =====================================

print("\n===== 3. COMPARISON WITH PYTORCH =====")

try:
    import torch

    inputs = {"x1": 2.0, "w1": -3.0, "x2": 0.0, "w2": 1.0, "b": 6.8813735870195432}

    # Custom autograd
    custom = {name: Value(v) for name, v in inputs.items()}
    n = custom["x1"] * custom["w1"] + custom["x2"] * custom["w2"] + custom["b"]
    o = n.tanh()
    o.backward()

    # PyTorch
    tensors = {
        name: torch.tensor(v, dtype=torch.double, requires_grad=True)
        for name, v in inputs.items()
    }
    n_t = (tensors["x1"] * tensors["w1"] + tensors["x2"] * tensors["w2"] + tensors["b"])
    o_t = torch.tanh(n_t)
    o_t.backward()

    torch_comparison = pd.DataFrame({
        "Custom grad": {name: custom[name].grad for name in inputs},
        "PyTorch grad": {name: tensors[name].grad.item() for name in inputs}
    })
    torch_comparison["Abs diff"] = (
        torch_comparison["Custom grad"] - torch_comparison["PyTorch grad"]
    ).abs()

    print(f"Output - custom: {o.data:.10f} | pytorch: {o_t.item():.10f}")
    print(torch_comparison)

except ImportError:
    print("PyTorch is not installed, skipping this section.")

# =====================================
# 4. Tiny neural network on a toy dataset
# =====================================

print("\n===== 4. TOY DATASET: 4 SAMPLES, MLP(3 -> 4 -> 4 -> 1) =====")

xs_toy = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]
ys_toy = [1.0, -1.0, -1.0, 1.0]

toy_model = MultiLayerPerceptron(3, [4, 4, 1])

print("Number of parameters:", len(toy_model.parameters()))

toy_loss_history = []

for k in range(100):
    # Forward propagation
    y_pred = [toy_model(x) for x in xs_toy]
    loss = sum((y_out - y_gt)**2 for y_gt, y_out in zip(ys_toy, y_pred))

    # Backpropagation (reset gradients first, they accumulate with +=)
    for p in toy_model.parameters():
        p.grad = 0.0
    loss.backward()

    # Update parameters
    for p in toy_model.parameters():
        p.data -= 0.05 * p.grad

    toy_loss_history.append(loss.data)

    if k % 10 == 0:
        print(f"Iteration {k}: Loss = {loss.data:.6f}")

print("\nTargets    :", ys_toy)
print("Predictions:", [round(toy_model(x).data, 4) for x in xs_toy])

# =====================================
# 5. Binary classification: make_moons
# =====================================

X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n===== 5. MAKE_MOONS CLASSIFICATION =====")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# The network outputs tanh in (-1, 1), so labels are mapped from {0, 1} to {-1, +1}
y_train_signed = 2 * y_train - 1

# --- Custom MLP trained with autograd ---

EPOCHS = 300

custom_model = MultiLayerPerceptron(2, [8, 8, 1])

print("Number of parameters:", len(custom_model.parameters()))

loss_history = []
accuracy_history = []

start = time.time()

for epoch in range(EPOCHS):
    # Forward propagation (mean squared error)
    y_pred = [custom_model(list(x)) for x in X_train]
    loss = sum((y_out - float(y_gt))**2 for y_gt, y_out in zip(y_train_signed, y_pred)) * (1.0 / len(X_train))

    # Backpropagation
    for p in custom_model.parameters():
        p.grad = 0.0
    loss.backward()

    # Update parameters with a decaying learning rate
    learning_rate = 0.5 - 0.45 * epoch / EPOCHS
    for p in custom_model.parameters():
        p.data -= learning_rate * p.grad

    accuracy = np.mean([(out.data > 0) == (t > 0) for out, t in zip(y_pred, y_train_signed)])

    loss_history.append(loss.data)
    accuracy_history.append(accuracy)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}: Loss = {loss.data:.6f}, Accuracy = {accuracy:.4f}")

custom_time = time.time() - start

def custom_predict(model, X):
    return np.array([1 if model(list(x)).data > 0 else 0 for x in X])

def custom_scores(model, X):
    return np.array([model(list(x)).data for x in X])

custom_predictions = custom_predict(custom_model, X_test)

# --- Scikit-learn MLP ---

start = time.time()

sklearn_model = MLPClassifier(
    hidden_layer_sizes=(8, 8),
    activation="tanh",
    max_iter=2000,
    random_state=42
)

sklearn_model.fit(X_train, y_train)

sklearn_time = time.time() - start

sklearn_predictions = sklearn_model.predict(X_test)

# =====================================
# 6. Evaluation
# =====================================

def evaluate_model(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0)
    }

train_predictions = custom_predict(custom_model, X_train)

results = pd.DataFrame(
    [
        evaluate_model(y_train, train_predictions),
        evaluate_model(y_test, custom_predictions),
        evaluate_model(y_test, sklearn_predictions)
    ],
    index=["Custom autograd MLP (train)", "Custom autograd MLP (test)", "Scikit-learn MLP (test)"]
)

print("\n===== MODEL COMPARISON =====")
print(results)

print("\n===== TRAINING TIME (seconds) =====")
print(f"Custom autograd: {custom_time:.2f}")
print(f"Scikit-learn   : {sklearn_time:.2f}")

print("\n===== CONFUSION MATRIX (test) =====")
print("Custom:")
print(confusion_matrix(y_test, custom_predictions))
print("Scikit-learn:")
print(confusion_matrix(y_test, sklearn_predictions))

# =====================================
# 7. Plot training loss and accuracy
# =====================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(toy_loss_history)
axes[0].set_title("Toy Dataset - Training Loss")
axes[0].set_xlabel("Iteration")
axes[0].set_ylabel("Sum of Squared Errors")
axes[0].grid(True)

axes[1].plot(loss_history)
axes[1].set_title("Moons - Training Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean Squared Error")
axes[1].grid(True)

axes[2].plot(accuracy_history)
axes[2].set_title("Moons - Training Accuracy")
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].grid(True)

plt.tight_layout()
plt.show()

# =====================================
# 8. Plot decision boundaries
# =====================================

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 40),
    np.linspace(y_min, y_max, 40)
)

grid = np.c_[xx.ravel(), yy.ravel()]

custom_grid = custom_scores(custom_model, grid).reshape(xx.shape)
sklearn_grid = sklearn_model.predict_proba(grid)[:, 1].reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, zz, level, title in zip(
    axes,
    [custom_grid, sklearn_grid],
    [0.0, 0.5],
    ["Custom autograd MLP", "Scikit-learn MLP"]
):
    ax.contourf(xx, yy, zz, levels=20, cmap="RdBu", alpha=0.6)
    ax.contour(xx, yy, zz, levels=[level], colors="black")

    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap="RdBu", edgecolors="k", label="Train")
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="RdBu", edgecolors="k", marker="s", label="Test")

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend()

plt.tight_layout()
plt.show()