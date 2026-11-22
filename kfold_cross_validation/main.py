
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from kfold_cross_validation import cross_validation_score

# Load dataset
X, y = load_iris(return_X_y=True)

# Initialize model
model = LogisticRegression(max_iter=1000)

# Run 5-fold cross validation
scores = cross_validation_score(
    model,
    X,
    y,
    k=5
)

print("Fold scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std:", scores.std())