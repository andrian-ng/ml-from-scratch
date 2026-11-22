import numpy as np

def k_fold_split(X, y, k=5, shuffle=True, random_state=None):
    n_samples = len(X)

    # Create indices
    indices = np.arange(n_samples)

    # Shuffle indices
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)

    # Split indices to k folds
    folds = np.array_split(indices, k)

    # Loop folds
    for i in range(k):
        # Validation indices
        val_indices = folds[i]

        # Training indices: all remaining folds
        train_indices = np.concatenate([folds[j] for j in range(k) if j != i])

        # Train data
        X_train = X[train_indices]
        y_train = y[train_indices]

        # Validation data
        X_val = X[val_indices]
        y_val = y[val_indices]

        yield X_train, X_val, y_train, y_val

def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)

def cross_validation_score(model, X, y, k=5):
    scores = []
    for X_train, X_val, y_train, y_val in k_fold_split(X, y, k=k):

        # Train model
        model.fit(X_train, y_train)

        # Predict validation set
        y_pred = model.predict(X_val)

        # Evaluate
        score = accuracy_score(y_val, y_pred)
        scores.append(score)

    return np.array(scores)