import numpy as np
from collections import Counter


class KNN:
    """k-Nearest Neighbours classifier using Euclidean distance."""

    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        self.X_train_ = np.array(X, dtype=float)
        self.y_train_ = np.array(y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        sq_dists = np.sum((self.X_train_ - x) ** 2, axis=1)
        k_idx = np.argpartition(sq_dists, self.n_neighbors)[:self.n_neighbors]
        return Counter(self.y_train_[k_idx]).most_common(1)[0][0]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
