import numpy as np


class PCA:
    """Principal Component Analysis via truncated SVD."""

    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        _, S, Vt = np.linalg.svd(X - self.mean_, full_matrices=False)
        n = len(X)
        ev = S ** 2 / (n - 1)
        self.components_ = Vt[: self.n_components]
        self.explained_variance_ = ev[: self.n_components]
        total = ev.sum()
        self.explained_variance_ratio_ = ev[: self.n_components] / total
        return self

    def transform(self, X):
        return (np.array(X, dtype=float) - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
