import numpy as np


class StandardScaler:
    """Zero-mean, unit-variance normalisation fit on training data."""

    def fit(self, X):
        X = np.array(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ == 0] = 1.0  # constant features: don't divide by zero
        return self

    def transform(self, X):
        return (np.array(X, dtype=float) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        return np.array(X, dtype=float) * self.scale_ + self.mean_
