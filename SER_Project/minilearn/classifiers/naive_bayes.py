import numpy as np


class GaussianNaiveBayes:
    """Gaussian Naive Bayes: models each feature as an independent Gaussian per class."""

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.priors_, self.means_, self.vars_ = {}, {}, {}
        for c in self.classes_:
            X_c = X[y == c]
            self.priors_[c] = len(X_c) / len(X)
            self.means_[c] = X_c.mean(axis=0)
            self.vars_[c] = X_c.var(axis=0) + 1e-9  # add small constant to avoid log(0)
        return self

    def _log_likelihood(self, X, c):
        mean, var = self.means_[c], self.vars_[c]
        return -0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)

    def predict_log_proba(self, X):
        X = np.array(X, dtype=float)
        return np.column_stack([
            np.log(self.priors_[c]) + self._log_likelihood(X, c)
            for c in self.classes_
        ])

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_log_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
