import numpy as np


class LogisticRegression:
    """Multiclass logistic regression via softmax + gradient descent."""

    def __init__(self, lr=0.1, max_iter=500, tol=1e-4, random_state=None):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)  # shift for numerical stability
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        y_idx = np.searchsorted(self.classes_, y)          # integer labels
        Y = np.eye(n_classes)[y_idx]                       # one-hot (N, K)

        self.W_ = rng.normal(0, 0.01, (n_features, n_classes))
        self.b_ = np.zeros(n_classes)

        for _ in range(self.max_iter):
            probs = self._softmax(X @ self.W_ + self.b_)   # (N, K)
            grad_W = (X.T @ (probs - Y)) / n_samples
            grad_b = (probs - Y).mean(axis=0)
            self.W_ -= self.lr * grad_W
            self.b_ -= self.lr * grad_b
            if np.max(np.abs(self.lr * grad_W)) < self.tol:
                break

        return self

    def predict_proba(self, X):
        return self._softmax(np.array(X, dtype=float) @ self.W_ + self.b_)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
