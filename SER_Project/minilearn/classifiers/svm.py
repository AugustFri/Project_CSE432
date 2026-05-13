import numpy as np


class LinearSVM:
    """Multi-class linear SVM via one-vs-rest, batch sub-gradient descent (hinge + L2)."""

    def __init__(self, C=1.0, lr=0.01, max_iter=500, random_state=None):
        self.C = C            # higher C → less regularisation, fits training data more tightly
        self.lr = lr
        self.max_iter = max_iter
        self.random_state = random_state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _fit_binary(self, X, y_bin):
        """Train one binary SVM.  y_bin ∈ {-1, +1}."""
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0

        for t in range(1, self.max_iter + 1):
            # Pegasos-style 1/√t decay preserves O(1/√T) convergence for sub-gradient descent
            lr_t = self.lr / np.sqrt(t)
            margins = y_bin * (X @ w + b)      # (n,)
            # hinge loss is zero when margin ≥ 1; only violated points contribute a gradient
            violated = margins < 1

            if violated.any():
                grad_w = w - self.C * (y_bin[violated, None] * X[violated]).sum(axis=0) / n
                grad_b = -self.C * y_bin[violated].sum() / n
            else:
                grad_w = w.copy()
                grad_b = 0.0

            w -= lr_t * grad_w
            b -= lr_t * grad_b

        return w, b

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        self.classes_ = np.unique(y)
        self.weights_, self.biases_ = [], []
        for cls in self.classes_:
            y_bin = np.where(y == cls, 1.0, -1.0)
            w, b = self._fit_binary(X, y_bin)
            self.weights_.append(w)
            self.biases_.append(b)
        return self

    def decision_function(self, X):
        X = np.array(X, dtype=float)
        return np.column_stack([X @ w + b for w, b in zip(self.weights_, self.biases_)])

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
