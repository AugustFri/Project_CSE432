import numpy as np


class MLPClassifier:
    """Multi-Layer Perceptron: ReLU hidden layers, softmax output, Adam optimiser."""

    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        lr=0.001,
        max_iter=100,
        batch_size=64,
        random_state=None,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state

    # ------------------------------------------------------------------
    # activations
    # ------------------------------------------------------------------

    @staticmethod
    def _relu(z):        return np.maximum(0.0, z)
    @staticmethod
    def _relu_d(z):      return (z > 0).astype(float)
    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    # weight init (He)
    # ------------------------------------------------------------------

    def _init_params(self, layer_sizes, rng):
        self.W_, self.b_ = [], []
        for i in range(len(layer_sizes) - 1):
            # He init: var = 2/fan_in compensates for ReLU zeroing ~half of activations
            scale = np.sqrt(2.0 / layer_sizes[i])
            self.W_.append(rng.normal(0, scale, (layer_sizes[i], layer_sizes[i + 1])))
            self.b_.append(np.zeros(layer_sizes[i + 1]))

    # ------------------------------------------------------------------
    # forward / backward
    # ------------------------------------------------------------------

    def _forward(self, X):
        acts, zs = [X], []
        a = X
        for i, (W, b) in enumerate(zip(self.W_, self.b_)):
            z = a @ W + b
            zs.append(z)
            a = self._relu(z) if i < len(self.W_) - 1 else self._softmax(z)
            acts.append(a)
        return zs, acts

    def _backward(self, zs, acts, Y):
        n = len(Y)
        dW = [np.zeros_like(w) for w in self.W_]
        db = [np.zeros_like(b) for b in self.b_]
        # softmax Jacobian composed with cross-entropy gradient collapses to (ŷ - y)
        delta = acts[-1] - Y
        for i in reversed(range(len(self.W_))):
            dW[i] = acts[i].T @ delta / n
            db[i] = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.W_[i].T) * self._relu_d(zs[i - 1])
        return dW, db

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        X = np.array(X, dtype=float)
        y = np.array(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples, n_features = X.shape

        y_idx = np.searchsorted(self.classes_, y)
        Y     = np.eye(n_classes)[y_idx]

        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [n_classes]
        self._init_params(layer_sizes, rng)

        # Adam state — beta1/beta2/eps are the values from the original paper (Kingma & Ba 2015)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        mW = [np.zeros_like(w) for w in self.W_]
        vW = [np.zeros_like(w) for w in self.W_]
        mb = [np.zeros_like(b) for b in self.b_]
        vb = [np.zeros_like(b) for b in self.b_]

        self.loss_curve_ = []
        step = 0

        for epoch in range(self.max_iter):
            idx    = rng.permutation(n_samples)
            X_s, Y_s = X[idx], Y[idx]
            losses = []

            for start in range(0, n_samples, self.batch_size):
                Xb = X_s[start : start + self.batch_size]
                Yb = Y_s[start : start + self.batch_size]

                zs, acts = self._forward(Xb)
                dW, db   = self._backward(zs, acts, Yb)

                # step counts total mini-batch updates so bias correction reflects actual iterations
                step += 1
                for i in range(len(self.W_)):
                    mW[i] = beta1 * mW[i] + (1 - beta1) * dW[i]
                    vW[i] = beta2 * vW[i] + (1 - beta2) * dW[i] ** 2
                    mb[i] = beta1 * mb[i] + (1 - beta1) * db[i]
                    vb[i] = beta2 * vb[i] + (1 - beta2) * db[i] ** 2
                    mW_h  = mW[i] / (1 - beta1 ** step)   # bias-corrected first moment
                    vW_h  = vW[i] / (1 - beta2 ** step)   # bias-corrected second moment
                    mb_h  = mb[i] / (1 - beta1 ** step)
                    vb_h  = vb[i] / (1 - beta2 ** step)
                    self.W_[i] -= self.lr * mW_h / (np.sqrt(vW_h) + eps)
                    self.b_[i] -= self.lr * mb_h / (np.sqrt(vb_h) + eps)

                loss = -np.mean(np.sum(Yb * np.log(acts[-1] + 1e-12), axis=1))
                losses.append(loss)

            epoch_loss = float(np.mean(losses))
            self.loss_curve_.append(epoch_loss)
            if (epoch + 1) % 10 == 0:
                print(f'  epoch {epoch+1:3d}/{self.max_iter}  loss={epoch_loss:.4f}')

        return self

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict_proba(self, X):
        _, acts = self._forward(np.array(X, dtype=float))
        return acts[-1]

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
