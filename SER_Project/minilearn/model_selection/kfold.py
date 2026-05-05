import numpy as np


class KFold:
    """K-Fold cross-validator."""

    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n = len(X)
        idx = np.arange(n)
        if self.shuffle:
            idx = np.random.default_rng(self.random_state).permutation(idx)
        fold_sizes = np.full(self.n_splits, n // self.n_splits)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for size in fold_sizes:
            test_idx = idx[current : current + size]
            train_idx = np.concatenate([idx[:current], idx[current + size :]])
            yield train_idx, test_idx
            current += size
