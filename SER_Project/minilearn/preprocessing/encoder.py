import numpy as np


class LabelEncoder:
    """Map arbitrary class labels to integer indices 0..n-1."""

    def fit(self, y):
        self.classes_ = np.unique(y)
        self._map = {c: i for i, c in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self._map[c] for c in y])

    def fit_transform(self, y):
        return self.fit(y).transform(y)

    def inverse_transform(self, y):
        return self.classes_[np.array(y)]
