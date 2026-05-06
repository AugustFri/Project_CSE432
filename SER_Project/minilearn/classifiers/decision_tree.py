import numpy as np
from collections import Counter


class _Node:
    __slots__ = ('feature', 'threshold', 'left', 'right', 'value')

    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # non-None only in leaf nodes


class DecisionTreeClassifier:
    """CART decision tree using Gini impurity (or entropy)."""

    def __init__(self, max_depth=None, min_samples_split=2, criterion='gini',
                 max_thresholds=64, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        # cap candidate thresholds per feature to keep training tractable
        self.max_thresholds = max_thresholds
        # number of features to consider at each split; 'sqrt', int, or None (all)
        self.max_features = max_features
        self.random_state = random_state

    # ------------------------------------------------------------------
    # impurity
    # ------------------------------------------------------------------

    def _impurity(self, y):
        counts = np.array(list(Counter(y).values()), dtype=float)
        p = counts / counts.sum()
        if self.criterion == 'gini':
            return 1.0 - float(np.sum(p ** 2))
        return float(-np.sum(p * np.log2(p + 1e-12)))  # entropy

    # ------------------------------------------------------------------
    # split search
    # ------------------------------------------------------------------

    def _feature_indices(self, n_features):
        """Return the subset of feature indices to evaluate at this split."""
        if self.max_features is None:
            return range(n_features)
        if self.max_features == 'sqrt':
            k = max(1, int(np.sqrt(n_features)))
        elif isinstance(self.max_features, float):
            k = max(1, int(self.max_features * n_features))
        else:
            k = max(1, min(int(self.max_features), n_features))
        return self._rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y):
        best_gain, best_feat, best_thresh = -1.0, None, None
        n = len(y)
        parent_imp = self._impurity(y)

        for feat in self._feature_indices(X.shape[1]):
            vals = np.unique(X[:, feat])
            # use percentile midpoints when there are many unique values
            if len(vals) > self.max_thresholds:
                percs = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
                thresholds = np.percentile(X[:, feat], percs)
            else:
                # midpoints between sorted unique values
                thresholds = (vals[:-1] + vals[1:]) / 2

            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) == 0 or len(right) == 0:
                    continue
                gain = parent_imp - (len(left) / n * self._impurity(left) +
                                     len(right) / n * self._impurity(right))
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, thresh

        return best_feat, best_thresh

    # ------------------------------------------------------------------
    # recursive build
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        # stopping conditions → leaf
        if (len(y) < self.min_samples_split or
                len(np.unique(y)) == 1 or
                (self.max_depth is not None and depth >= self.max_depth)):
            return _Node(value=Counter(y).most_common(1)[0][0])

        feat, thresh = self._best_split(X, y)
        if feat is None:
            return _Node(value=Counter(y).most_common(1)[0][0])

        left_mask = X[:, feat] <= thresh
        return _Node(
            feature=feat,
            threshold=thresh,
            left=self._build(X[left_mask], y[left_mask], depth + 1),
            right=self._build(X[~left_mask], y[~left_mask], depth + 1),
        )

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.n_features_ = np.array(X).shape[1]
        self._rng = np.random.RandomState(self.random_state)
        self.root_ = self._build(np.array(X, dtype=float), np.array(y), depth=0)
        return self

    def _traverse(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse(x, node.left)
        return self._traverse(x, node.right)

    def predict(self, X):
        X = np.array(X, dtype=float)
        return np.array([self._traverse(x, self.root_) for x in X])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
