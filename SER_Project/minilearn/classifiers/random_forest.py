import numpy as np
from collections import Counter
from .decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    """Ensemble of CART trees trained on bootstrap samples with feature subsampling.

    Each tree sees a bootstrap replicate of the training data, and at every split
    only a random subset of features (`max_features`) is considered. Predictions
    are made by majority vote across all trees.
    """

    def __init__(self, n_estimators=50, max_depth=None, max_features='sqrt',
                 min_samples_split=2, max_thresholds=32, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.max_thresholds = max_thresholds
        self.random_state = random_state

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y)
        n_samples = X.shape[0]
        rng = np.random.RandomState(self.random_state)

        self.estimators_ = []
        for _ in range(self.n_estimators):
            # bootstrap resample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_thresholds=self.max_thresholds,
                max_features=self.max_features,
                # cast to Python int — numpy int64 breaks RandomState seeding in some builds
                random_state=int(rng.randint(0, 2 ** 31)),
            )
            tree.fit(X_boot, y_boot)
            self.estimators_.append(tree)

        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X = np.array(X, dtype=float)
        # shape: (n_estimators, n_samples)
        all_preds = np.array([t.predict(X) for t in self.estimators_])
        return np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.array(y)))
