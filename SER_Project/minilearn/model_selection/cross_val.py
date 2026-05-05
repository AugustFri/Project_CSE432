import copy
import numpy as np
from .kfold import KFold
from ..metrics import accuracy_score, f1_score


def cross_val_score(estimator, X, y, cv=5, scoring='accuracy', random_state=0):
    """Evaluate an estimator using k-fold cross-validation.

    Returns an array of scores, one per fold.
    scoring: 'accuracy' | 'f1_macro'
    """
    X, y = np.array(X), np.array(y)
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in kf.split(X):
        clf = copy.deepcopy(estimator)
        clf.fit(X[train_idx], y[train_idx])
        y_pred = clf.predict(X[test_idx])
        if scoring == 'f1_macro':
            scores.append(f1_score(y[test_idx], y_pred, average='macro'))
        else:
            scores.append(accuracy_score(y[test_idx], y_pred))
    return np.array(scores)
