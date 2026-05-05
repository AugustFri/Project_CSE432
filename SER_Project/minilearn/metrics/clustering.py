import numpy as np


def silhouette_score(X, labels):
    """Mean silhouette coefficient for all samples.

    s(i) = (b - a) / max(a, b)
    where a = mean intra-cluster distance, b = mean nearest-cluster distance.
    """
    X = np.array(X, dtype=float)
    labels = np.array(labels)
    unique = np.unique(labels)
    scores = []

    for i in range(len(X)):
        lbl = labels[i]
        same_mask = labels == lbl
        same_mask[i] = False          # exclude self
        if not same_mask.any():
            scores.append(0.0)
            continue

        a = np.mean(np.sqrt(np.sum((X[same_mask] - X[i]) ** 2, axis=1)))

        b_vals = []
        for other_lbl in unique:
            if other_lbl == lbl:
                continue
            other = X[labels == other_lbl]
            b_vals.append(np.mean(np.sqrt(np.sum((other - X[i]) ** 2, axis=1))))
        b = min(b_vals) if b_vals else 0.0

        denom = max(a, b)
        scores.append((b - a) / denom if denom > 0 else 0.0)

    return float(np.mean(scores))
