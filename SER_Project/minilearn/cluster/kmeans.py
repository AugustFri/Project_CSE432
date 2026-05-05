import numpy as np


class KMeans:
    """K-Means clustering with k-means++ initialisation."""

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, n_init=10, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _init_centroids(self, X, rng):
        """k-means++ seeding: spread initial centroids with prob ∝ distance²."""
        idx = rng.integers(0, len(X))
        centroids = [X[idx]]
        for _ in range(self.n_clusters - 1):
            C = np.array(centroids)
            # squared distance from each point to its nearest centroid
            diffs = X[:, np.newaxis, :] - C[np.newaxis, :, :]   # (n, k, d)
            sq_dists = np.min(np.sum(diffs ** 2, axis=2), axis=1)  # (n,)
            probs = sq_dists / sq_dists.sum()
            centroids.append(X[rng.choice(len(X), p=probs)])
        return np.array(centroids)

    def _assign(self, X, centroids):
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]  # (n, k, d)
        return np.argmin(np.sum(diffs ** 2, axis=2), axis=1)        # (n,)

    def _run_once(self, X, rng):
        centroids = self._init_centroids(X, rng)
        for _ in range(self.max_iter):
            labels = self._assign(X, centroids)
            new_centroids = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(self.n_clusters)
            ])
            if np.max(np.abs(new_centroids - centroids)) < self.tol:
                centroids = new_centroids
                break
            centroids = new_centroids
        inertia = float(sum(
            np.sum((X[labels == k] - centroids[k]) ** 2)
            for k in range(self.n_clusters) if np.any(labels == k)
        ))
        return labels, centroids, inertia

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def fit(self, X):
        X = np.array(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        best = (None, None, float('inf'))
        for _ in range(self.n_init):
            labels, centroids, inertia = self._run_once(X, rng)
            if inertia < best[2]:
                best = (labels, centroids, inertia)
        self.labels_, self.cluster_centers_, self.inertia_ = best
        return self

    def predict(self, X):
        return self._assign(np.array(X, dtype=float), self.cluster_centers_)

    def fit_predict(self, X):
        return self.fit(X).labels_
