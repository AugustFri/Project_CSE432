import numpy as np


def train_test_split(*arrays, test_size=0.2, random_state=None):
    """Split arrays into random train and test subsets."""
    rng = np.random.default_rng(random_state)
    n = len(arrays[0])
    idx = rng.permutation(n)
    n_test = max(1, int(np.ceil(n * test_size)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    result = []
    for arr in arrays:
        arr = np.array(arr)
        result.append(arr[train_idx])
        result.append(arr[test_idx])
    return result
