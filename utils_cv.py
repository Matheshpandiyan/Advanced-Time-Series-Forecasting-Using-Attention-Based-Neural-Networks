import numpy as np
from sklearn.model_selection import TimeSeriesSplit

def rolling_origin_splits(n_samples, n_splits=5, train_size=None, test_size=None):
    "+""Return list of (train_idx, val_idx) pairs for rolling-origin CV."""
    if train_size is None:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        return list(tscv.split(np.arange(n_samples)))
    else:
        if test_size is None:
            raise ValueError('test_size must be provided when using fixed train_size')
        splits = []
        start = 0
        while (start + train_size + test_size) <= n_samples:
            train_idx = np.arange(start, start+train_size)
            val_idx = np.arange(start+train_size, start+train_size+test_size)
            splits.append((train_idx, val_idx))
            start += test_size
        return splits
