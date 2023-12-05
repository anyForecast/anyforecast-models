import numpy as np


class WalkForwardCV:
    """Cross-validator for grouped Time Series

    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals, in train/test sets.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.

    Parameters
    ----------
    group_ids : list of str
         List of column names identifying a time series

    max_train_size : int, default=None
        Maximum size for a single training set.

    test_size : int, default=None
        Used to limit the size of the test set. Defaults to
        ``n_samples // (n_splits + 1)``, which is the maximum allowed value.
    """

    def __init__(self, group_ids, n_splits=5, max_train_size=None,
                 test_size=None):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.group_ids = group_ids
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, X, y=None, groups=None):
        """
        Parameters
        ----------
        X : Slice dataset
            Slice version of training dataset. Contains both training and
            validation/test data

        y : None
            Always ignored, exists for compatibility.

        groups : None
            Always ignored, exists for compatibility.

        Returns
        -------
        train-test indices : list of (train, test) tuples
        """
        groups_count = X.decoded_index.groupby(
            self.group_ids, sort=False).size().to_dict()
        indices = dict.fromkeys(
            range(self.n_splits),
            (np.array([], dtype=int), np.array([], dtype=int))
        )
        j = 0
        for groups, count in groups_count.items():
            dummy_array = np.zeros((count, 1))
            tscv = sklearn.model_selection.TimeSeriesSplit(
                n_splits=self.n_splits,
                max_train_size=self.max_train_size,
                test_size=self.test_size
            )
            for i, (train_idx, test_idx) in enumerate(tscv.split(dummy_array)):
                i_split_train, i_split_test = indices[i]
                i_split_train = np.append(i_split_train, train_idx + j)
                i_split_test = np.append(i_split_test, test_idx + j)
                indices[i] = i_split_train, i_split_test
            j += count
        return list(indices.values())
