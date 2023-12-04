from typing import List
from typing import Optional

import math
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import data, validation


def _identity(X):
    """The identity function."""
    return X


class IdentityTransformer(BaseEstimator, TransformerMixin):
    """Identity Transformer
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return _identity(X)

    def inverse_transform(self, X):
        return _identity(X)


class UnitCircleProjector(BaseEstimator, TransformerMixin):
    """Unit circle projector.

    Projects X into a unit circle by computing its sine and cosine Fourier
    series.

    Parameters
    ----------
    feature_names_out_prefix : str, default=None
        If given, ``feature_names_out`` will have this prefix.
    """

    def __init__(self, feature_names_out_prefix: Optional[str] = None):
        self.feature_names_out_prefix = feature_names_out_prefix

    @property
    def dtype(self):
        return float

    def fit(self, X, y=None):
        self.data_max_ = X.max()
        return self

    def transform(self, X):
        X = sklearn.utils.validation.check_array(X)

        sine = self._sine_transform(X)
        cosine = self._cosine_transform(X)
        return np.concatenate((sine, cosine), axis=1)

    def _sine_transform(self, X):
        """Fourier sine transformation on X.
        """
        return np.sin((2 * np.pi * X) / self.data_max_)

    def _cosine_transform(self, X):
        """Fourier cosine transformation on X.
        """
        return np.cos((2 * np.pi * X) / self.data_max_)

    def get_feature_names_out(self):
        if self.feature_names_out_prefix is None:
            prefix = ''
        else:
            prefix = self.feature_names_out_prefix

        return np.array([prefix + '_sine', prefix + '_cosine'])


def inverse_transform_sliding_sequences(
        sliding_sequences: np.ndarray,
        sequence_length: int
):
    """Inverse transformation for sliding sequences data structure.

       Parameters
       ----------
        sliding_sequences : np.ndarray
           Numpy ndarray containing arrays in sliding format.

        sequence_length : int
            Length of each sequence.

       Returns
       -------
       inv : np.ndarray

       Examples
       --------
       # Undo sliding window with sequence_length=3 and step=1
       >>> sliding_seqs = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
       >>> inverse_transform_sliding_sequences(sliding_seqs, 3)
       array([[1],
              [2],
              [3],
              [4],
              [5]])
       """
    validation.check_nd(sliding_sequences)

    output_length = len(sliding_sequences)
    sequences = sliding_sequences[::sequence_length]
    sequences = data.collapse_first_dim(sequences)

    mod = (output_length - 1) % sequence_length
    if mod > 0:
        # Get the last element of each of the last `mod` sequences.
        mod_sequences = sliding_sequences[-mod:, -1]
        sequences = np.concatenate((sequences, mod_sequences))

    return sequences


class SlidingWindowTransformer(BaseEstimator, TransformerMixin):
    """Transforms n-dimensional data into sliding sequences.

    Parameters
    ----------
    columns : list of str
        Columns for which the sliding sequences will be created

    sequence_length : int
        Length for each sequence

    step : int
        Gap between sequences
    """

    def __init__(self, columns: List[str], sequence_length: int, step: int):
        self.columns = columns
        self.sequence_length = sequence_length
        self.step = step

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Transforms n-dimensional data into sliding sequences

        Parameters
        ----------
        X : pd.DataFrame

        Returns
        -------
        np.ndarray
            np.ndarray of shape (N, ``sequence_length``, ``columns``) where
            N is the total number of sliding sequences created
        """
        Xs = self._fill_sliding_seqs(X)
        Xs = self._slice_sliding_seqs(Xs, len(X))
        return Xs

    def inverse_transform(self, X):
        """Undoes sliding window and recovers original ``X``

        Parameters
        ----------
        X : numpy ndarray

        Returns
        -------
        inv trans X : np.ndarray
        """
        return inverse_transform_sliding_sequences(X, self.sequence_length)

    def _fill_sliding_seqs(self, X):
        """The sliding sequences are populated using the pandas shift operator.
        """
        numeric_cols = X[self.columns].select_dtypes(include=np.number).columns
        dtype = float if len(numeric_cols) == len(self.columns) else object
        # The ceil operator allows to keep the last possible sequence
        # containing nans
        len_sliding_sequences = math.ceil(len(X) / self.step)
        # Holder for sliding sequences
        sliding_sequences = np.zeros(
            (len_sliding_sequences, self.sequence_length, len(self.columns)),
            dtype=dtype
        )
        for j in range(self.sequence_length):
            sliding_sequences[:, j, :] = X[self.columns].shift(-j)[::self.step]
        return sliding_sequences

    def _slice_sliding_seqs(self, X, original_length):
        """Drops sliding windows nans.

        There are two sources of nans:

        - Those generated by the shift operator in _fill_sliding_sequences
        - The training sequences whose target sequences will contain nans
        """
        all_starting_index = np.array(range(original_length)[::self.step])
        all_ending_lengths = all_starting_index + self.sequence_length
        arrays_with_nan = len(
            all_ending_lengths[all_ending_lengths > original_length]
        )
        return X[0:-arrays_with_nan] if arrays_with_nan != 0 else X

    def _more_tags(self):
        return {'stateless': True}
