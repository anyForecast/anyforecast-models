from typing import (
    List,
    Optional,
    Union
)
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset as TorchDataset


class SliceDataset(Sequence, TorchDataset):
    """Makes Dataset sliceable.

    Helper class that wraps a torch dataset to make it work with
    sklearn. That is, sometime sklearn will touch the input data, e.g. when
    splitting the data for a grid search. This will fail when the input data is
    a torch dataset. To prevent this, use this wrapper class for your
    dataset.

    ``dataset`` attributes are also available from :class:`SliceDataset`
    object (see Examples section).

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
      A valid torch dataset.

    indices : list, np.ndarray, or None (default=None)
      If you only want to return a subset of the dataset, indicate
      which subset that is by passing this argument. Typically, this
      can be left to be None, which returns all the data.



    Notes
    -----
    This class will only return the X value by default (i.e. the
    first value returned by indexing the original dataset). Sklearn,
    and hence skorch, always require 2 values, X and y. Therefore, you
    still need to provide the y data separately.

    This class behaves similarly to a PyTorch
    :class:`~torch.utils.data.Subset` when it is indexed by a slice or
    numpy array: It will return another ``SliceDataset`` that
    references the subset instead of the actual values. Only when it
    is indexed by an int does it return the actual values. The reason
    for this is to avoid loading all data into memory when sklearn,
    for instance, creates a train/validation split on the
    dataset. Data will only be loaded in batches during the fit loop.
    """

    def __init__(
            self,
            dataset: TorchDataset,
            indices: Optional[Union[List[int], np.array]] = None
    ):
        self.dataset = dataset
        self.indices = indices
        self.indices_ = (
            self.indices if self.indices is not None
            else np.arange(len(self.dataset))
        )
        self.ndim = 1

    @property
    def shape(self):
        return len(self)

    def transform(self, data):
        """Additional transformations on ``data``.

        Notes
        -----
        If you use this in conjunction with PyTorch
        :class:`~torch.utils.data.DataLoader`, the latter will call
        the dataset for each row separately, which means that the
        incoming ``data`` is a single rows.

        """
        return data

    def __getattr__(self, attr):
        """If attr is not in self, look in self.dataset.

        Notes
        -----
        Issues with serialization were solved with the following discussion:
        https://stackoverflow.com/questions/49380224/how-to-make-classes-with-getattr-pickable
        """
        if 'dataset' not in vars(self):
            raise AttributeError
        return getattr(self.dataset, attr)

    def __len__(self):
        return len(self.indices_)

    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xn = self.dataset[self.indices_[i]]
            return self.transform(Xn)
        if isinstance(i, slice):
            return SliceDataset(self.dataset, indices=self.indices_[i])
        if isinstance(i, np.ndarray):
            if i.ndim != 1:
                raise IndexError(
                    "SliceDataset only supports slicing with 1 "
                    "dimensional arrays, got {} dimensions "
                    "instead".format(i.ndim)
                )
            if i.dtype == np.bool:
                i = np.flatnonzero(i)
        return SliceDataset(self.dataset, indices=self.indices_[i])
