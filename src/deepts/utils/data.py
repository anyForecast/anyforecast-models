import numpy as np


def hstack(arrays, cast_to_object=True):
    """Horizontally stacks numpy arrays.

    Parameters
    ----------
    arrays : sequence of ndarrays
        The arrays must have the same shape along all but the second axis,
        except 1-D arrays which can be any length.

    cast_to_object : bool, default=True
        If ``np.stack`` raises TypeError, converts all arrays to object
        dtype and tries again.

    Returns
    -------
    stacked : ndarray
        The array formed by stacking the given arrays.
    """
    try:
        return np.hstack(arrays)
    except TypeError as e:
        if cast_to_object:
            obj_arrays = [arr.astype(object) for arr in arrays]
            return np.hstack(obj_arrays)
        raise e
