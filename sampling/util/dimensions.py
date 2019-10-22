from collections import Iterable, MutableMapping
import numpy as np


def ensure_iterable(arg, length=1):
    """Helper function to make argument iterable before continuing."""

    if isinstance(arg, Iterable) and not (isinstance(arg, MutableMapping) or
                                          isinstance(arg, str)):
        assert length in (len(arg),1), \
            'mismatched lengths: expected {}, got {}'.format(length,len(arg))
        return arg
    else:
        return [arg]*length


def ensure_1d_numeric(a):
    array = np.asarray(a)
    if array.dtype == 'object':
        raise ValueError('array is not numeric: element type is {}'
                         ''.format(type(array[0])))
    if array.ndim != 1:
        raise ValueError('array is not a 1d vector: shape is {}'
                         ''.format(array.shape))
    return array
