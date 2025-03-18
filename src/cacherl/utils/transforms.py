import collections
from collections.abc import Iterable, Sequence


def recursive_tuple(arr):
    if not isinstance(arr, Iterable):
        return arr
    return tuple(recursive_tuple(element) for element in arr)

def flatten_collection(nested_collection):
    flattened = []
    for item in nested_collection:
        if isinstance(item, collections.abc.Iterable):
            flattened.extend(flatten_collection(item))
        else:
            flattened.append(item)
    return tuple(flattened)