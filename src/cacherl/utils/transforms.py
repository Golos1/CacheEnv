from typing import Iterable

def recursive_tuple(arr):
    if not isinstance(arr, Iterable):
        return arr
    return tuple(recursive_tuple(element) for element in arr)
