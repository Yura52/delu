import functools
import itertools
from typing import Callable, Iterable

import numpy as np
import torch

from .hardware import to_device


def concat(iterable: Iterable):
    # TODO (docs): the first batch determines everything
    data = iterable if isinstance(iterable, list) else list(iterable)
    assert data, 'iterable must be non-empty'

    def concat_fn(sequence):
        if not isinstance(sequence, list):
            sequence = list(sequence)
        x = sequence[0]
        if x is None or isinstance(x, (int, float)):
            return sequence
        elif isinstance(x, list):
            return list(itertools.chain.from_iterable(sequence))
        elif isinstance(x, np.ndarray):
            return np.concatenate(sequence)
        elif isinstance(x, torch.Tensor):
            return torch.cat(sequence)
        else:
            raise ValueError()

    first = data[0]
    return (
        tuple(concat_fn(x[i] for x in data) for i, _ in enumerate(first))
        if isinstance(first, tuple)
        else dict((key, concat_fn(x[key] for x in data)) for key in first)
        if isinstance(first, dict)
        else concat_fn(data)
    )


def dmap(
    fn: Callable,
    iterable: Iterable,
    in_device=None,
    out_device=None,
    non_blocking=False,
    star=False,
):
    @functools.wraps(fn)
    def wrapper(x):
        if in_device is not None:
            x = to_device(x, in_device, non_blocking)
        result = fn(*x) if star else fn(x)
        if out_device is not None:
            result = to_device(result, out_device, non_blocking)
        return result

    return map(wrapper, iterable)
