import functools
import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import numpy as np
import torch

from .hardware import to_device
from .types import Device, S, T

OutputItem = Union[List, np.ndarray, torch.Tensor]


def concat(
    iterable: Iterable[T],
) -> Union[OutputItem, Tuple[OutputItem], Dict[Any, OutputItem]]:
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
    fn: Union[Callable[[T], S], Callable[..., S]],
    iterable: Iterable[T],
    in_device: Device = None,
    out_device: Device = None,
    non_blocking: bool = False,
    star: bool = False,
) -> Iterator[S]:
    @functools.wraps(fn)
    def wrapper(x):
        if in_device is not None:
            x = to_device(x, in_device, non_blocking)
        result = fn(*x) if star else fn(x)
        if out_device is not None:
            # mypy: NaN
            result = to_device(result, out_device, non_blocking)  # type: ignore
        return result

    return map(wrapper, iterable)
