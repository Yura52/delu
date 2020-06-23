__all__ = ['dmap', 'concat']

import itertools
from typing import Any, Callable, Dict, Iterable, Iterator, List, Tuple, Union

import numpy as np
import torch

from ._util import is_namedtuple
from .tensor import to_device
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
        return (
            list(itertools.chain.from_iterable(sequence))
            if isinstance(x, list)
            else np.concatenate(sequence)
            if isinstance(x, np.ndarray)
            else torch.cat(sequence)
            if isinstance(x, torch.Tensor)
            else sequence
        )

    first = data[0]
    return (
        type(first)._make(concat_fn(x[i] for x in data) for i, _ in enumerate(first))
        if is_namedtuple(first)
        else type(first)(concat_fn(x[i] for x in data) for i, _ in enumerate(first))
        if isinstance(first, tuple)
        else type(first)((key, concat_fn(x[key] for x in data)) for key in first)
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
    def wrapper(x):
        if in_device is not None:
            x = to_device(x, in_device, non_blocking)
        result = fn(*x) if star else fn(x)
        if out_device is not None:
            # mypy: NaN
            result = to_device(result, out_device, non_blocking)  # type: ignore
        return result

    return map(wrapper, iterable)
