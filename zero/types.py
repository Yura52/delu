"""Types used throughout Zero."""

__all__ = ['PathLike', 'ArrayIndex', 'TensorIndex', 'Recursive', 'JSON', 'Device']

from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, TypeVar, Union

import numpy as np
import torch

T = TypeVar('T')


PathLike = Union[Path, bytes, str]
""""""
ArrayIndex = Union[int, slice, List[int], np.ndarray]
""""""
TensorIndex = Union[ArrayIndex, torch.Tensor]
""""""

# mypy cannot resolve recursive types
Recursive = Union[T, Tuple['Recursive', ...], List['Recursive'], Dict[Any, 'Recursive']]  # type: ignore
"""
.. note::
    The following values are all "instances" of `Recursive[int]`:

    .. testcode::

        0
        (0, 1)
        [0, 1, 2]
        {'a': 0, 1: 2}
        [[[0], (1, 2, (3,)), {'a': {'b': [4]}}]]

        from collections import namedtuple
        Point = namedtuple('Point', ['x', 'y'])
        Point(0, 1)  # also `Recursive[int]`
"""

JSON = Union[None, bool, int, float, str, List['JSON'], Mapping[str, 'JSON']]  # type: ignore
"""
.. note::
    The following values are all "instances" of `JSON`:

    .. testcode::

        True
        0
        1.0
        'abc'
        [0, 1.0]
        {'a': [0, 1.0], 'b': False, 'c': 'abc'}
"""

# Inspired by:
# https://github.com/pytorch/pytorch/blob/ebdff07d4910eea464c0808e84861c14b7a3e270/torch/types.py#L31
# Not available in torch 1.5
Device = Union[torch.device, int, str, None]
""""""
