from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple, TypeVar, Union

import numpy as np
import torch

T = TypeVar('T')
S = TypeVar('S')
Number = TypeVar('Number', int, float)
PathLike = Union[Path, bytes, str]
ArrayIndex = Union[int, slice, List[int], np.ndarray]
TensorIndex = Union[ArrayIndex, torch.Tensor]

OneOrList = Union[T, List[T]]
# mypy cannot resolve recursive types
Recursive = Union[T, Tuple['Recursive'], List['Recursive'], Dict[Any, 'Recursive']]  # type: ignore
JSON = Union[None, bool, int, float, str, List['JSON'], Mapping[str, 'JSON']]  # type: ignore

# Inspired by:
# https://github.com/pytorch/pytorch/blob/ebdff07d4910eea464c0808e84861c14b7a3e270/torch/types.py#L31
# Not available in torch 1.5
Device = Union[torch.device, int, str, None]
