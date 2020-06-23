__all__ = ['ibackward', 'to_device']

import torch

from ._util import traverse
from .types import Device, Recursive


def ibackward(x: torch.Tensor, *args, **kwargs) -> float:
    x.backward(*args, **kwargs)
    return x.item()


def to_device(
    data: Recursive[torch.Tensor], device: Device, non_blocking: bool = False
) -> Recursive[torch.Tensor]:
    # int is missing in .pyi
    return traverse(lambda x: x.to(device, non_blocking=non_blocking), data)  # type: ignore
