r"""Tools for working with `torch.Tensor`."""

__all__ = ['ibackward', 'to_device']

import torch

from ._util import traverse
from .types import Device, Recursive


def ibackward(x: torch.Tensor, *args, **kwargs) -> float:
    """Do :code:`.backward()` and return :code:`.item()`.

    The function is useful when doing backward and extracting value are the only things
    you want from a loss tensor (see examples).

    Args:
        x: Tensor.
        *args: positional arguments for `torch.Tensor.backward`
        **kwargs: keyword arguments for `torch.Tensor.backward`
    Returns:
        The underlying scalar value.

    Examples:
        Before:

        .. code-block::

            loss = loss_fn(model(X), y)
            loss.backward()
            loss = loss.item()

        After:

        .. code-block::

            loss = ibackward(loss_fn(model(X), y))
    """
    x.backward(*args, **kwargs)
    return x.item()


def to_device(
    data: Recursive[torch.Tensor], device: Device, non_blocking: bool = False
) -> Recursive[torch.Tensor]:
    """Move tensor(s) to device.

    Move data consisting of tensors to the given device using `torch.Tensor.to`.

    Args:
        data (`Recursive[torch.Tensor] <zero.types.Recursive>`)
        device (`Device <zero.types.Device>`)
        non_blocking: is forwarded to `torch.Tensor.to`
    Returns:
        `Recursive[torch.Tensor] <zero.types.Recursive>`:
            The same data, but moved to the given device.

    Examples:
        .. testcode ::

            to_device(torch.tensor(0), 'cpu')
            to_device({'a': torch.tensor(0), 'b': [(torch.tensor(0),)]}, 'cpu')
    """
    # int is missing in .pyi
    return traverse(lambda x: x.to(device, non_blocking=non_blocking), data)  # type: ignore
