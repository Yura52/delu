"""Missing batteries from `torch.nn`."""

from typing import Callable

import torch.nn


class Lambda(torch.nn.Module):
    """A parameters-free module for wrapping callables.

    Examples:

        .. testcode::

            assert zero.nn.Lambda(lambda: 0)() == 0
            assert zero.nn.Lambda(lambda x: x)(1) == 1
            assert zero.nn.Lambda(lambda x, y, z: x + y + z)(1, 2, z=3) == 6
    """

    def __init__(self, fn: Callable):
        """Initialize self."""
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Perform the forward pass."""
        return self.fn(*args, **kwargs)
