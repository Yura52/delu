"""An addition to `torch.nn`."""

from typing import Callable

import torch.nn


class Lambda(torch.nn.Module):
    """A parameter-free module for wrapping callables.

    Examples:
        .. testcode::

            module = delu.nn.Lambda(torch.square)
            assert torch.equal(module(torch.tensor(3)), torch.tensor(9))

            # Any callable can be wrapped in Lambda:
            module = delu.nn.Lambda(lambda x, y, z: x + y + z)
            assert module(1, 2, z=3) == 6
    """

    def __init__(self, fn: Callable):
        """Initialize self."""
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        """Perform the forward pass."""
        return self.fn(*args, **kwargs)
