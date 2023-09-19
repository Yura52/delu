"""An addition to `torch.nn`."""

import inspect
from typing import Callable

import torch.nn

__all__ = ['Lambda']


class Lambda(torch.nn.Module):
    """A wrapper for functions from `torch` and methods of `torch.Tensor`.

    An important "feature" of this module is that it is intentionally limited:

    - only the functions from the `torch` module and the methods of `torch.Tensor`
      are allowed
    - the passed callable must accept a single `torch.Tensor`
      and return a single `torch.Tensor`
    - the allowed keyword arguments must be of simple types (see the docstring).

    .. note::
        The above limitations are introduced to guarantee
        that `delu.nn.Lambda` modules are always simple and serializable.

    **Usage**

    >>> m = delu.nn.Lambda(torch.squeeze, dim=1)
    >>> m(torch.randn(2, 1, 3, 1)).shape
    torch.Size([2, 3, 1])
    >>> m = delu.nn.Lambda(torch.Tensor.abs)
    >>> m(torch.tensor(-1.0))
    tensor(1.)
    >>> # Custom functions are not allowed.
    >>> m = delu.nn.Lambda(lambda x: torch.abs(x))
    Traceback (most recent call last):
        ...
    ValueError: fn must be a function from `torch` or a method of `torch.Tensor`, but ...
    >>> # Non-trivial keyword arguments are not allowed.
    >>> m = delu.nn.Lambda(torch.mul, other=torch.tensor(2.0))
    Traceback (most recent call last):
        ...
    ValueError: For kwargs, the allowed value types include: ...
    """  # noqa: E501

    def __init__(self, fn: Callable[..., torch.Tensor], /, **kwargs) -> None:
        """
        Args:
            fn: the callable.
            kwargs: the keyword arguments for ``fn``. The allowed values types include:
                None, bool, int, float, bytes, str
                and (nested) tuples of these simple types.
        """
        super().__init__()
        if not callable(fn) or (
            fn not in vars(torch).values()
            and (
                fn not in (member for _, member in inspect.getmembers(torch.Tensor))
                or inspect.ismethod(fn)  # Check if fn is a @classmethod.
            )
        ):
            raise ValueError(
                'fn must be a function from `torch` or a method of `torch.Tensor`,'
                f' but this is not true for the passed {fn=}'
            )

        def is_valid_value(x):
            return (
                x is None
                or isinstance(x, (bool, int, float, bytes, str))
                or isinstance(x, tuple)
                and all(map(is_valid_value, x))
            )

        for k, v in kwargs.items():
            if not is_valid_value(v):
                raise ValueError(
                    'For kwargs, the allowed value types include:'
                    ' None, bool, int, float, bytes, str and (nested) tuples containing'
                    ' values of these simple types. This is not true for the passed'
                    f' argument {k} with the value {v}'
                )

        self._function = fn
        self._function_kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Do the forward pass."""
        return self._function(x, **self._function_kwargs)
