"""Easier training and evaluation."""

__all__ = ['evaluation']

import torch
import torch.nn as nn


class evaluation(torch.no_grad):
    """Context-manager for models evaluation. Can be used as a decorator.

    Note:
        The training status of modules is undefined once a context is finished or a
        decorated function returns.

    Warning:
        The function must be used only as a context manager or a decorator as shown
        below in the examples. Otherwise, the behaviour is undefined.

    This code...::

        with evaluation(model):
            ...

        @evaluation(model)
        def f():
            ...

    ...is equivalent to ::

        model.eval()
        with torch.no_grad():
            ...

        @torch.no_grad()
        def f():
            model.eval()

    Args:
        modules

    Examples:
        .. testcode::

            a = torch.nn.Linear(1, 1)
            b = torch.nn.Linear(2, 2)
            with evaluation(a):
                ...
            with evaluation(a, b):
                ...

            @evaluate(a)
            def f():
                ...

            @evaluate(a, b)
            def f():
                ...

        .. testcode::

            model = torch.nn.Linear(1, 1)
            for grad in False, True:
                for train in False, True:
                    torch.set_grad_enabled(grad)
                    model.train(train)
                    with evaluation(model):
                        assert not model.training
                        assert not torch.is_grad_enabled()
                        ...
                    assert torch.is_grad_enabled() == grad_before_context
                    # model.training is unspecified here
    """

    def __init__(self, *modules: nn.Module) -> None:
        assert modules
        self._modules = modules

    def __enter__(self) -> None:
        result = super().__enter__()
        for m in self._modules:
            m.eval()
        return result
