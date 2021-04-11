"""Easier training process."""

__all__ = ['evaluation']

import contextlib

import torch
import torch.nn as nn


@contextlib.contextmanager
def _evaluation(*modules: nn.Module):
    assert modules
    for x in modules:
        x.eval()
    no_grad_context = torch.no_grad()
    no_grad_context.__enter__()
    try:
        yield
    finally:
        no_grad_context.__exit__(None, None, None)


def evaluation(*modules: nn.Module):
    """Context-manager for models evaluation.

    Warning:
        The function must be used only as a context manager as shown below in the
        examples. The behaviour for call without the `with` keyword is unspecified.

    This code...::

        model.eval()
        with torch.no_grad():
            ...

    ...is equivalent to ::

        with evaluation(model):
            ...

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
    return _evaluation(*modules)
