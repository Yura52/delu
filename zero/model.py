r"""Tools for working with `torch.nn.Module`."""

__all__ = ['Eval']

from typing import ClassVar, List, Optional

import torch


class _ModelsContext:
    _train: ClassVar[bool] = NotImplemented
    _grad: ClassVar[bool] = NotImplemented

    def __init__(self, *models: torch.nn.Module) -> None:
        assert models
        self._models = models
        self._training: Optional[List[bool]] = None
        self._grad_context = torch.enable_grad() if self._grad else torch.no_grad()

    def __enter__(self) -> None:
        self._training = []
        for x in self._models:
            self._training.append(x.training)
            x.train(self._train)
        self._grad_context.__enter__()  # type: ignore

    def __exit__(self, *args) -> bool:  # type: ignore
        for model, train in zip(self._models, self._training):  # type: ignore
            model.train(train)
        self._grad_context.__exit__(*args)  # type: ignore
        return False


class Eval(_ModelsContext):
    r"""A context-manager for models evaluation.

    Switches one or more models to the evaluation mode and turns off gradients
    **when enters a context** (not when constructed!) and reverts all the changes to the
    previous state when exits the context.

    Args:
        *models (`torch.nn.Module`)

    Examples:
        .. testcode::

            a = torch.nn.Linear(1, 1)
            b = torch.nn.Linear(2, 2)
            with Eval(a):
                ...
            with Eval(a, b):
                ...

    .. rubric:: Tutorial

    .. testcode::

        model = torch.nn.Linear(1, 1)
        grad_before_context = torch.is_grad_enabled()
        for training_before_context in False, True:
            model.train(training_before_context)
            with Eval(model):
                assert not model.training
                assert not torch.is_grad_enabled()
            assert model.training == training_before_context
            assert torch.is_grad_enabled() == grad_before_context
    """
    _train = False
    _grad = False
