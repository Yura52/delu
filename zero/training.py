from contextlib import ExitStack
from typing import Optional

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from ._util import to_list
from .metrics import Metric
from .types import OneOrList


class _ModelsContext:
    def __init__(self, models, train):
        self._models = models
        self._train = train
        self._training = None

    def __enter__(self):
        self._training = []
        for x in self._models:
            self._training.append(x.training)
            x.train(self._train)

    def __exit__(self, *args):
        for model, train in zip(self._models, self._training):  # type: ignore
            model.train(train)


class TrainContext:
    def __init__(
        self,
        models: OneOrList[nn.Module],
        optimizers: OneOrList[Optimizer],
        n_backwards: int = 1,
    ) -> None:
        assert n_backwards >= 0
        self._models = to_list(models)
        self._optimizers = to_list(optimizers)
        self._n_backwards = n_backwards
        self._current_n_backwards = 0
        self._exit_stack: Optional[ExitStack] = None

    @property
    def _in_context(self) -> bool:
        return self._exit_stack is not None

    def __enter__(self) -> 'TrainContext':
        assert not self._in_context
        self._exit_stack = ExitStack().__enter__()
        self._exit_stack.enter_context(_ModelsContext(self._models, True))
        self._exit_stack.enter_context(torch.enable_grad())
        for x in self._optimizers:
            x.zero_grad()
        return self

    def backward(self, x: torch.Tensor, *args, **kwargs) -> float:
        assert self._in_context
        assert self._current_n_backwards < self._n_backwards
        x.backward(*args, **kwargs)
        self._current_n_backwards += 1
        return x.item()

    # https://github.com/python/mypy/pull/7655
    def __exit__(self, *args) -> bool:  # type: ignore
        # TODO (docs): mention that .step() is performed only if no exceptions occur
        if args == (None, None, None):
            assert self._current_n_backwards == self._n_backwards
            for x in self._optimizers:
                x.step()
        self._current_n_backwards = 0
        assert self._exit_stack is not None  # help mypy
        self._exit_stack.__exit__(*args)
        self._exit_stack = None
        return False


class EvalContext:
    def __init__(self, models: OneOrList[nn.Module], metric: Optional[Metric]) -> None:
        self._models = to_list(models)
        self._metric = metric
        self._exit_stack: Optional[ExitStack] = None

    def __enter__(self) -> None:
        self._exit_stack = ExitStack().__enter__()
        self._exit_stack.enter_context(_ModelsContext(self._models, False))
        self._exit_stack.enter_context(torch.no_grad())
        if self._metric is not None:
            self._metric.reset()

    def __exit__(self, *args) -> bool:  # type: ignore
        assert self._exit_stack is not None  # help mypy
        self._exit_stack.__exit__(*args)
        self._exit_stack = None
        return False
