from contextlib import ExitStack

import torch

from ._util import to_list, traverse


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
        for model, train in zip(self._models, self._training):
            model.train(train)


class TrainContext:
    def __init__(self, models, optimizers):
        self._models = to_list(models)
        self._optimizers = to_list(optimizers)
        self._backward = False
        self._exit_stack = None

    @property
    def _in_context(self):
        return self._exit_stack is not None

    def __enter__(self):
        assert not self._in_context
        self._exit_stack = ExitStack().__enter__()
        self._exit_stack.enter_context(_ModelsContext(self._models, True))
        self._exit_stack.enter_context(torch.enable_grad())
        for x in self._optimizers:
            x.zero_grad()
        self._backward = False
        return self

    def backward(self, loss):
        assert self._in_context
        assert not self._backward

        def f(x):
            x.backward()
            return x.item()

        result = traverse(f, loss)
        self._backward = True
        return result

    def __exit__(self, *args):
        # TODO (docs): mention that .step() is performed only if no exceptions occur
        if args == (None, None, None):
            assert self._backward
            for x in self._optimizers:
                x.step()
        self._backward = False
        self._exit_stack.__exit__(*args)
        self._exit_stack = None


class EvalContext:
    def __init__(self, models, metric):
        self._models = to_list(models)
        self._metric = metric
        self._exit_stack = None

    def __enter__(self):
        self._exit_stack = ExitStack().__enter__()
        self._exit_stack.enter_context(_ModelsContext(self._models, False))
        self._exit_stack.enter_context(torch.no_grad())
        if self._metric is not None:
            self._metric.reset()

    def __exit__(self, *args):
        self._exit_stack.__exit__(*args)
        self._exit_stack = None
