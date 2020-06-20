__all__ = [  # noqa
    'ASGD',
    'Adadelta',
    'Adagrad',
    'Adam',
    'AdamW',
    'Adamax',
    'RMSprop',
    'Rprop',
    'SGD',
    'SparseAdam',
]

import inspect

import torch.optim as optim


class _ZeroOptimizer:
    # TODO (docs): step only if no exceptions
    def __enter__(self) -> None:
        self.zero_grad()  # type: ignore

    def __exit__(self, *args) -> bool:  # type: ignore
        if args == (None, None, None):
            self.step()  # type: ignore
        return False


def make_zero_optimizer(cls):
    return type(cls.__name__, (cls, _ZeroOptimizer), {})


def _is_supported_optimizer_name(name):
    if (
        name not in dir(optim)
        or name.startswith(('_', 'Base'))
        or name.endswith('Base')
        or 'Mixin' in name
    ):
        return False
    cls = getattr(optim, name)
    return (
        inspect.isclass(cls)
        and cls is not optim.Optimizer  # type: ignore
        and issubclass(cls, optim.Optimizer)  # type: ignore
        # Optimizers that require closure are not supported
        and inspect.signature(cls.step).parameters['closure'].default is None
    )


_OPTIMIZER_NAMES = list(filter(_is_supported_optimizer_name, __all__))


for name in _OPTIMIZER_NAMES:
    globals()[name] = make_zero_optimizer(getattr(optim, name))
