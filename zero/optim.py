_optimizers = [
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
__all__ = _optimizers

import torch.optim as optim  # noqa


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


for name in _optimizers:
    globals()[name] = make_zero_optimizer(getattr(optim, name))
