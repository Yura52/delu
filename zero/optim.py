__all__ = [
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
    # TODO (docs): optimizers that require closure in .step() are not supported
    return type(cls.__name__, (cls, _ZeroOptimizer), {})


# for whatever reasons, mypy doesn't undestand what is going on
ASGD = make_zero_optimizer(optim.ASGD)  # type: ignore
Adadelta = make_zero_optimizer(optim.Adadelta)  # type: ignore
Adagrad = make_zero_optimizer(optim.Adagrad)  # type: ignore
Adam = make_zero_optimizer(optim.Adam)  # type: ignore
AdamW = make_zero_optimizer(optim.AdamW)  # type: ignore
Adamax = make_zero_optimizer(optim.Adamax)  # type: ignore
RMSprop = make_zero_optimizer(optim.RMSprop)  # type: ignore
Rprop = make_zero_optimizer(optim.Rprop)  # type: ignore
SGD = make_zero_optimizer(optim.SGD)  # type: ignore
SparseAdam = make_zero_optimizer(optim.SparseAdam)  # type: ignore
