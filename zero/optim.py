"""Additional features for `torch.optim.Optimizer`.

Replace this::

    from torch.optim import SGD  # or any other optimizer

with this::

    from zero.optim import SGD  # or any other optimizer

and enjoy additional features. See :ref:`available-optimizers` for the list of available
optimizers. For adding new features to third-party optimizers, use
`make_zero_optimizer`.

Features
^^^^^^^^

.. rubric:: Support for the `with` keyword

.. code-block::

    with optimizer:
        ...
        <do backward>
        ...

Is equivalent to:

.. code-block::

    optimizer.zero_grad()
    ...
    <do backward>
    ...
    optimizer.step()

Warning:
    :code:`.step()` is performed only if no exceptions are raised in the context.
"""

__all__ = [
    'make_zero_optimizer',
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


class _ZeroOptimizerMixin:
    """Mixin adding additional functionality to `torch.optim.Optimizer`.

    Warning:
        Do not use this class directly. Instead, use `make_zero_optimizer`.
    """

    def __enter__(self) -> None:
        self.zero_grad()  # type: ignore

    def __exit__(self, *args) -> bool:  # type: ignore
        if args == (None, None, None):
            self.step()  # type: ignore
        return False


def make_zero_optimizer(cls):
    """Make *the same optimizer class*, but with additional functionality.

    Args:
        cls: class, inherited from `torch.optim.Optimizer`. **Optimizers that require
            closure in .step() are not supported**.
    Returns:
        Enhanced cls.
    """
    doc = getattr(cls, '__doc__')
    attrs = {} if doc is None else {'__doc__': doc}
    return type(cls.__name__, (cls, _ZeroOptimizerMixin), attrs)


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
