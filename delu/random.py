"""Random sampling utilities."""

import random
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import torch


def seed(base_seed: int, one_cuda_seed: bool = False) -> None:
    """Set seeds in `random`, `numpy` and `torch`.

    For all libraries, different seeds (which are *deterministically* calculated based
    on the ``base_seed`` argument) are set (see the note below).

    Args:
        base_seed: the number used to determine random seeds for all libraries and
            hardware. Must be a non-negative number less than ``2 ** 32 - 10000``.
        one_cuda_seed: if `True`, then the same seed will be set for all cuda devices,
            otherwise, different seeds will be set for all cuda devices.
    Raises:
        AssertionError: if the seed is not within the required interval

    Note:
        Different seeds are set to avoid situations where different libraries or devices
        generate the same random sequences. See
        `this comment <https://github.com/PyTorchLightning/pytorch-lightning/pull/6960#issuecomment-818393659>`_
        for details.

    Examples:
        .. testcode::

            delu.random.seed(0)
    """  # noqa: E501
    assert 0 <= base_seed < 2**32 - 10000
    random.seed(base_seed)
    np.random.seed(base_seed + 1)
    torch.manual_seed(base_seed + 2)
    cuda_seed = base_seed + 3
    if one_cuda_seed:
        torch.cuda.manual_seed_all(cuda_seed)
    elif torch.cuda.is_available():
        # the following check should never succeed since torch.manual_seed also calls
        # torch.cuda.manual_seed_all() inside; but let's keep it just in case
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        # Source: https://github.com/pytorch/pytorch/blob/2f68878a055d7f1064dded1afac05bb2cb11548f/torch/cuda/random.py#L109
        for i in range(torch.cuda.device_count()):
            default_generator = torch.cuda.default_generators[i]
            default_generator.manual_seed(cuda_seed + i)


def get_state() -> Dict[str, Any]:
    """Aggregate global random states from `random`, `numpy` and `torch`.

    The function is useful for creating checkpoints that allow to resume data streams or
    other activities dependent on **global** random number generator (see the note below
    ). The result of this function can be passed to `set_state`.

    Returns:
        state

    See also:
        `set_state`

    Examples:
        .. testcode::

            model = torch.nn.Linear(1, 1)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            ...
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
            }
            ...
            # Later:
            # torch.save(checkpoint, 'checkpoint.pt')
            # ...
            # delu.random.set_state(torch.load('checkpoint.pt')['random_state'])
    """
    return {
        'random': random.getstate(),
        'numpy.random': np.random.get_state(),
        'torch.random': torch.random.get_rng_state(),
        'torch.cuda': torch.cuda.get_rng_state_all(),  # type: ignore
    }


def set_state(state: Dict[str, Any]) -> None:
    """Set global random states in `random`, `numpy` and `torch`.

    Args:
        state: global RNG states. Must be produced by `get_state`. The size of the list
            ``state['torch.cuda']`` must be equal to the number of available cuda
            devices.

    See also:
        `get_state`

    Raises:
        AssertionError: if ``torch.cuda.device_count() != len(state['torch.cuda'])``
    """
    random.setstate(state['random'])
    np.random.set_state(state['numpy.random'])
    torch.random.set_rng_state(state['torch.random'])
    assert torch.cuda.device_count() == len(state['torch.cuda'])
    torch.cuda.set_rng_state_all(state['torch.cuda'])  # type: ignore


@contextmanager
def preserve_state():
    """Decorator and context manager for preserving global random state.

    Examples:

        .. testcode::

            import random

            f = lambda: (
                random.randint(0, 10),
                np.random.randint(10),
                torch.randint(10, (1,)).item()
            )
            with delu.random.preserve_state():
                a = f()
            b = f()
            assert a == b

            @delu.random.preserve_state()
            def g():
                return f()

            a = g()
            b = f()
            assert a == b

    """
    state = get_state()
    try:
        yield
    finally:
        set_state(state)