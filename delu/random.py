"""Random sampling utilities."""

import random
from contextlib import contextmanager
from typing import Any, Dict

import numpy as np
import torch

__all__ = ['seed', 'get_state', 'set_state', 'preserve_state']


def seed(base_seed: int, /, *, one_cuda_seed: bool = False) -> None:
    """Set the global random seeds in `random`, `numpy` and `torch`.

    For all libraries, different seeds (which are *deterministically* computed based
    on the ``base_seed`` argument) are set (see the note below).

    **Usage**

    >>> import random
    >>> import numpy as np
    >>>
    >>> def f():
    ...     return random.random(), np.random.rand(), torch.rand(1).item()
    ...
    >>> # Randomly sampled numbers are different.
    >>> a = f()
    >>> b = f()
    >>> a == b
    False
    >>> # But they are equal when sampled under the same random seed.
    >>> delu.random.seed(0)
    >>> a = f()
    >>> delu.random.seed(0)
    >>> b = f()
    >>> a == b
    True

    Args:
        base_seed: the number used to determine random seeds for all libraries and
            hardware. Must be a non-negative number less than ``2 ** 32 - 10000``.
        one_cuda_seed: if `True`, then the same seed will be set for all cuda devices,
            otherwise, different seeds will be set for all cuda devices.

    Note:
        Different seeds are set to avoid situations where different libraries or devices
        generate the same random sequences. See
        `this comment <https://github.com/PyTorchLightning/pytorch-lightning/pull/6960#issuecomment-818393659>`_
        for details.
    """
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
    """Aggregate the global random generator states from `random`, `numpy` and `torch`.

    The result of this function can be passed to `delu.random.set_state`.
    An important use case is saving the random states to a checkpoint.

    **Usage**

    First, a technical example:

    >>> import random
    >>> import numpy as np
    >>>
    >>> def f():
    ...     return random.random(), np.random.rand(), torch.rand(1).item()
    ...
    >>> # Save the current state:
    >>> state = delu.random.get_state()
    >>>
    >>> # The first call changes the state,
    >>> # so the second call produces different results.
    >>> a1, b1, c1 = f()
    >>> a2, b2, c2 = f()
    >>> print(a1 == a2, b1 == b2, c1 == c2)
    False False False
    >>>
    >>> # Restore the initial global state:
    >>> delu.random.set_state(state)
    >>> a3, b3, c3 = f()
    >>> print(a1 == a3, b1 == b3, c1 == c3)
    True True True

    An example pseudocode
    for saving/loading the global state to/from a checkpoint::

        # Resuming from a checkpoint:
        checkpoint_path = ...
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            delu.random.set_state(checkpoint['random_state'])
        ...
        # Training:
        for batch in batches:
            ...
            if step % checkpoint_frequency == 0:
                torch.save(
                    {
                        'model': ...,
                        'optimizer': ...,
                        'random_state': delu.random.get_state(),
                    },
                    checkpoint_path,
                )

    Returns:
        The aggregated random states.
    """
    return {
        'random': random.getstate(),
        'numpy.random': np.random.get_state(),
        'torch.random': torch.random.get_rng_state(),
        'torch.cuda': torch.cuda.get_rng_state_all(),  # type: ignore
    }


def set_state(state: Dict[str, Any], /) -> None:
    """Set the global random number generator states in `random`, `numpy` and `torch`.

    **Usage**

    See `delu.random.get_state` for usage examples.

    Args:
        state: the dict with the states produced by `delu.random.get_state`.
    """
    if torch.cuda.device_count() != len(state['torch.cuda']):
        raise RuntimeError(
            'The provided state of the global RNGs is not compatible with the current'
            ' hardware, because:'
            f' {torch.cuda.device_count()=} != {len(state["torch.cuda"])=}'
        )
    random.setstate(state['random'])
    np.random.set_state(state['numpy.random'])
    torch.random.set_rng_state(state['torch.random'])
    torch.cuda.set_rng_state_all(state['torch.cuda'])  # type: ignore


@contextmanager
def preserve_state():
    """A decorator and context manager for preserving the global random generators state.

    The function saves the global random generators state
    when entering a context/function and restores it on exit/return.

    **Usage**

    As a context manager
    (the state after the context is the same as before the context):

    >>> import random
    >>> import numpy as np
    >>>
    >>> def f():
    ...     return random.random(), np.random.rand(), torch.rand(1).item()
    ...
    >>> with delu.random.preserve_state():
    ...     a = f()
    >>> b = f()
    >>> a == b
    True

    As a decorator
    (the state after the call `g()` is the same as before the call):

    >>> @delu.random.preserve_state()
    ... def g():
    ...     return f()
    ...
    >>> a = g()
    >>> b = f()
    >>> a == b
    True
    """  # noqa: E501
    state = get_state()
    try:
        yield
    finally:
        set_state(state)
