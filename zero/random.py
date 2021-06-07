"""Random sampling utilities."""

import random
from typing import Any, Dict

import numpy as np
import torch


def seed(seed: int) -> None:
    """Set seeds in `random`, `numpy` and `torch`.

    Args:
        seed: the seed for all mentioned libraries. Must be less
            than :code:`2 ** 32 - 1`.
    Raises:
        AssertionError: if the seed is not less than :code:`2 ** 32 - 1`

    Examples:
        .. testcode::

            zero.random.seed(0)
    """
    assert seed < 2 ** 32 - 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mypy doesn't know about the following functions
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # type: ignore


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
                'random_state': zero.random.get_state(),
            }
            # later
            # torch.save(checkpoint, 'checkpoint.pt')
            # ...
            # zero.random.set_state(torch.load('checkpoint.pt')['random_state'])
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
            :code:`state['torch.cuda']` must be equal to the number of available cuda
            devices.

    See also:
        `get_state`

    Raises:
        AssertionError: if :code:`torch.cuda.device_count() != len(state['torch.cuda'])`
    """
    random.setstate(state['random'])
    np.random.set_state(state['numpy.random'])
    torch.random.set_rng_state(state['torch.random'])
    assert torch.cuda.device_count() == len(state['torch.cuda'])
    torch.cuda.set_rng_state_all(state['torch.cuda'])  # type: ignore
