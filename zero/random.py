"""Random sampling utilities."""

import random
import secrets
from typing import Any, Dict, Optional

import numpy as np
import torch


def init(
    seed: Optional[int],
    cudnn_deterministic: bool = True,
    cudnn_benchmark: bool = False,
) -> int:
    """Set seeds and settings in `random`, `numpy` and `torch`.

    Sets random seed in `random`, `numpy.random`, `torch`, `torch.cuda` and sets
    settings in :code:`torch.backends.cudnn`.

    Args:
        seed: the seed for all mentioned libraries. If `None`, a **high-quality** seed
            is generated and used instead.
        cudnn_deterministic: value for :code:`torch.backends.cudnn.deterministic`
        cudnn_benchmark: value for :code:`torch.backends.cudnn.benchmark`

    Returns:
        seed: if :code:`seed` is set to `None`, the generated seed is returned;
            otherwise the :code:`seed` argument is returned as is

    Note:
        If you don't want to set the seed by hand, but still want to have a chance to
        reproduce things, you can use the following pattern::

            print('Seed:', zero.random.init(None))

    Examples:
        .. testcode::

            assert zero.random.init(0) == 0
            generated_seed = zero.random.init(None)
    """
    torch.backends.cudnn.deterministic = cudnn_deterministic  # type: ignore
    torch.backends.cudnn.benchmark = cudnn_benchmark  # type: ignore
    raw_seed = seed
    if raw_seed is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        raw_seed = secrets.randbits(128)
    seed = raw_seed % (2 ** 32 - 1)
    torch.manual_seed(seed)
    # mypy doesn't know about the following functions
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # type: ignore
    np.random.seed(seed)
    random.seed(seed)
    return seed


def get_state() -> Dict[str, Any]:
    """Aggregate global random states from `random`, `numpy` and `torch`.

    The function is useful for creating checkpoints that allow to resume data streams or
    other activities dependent on **global** random number generator (see the note below
    ). The result of this function can be passed to `set_state`.

    Returns:
        state

    Note:
        The most reliable way to guarantee reproducibility and to make your data streams
        resumable is to create separate random number generators and manage them
        manually (for example, `torch.utils.data.DataLoader` accepts the
        argument :code:`generator` for that purposes). However, if you rely on the
        global random state, this function along with `set_state` does everything
        just right.

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
    """Set global random states from `random`, `numpy` and `torch`.

    The argument must be produced by `get_state`.

    Note:
        The size of list :code:`state['torch.cuda']` must be equal to the number of
        available cuda devices. If random state of cuda devices is not important, remove
        the entry 'torch.cuda' from the state beforehand, or, **at your own risk**
        adjust its value.

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
