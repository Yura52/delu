"""Random sampling utilities."""

__all__ = ['set_randomness']

import random
import secrets
from typing import Optional

import numpy as np
import torch


def set_randomness(
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

            print('Seed:', set_randomness(None))

    Examples:
        .. testcode::

            assert set_randomness(0) == 0
            assert set_randomness(), '0 was generated as the seed, which is almost impossible!'
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
