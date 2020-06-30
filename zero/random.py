"""Random sampling utilities."""

__all__ = ['set_seed_everywhere']

import random
import secrets
from typing import Any, Callable, Optional

import numpy as np
import torch
from numpy.random import Generator, default_rng


def _default_callback(seed):
    print(f'Seed: {seed} (see zero.random.set_seed_everywhere)')


def set_seed_everywhere(
    seed: Optional[int] = None,
    callback: Optional[Callable[[int], Any]] = _default_callback,
) -> Generator:
    r"""Set random seed for `random`, `numpy.random`, `torch`, `torch.cuda`.

    Args:
        seed: the seed for all mentioned libraries. If omitted, a high-quality seed is
            generated (an integer that **does not fit in int64**). In any case,
            :code:`seed % (2 ** 32 - 1)` will be used for everything except for building
            `numpy.random.Generator`.
        callback: a function that takes the seed as the only argument. The default
            callback simply prints the seed via `print` which is convenient when `seed`
            is set to `None`.
    Returns:
        `numpy.random.Generator`: A new style numpy random number generator constructed
        via `numpy.random.default_rng` (it should be used instead of functions from
        `np.random`, see
        `the document <https://numpy.org/doc/stable/reference/random/index.html>`_).

    Examples:
        .. testcode ::

            rng = set_seed_everywhere()  # seed will be generated
            rng = set_seed_everywhere(0)

        .. testoutput ::

            Seed: ... (see zero.random.set_seed_everywhere)
            Seed: 0 (see zero.random.set_seed_everywhere)
    """
    if seed is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        seed = secrets.randbits(128)
    raw_seed = seed
    seed = raw_seed % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mypy doesn't see the following functions on CPU-only machines (all machines?)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # type: ignore
    if callback is not None:
        callback(raw_seed)
    return default_rng(raw_seed)
