import random
import secrets
from typing import Any, Callable, Optional

import numpy as np
import torch
from numpy.random import Generator, default_rng


def _default_callback(seed_sequence):
    print(f'Seed sequence: {seed_sequence} (see zero.random.set_seed_everywhere)')


def set_seed_everywhere(
    seed_sequence: Optional[int] = None,
    callback: Optional[Callable[[int], Any]] = _default_callback,
) -> Generator:
    if seed_sequence is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        seed_sequence = secrets.randbits(128)
    seed = seed_sequence % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # mypy doesn't see the following functions on CPU-only machines (all machines?)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)  # type: ignore
    if callback is not None:
        callback(seed_sequence)
    return default_rng(seed_sequence)
