import random
import secrets

import numpy as np
import torch
from numpy.random import Generator, default_rng

_DEFAULT_CALLBACK = object()


def set_seed_everywhere(seed_sequence=None, callback=_DEFAULT_CALLBACK) -> Generator:
    if seed_sequence is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        seed_sequence = secrets.randbits(128)
    seed = seed_sequence % (2 ** 32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if callback is not None:
        if callback is _DEFAULT_CALLBACK:
            print(
                f'Seed sequence: {seed_sequence} (see zero.random.set_seed_everywhere)'
            )
        else:
            callback(seed_sequence)
    return default_rng(seed_sequence)
