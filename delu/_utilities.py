import secrets
from typing import Optional

import torch

from . import random as delu_random


def improve_reproducibility(
    base_seed: Optional[int], one_cuda_seed: bool = False
) -> int:
    """Set seeds and turn off non-deterministic algorithms.

    Do everything possible to improve reproducibility for code that relies on global
    random number generators from the aforementioned modules. See also the note below.

    Sets:

    1. seeds in `random`, `numpy.random`, `torch`, `torch.cuda`
    2. `torch.backends.cudnn.benchmark` to `False`
    3. `torch.backends.cudnn.deterministic` to `True`

    Args:
        base_seed: the argument for `delu.random.seed`. If `None`, a high-quality base
            seed is generated instead.
        one_cuda_seed: the argument for `delu.random.seed`.

    Returns:
        base_seed: if ``base_seed`` is set to `None`, the generated base seed is
            returned; otherwise, ``base_seed`` is returned as is

    Note:
        If you don't want to choose the base seed, but still want to have a chance to
        reproduce things, you can use the following pattern::

            print('Seed:', delu.improve_reproducibility(None))

    Note:
        100% reproducibility is not always possible in PyTorch. See
        `this page <https://pytorch.org/docs/stable/notes/randomness.html>`_ for
        details.

    Examples:
        .. testcode::

            assert delu.improve_reproducibility(0) == 0
            seed = delu.improve_reproducibility(None)
    """
    torch.backends.cudnn.benchmark = False  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    if base_seed is None:
        # See https://numpy.org/doc/1.18/reference/random/bit_generators/index.html#seeding-and-entropy  # noqa
        base_seed = secrets.randbits(128) % (2**32 - 1024)
    else:
        assert base_seed < (2**32 - 1024)
    delu_random.seed(base_seed, one_cuda_seed)
    return base_seed
