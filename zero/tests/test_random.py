import random
from unittest.mock import Mock

import numpy as np
import torch

from zero.random import set_randomness

from .util import requires_gpu


def test_fix_randomness_common():
    assert isinstance(set_randomness(), np.random.Generator)
    mock = Mock()
    set_randomness(1, callback=mock)
    mock.assert_called_once_with(1)


def _test_fix_randomness(functions):
    high = 1000000
    for seed_sequence in range(10):
        x = [None, None]
        for i in range(2):
            rng = set_randomness(seed_sequence, callback=None)
            x[i] = [f(rng, high) for f in functions]
        assert x[0] == x[1]


def test_fix_randomness_cpu():
    _test_fix_randomness(
        [
            lambda _, x: random.randint(0, x),
            lambda _, x: np.random.randint(x),
            lambda _, x: torch.randint(x, (1,))[0].item(),
            lambda rng, x: rng.integers(x),
        ]
    )


@requires_gpu
def test_fix_randomness_gpu():
    functions = []
    for i in range(torch.cuda.device_count()):

        def f(_, x):
            return (torch.randint(x, (1,), device=f'cuda:{i}')[0].item(),)

        functions.append(f)
    _test_fix_randomness(functions)
