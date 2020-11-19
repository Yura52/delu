import random

import numpy as np
import torch

from zero.random import set_randomness

from .util import requires_gpu


def _test_set_randomness(functions):
    assert isinstance(set_randomness(None), int)
    high = 1000000
    for seed in range(10):
        x = [None, None]
        for i in range(2):
            set_randomness(seed)
            x[i] = [f(high) for f in functions]
        assert x[0] == x[1]


def test_set_randomness_cpu():
    _test_set_randomness(
        [
            lambda x: random.randint(0, x),
            lambda x: np.random.randint(x),
            lambda x: torch.randint(x, (1,))[0].item(),
        ]
    )


@requires_gpu
def test_set_randomness_gpu():
    functions = []
    for i in range(torch.cuda.device_count()):

        def f(x):
            return (torch.randint(x, (1,), device=f'cuda:{i}')[0].item(),)

        functions.append(f)
    _test_set_randomness(functions)
