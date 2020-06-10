import random
from unittest.mock import Mock

import numpy as np
import torch as tr

from zero.random import set_seed_everywhere

from .util import requires_gpu


def test_set_seed_everywhere_common():
    assert isinstance(set_seed_everywhere(), np.random.Generator)
    mock = Mock()
    set_seed_everywhere(1, mock)
    mock.assert_called_once_with(1)


def _test_set_seed_everywhere(functions):
    high = 1000000
    for seed_sequence in range(10):
        x = [None, None]
        for i in range(2):
            rng = set_seed_everywhere(seed_sequence, None)
            x[i] = [f(rng, high) for f in functions]
        assert x[0] == x[1]


def test_set_seed_everywhere_cpu():
    _test_set_seed_everywhere(
        [
            lambda _, x: random.randint(0, x),
            lambda _, x: np.random.randint(x),
            lambda _, x: tr.randint(x, (1,))[0].item(),
            lambda rng, x: rng.integers(x),
        ]
    )


@requires_gpu
def test_set_seed_everywhere_gpu():
    functions = []
    for i in range(tr.cuda.device_count()):

        def f(_, x):
            return (tr.randint(x, (1,), device=f'cuda:{i}')[0].item(),)

        functions.append(f)
    _test_set_seed_everywhere(functions)
