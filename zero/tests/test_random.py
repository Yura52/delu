import random
from unittest.mock import Mock

import numpy.random as np_random
import torch

from zero.random import set_seed_everywhere


def test_set_seed_everywhere():
    assert isinstance(set_seed_everywhere(), np_random.Generator)

    high = 1000000
    for seed_sequence in range(10):
        x = [None, None]
        for i in range(2):
            rng = set_seed_everywhere(seed_sequence, None)
            x[i] = [
                random.randint(0, high),
                np_random.randint(high),
                torch.randint(high, (1,))[0].item(),
                rng.integers(high),
            ]
        assert x[0] == x[1]

    mock = Mock()
    set_seed_everywhere(1, mock)
    mock.assert_called_once_with(1)
