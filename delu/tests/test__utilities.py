import random

import numpy as np
import torch

import delu


def test_improve_reproducibility():
    def f():
        upper_bound = 100
        return [
            random.randint(0, upper_bound),
            np.random.randint(upper_bound),
            torch.randint(upper_bound, (1,))[0].item(),
        ]

    for seed in [None, 0, 1, 2]:
        seed = delu.improve_reproducibility(seed)
        assert not torch.backends.cudnn.benchmark
        assert torch.backends.cudnn.deterministic
        results = f()
        delu.random.seed(seed)
        assert results == f()
