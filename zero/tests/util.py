from collections import namedtuple

import pytest
import torch

from zero.metrics import Metric

Point = namedtuple('Point', ['x', 'y'])


class ObjectCounter(Metric):
    def __init__(self, sign):
        self.sign = sign
        self.reset()

    def reset(self):
        self.count = 0

    def update(self, data):
        self.count += len(data[0])

    def compute(self):
        assert not self.empty
        return self.sign * self.count

    @property
    def empty(self):
        return not self.count


requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU is required for this test"
)
