from collections import namedtuple

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
        assert self.count
        return self.sign * self.count

    def empty(self):
        return not self.count
