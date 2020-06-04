from zero.metrics import Metric


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
        return bool(self.count)
