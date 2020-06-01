# The design is heavily inspired by Ignite: https://pytorch.org/ignite.
from abc import ABC, abstractmethod


class Metric(ABC):
    @abstractmethod
    def reset(self):
        ...  # pragma: no cover

    @abstractmethod
    def update(self, data):
        ...  # pragma: no cover

    @abstractmethod
    def compute(self):
        ...  # pragma: no cover

    # @typing.final (available in 3.8)
    def apply(self, data):
        self.reset()
        self.update(data)
        result = self.compute()
        self.reset()
        return result


class MetricsList(Metric):
    def __init__(self, metrics):
        self._metrics = metrics

    def reset(self):
        for x in self._metrics:
            x.reset()

    def update(self, data):
        for x in self._metrics:
            x.update(data)

    def compute(self):
        return [x.compute() for x in self._metrics]

    def __getitem__(self, i):
        return self._metrics[i]


class MetricsDict(Metric):
    def __init__(self, metrics):
        self._metrics = metrics

    def reset(self):
        for x in self._metrics.values():
            x.reset()

    def update(self, data):
        for x in self._metrics.values():
            x.update(data)

    def compute(self):
        return {k: v.compute() for k, v in self._metrics.items()}

    def __getitem__(self, key):
        return self._metrics[key]
