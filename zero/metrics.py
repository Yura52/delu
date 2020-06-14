# The design is heavily inspired by Ignite: https://pytorch.org/ignite.
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Metric(ABC):
    # TODO (docs): pattern metric_fn.reset().update(x).compute()
    @abstractmethod
    def reset(self) -> 'Metric':
        ...  # pragma: no cover

    @abstractmethod
    def update(self, data) -> 'Metric':
        ...  # pragma: no cover

    @abstractmethod
    def compute(self):
        ...  # pragma: no cover


class MetricsList(Metric):
    def __init__(self, metrics: List[Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> 'MetricsList':
        for x in self._metrics:
            x.reset()
        return self

    def update(self, data) -> 'MetricsList':
        for x in self._metrics:
            x.update(data)
        return self

    def compute(self) -> List:
        return [x.compute() for x in self._metrics]

    def __getitem__(self, i: int) -> Metric:
        return self._metrics[i]


class MetricsDict(Metric):
    def __init__(self, metrics: Dict[Any, Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> 'MetricsDict':
        for x in self._metrics.values():
            x.reset()
        return self

    def update(self, data) -> 'MetricsDict':
        for x in self._metrics.values():
            x.update(data)
        return self

    def compute(self) -> Dict:
        return {k: v.compute() for k, v in self._metrics.items()}

    def __getitem__(self, key) -> Metric:
        return self._metrics[key]
