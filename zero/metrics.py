# The design is heavily inspired by Ignite: https://pytorch.org/ignite.
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class Metric(ABC):
    @abstractmethod
    def reset(self) -> None:
        ...  # pragma: no cover

    @abstractmethod
    def update(self, data) -> None:
        ...  # pragma: no cover

    @abstractmethod
    def compute(self):
        ...  # pragma: no cover


class MetricsList(Metric):
    def __init__(self, metrics: List[Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> None:
        for x in self._metrics:
            x.reset()

    def update(self, data) -> None:
        for x in self._metrics:
            x.update(data)

    def compute(self) -> List:
        return [x.compute() for x in self._metrics]

    def __getitem__(self, i: int) -> Metric:
        return self._metrics[i]


class MetricsDict(Metric):
    def __init__(self, metrics: Dict[Any, Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> None:
        for x in self._metrics.values():
            x.reset()

    def update(self, data) -> None:
        for x in self._metrics.values():
            x.update(data)

    def compute(self) -> Dict:
        return {k: v.compute() for k, v in self._metrics.items()}

    def __getitem__(self, key) -> Metric:
        return self._metrics[key]


def apply_metric(metric_fn: Metric, data):
    metric_fn.reset()
    metric_fn.update(data)
    result = metric_fn.compute()
    metric_fn.reset()
    return result
