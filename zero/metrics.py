__all__ = ['Metric', 'MetricsList', 'MetricsDict', 'IgniteMetric']

# The API intentially follows that of Ignite: https://pytorch.org/ignite/metrics.html
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence


class Metric(ABC):
    # TODO (docs): pattern metric_fn.reset().update(x).compute()
    @abstractmethod
    def reset(self) -> Any:
        ...  # pragma: no cover

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        ...  # pragma: no cover

    @abstractmethod
    def compute(self) -> Any:
        ...  # pragma: no cover

    def __enter__(self) -> None:
        self.reset()

    def __exit__(self, *args) -> bool:  # type: ignore
        self.reset()
        return False


class MetricsList(Metric):
    def __init__(self, metrics: Sequence[Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> 'MetricsList':
        for x in self._metrics:
            x.reset()
        return self

    def update(self, *args, **kwargs) -> 'MetricsList':
        for x in self._metrics:
            x.update(*args, **kwargs)
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

    def update(self, *args, **kwargs) -> 'MetricsDict':
        for x in self._metrics.values():
            x.update(*args, **kwargs)
        return self

    def compute(self) -> Dict:
        return {k: v.compute() for k, v in self._metrics.items()}

    def __getitem__(self, key) -> Metric:
        return self._metrics[key]


class IgniteMetric(Metric):
    def __init__(self, ignite_metric) -> None:
        self.metric = ignite_metric

    def reset(self) -> 'IgniteMetric':
        self.metric.reset()
        return self

    def update(self, *args, **kwargs) -> 'IgniteMetric':
        self.metric.update(*args, **kwargs)
        return self

    def compute(self) -> Any:
        return self.metric.compute()
