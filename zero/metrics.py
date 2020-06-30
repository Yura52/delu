"""Tiny ecosystem for metrics.

Using this library together with `zero.model.Eval` makes evaluation look like this:

.. code-block::

    with Eval(model), metric_fn:
        for batch in val_loader:
            metric_fn.update(predict(batch))
        metrics = metric_fn.compute()

In order to create your own metric, inherit from `Metric` and implement its interface.
The API throughout the module intentionally follows that of
https://pytorch.org/ignite/metrics.html, hence, Ignite metrics are supported almost
everywhere where `Metric` is supported. For giving Ignite metrics all functionality of
`Metric`, use `IgniteMetric`.
"""

__all__ = ['Metric', 'MetricsList', 'MetricsDict', 'IgniteMetric']

# The API intentially follows that of Ignite: https://pytorch.org/ignite/metrics.html
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Sequence


class Metric(ABC):
    """Base class for metrics.

    In order to create your own metric, inherit from this class and implement all
    methods marked with `@abstractmethod`. Context-manager functionality is already
    implemented (see `Metric.__enter__` and `Metric.__exit__`).

    .. rubric:: Tutorial

    .. testcode ::

        # NOTE: Ending `reset` and `update` with `return self` is not required,
        # but it enables the pattern `metric_fn.reset().update(...).compute()`.
        class Accuracy(Metric):
            def __init__(self):
                self.reset()

            def reset(self):
                self.n_objects = 0
                self.n_correct = 0
                return self

            def update(self, y_pred, y):
                self.n_objects += len(y)
                self.n_correct += (y_pred == y).sum().item()
                return self

            def compute(self):
                assert self.n_objects
                return self.n_correct / self.n_objects
    """

    # TODO (docs): pattern metric_fn.reset().update(x).compute()
    @abstractmethod
    def reset(self) -> Any:
        """Reset the metric's state."""
        ...  # pragma: no cover

    @abstractmethod
    def update(self, *args, **kwargs) -> Any:
        """Update the metric's state."""
        ...  # pragma: no cover

    @abstractmethod
    def compute(self) -> Any:
        """Compute the metric."""
        ...  # pragma: no cover

    def __enter__(self) -> None:
        """Reset the metric."""
        self.reset()

    def __exit__(self, *args) -> bool:  # type: ignore
        """Reset the metric."""
        self.reset()
        return False


class MetricsList(Metric):
    """List for metrics.

    Args:
        metrics

    Examples:
        .. code-block::

            metric_fn = MetricList([FirstMetric(), SecondMetric()])

    .. rubric:: Tutorial

    .. code-block::

        from ignite.metrics import Precision

        class MyMetric(Metric):
            ...

        a = MyMetric()
        b = IgniteMetric(Precision())
        metric_fn = MetricList([a, b])
        metric_fn.reset()  # reset all metrics
        metric_fn.update(...)  # update all metrics
        metric_fn.compute()  # [<my metric>, <precision>]
        assert metric_fn[0] is a and metric[1] is b
    """

    def __init__(self, metrics: Sequence[Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> 'MetricsList':
        """Reset all underlying metrics.

        Returns:
            self
        """
        for x in self._metrics:
            x.reset()
        return self

    def update(self, *args, **kwargs) -> 'MetricsList':
        """Update all underlying metrics.

        Args:
            *args: positional arguments forwarded to `update()` for all metrics
            *kwargs: keyword arguments forwarded to `update()` for all metrics
        Returns:
            self
        """
        for x in self._metrics:
            x.update(*args, **kwargs)
        return self

    def compute(self) -> List:
        """Compute the results.

        The order is the same as in the constructor.

        Returns:
            List with results of `.compute()` of the underlying metrics.
        """
        return [x.compute() for x in self._metrics]

    def __getitem__(self, index: int) -> Metric:
        """Access a metric by index.

        Args:
            index
        Returns:
            The metric corresponding to the index.
        """
        return self._metrics[index]


class MetricsDict(Metric):
    """Dictionary for metrics.

    Args:
        metrics

    Examples:
        .. code-block::

            metric_fn = MetricList([FirstMetric(), SecondMetric()])

    .. rubric:: Tutorial

    .. code-block::

        from ignite.metrics import Precision

        class MyMetric(Metric):
            ...

        a = MyMetric()
        b = IgniteMetric(Precision())
        metric_fn = MetricsDict({'a': a, 'b': b})
        metric_fn.reset()  # reset all metrics
        metric_fn.update(...)  # update all metrics
        metric_fn.compute()  # {'a': <my metric>, 'b': <precision>}
        assert metric_fn['a'] is a and metric['b'] is b
    """

    def __init__(self, metrics: Dict[Any, Metric]) -> None:
        self._metrics = metrics

    def reset(self) -> 'MetricsDict':
        """Reset all underlying metrics.

        Returns:
            self
        """
        for x in self._metrics.values():
            x.reset()
        return self

    def update(self, *args, **kwargs) -> 'MetricsDict':
        """Update all underlying metrics.

        Args:
            *args: positional arguments forwarded to `update()` for all metrics
            *kwargs: keyword arguments forwarded to `update()` for all metrics
        Returns:
            self
        """
        for x in self._metrics.values():
            x.update(*args, **kwargs)
        return self

    def compute(self) -> Dict:
        """Compute the results.

        The keys are the same as in the constructor.

        Returns:
            Dictionary with results of `.compute()` of the underlying metrics.
        """
        return {k: v.compute() for k, v in self._metrics.items()}

    def __getitem__(self, key) -> Metric:
        """Access a metric by key.

        Args:
            key
        Returns:
            The metric corresponding to the key.
        """
        return self._metrics[key]


class IgniteMetric(Metric):
    """Wrapper for metrics from `ignite.metrics`.

    Args:
        metric (`ignite.metrics.Metric`)

    Examples:
        .. code-block::

            from ignite import Precision
            metric_fn = IgniteMetric(Precision())
    """

    def __init__(self, ignite_metric) -> None:
        self._metric = ignite_metric

    @property
    def metric(self):
        """Get the underlying metric.

        Returns:
            `ignite.metrics.Metric`: the underlying metric.
        """
        return self._metric

    def reset(self) -> 'IgniteMetric':
        """Reset the underlying metric.

        Returns:
            self
        """
        self.metric.reset()
        return self

    def update(self, *args, **kwargs) -> 'IgniteMetric':
        """Update the underlying metric.

        Args:
            *args: positional arguments forwarded to :code:`update`
            *kwargs: keyword arguments forwarded to :code:`update`
        Returns:
            self
        """
        self.metric.update(*args, **kwargs)
        return self

    def compute(self) -> Any:
        """Compute the result.

        Returns:
            The result of :code:`compute` of the underlying metric.
        """
        return self.metric.compute()
