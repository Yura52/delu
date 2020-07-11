"""Tiny ecosystem for metrics.

TL;DR: with this module, evaluation looks like this:

.. code-block::

    metrics = metric_fn.calculate_iter(map(predict_batch, val_loader))

In order to create your own metric, inherit from `Metric` and implement its interface
(see `Metric`'s docs for examples). The API throughout the module intentionally follows
that of `ignite.metrics <https://pytorch.org/ignite/metrics.html>`_, hence, Ignite
metrics are supported almost everywhere where `Metric` is supported. For giving Ignite
metrics full functionality of `Metric`, use `IgniteMetric`.

Warning:
    Distributed settings are not supported out-of-the-box. In such cases, you have the
    following options:

    - wrap a metric from Ignite in `IgniteMetric`
    - use `ignite.metrics.metric.sync_all_reduce` and
      `ignite.metrics.metric.reinit__is_reduced`
    - manually take care of everything
"""

__all__ = ['Metric', 'MetricsDict', 'IgniteMetric']

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class Metric(ABC):
    """The base class for metrics.

    In order to create your own metric, inherit from this class and implement all
    methods marked with `@abstractmethod`. High-level functionality (`Metric.calculate`,
    `Metric.calculate_iter`) is already implemented.

    .. rubric:: Tutorial

    .. testcode::

        class Accuracy(Metric):
            def __init__(self):
                self.reset()

            def reset(self):
                self.n_objects = 0
                self.n_correct = 0

            def update(self, y_pred, y):
                self.n_objects += len(y)
                self.n_correct += (y_pred == y).sum().item()

            def compute(self):
                assert self.n_objects
                return self.n_correct / self.n_objects

        metric_fn = Accuracy()
        y_pred = torch.tensor([0, 0, 0, 0])
        y = torch.tensor([0, 1, 0, 1])
        assert metric_fn.calculate(y_pred, y) == 0.5

        from zero.all import iter_batches
        y = torch.randint(2, size=(10,))
        X = torch.randn(len(y), 3)
        batches = iter_batches((X, y), batch_size=2)

        def perfect_prediction(batch):
            X, y = batch
            y_pred = y
            return y_pred, y

        score = metric_fn.calculate_iter(map(perfect_prediction, batches), star=True)
        assert score == 1.0
    """

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

    def calculate(self, *args, **kwargs) -> Any:
        """Calculate metric for a single input.

        The method does the following:

        #. **resets the metric**
        #. updates the metric with :code:`(*args, **kwargs)`
        #. computes the result
        #. **resets the metric**
        #. returns the result

        Args:
            *args: arguments for `Metric.update`
            **kwargs arguments for `Metric.update`
        Returns:
            The result of `Metric.compute`.
        """
        self.reset()
        self.update(*args, **kwargs)
        result = self.compute()
        self.reset()
        return result

    def calculate_iter(self, iterable: Iterable, star: bool = False) -> Any:
        """Calculate metric for iterable.

        The method does the following:

        #. **resets the metric**
        #. sequentially updates the metric with every value from :code:`iterable`
        #. computes the result
        #. **resets the metric**
        #. returns the result

        Args:
            iterable: data for `Metric.update`
            star: if `True`, then :code:`update(*x)` is performed instead of
                :code:`update(x)`
        Returns:
            The result of `Metric.compute`.

        Examples:
            .. code-block::

                metrics = metric_fn.calculate_iter(map(predict_batch, val_loader))
        """
        self.reset()
        for x in iterable:
            if star:
                self.update(*x)
            else:
                self.update(x)
        result = self.compute()
        self.reset()
        return result


class MetricsDict(Metric):
    """Dictionary for metrics.

    The container is suitable when all metrics take input in the same form.

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
            metric_fn.calculate(...)
            metric_fn.calculate_iter(...)
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
