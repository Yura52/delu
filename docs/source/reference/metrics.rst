.. _metrics:

zero.metrics
============

.. currentmodule:: zero.metrics

.. automodule:: zero.metrics

.. _zero.metrics.Metric:

Metric
^^^^^^

.. autoclass:: Metric

.. autosummary::
    :toctree: api

    Metric.reset
    Metric.update
    Metric.compute
    Metric.__enter__
    Metric.__exit__

MetricsList
^^^^^^^^^^^

.. autoclass:: MetricsList

.. autosummary::
    :toctree: api

    MetricsList.reset
    MetricsList.update
    MetricsList.compute
    MetricsList.__getitem__


MetricsDict
^^^^^^^^^^^

.. autoclass:: MetricsDict

.. autosummary::
    :toctree: api

    MetricsDict.reset
    MetricsDict.update
    MetricsDict.compute
    MetricsDict.__getitem__

IgniteMetric
^^^^^^^^^^^^

.. autoclass:: IgniteMetric

.. autosummary::
    :toctree: api

    IgniteMetric.metric
    IgniteMetric.reset
    IgniteMetric.update
    IgniteMetric.compute
