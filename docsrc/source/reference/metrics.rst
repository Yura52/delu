.. _zero.metrics:

zero.metrics
============

.. currentmodule:: zero.metrics

.. automodule:: zero.metrics


Metric
^^^^^^

.. autoclass:: Metric

.. autosummary::
    :toctree: api

    Metric.reset
    Metric.update
    Metric.compute
    Metric.calculate
    Metric.calculate_iter


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
