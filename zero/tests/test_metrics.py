import torch as tr
from pytest import raises

from zero.metrics import MetricsDict, MetricsList

from .util import ObjectCounter


def test_metrics_containers():
    data = (tr.tensor([0, 0]), tr.tensor([1, 1]))
    a = ObjectCounter(1)
    b = ObjectCounter(-1)

    for metric_fn, first, second, keys in [
        (MetricsList([a, b]), [2, -2], [4, -4], [0, 1]),
        (MetricsDict({'a': a, 'b': b}), {'a': 2, 'b': -2}, {'a': 4, 'b': -4}, ['a', 'b']),
    ]:
        assert metric_fn.apply(data) == first
        metric_fn.update(data)
        metric_fn.update(data)
        assert metric_fn.compute() == second
        for key in keys:
            assert metric_fn[key].compute() == metric_fn.compute()[key]
        metric_fn.reset()
        with raises(AssertionError):
            metric_fn.compute()


def test_metrics_dict():
    metric_fn = MetricsDict({'a': ObjectCounter(1), 'b': ObjectCounter(-1)})
    data = (tr.tensor([0, 0]), tr.tensor([1, 1]))
    assert metric_fn.apply(data) == {'a': 2, 'b': -2}
    metric_fn.update(data)
    metric_fn.update(data)
    assert metric_fn.compute() == {'a': 4, 'b': -4}
