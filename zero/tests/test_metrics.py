import torch
from pytest import raises

from zero.metrics import MetricsDict, MetricsList

from .util import ObjectCounter


def apply_metric(metric_fn, data):
    result = metric_fn.reset().update(data).compute()
    metric_fn.reset()
    return result


def test_metrics_containers():
    data = (torch.tensor([0, 0]), torch.tensor([1, 1]))
    a = ObjectCounter(1)
    b = ObjectCounter(-1)

    params = [
        (MetricsList([a, b]), [2, -2], [4, -4], [0, 1]),
        (
            MetricsDict({'a': a, 'b': b}),
            {'a': 2, 'b': -2},
            {'a': 4, 'b': -4},
            ['a', 'b'],
        ),
    ]
    for metric_fn, first, second, keys in params:
        assert apply_metric(metric_fn, data) == first
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
    data = (torch.tensor([0, 0]), torch.tensor([1, 1]))
    assert apply_metric(metric_fn, data) == {'a': 2, 'b': -2}
    metric_fn.update(data)
    metric_fn.update(data)
    assert metric_fn.compute() == {'a': 4, 'b': -4}
