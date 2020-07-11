import torch
from pytest import raises

from zero.metrics import MetricsDict

from .util import ObjectCounter


def test_metrics_dict():
    data = (torch.tensor([0, 0]), torch.tensor([1, 1]))
    metric_fn = MetricsDict({'a': ObjectCounter(1), 'b': ObjectCounter(-1)})
    assert metric_fn.calculate(data) == {'a': 2, 'b': -2}
    metric_fn.update(data)
    metric_fn.update(data)
    result = {'a': 4, 'b': -4}
    assert metric_fn.compute() == result
    for key in 'a', 'b':
        assert metric_fn[key].compute() == metric_fn.compute()[key]
    metric_fn.reset()
    with raises(AssertionError):
        metric_fn.compute()
    assert metric_fn.calculate_iter([data, data]) == result
    assert metric_fn.calculate_iter([(data,), (data,)], True) == result
