import itertools
import math

from pytest import mark, raises

from zero.flow import Flow


def test_properties():
    loader = [0]
    flow = Flow(loader)
    assert flow.epoch == 0
    assert flow.iteration == 0
    assert flow.count == 0
    assert flow.loader is loader
    for attr in 'epoch', 'iteration', 'count':
        with raises(AttributeError):
            setattr(flow, attr, 0)
    with raises(AttributeError):
        flow.loader = flow.loader


def test_epoch_iteration():
    with raises(AssertionError):
        Flow([])

    flow = Flow(range(3))
    assert flow.epoch == 0
    assert flow.iteration == 0

    n = 4
    for i in range(1, n + 1):
        flow.increment_epoch()
        assert flow.epoch == i
        flow.increment_iteration()
        assert flow.iteration == i
    assert not flow.increment_epoch(n)
    assert flow.epoch == n


@mark.parametrize('n', range(1, 5))
def test_count_next(n):
    flow = Flow(range(n))
    assert flow.count == 0

    for epoch in range(2):
        for x in range(n):
            assert flow.next() == x
            assert flow.iteration == x + epoch * n + 1
            assert flow.count == flow.iteration
    for i in range(1, 3):
        flow.next(False)
        assert flow.count == flow.iteration + i

    flow = Flow(iter(range(n)))
    for _ in range(n):
        flow.next()
    with raises(StopIteration):
        flow.next()


@mark.parametrize('n', range(1, 6))
def test_data(n):
    # iterables that are not iterators
    n_epoches = 10
    flow = Flow(range(n))
    assert [list(flow.data(0)) for _ in range(n_epoches)] == [[]] * n_epoches

    for epoch_size in [None] + list(range(1, 2 * n)):
        effective_epoch_size = n if epoch_size is None else epoch_size
        max_count = n_epoches * effective_epoch_size
        flow = Flow(range(n))
        actual = []
        while flow.count < max_count:
            actual.append(list(flow.data(epoch_size)))
        assert flow.iteration == max_count
        assert flow.count == max_count
        flat_correct = [x % n for x in range(max_count)]
        correct = [
            flat_correct[i * effective_epoch_size : (i + 1) * effective_epoch_size]
            for i in range(n_epoches)
        ]
        assert actual == correct

    # count=inf
    flow = Flow(range(n))
    assert [x for _, x in zip(range(1000), flow.data(math.inf))] == [
        x % n for x in range(1000)
    ]

    # increment_iteration=False
    flow = Flow(range(n))
    for x in flow.data(None, False):
        flow.increment_iteration()
    assert flow.iteration == n
    assert flow.count == n

    # infinite iterators
    flow = Flow(itertools.count())
    with raises(ValueError):
        flow.data()
    n_epoches = 10
    for epoch_size in range(1, n):
        flow = Flow(itertools.count())
        actual = [list(flow.data(epoch_size)) for _ in range(n_epoches)]
        flat_correct = list(range(n_epoches * epoch_size))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size]
            for i in range(n_epoches)
        ]
        assert actual == correct

    # finite iterators
    flow = Flow(iter(range(n)))
    with raises(ValueError):
        flow.data()
    n_epoches = 10
    for epoch_size in range(1, 2 * n):
        flow = Flow(iter(range(n)))
        actual = []
        for _ in range(n_epoches):
            actual.append(list(flow.data(epoch_size)))
        flat_correct = list(range(n))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size]
            for i in range(n_epoches)
        ]
        assert actual == correct


def test_reset_iterator():
    n = 10
    flow = Flow(range(n))
    for x in range(n):
        assert flow.next() == 0
        flow.reset_iterator()

    flow = Flow(iter(range(n)))
    for x in range(n):
        assert flow.next() == x
        flow.reset_iterator()
    with raises(StopIteration):
        flow.next()


def test_set_loader():
    a = itertools.repeat(0)
    b = itertools.repeat(1)
    flow = Flow([2])
    flow.set_loader(a)
    for x in range(10):
        assert flow.next() == next(b if x % 2 else a)
        flow.set_loader(a if x % 2 else b)
