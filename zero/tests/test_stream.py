import itertools
import math

from pytest import mark, raises

from zero.stream import Stream


def test_properties():
    loader = [0]
    stream = Stream(loader)
    assert stream.iteration == 0
    assert stream.epoch == 0
    assert stream.loader is loader
    for attr in 'iteration', 'epoch':
        with raises(AttributeError):
            setattr(stream, attr, 1)
    with raises(AttributeError):
        stream.loader = stream.loader


def test_bad_loader():
    with raises(AssertionError):
        Stream([])


def test_increment_epoch():
    stream = Stream(range(3))
    n = 4
    for i in range(1, n + 1):
        stream.increment_epoch()
        assert stream.epoch == i
    assert not stream.increment_epoch(n)
    assert stream.epoch == n

    with raises(AssertionError):
        stream.increment_epoch(10.0)
    for _ in range(1000):
        assert stream.increment_epoch(math.inf)


@mark.parametrize('n', range(1, 5))
def test_next_iteration(n):
    stream = Stream(range(n))
    assert stream.iteration == 0

    for epoch in range(2):
        for x in range(n):
            assert stream.next() == x
            assert stream.iteration == x + epoch * n + 1

    stream = Stream(iter(range(n)))
    for _ in range(n):
        stream.next()
    with raises(StopIteration):
        stream.next()
    with raises(StopIteration):
        stream.next()


@mark.parametrize('n', range(1, 6))
def test_data(n):
    # iterables that are not iterators
    n_epoches = 10
    stream = Stream(range(n))
    assert [list(stream.data(0)) for _ in range(n_epoches)] == [[]] * n_epoches

    for epoch_size in [None] + list(range(1, 2 * n)):
        effective_epoch_size = n if epoch_size is None else epoch_size
        max_iteration = n_epoches * effective_epoch_size
        stream = Stream(range(n))
        actual = []
        while stream.iteration < max_iteration:
            actual.append(list(stream.data(epoch_size)))
        assert stream.iteration == max_iteration
        flat_correct = [x % n for x in range(max_iteration)]
        correct = [
            flat_correct[i * effective_epoch_size : (i + 1) * effective_epoch_size]
            for i in range(n_epoches)
        ]
        assert actual == correct

    # count=inf
    stream = Stream(range(n))
    assert [x for _, x in zip(range(1000), stream.data(math.inf))] == [
        x % n for x in range(1000)
    ]

    # infinite iterators
    stream = Stream(itertools.count())
    with raises(ValueError):
        stream.data()
    n_epoches = 10
    for epoch_size in range(1, n):
        stream = Stream(itertools.count())
        actual = [list(stream.data(epoch_size)) for _ in range(n_epoches)]
        flat_correct = list(range(n_epoches * epoch_size))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size]
            for i in range(n_epoches)
        ]
        assert actual == correct

    # finite iterators
    stream = Stream(iter(range(n)))
    with raises(ValueError):
        stream.data()
    n_epoches = 10
    for epoch_size in range(1, 2 * n):
        stream = Stream(iter(range(n)))
        actual = []
        for _ in range(n_epoches):
            actual.append(list(stream.data(epoch_size)))
        flat_correct = list(range(n))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size]
            for i in range(n_epoches)
        ]
        assert actual == correct

    # bad input
    with raises(AssertionError):
        stream.data(10.0)


def test_reload_iterator():
    n = 10
    stream = Stream(range(n))
    for x in range(n):
        assert stream.next() == 0
        stream.reload_iterator()

    stream = Stream(iter(range(n)))
    for x in range(n):
        assert stream.next() == x
        stream.reload_iterator()
    with raises(StopIteration):
        stream.next()


def test_set_loader():
    a = itertools.repeat(0)
    b = itertools.repeat(1)
    stream = Stream([2])
    stream.set_loader(a)
    for x in range(10):
        assert stream.next() == next(b if x % 2 else a)
        stream.set_loader(a if x % 2 else b)
