import itertools

from pytest import mark, raises

from delu.data import Stream


def test_properties():
    data = [0]
    stream = Stream(data)
    assert stream.data is data
    with raises(AttributeError):
        stream.data = []


def test_bad_data():
    with raises(AssertionError):
        Stream([])


def test_set_data():
    a = itertools.repeat(0)
    b = itertools.repeat(1)
    stream = Stream([2])
    stream.set_data(a)
    for x in range(10):
        assert stream.next() == next(b if x % 2 else a)
        stream.set_data(a if x % 2 else b)


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


@mark.parametrize('n', range(1, 5))
def test_next(n):
    stream = Stream(range(n))
    for epoch in range(2):
        for x in range(n):
            assert stream.next() == x

    stream = Stream(iter(range(n)))
    for _ in range(n):
        stream.next()
    with raises(StopIteration):
        stream.next()
    with raises(StopIteration):
        stream.next()


@mark.parametrize('n', range(1, 6))
def test_next_n(n):
    # iterables that are not iterators
    n_epochs = 10
    stream = Stream(range(n))
    assert [list(stream.next_n(0)) for _ in range(n_epochs)] == [[]] * n_epochs

    for epoch_size in range(1, 2 * n):
        stream = Stream(range(n))
        n_items = n_epochs * epoch_size
        actual = [list(stream.next_n(epoch_size)) for _ in range(n_epochs)]
        assert n_items == sum(map(len, actual))
        flat_correct = [x % n for x in range(sum(map(len, actual)))]
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # count=inf
    stream = Stream(range(n))
    assert [x for _, x in zip(range(1000), stream.next_n('inf'))] == [
        x % n for x in range(1000)
    ]

    # infinite iterators
    stream = Stream(itertools.count())
    n_epochs = 10
    for epoch_size in range(1, n):
        stream = Stream(itertools.count())
        actual = [list(stream.next_n(epoch_size)) for _ in range(n_epochs)]
        flat_correct = list(range(n_epochs * epoch_size))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # finite iterators
    stream = Stream(iter(range(n)))
    n_epochs = 10
    for epoch_size in range(1, 2 * n):
        stream = Stream(iter(range(n)))
        actual = []
        for _ in range(n_epochs):
            actual.append(list(stream.next_n(epoch_size)))
        flat_correct = list(range(n))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # bad input
    with raises(AssertionError):
        stream.next_n('inf ')
