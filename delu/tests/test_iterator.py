import itertools

from pytest import mark, raises

from delu import Iterator


def test_properties():
    data = [0]
    dataiter = Iterator(data)
    assert dataiter.source is data
    with raises(AttributeError):
        dataiter.source = []


def test_bad_data():
    with raises(AssertionError):
        Iterator([])


def test_set_data():
    a = itertools.repeat(0)
    b = itertools.repeat(1)
    dataiter = Iterator([2])
    dataiter.set_source(a)
    for x in range(10):
        assert dataiter.next() == next(b if x % 2 else a)
        dataiter.set_source(a if x % 2 else b)


def test_reload_iterator():
    n = 10
    dataiter = Iterator(range(n))
    for x in range(n):
        assert dataiter.next() == 0
        dataiter.reload_iterator()

    dataiter = Iterator(iter(range(n)))
    for x in range(n):
        assert dataiter.next() == x
        dataiter.reload_iterator()
    with raises(StopIteration):
        dataiter.next()


@mark.parametrize('n', range(1, 5))
def test_next(n):
    dataiter = Iterator(range(n))
    for epoch in range(2):
        del epoch
        for x in range(n):
            assert dataiter.next() == x

    dataiter = Iterator(iter(range(n)))
    for _ in range(n):
        dataiter.next()
    with raises(StopIteration):
        dataiter.next()
    with raises(StopIteration):
        dataiter.next()


@mark.parametrize('n', range(1, 6))
def test_next_n(n):
    # iterables that are not iterators
    n_epochs = 10
    dataiter = Iterator(range(n))
    assert [list(dataiter.next_n(0)) for _ in range(n_epochs)] == [[]] * n_epochs

    for epoch_size in range(1, 2 * n):
        dataiter = Iterator(range(n))
        n_items = n_epochs * epoch_size
        actual = [list(dataiter.next_n(epoch_size)) for _ in range(n_epochs)]
        assert n_items == sum(map(len, actual))
        flat_correct = [x % n for x in range(sum(map(len, actual)))]
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # count=inf
    dataiter = Iterator(range(n))
    assert [x for _, x in zip(range(1000), dataiter.next_n('inf'))] == [
        x % n for x in range(1000)
    ]

    # infinite iterators
    dataiter = Iterator(itertools.count())
    n_epochs = 10
    for epoch_size in range(1, n):
        dataiter = Iterator(itertools.count())
        actual = [list(dataiter.next_n(epoch_size)) for _ in range(n_epochs)]
        flat_correct = list(range(n_epochs * epoch_size))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct

    # finite iterators
    dataiter = Iterator(iter(range(n)))
    n_epochs = 10
    for epoch_size in range(1, 2 * n):
        dataiter = Iterator(iter(range(n)))
        actual = []
        for _ in range(n_epochs):
            actual.append(list(dataiter.next_n(epoch_size)))
        flat_correct = list(range(n))
        correct = [
            flat_correct[i * epoch_size : (i + 1) * epoch_size] for i in range(n_epochs)
        ]
        assert actual == correct
