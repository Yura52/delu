import numpy as np
import torch
from pytest import mark, raises
from torch.utils.data import DataLoader, TensorDataset

import zero.data as zd

from .util import Point


def test_enumerate():
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    x = zd.Enumerate(dataset)
    assert x.dataset is dataset
    assert len(x) == 10
    assert x[3] == (3, (torch.tensor(3), torch.tensor(3)))


def test_fndataset():
    dataset = zd.FnDataset(lambda x: x * 2, 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 2
    assert dataset[2] == 4

    dataset = zd.FnDataset(lambda x: x * 2, 3, lambda x: x * 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 6
    assert dataset[2] == 12

    dataset = zd.FnDataset(lambda x: x * 2, [1, 10, 100])
    assert len(dataset) == 3
    assert dataset[0] == 2
    assert dataset[1] == 20
    assert dataset[2] == 200

    dataset = zd.FnDataset(lambda x: x * 2, (x for x in range(0, 10, 4)))
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 8
    assert dataset[2] == 16


def test_iloader():
    with raises(AssertionError):
        zd.IndexLoader(0)

    for x in range(1, 10):
        assert len(zd.IndexLoader(x)) == x

    data = torch.arange(10)
    for batch_size in range(1, len(data) + 1):
        torch.manual_seed(batch_size)
        correct = list(DataLoader(data, batch_size, shuffle=True, drop_last=True))
        torch.manual_seed(batch_size)
        actual = list(
            zd.IndexLoader(len(data), batch_size, shuffle=True, drop_last=True)
        )
        for x, y in zip(actual, correct):
            assert torch.equal(x, y)


@mark.parametrize('batch_size', list(range(1, 11)))
def test_iter_batches(batch_size):
    def check(batches, correct):
        sizes = list(map(len, batches))
        assert sum(sizes) == len(correct)
        assert (
            set(sizes) == {batch_size}
            or set(sizes[:-1]) == {batch_size}
            and sizes[-1] == len(correct) % batch_size
        )
        assert torch.equal(torch.cat(batches), correct)

    n = 10

    # test batch size only
    data = torch.arange(n)
    batches = list(zd.iter_batches(data, batch_size))
    check(batches, data)

    data = Point(torch.arange(n), torch.arange(n))
    batches = list(zd.iter_batches(data, batch_size))
    assert all(isinstance(t, Point) for t in batches)
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = (torch.arange(n), torch.arange(n))
    batches = list(zd.iter_batches(data, batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = {'a': torch.arange(n), 'b': torch.arange(n)}
    batches = list(zd.iter_batches(data, batch_size))
    for key in data:
        batches_i = list(x[key] for x in batches)
        check(batches_i, data[key])

    data = (torch.arange(n), torch.arange(n))
    batches = list(zd.iter_batches(TensorDataset(*data), batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    # test DataLoader kwargs
    data = torch.arange(n)
    kwargs = {'shuffle': True, 'drop_last': True}
    torch.manual_seed(0)
    correct_batches = torch.cat(list(DataLoader(data, batch_size, **kwargs)))
    torch.manual_seed(0)
    actual_batches = torch.cat(list(zd.iter_batches(data, batch_size, **kwargs)))
    assert torch.equal(actual_batches, correct_batches)


def test_iter_batches_bad_input():
    with raises(AssertionError):
        zd.iter_batches((), 1)
    with raises(AssertionError):
        zd.iter_batches({}, 1)
    with raises(AssertionError):
        zd.iter_batches(torch.empty(0), 1)


def test_concat():
    a = [0, 1, 2]
    b = list(map(float, range(3)))
    assert zd.concat(a) == [0, 1, 2]
    assert zd.concat(zip(a, b)) == (a, b)
    correct = Point(a, b)
    actual = zd.concat(map(Point._make, zip(a, b)))
    assert isinstance(actual, Point) and actual == correct
    assert zd.concat({'a': x, 'b': y} for x, y in zip(a, b)) == {'a': a, 'b': b}

    for container, equal in (np.array, np.array_equal), (torch.tensor, torch.equal):
        a = [container([0, 1]), container([2, 3])]
        b = [container([[0, 1]]), container([[2, 3]])]
        a_correct = container([0, 1, 2, 3])
        b_correct = container([[0, 1], [2, 3]])
        assert equal(zd.concat(a), a_correct)
        actual = zd.concat(zip(a, b))
        assert isinstance(actual, tuple) and len(actual) == 2
        assert equal(actual[0], a_correct) and equal(actual[1], b_correct)
        actual = zd.concat([{'a': a[0], 'b': b[0]}, {'a': a[1], 'b': b[1]}])
        assert list(actual) == ['a', 'b']
        assert equal(actual['a'], a_correct) and equal(actual['b'], b_correct)

    a0 = 0
    b0 = [0, 0]
    c0 = np.array([0, 0])
    d0 = torch.tensor([[0, 0]])
    a1 = 1
    b1 = [1, 1]
    c1 = np.array([1, 1])
    d1 = torch.tensor([[1, 1]])
    a_correct = [0, 1]
    b_correct = [0, 0, 1, 1]
    c_correct = np.array([0, 0, 1, 1])
    d_correct = torch.tensor([[0, 0], [1, 1]])

    def assert_correct(actual, keys):
        assert actual[keys[0]] == a_correct
        assert actual[keys[1]] == b_correct
        assert np.array_equal(actual[keys[2]], c_correct)
        assert torch.equal(actual[keys[3]], d_correct)

    data = [(a0, b0, c0, d0), (a1, b1, c1, d1)]
    actual = zd.concat(data)
    assert_correct(actual, list(range(4)))
    data = [{'a': a0, 'b': b0, 'c': c0, 'd': d0}, {'a': a1, 'b': b1, 'c': c1, 'd': d1}]
    actual = zd.concat(data)
    assert list(actual) == ['a', 'b', 'c', 'd']
    assert_correct(actual, ['a', 'b', 'c', 'd'])

    data = ['a', 0, (1, 2), {'1', '2'}]
    assert zd.concat(data) is data

    with raises(AssertionError):
        zd.concat([])


def test_collate():
    # just test that the function is still a valid alias
    assert torch.equal(zd.collate([1])[0], torch.tensor(1))
