import dataclasses
from collections.abc import Mapping, Sequence
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

import delu

from .util import Point


def flatten(data):
    if isinstance(data, torch.Tensor):
        yield data
    elif isinstance(data, (str, bytes)):
        # mypy: NaN
        yield data  # type: ignore
    elif isinstance(data, Sequence):
        for x in data:
            yield from flatten(x)
    elif isinstance(data, Mapping):
        for x in data.values():
            yield from flatten(x)
    elif isinstance(data, SimpleNamespace):
        for x in vars(data).values():
            yield from flatten(x)
    elif dataclasses.is_dataclass(data):
        for x in vars(data).values():
            yield from flatten(x)
    else:
        yield data


def test_to():
    with pytest.raises(ValueError):
        delu.to(None)
    with pytest.raises(ValueError):
        delu.to([None, None])

    t = lambda x: torch.tensor(0, dtype=x)  # noqa
    f32 = torch.float32
    i64 = torch.int64

    for dtype in f32, i64:
        x = t(dtype)
        assert delu.to(x, dtype) is x
    assert delu.to(t(f32), i64).dtype is i64

    for Container in tuple, Point, list:
        constructor = Container._make if Container is Point else Container
        for dtype in [f32, i64]:
            x = constructor([t(f32), t(f32)])
            out = delu.to(x, dtype)
            assert isinstance(out, Container)
            assert all(x.dtype is dtype for x in out)
            if dtype is f32:
                for x, y in zip(out, x):
                    assert x is y

    data = [t(f32), t(f32)]
    for x, y in zip(delu.to(data, f32), data):
        assert x is y
    assert all(x.dtype is i64 for x in delu.to(data, i64))

    @dataclasses.dataclass
    class A:
        a: torch.Tensor

    data = {
        'a': [t(f32), (t(f32), t(f32))],
        'b': {'c': {'d': [[[t(f32)]]]}},
        'c': Point(t(f32), {'d': t(f32)}),
        'f': SimpleNamespace(g=t(f32), h=A(t(f32))),
    }
    for x, y in zip(flatten(delu.to(data, f32)), flatten(data)):
        assert x is y
    for x, y in zip(flatten(delu.to(data, i64)), flatten(data)):
        assert x.dtype is i64
        assert type(x) is type(y)


def test_concat():
    a = [0, 1, 2]
    b = list(map(float, range(3)))
    assert delu.concat(a) == [0, 1, 2]
    assert delu.concat(zip(a, b)) == (a, b)
    correct = Point(a, b)
    actual = delu.concat(map(Point._make, zip(a, b)))
    assert isinstance(actual, Point) and actual == correct
    assert delu.concat({'a': x, 'b': y} for x, y in zip(a, b)) == {'a': a, 'b': b}

    for container, equal in (np.array, np.array_equal), (torch.tensor, torch.equal):
        a = [container([0, 1]), container([2, 3])]
        b = [container([[0, 1]]), container([[2, 3]])]
        a_correct = container([0, 1, 2, 3])
        b_correct = container([[0, 1], [2, 3]])
        assert equal(delu.concat(a), a_correct)
        actual = delu.concat(zip(a, b))
        assert isinstance(actual, tuple) and len(actual) == 2
        assert equal(actual[0], a_correct) and equal(actual[1], b_correct)
        actual = delu.concat([{'a': a[0], 'b': b[0]}, {'a': a[1], 'b': b[1]}])
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
    actual = delu.concat(data)
    assert_correct(actual, list(range(4)))
    data = [{'a': a0, 'b': b0, 'c': c0, 'd': d0}, {'a': a1, 'b': b1, 'c': c1, 'd': d1}]
    actual = delu.concat(data)
    assert list(actual) == ['a', 'b', 'c', 'd']
    assert_correct(actual, ['a', 'b', 'c', 'd'])

    data = ['a', 0, (1, 2), {'1', '2'}]
    assert delu.concat(data) is data

    with pytest.raises(AssertionError):
        delu.concat([])


@pytest.mark.parametrize('batch_size', list(range(1, 11)))
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
    batches = list(delu.iter_batches(data, batch_size))
    check(batches, data)

    data = Point(torch.arange(n), torch.arange(n))
    batches = list(delu.iter_batches(data, batch_size))
    assert all(isinstance(t, Point) for t in batches)
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = (torch.arange(n), torch.arange(n))
    batches = list(delu.iter_batches(data, batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = {'a': torch.arange(n), 'b': torch.arange(n)}
    batches = list(delu.iter_batches(data, batch_size))
    for key in data:
        batches_i = list(x[key] for x in batches)
        check(batches_i, data[key])

    data = (torch.arange(n), torch.arange(n))
    batches = list(delu.iter_batches(TensorDataset(*data), batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    # test DataLoader kwargs
    data = torch.arange(n)
    kwargs = {'shuffle': True, 'drop_last': True}
    torch.manual_seed(0)
    correct_batches = torch.cat(list(DataLoader(data, batch_size, **kwargs)))
    torch.manual_seed(0)
    actual_batches = torch.cat(list(delu.iter_batches(data, batch_size, **kwargs)))
    assert torch.equal(actual_batches, correct_batches)


def test_iter_batches_bad_input():
    with pytest.raises(AssertionError):
        delu.iter_batches((), 1)
    with pytest.raises(AssertionError):
        delu.iter_batches({}, 1)
    with pytest.raises(ValueError):
        delu.iter_batches(torch.empty(0), 1)
