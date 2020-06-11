import torch
from pytest import mark, raises
from torch.utils.data import DataLoader, TensorDataset

from zero.data import Enumerate, NamedTensorDataset, iloader, iter_batches

from .util import Point


def test_named_tensor_dataset():
    a = torch.arange(3)
    b = torch.arange(3)
    c = torch.arange(4)
    with raises(AssertionError):
        NamedTensorDataset(names=[])
    with raises(AssertionError):
        NamedTensorDataset(a, b, names=('a',))
    with raises(AssertionError):
        NamedTensorDataset(torch.tensor(0), names=('a',))
    with raises(AssertionError):
        NamedTensorDataset(a, c, names=('a', 'c'))

    for x in [
        NamedTensorDataset(a, b, names=('a', 'b')),
        NamedTensorDataset.from_dict({'a': a, 'b': b}),
    ]:
        assert x.names == ('a', 'b')
        assert x.a is a and x.b is b
        assert len(x) == 3
        assert x[1] == x._tuple_cls(torch.tensor(1), torch.tensor(1))
        for correct, actual in zip(
            zip(['a', 'b'], [a, b]), x.tensors._asdict().items()
        ):
            assert actual[0] == correct[0] and torch.equal(actual[1], correct[1])
        with raises(AssertionError):
            x.a = 0


def test_enumerate():
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    x = Enumerate(dataset)
    assert x.dataset is dataset
    assert len(x) == 10
    assert x[3] == (3, (torch.tensor(3), torch.tensor(3)))


def test_iloader():
    with raises(AssertionError):
        iloader(0)

    for x in range(1, 10):
        assert len(iloader(x)) == x

    data = torch.arange(10)
    for batch_size in range(1, len(data) + 1):
        torch.manual_seed(batch_size)
        correct = list(DataLoader(data, batch_size, shuffle=True, drop_last=True))
        torch.manual_seed(batch_size)
        actual = list(iloader(len(data), batch_size, shuffle=True, drop_last=True))
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
    batches = list(iter_batches(data, batch_size))
    check(batches, data)

    data = Point(torch.arange(n), torch.arange(n))
    batches = list(iter_batches(data, batch_size))
    assert all(isinstance(t, Point) for t in batches)
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = (torch.arange(n), torch.arange(n))
    batches = list(iter_batches(data, batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = {'a': torch.arange(n), 'b': torch.arange(n)}
    batches = list(iter_batches(data, batch_size))
    for key in data:
        batches_i = list(x[key] for x in batches)
        check(batches_i, data[key])

    data = (torch.arange(n), torch.arange(n))
    batches = list(iter_batches(TensorDataset(*data), batch_size))
    for i in range(2):
        batches_i = list(x[i] for x in batches)
        check(batches_i, data[i])

    data = {'a': torch.arange(n), 'b': torch.arange(n)}
    batches = list(iter_batches(NamedTensorDataset.from_dict(data), batch_size))
    for key in data:
        batches_i = list(getattr(x, key) for x in batches)
        check(batches_i, data[key])

    # test DataLoader kwargs
    data = torch.arange(n)
    kwargs = {'shuffle': True, 'drop_last': True}
    torch.manual_seed(0)
    correct_batches = torch.cat(list(DataLoader(data, batch_size, **kwargs)))
    torch.manual_seed(0)
    actual_batches = torch.cat(list(iter_batches(data, batch_size, **kwargs)))
    assert torch.equal(actual_batches, correct_batches)


def test_iter_batches_bad_input():
    with raises(AssertionError):
        iter_batches((), 1)
    with raises(AssertionError):
        iter_batches({}, 1)
    with raises(AssertionError):
        iter_batches(torch.empty(0), 1)
