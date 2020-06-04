import numpy as np
import torch as tr
from pytest import mark, raises
from torch.utils.data import DataLoader, TensorDataset

from zero.data import Enumerate, NamedTensorDataset, iloader, iter_batches


def test_named_tensor_dataset():
    a = tr.arange(3)
    b = tr.arange(3)
    c = tr.arange(4)
    with raises(AssertionError):
        NamedTensorDataset(names=[])
    with raises(AssertionError):
        NamedTensorDataset(a, b, names=('a',))
    with raises(AssertionError):
        NamedTensorDataset(tr.tensor(0), names=('a',))
    with raises(AssertionError):
        NamedTensorDataset(a, c, names=('a', 'c'))

    for x in [
        NamedTensorDataset(a, b, names=('a', 'b')),
        NamedTensorDataset.from_dict({'a': a, 'b': b})
    ]:
        assert x.names == ('a', 'b')
        assert x.a is a and x.b is b
        assert len(x) == 3
        assert x[1] == x._tuple_cls(tr.tensor(1), tr.tensor(1))
        for correct, actual in zip(zip(['a', 'b'], [a, b]), x.tensors._asdict().items()):
            assert actual[0] == correct[0] and tr.equal(actual[1], correct[1])
        with raises(AssertionError):
            x.a = 0


def test_enumerate():
    dataset = TensorDataset(tr.arange(10), tr.arange(10))
    x = Enumerate(dataset)
    assert x.dataset is dataset
    assert len(x) == 10
    assert x[3] == (3, (tr.tensor(3), tr.tensor(3)))


def test_iloader():
    with raises(AssertionError):
        iloader(0)

    for x in range(1, 10):
        assert len(iloader(x)) == x

    data = tr.arange(10)
    for batch_size in range(1, len(data) + 1):
        tr.manual_seed(batch_size)
        correct = list(DataLoader(data, batch_size, shuffle=True, drop_last=True))
        tr.manual_seed(batch_size)
        actual = list(iloader(len(data), batch_size, shuffle=True, drop_last=True))
        for x, y in zip(actual, correct):
            assert tr.equal(x, y)


@mark.parametrize('batch_size', list(range(1, 11)))
def test_iter_batches(batch_size):
    # test batch size only
    x = np.arange(10)
    assert np.array_equal(np.hstack(tuple(iter_batches(x, batch_size))), x)

    x = tr.arange(10)
    assert tr.equal(tr.cat(tuple(iter_batches(x, batch_size))), x)

    x = (np.arange(10), np.arange(10))
    batches = tuple(iter_batches(x, batch_size))
    batches = [np.hstack(tuple(x[i] for x in batches)) for i in range(2)]
    assert np.array_equal(batches[0], x[0]) and np.array_equal(batches[1], x[1])

    x = (tr.arange(10), tr.arange(10))
    batches = tuple(iter_batches(x, batch_size))
    batches = [tr.cat(tuple(x[i] for x in batches)) for i in range(2)]
    assert tr.equal(batches[0], x[0]) and tr.equal(batches[1], x[1])

    batches = tuple(iter_batches(TensorDataset(*x), batch_size))
    batches = [tr.cat(tuple(x[i] for x in batches)) for i in range(2)]
    assert tr.equal(batches[0], x[0]) and tr.equal(batches[1], x[1])

    names = ('a', 'b')
    batches = tuple(iter_batches(NamedTensorDataset(*x, names=names), batch_size))
    batches = [tr.cat(tuple(getattr(x, n) for x in batches)) for n in names]
    assert tr.equal(batches[0], x[0]) and tr.equal(batches[1], x[1])

    # test DataLoader kwargs
    x = np.arange(10)
    kwargs = {'shuffle': True, 'drop_last': True}
    tr.manual_seed(0)
    correct_batches = np.hstack(tuple(DataLoader(x, batch_size, **kwargs)))
    tr.manual_seed(0)
    actual_batches = np.hstack(tuple(iter_batches(x, batch_size, **kwargs)))
    assert np.array_equal(actual_batches, correct_batches)
