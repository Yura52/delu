import numpy as np
import torch
from pytest import mark, raises

from zero.concat_dmap import concat, dmap

from .util import Point


def test_concat():
    a = [0, 1, 2]
    b = list(map(float, range(3)))
    assert concat(a) == [0, 1, 2]
    assert concat(zip(a, b)) == (a, b)
    correct = Point(a, b)
    actual = concat(map(Point._make, zip(a, b)))
    assert isinstance(actual, Point) and actual == correct
    assert concat({'a': x, 'b': y} for x, y in zip(a, b)) == {'a': a, 'b': b}

    for container, equal in (np.array, np.array_equal), (torch.tensor, torch.equal):
        a = [container([0, 1]), container([2, 3])]
        b = [container([[0, 1]]), container([[2, 3]])]
        a_correct = container([0, 1, 2, 3])
        b_correct = container([[0, 1], [2, 3]])
        assert equal(concat(a), a_correct)
        actual = concat(zip(a, b))
        assert isinstance(actual, tuple) and len(actual) == 2
        assert equal(actual[0], a_correct) and equal(actual[1], b_correct)
        actual = concat([{'a': a[0], 'b': b[0]}, {'a': a[1], 'b': b[1]}])
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
    actual = concat(data)
    assert_correct(actual, list(range(4)))
    data = [{'a': a0, 'b': b0, 'c': c0, 'd': d0}, {'a': a1, 'b': b1, 'c': c1, 'd': d1}]
    actual = concat(data)
    assert list(actual) == ['a', 'b', 'c', 'd']
    assert_correct(actual, ['a', 'b', 'c', 'd'])

    data = ['a', 0, (1, 2), {'1', '2'}]
    assert concat(data) is data

    with raises(AssertionError):
        concat([])


def test_dmap_correctness():
    assert concat(dmap(lambda x: x * 2, range(3))) == [0, 2, 4]

    actual = concat(
        dmap(lambda x: (x, [x], [[x]], np.array([x]), torch.tensor([[x]])), range(2))
    )
    correct = ([0, 1], [0, 1], [[0], [1]], np.array([0, 1]), torch.tensor([[0], [1]]))
    assert actual[0] == correct[0]
    assert actual[1] == correct[1]
    assert actual[2] == correct[2]
    assert np.array_equal(actual[3], correct[3])
    assert torch.equal(actual[4], correct[4])

    assert concat(dmap(lambda a, b: a + b, zip(range(3), range(3)), star=True)) == [
        0,
        2,
        4,
    ]


_devices = ['cpu', 'cuda'] if torch.cuda.is_available() else ['cpu']


@mark.parametrize('in_device', _devices)
@mark.parametrize('out_device', _devices)
def test_dmap_devices(in_device, out_device):
    model = torch.nn.Linear(3, 1)
    model.to(in_device)
    model.weight.requires_grad = False
    model.bias.requires_grad = False
    model.weight.zero_()
    model.bias.zero_()
    dataset = torch.utils.data.TensorDataset(torch.randn(5, 3))
    loader = torch.utils.data.DataLoader(dataset, 2)
    actual = concat(
        dmap(model, loader, in_device=in_device, out_device=out_device, star=True)
    )
    correct = torch.tensor([[0.0], [0.0], [0.0], [0.0], [0.0]], device=out_device)
    assert actual.device.type == out_device
    assert torch.equal(actual, correct)
