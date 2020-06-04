import numpy as np
import torch as tr
from pytest import raises

from zero.map_concat import concat, zmap


def test_concat():
    a = [0, 1, 2]
    b = list(map(float, range(3)))
    assert concat(a) == [0, 1, 2]
    assert concat(zip(a, b)) == (a, b)
    assert concat({'a': x, 'b': y} for x, y in zip(a, b)) == {'a': a, 'b': b}

    for container, equal in (np.array, np.array_equal), (tr.tensor, tr.equal):
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
    d0 = tr.tensor([[0, 0]])
    a1 = 1
    b1 = [1, 1]
    c1 = np.array([1, 1])
    d1 = tr.tensor([[1, 1]])
    a_correct = [0, 1]
    b_correct = [0, 0, 1, 1]
    c_correct = np.array([0, 0, 1, 1])
    d_correct = tr.tensor([[0, 0], [1, 1]])

    def assert_correct(actual, keys):
        assert actual[keys[0]] == a_correct
        assert actual[keys[1]] == b_correct
        assert np.array_equal(actual[keys[2]], c_correct)
        assert tr.equal(actual[keys[3]], d_correct)

    data = [(a0, b0, c0, d0), (a1, b1, c1, d1)]
    actual = concat(data)
    assert_correct(actual, list(range(4)))
    data = [{'a': a0, 'b': b0, 'c': c0, 'd': d0}, {'a': a1, 'b': b1, 'c': c1, 'd': d1}]
    actual = concat(data)
    assert list(actual) == ['a', 'b', 'c', 'd']
    assert_correct(actual, ['a', 'b', 'c', 'd'])

    with raises(AssertionError):
        concat([])
    with raises(ValueError):
        concat(['a', 0])


def test_zmap():
    assert concat(zmap(lambda x: x * 2, range(3))) == [0, 2, 4]

    # fmt: off
    actual = concat(zmap(
        lambda x: (x, [x], [[x]], np.array([x]), tr.tensor([[x]])), range(2)
    ))
    correct = ([0, 1], [0, 1], [[0], [1]], np.array([0, 1]), tr.tensor([[0], [1]]))
    assert actual[0] == correct[0]
    assert actual[1] == correct[1]
    assert actual[2] == correct[2]
    assert np.array_equal(actual[3], correct[3])
    assert tr.equal(actual[4], correct[4])

    assert (
        concat(zmap(lambda a, b: a + b, zip(range(3), range(3)), star=True)) == [0, 2, 4]
    )
    # fmt: on

    model = tr.nn.Linear(3, 1)
    model.weight.requires_grad = False
    model.bias.requires_grad = False
    model.weight.zero_()
    model.bias.zero_()
    dataset = tr.utils.data.TensorDataset(tr.randn(5, 3))
    loader = tr.utils.data.DataLoader(dataset, 2)
    assert tr.equal(
        concat(zmap(model, loader, star=True)),
        tr.tensor([[0.0], [0.0], [0.0], [0.0], [0.0]]),
    )
