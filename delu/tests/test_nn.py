import torch
from pytest import raises

import delu


def test_lambda():
    m = delu.nn.Lambda(torch.square)
    assert torch.allclose(m(torch.tensor(3.0)), torch.tensor(9.0))

    m = delu.nn.Lambda(torch.squeeze)
    assert m(torch.zeros(2, 1, 3, 1)).shape == (2, 3)

    m = delu.nn.Lambda(torch.squeeze, dim=1)
    assert m(torch.zeros(2, 1, 3, 1)).shape == (2, 3, 1)

    with raises(ValueError):
        delu.nn.Lambda(lambda x: torch.square(x))()

    with raises(ValueError):
        delu.nn.Lambda(torch.mul, other=torch.tensor(2.0))()
