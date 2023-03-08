import torch
from torch.utils.data import TensorDataset

import delu.data as dd


def test_enumerate():
    dataset = TensorDataset(torch.arange(10), torch.arange(10))
    x = dd.Enumerate(dataset)
    assert x.dataset is dataset
    assert len(x) == 10
    assert x[3] == (3, (torch.tensor(3), torch.tensor(3)))


def test_fndataset():
    dataset = dd.FnDataset(lambda x: x * 2, 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 2
    assert dataset[2] == 4

    dataset = dd.FnDataset(lambda x: x * 2, 3, lambda x: x * 3)
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 6
    assert dataset[2] == 12

    dataset = dd.FnDataset(lambda x: x * 2, [1, 10, 100])
    assert len(dataset) == 3
    assert dataset[0] == 2
    assert dataset[1] == 20
    assert dataset[2] == 200

    dataset = dd.FnDataset(lambda x: x * 2, (x for x in range(0, 10, 4)))
    assert len(dataset) == 3
    assert dataset[0] == 0
    assert dataset[1] == 8
    assert dataset[2] == 16
